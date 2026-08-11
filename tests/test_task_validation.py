"""
End-to-end validation of the TaskState + completion enforcement fix.

Eight scenarios prove Ultron no longer stops after completing only the first
action of a multi-step task:

  S1  single-step task still works normally
  S2  multi-step file task does not stop after the write
  S3  todo-app-style task treats ``mkdir`` as an intermediate action
  S4  confirmation interruption preserves the full task state
  S5  a failed intermediate action is recorded — never success
  S6  a model falsely claiming completion is overruled by TaskState
  S7  max_iterations bounds execution and reports incomplete
  S8  the security boundary / guardrails are not bypassed

The tests script a FakeEngine (no real LLM) and drive the same lifecycle
main.py uses: pending action -> execute -> feed observation back -> the agent
continues -> verification -> only then task completes. Real tools (mkdir,
write_file, read_file) run inside a temp sandbox.
"""

import asyncio
import json

import pytest

from ultron.core.agents.react import ReActAgent
from ultron.core.tools import paths
from ultron.core.types import Role, TaskStatus
from ultron.main import (
    continue_task_after_confirmation,
    execute_pending_action,
)


class FakeEngine:
    """Scripted engine — no real LLM or network access needed."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    async def generate(self, messages, **kwargs):
        self.calls.append(messages)
        return self._responses.pop(0) if self._responses else ""

    async def stream(self, messages, **kwargs):
        yield ""


def _run(coro):
    return asyncio.run(coro)


def _tool(tool, **arguments):
    return f"```json\n{json.dumps({'tool': tool, 'arguments': arguments})}\n```"


def _done_json(*requirements):
    return json.dumps([{"description": d, "satisfied": s} for d, s in requirements])


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Points cwd + the file-policy allowlist at a temp dir so real tools are safe."""
    monkeypatch.setattr(paths, "ALLOWED_BASE_DIR", tmp_path)
    monkeypatch.chdir(tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# S1 — Simple single-step task
# ---------------------------------------------------------------------------


def test_s1_single_step_task_still_works(sandbox):
    engine = FakeEngine(
        [
            _tool("run_command", command="mkdir TestDir"),
            "Done, TestDir created.",
            _done_json(("TestDir directory exists", True)),
        ]
    )
    agent = ReActAgent(engine)
    msg = _run(agent.run("create a directory called TestDir"))

    # Task created; command gated as usual.
    assert msg.pending_action is not None
    task = msg.task_state
    assert task is not None
    assert task.goal == "create a directory called TestDir"

    # Command executes after approval; observation feeds back; verified.
    result = _run(execute_pending_action(msg.pending_action))
    assert "Exit code: 0" in result
    final = _run(continue_task_after_confirmation(agent, task, result, []))
    assert final.pending_action is None
    assert task.is_complete() is True
    assert task.status == TaskStatus.TASK_COMPLETED
    assert task.remaining_requirements() == []
    assert task.execution_history[0].target == "mkdir TestDir"
    assert (sandbox / "TestDir").is_dir()


# ---------------------------------------------------------------------------
# S2 — Multi-step file task
# ---------------------------------------------------------------------------


def test_s2_multistep_file_task_does_not_stop_after_write(sandbox):
    engine = FakeEngine(
        [
            _tool("write_file", filename="app.txt", content="hello"),
            _tool("read_file", file_path="app.txt"),
            "app.txt contains: hello",
            _done_json(("app.txt exists", True), ("content is hello", True)),
        ]
    )
    agent = ReActAgent(engine)
    msg = _run(
        agent.run("create a file app.txt, write hello into it, then read it")
    )
    task = msg.task_state

    # Step 1: write is confirm-gated → executes → observation → continue.
    assert msg.pending_action.action_type == "write_file"
    result = _run(execute_pending_action(msg.pending_action))
    assert "Successfully wrote" in result
    msg = _run(continue_task_after_confirmation(agent, task, result, []))

    # Step 2 (read) ran inside the loop; only then did the task complete.
    assert msg.pending_action is None
    assert task.is_complete() is True
    assert [e.tool_name for e in task.execution_history] == [
        "write_file",
        "read_file",
    ]
    assert (sandbox / "app.txt").read_text() == "hello"


# ---------------------------------------------------------------------------
# S3 — TodoList-style software task: mkdir is only an intermediate action
# ---------------------------------------------------------------------------


def test_s3_todolist_app_mkdir_is_only_intermediate(sandbox):
    engine = FakeEngine(
        [
            _tool("run_command", command="mkdir TodoList"),
            _tool("write_file", filename="TodoList/index.html", content="<html>todo</html>"),
            _tool("write_file", filename="TodoList/app.js", content="console.log('hi')"),
            "The application is complete.",
            _done_json(
                ("app directory exists", True),
                ("application files exist", True),
            ),
        ]
    )
    agent = ReActAgent(engine)
    msg = _run(
        agent.run("create a TodoList application in a separate folder TodoList")
    )
    task = msg.task_state

    # mkdir executes successfully — but the task must NOT be complete.
    result = _run(execute_pending_action(msg.pending_action))
    assert "Exit code: 0" in result
    assert task.is_complete() is False

    # The agent continues the original task rather than stopping.
    msg = _run(continue_task_after_confirmation(agent, task, result, []))
    assert msg.pending_action is not None
    assert msg.pending_action.action_type == "write_file"
    assert task.is_complete() is False

    result = _run(execute_pending_action(msg.pending_action))
    msg = _run(continue_task_after_confirmation(agent, task, result, []))
    assert msg.pending_action.action_type == "write_file"

    result = _run(execute_pending_action(msg.pending_action))
    msg = _run(continue_task_after_confirmation(agent, task, result, []))

    # Only after verification does the task complete.
    assert msg.pending_action is None
    assert task.is_complete() is True
    assert task.status == TaskStatus.TASK_COMPLETED
    assert task.remaining_requirements() == []
    assert [e.tool_name for e in task.execution_history] == [
        "run_command",
        "write_file",
        "write_file",
    ]
    assert (sandbox / "TodoList" / "app.js").exists()


# ---------------------------------------------------------------------------
# S4 — Confirmation interruption: everything survives the boundary
# ---------------------------------------------------------------------------


def test_s4_confirmation_interruption_preserves_task_state(sandbox):
    engine = FakeEngine(
        [
            _tool("run_command", command="mkdir TestDir"),
            _tool("write_file", filename="app.txt", content="x"),
            "Done.",
            _done_json(("TestDir exists", True), ("app.txt exists", True)),
        ]
    )
    agent = ReActAgent(engine)
    goal = "create a directory called TestDir and a file app.txt"
    msg = _run(agent.run(goal))
    task = msg.task_state

    # Before confirmation: goal, state, step, requirements, history.
    assert task.goal == goal
    assert task.status == TaskStatus.WAITING_CONFIRMATION
    assert task.current_step == 0
    assert task.requirements == []
    assert task.execution_history == []

    # Confirm + execute + resume: the observation/step is recorded.
    result = _run(execute_pending_action(msg.pending_action))
    msg = _run(continue_task_after_confirmation(agent, task, result, []))
    assert task.goal == goal
    assert task.current_step == 1
    assert len(task.execution_history) == 1
    assert task.execution_history[0].tool_name == "run_command"
    assert msg.pending_action.action_type == "write_file"

    # Second confirmation.
    result = _run(execute_pending_action(msg.pending_action))
    msg = _run(continue_task_after_confirmation(agent, task, result, []))

    # After verification: completed + remaining requirements consistent.
    assert msg.pending_action is None
    assert task.goal == goal
    assert task.status == TaskStatus.TASK_COMPLETED
    assert task.is_complete() is True
    assert task.current_step == 2
    assert len(task.completed_requirements) == 2
    assert task.remaining_requirements() == []
    assert len(task.execution_history) == 2


# ---------------------------------------------------------------------------
# S5 — Failed intermediate action
# ---------------------------------------------------------------------------


def test_s5_failed_intermediate_action_never_reports_success(sandbox):
    engine = FakeEngine(
        [
            _tool("run_command", command="mkdir TestDir"),
            "Done! TestDir created.",
            _done_json(("TestDir directory exists", False)),
            _tool("run_command", command="mkdir TestDir"),  # recovery attempt
            "Now it exists.",
            _done_json(("TestDir directory exists", True)),
        ]
    )
    agent = ReActAgent(engine)
    msg = _run(agent.run("create a directory called TestDir"))
    task = msg.task_state

    # First action FAILS at execution time.
    failure = "Error: mkdir: permission denied"
    msg = _run(continue_task_after_confirmation(agent, task, failure, []))

    # Failure recorded; the model's claim overruled; requirements preserved.
    assert task.execution_history[0].success is False
    assert task.is_complete() is False
    assert msg.pending_action is not None  # agent attempted recovery
    assert msg.pending_action.action_type == "run_command"
    assert [r.description for r in task.remaining_requirements()] == [
        "TestDir directory exists"
    ]

    # Recovery succeeds → verified completion.
    result = _run(execute_pending_action(msg.pending_action))
    final = _run(continue_task_after_confirmation(agent, task, result, []))
    assert final.pending_action is None
    assert task.is_complete() is True
    assert task.remaining_requirements() == []


# ---------------------------------------------------------------------------
# S6 — Model falsely claims completion (critical regression)
# ---------------------------------------------------------------------------


def test_s6_model_false_completion_claim_is_overruled(sandbox):
    engine = FakeEngine(
        [
            _tool("run_command", command="mkdir TodoList"),
            "Done! The application has been created.",  # false claim
            _done_json(
                ("TodoList directory exists", True),
                ("application files exist", False),
            ),
            _tool("write_file", filename="TodoList/app.js", content="console.log('x')"),
            "The application is now complete.",
            _done_json(
                ("TodoList directory exists", True),
                ("application files exist", True),
            ),
        ]
    )
    agent = ReActAgent(engine)
    msg = _run(
        agent.run("create a TodoList application in a separate folder TodoList")
    )
    task = msg.task_state

    # The false claim was NOT accepted — execution continued.
    msg = _run(continue_task_after_confirmation(agent, task, "Exit code: 0", []))
    assert task.is_complete() is False
    assert msg.pending_action is not None
    assert msg.pending_action.action_type == "write_file"
    assert [r.description for r in task.remaining_requirements()] == [
        "application files exist"
    ]

    # Once the remaining requirement is fulfilled, the task completes.
    result = _run(execute_pending_action(msg.pending_action))
    final = _run(continue_task_after_confirmation(agent, task, result, []))
    assert final.pending_action is None
    assert task.is_complete() is True
    assert task.remaining_requirements() == []


# ---------------------------------------------------------------------------
# S7 — Max iterations
# ---------------------------------------------------------------------------


def test_s7_max_iterations_bounds_and_reports_incomplete():
    never_done = json.dumps(
        [{"description": "application files exist", "satisfied": False}]
    )
    engine = FakeEngine(
        [
            _tool("run_command", command="mkdir TodoList"),
            "Almost there...",
            never_done,
            "Almost there...",
            never_done,
        ]
    )
    agent = ReActAgent(engine, max_iterations=2)
    msg = _run(
        agent.run("create a TodoList application in a separate folder TodoList")
    )
    task = msg.task_state
    final = _run(continue_task_after_confirmation(agent, task, "Exit code: 0", []))

    assert "incomplete" in final.content.lower()
    assert "remaining requirements" in final.content.lower()
    assert task.is_complete() is False
    assert task.status == TaskStatus.TASK_FAILED
    assert len(task.remaining_requirements()) == 1  # preserved
    assert len(engine.calls) <= 6  # bounded — no infinite loop


# ---------------------------------------------------------------------------
# S8 — Security boundary / guardrails are not bypassed
# ---------------------------------------------------------------------------


def test_s8_guardrail_secret_write_stays_blocked(sandbox):
    # A secret-bearing write is DENIED by the guardrails: never offered for
    # confirmation, never executed, no task is created.
    engine = FakeEngine(
        [
            _tool("write_file", filename="leak.txt", content="aws key AKIA1234567890ABCDEF"),
            "I could not write it.",
        ]
    )
    agent = ReActAgent(engine)
    msg = _run(agent.run("write a file with the key"))
    assert msg.pending_action is None  # never offered for confirmation
    assert msg.task_state is None  # a blocked action does not start a task
    assert not (sandbox / "leak.txt").exists()  # nothing written
    blocked = [m for m in engine.calls[1] if m.get("role") == "tool"]
    assert "Blocked by security" in blocked[0]["content"]


def test_s8_guardrail_path_escape_stays_blocked():
    engine = FakeEngine(
        [
            _tool("read_file", file_path="../../etc/passwd"),
            "Nothing to report.",
        ]
    )
    agent = ReActAgent(engine)
    msg = _run(agent.run("read the passwd file"))
    assert msg.pending_action is None
    blocked = [m for m in engine.calls[1] if m.get("role") == "tool"]
    assert "Blocked by security" in blocked[0]["content"]


def test_s8_confirm_gated_action_still_requires_approval(sandbox):
    # A state-changing action still flows through PendingAction — nothing runs
    # without the user's approval.
    engine = FakeEngine([_tool("run_command", command="mkdir TestDir")])
    agent = ReActAgent(engine)
    msg = _run(agent.run("create a directory called TestDir"))
    assert msg.pending_action is not None
    assert msg.pending_action.action_type == "run_command"
    assert not (sandbox / "TestDir").exists()  # nothing ran without approval


def test_s8_blocked_action_mid_task_does_not_execute(sandbox):
    # A guardrail denial mid-task feeds back as a blocked observation and the
    # action never executes — even with an active TaskState.
    engine = FakeEngine(
        [
            _tool("run_command", command="mkdir TestDir"),
            _tool("write_file", filename="leak.txt", content="aws key AKIA1234567890ABCDEF"),
            "The leak was blocked.",
        ]
    )
    agent = ReActAgent(engine)
    msg = _run(agent.run("create TestDir and write the key"))
    task = msg.task_state
    result = _run(execute_pending_action(msg.pending_action))
    msg = _run(continue_task_after_confirmation(agent, task, result, []))

    # The write was denied and never executed; the loop moved on.
    assert not (sandbox / "leak.txt").exists()
    assert msg.pending_action is None
    assert msg.task_state is not None  # task object still carried
    tool_msgs = [m for m in task.context if m.role == Role.TOOL]
    assert any("Blocked by security" in m.content for m in tool_msgs)
    # mkdir ran; the blocked write is recorded as a FAILED execution.
    assert [e.tool_name for e in task.execution_history] == ["run_command", "write_file"]
    assert task.execution_history[1].success is False
