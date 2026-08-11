"""
Integration tests for the TaskState-aware ReAct confirmation lifecycle.

Reproduces the original bug: "create a todo list application" must NOT be
considered complete after `mkdir TodoList` succeeds and is confirmed. The
tests script a FakeEngine (no real LLM) and drive the same lifecycle main.py
uses: pending action -> execute -> feed observation back -> agent continues ->
verification -> only then task completes.

The central invariants under test:
- a confirmed tool execution is an observation, never task completion
- the model cannot claim success while TaskState says work remains
- max_iterations never falsely reports success
"""

import asyncio
import json

import pytest

from ultron.core.agents.react import ReActAgent
from ultron.core.tools import paths
from ultron.core.tools import registry as reg
from ultron.core.types import PendingAction, Role, TaskStatus
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


def _tool_call(tool, **arguments):
    return f"```json\n{json.dumps({'tool': tool, 'arguments': arguments})}\n```"


def _mkdir():
    return _tool_call("run_command", command="mkdir TodoList")


def _write(path, content):
    return _tool_call("write_file", filename=path, content=content)


def _all_done():
    return json.dumps(
        [
            {"description": "TodoList directory exists", "satisfied": True},
            {"description": "application files exist", "satisfied": True},
            {"description": "application can run", "satisfied": True},
        ]
    )


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Points cwd + the file-policy allowlist at a temp dir so real tools are safe."""
    monkeypatch.setattr(paths, "ALLOWED_BASE_DIR", tmp_path)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_pending_action_carries_task_state():
    engine = FakeEngine([_mkdir()])
    agent = ReActAgent(engine)
    msg = _run(agent.run("create a todo list application"))
    assert msg.pending_action is not None
    assert msg.pending_action.action_type == "run_command"
    assert msg.pending_action.target == "mkdir TodoList"

    task = msg.task_state
    assert task is not None
    assert task.goal == "create a todo list application"
    assert task.status == TaskStatus.WAITING_CONFIRMATION
    assert task.is_complete() is False
    assert task.execution_history == []  # nothing executed yet
    assert task.requires_verification is True
    # The transcript preserves the goal and the pending tool call.
    assert task.context[0].role == Role.USER
    assert task.context[-1].role == Role.ASSISTANT
    assert len(engine.calls) == 1  # loop stops for confirmation, no silent execution


def test_mkdir_success_does_not_complete_task(sandbox):
    engine = FakeEngine(
        [
            _mkdir(),
            _write("TodoList/index.html", "<html></html>"),
            "The TodoList application is complete.",
            _all_done(),
        ]
    )
    agent = ReActAgent(engine)
    msg1 = _run(agent.run("create a todo list application"))
    task = msg1.task_state

    # Confirmation: the user approves, the real mkdir executes.
    result = _run(execute_pending_action(msg1.pending_action))
    assert "Exit code: 0" in result

    # mkdir succeeded — the task must NOT be complete.
    assert task.is_complete() is False
    assert task.status == TaskStatus.WAITING_CONFIRMATION

    # Feed the observation back; the agent resumes and continues the task.
    msg2 = _run(continue_task_after_confirmation(agent, task, result, []))
    assert msg2.pending_action is not None
    assert msg2.pending_action.action_type == "write_file"  # continued, not stopped
    # The resume happened (step recorded); the new action parks the task back
    # in WAITING_CONFIRMATION awaiting its own approval.
    assert task.current_step == 1
    assert task.status == TaskStatus.WAITING_CONFIRMATION

    # The observation was recorded in history + transcript.
    assert task.execution_history[0].tool_name == "run_command"
    assert task.execution_history[0].target == "mkdir TodoList"
    assert task.execution_history[0].success is True
    tool_msgs = [m for m in task.context if m.role == Role.TOOL]
    assert tool_msgs[-1].content == result


def test_todo_app_flow_continues_until_verified_complete(sandbox):
    # Full reproduction of the original bug flow:
    # mkdir -> confirm -> execute -> continue -> write x2 -> verify -> complete.
    engine = FakeEngine(
        [
            _mkdir(),
            _write("TodoList/index.html", "<h1>todo</h1>"),
            _write("TodoList/app.js", "console.log('hi')"),
            "The TodoList app is complete.",
            _all_done(),
        ]
    )
    agent = ReActAgent(engine)

    msg = _run(agent.run("create a todo list application"))
    task = msg.task_state

    # Confirmation 1: mkdir TodoList
    result1 = _run(execute_pending_action(msg.pending_action))
    assert "Exit code: 0" in result1
    msg = _run(continue_task_after_confirmation(agent, task, result1, []))

    # Confirmation 2: write index.html
    assert msg.pending_action.action_type == "write_file"
    result2 = _run(execute_pending_action(msg.pending_action))
    assert "Successfully wrote" in result2
    msg = _run(continue_task_after_confirmation(agent, task, result2, []))

    # Confirmation 3: write app.js
    assert msg.pending_action.action_type == "write_file"
    result3 = _run(execute_pending_action(msg.pending_action))
    msg = _run(continue_task_after_confirmation(agent, task, result3, []))

    # Final verification — only now does the task complete.
    assert msg.pending_action is None
    assert "complete" in msg.content.lower()
    assert task.is_complete() is True
    assert task.status == TaskStatus.TASK_COMPLETED
    assert task.remaining_requirements() == []
    assert [e.tool_name for e in task.execution_history] == [
        "run_command",
        "write_file",
        "write_file",
    ]
    assert all(e.success for e in task.execution_history)
    # Real artifacts landed on disk.
    assert (sandbox / "TodoList" / "index.html").exists()
    assert (sandbox / "TodoList" / "app.js").exists()


def test_model_cannot_claim_completion_while_work_remains():
    # mkdir succeeds and the model immediately claims success — TaskState
    # rejects the claim because requirements remain, and execution continues.
    not_done = json.dumps(
        [
            {"description": "TodoList directory exists", "satisfied": True},
            {"description": "application files exist", "satisfied": False},
        ]
    )
    engine = FakeEngine(
        [
            _mkdir(),
            "The TodoList application is complete!",  # premature claim
            not_done,
            _write("TodoList/index.html", "<html></html>"),
            "Done.",
            _all_done(),
        ]
    )
    agent = ReActAgent(engine)
    msg1 = _run(agent.run("create a todo list application"))
    task = msg1.task_state

    msg2 = _run(continue_task_after_confirmation(agent, task, "Exit code: 0", []))
    # The premature success claim was NOT accepted — the agent kept working.
    assert msg2.pending_action is not None
    assert msg2.pending_action.action_type == "write_file"
    assert task.is_complete() is False
    assert [r.description for r in task.remaining_requirements()] == [
        "application files exist"
    ]

    # Once the remaining requirement is fulfilled, the task completes.
    msg3 = _run(
        continue_task_after_confirmation(
            agent, task, "Successfully wrote index.html", []
        )
    )
    assert msg3.pending_action is None
    assert task.is_complete() is True
    assert task.remaining_requirements() == []


def test_max_iterations_reports_incomplete_with_remaining_requirements():
    # The model never finishes: TaskState keeps requirements unmet, the loop
    # hits the budget, and the agent must report incomplete — never success.
    not_done = json.dumps(
        [{"description": "application files exist", "satisfied": False}]
    )
    engine = FakeEngine(
        [
            _mkdir(),
            "Almost done.",  # iteration 1: final-looking text
            not_done,        # verification: still incomplete
            "Almost done.",  # iteration 2: final-looking text
            not_done,        # verification: still incomplete
        ]
    )
    agent = ReActAgent(engine, max_iterations=2)
    msg1 = _run(agent.run("create a todo list application"))
    task = msg1.task_state

    final = _run(continue_task_after_confirmation(agent, task, "Exit code: 0", []))
    assert "incomplete" in final.content.lower()
    assert "remaining requirements" in final.content.lower()
    assert "application files exist" in final.content
    assert task.is_complete() is False
    assert task.status == TaskStatus.TASK_FAILED
    assert len(task.errors) == 1
    assert [r.description for r in task.remaining_requirements()] == [
        "application files exist"
    ]


def test_failed_command_observation_recorded_as_failure():
    # A command that exits non-zero ("Exit code: 1") is a FAILED observation —
    # it must not be recorded as success in the task history.
    engine = FakeEngine([_mkdir(), "Done.", _all_done()])
    agent = ReActAgent(engine)
    msg1 = _run(agent.run("create a todo list application"))
    task = msg1.task_state
    final = _run(
        continue_task_after_confirmation(
            agent, task, "Exit code: 1\nError Output:\nmkdir: permission denied", []
        )
    )
    assert final.pending_action is None
    assert task.execution_history[0].tool_name == "run_command"
    assert task.execution_history[0].success is False


def test_denied_action_observation_fed_back_and_agent_continues():
    # A denied confirmation feeds "Action cancelled by user." back as an
    # observation (recorded as failure); the agent adapts and continues.
    engine = FakeEngine(
        [
            _mkdir(),
            _write("TodoList/index.html", "<html></html>"),
            "Done.",
            _all_done(),
        ]
    )
    agent = ReActAgent(engine)
    msg1 = _run(agent.run("create a todo list application"))
    task = msg1.task_state
    msg2 = _run(
        continue_task_after_confirmation(agent, task, "Action cancelled by user.", [])
    )
    assert msg2.pending_action is not None  # tried a different approach
    assert task.execution_history[0].success is False
    assert task.is_complete() is False


def test_readonly_flow_has_no_task_state(monkeypatch):
    # Read-only requests never enter task mode: no verification overhead, and
    # the final answer is returned as before.
    monkeypatch.setitem(reg.TOOLS, "read_file", lambda file_path: "hello world")
    engine = FakeEngine(
        [
            _tool_call("read_file", file_path="notes.txt"),
            "The file says: hello world",
        ]
    )
    agent = ReActAgent(engine)
    msg = _run(agent.run("read the notes file"))
    assert msg.pending_action is None
    assert msg.task_state is None
    assert len(engine.calls) == 2  # tool call + final answer


def test_execute_pending_action_dispatches_write_file(sandbox):
    action = PendingAction(action_type="write_file", target="a.txt", content="hello")
    result = _run(execute_pending_action(action))
    assert "Successfully wrote" in result
    assert (sandbox / "a.txt").read_text() == "hello"
