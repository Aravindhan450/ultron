"""
Fix #3 stage-1 tests: CodeContext, observations, structured command results,
TaskState integration, security gating, and confirmation preservation.

All filesystem tests use temporary directories; the real Ultron repository
is never modified. No real LLM is required (FakeEngine is used where the
agent loop is exercised).
"""

import asyncio
import json

import pytest

from ultron.core.agents.react import ReActAgent
from ultron.core.coding.command import (
    capture_command,
    parse_command_output,
)
from ultron.core.coding.context import CodeContext
from ultron.core.coding.edits import EditAction
from ultron.core.coding.observations import Observation, ObservationKind
from ultron.core.coding.workspace import discover_workspace
from ultron.core.tools import paths as tools_paths
from ultron.core.types import ChatMessage, PendingAction, TaskState, TaskType


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(tools_paths, "ALLOWED_BASE_DIR", tmp_path)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _run(coro):
    return asyncio.run(coro)


def _tool_call(tool, **arguments):
    return f"```json\n{json.dumps({'tool': tool, 'arguments': arguments})}\n```"


def _all_done():
    return json.dumps(
        [
            {"description": "code updated", "satisfied": True},
            {"description": "edit verified", "satisfied": True},
        ]
    )


# ---------------------------------------------------------------------------
# Observation model
# ---------------------------------------------------------------------------


def test_observation_builders():
    obs = Observation.error("run_command", "command failed", "traceback here")
    assert obs.kind is ObservationKind.ERROR
    assert obs.success is False
    assert "[error]" in obs.to_prompt_line()

    ok = Observation(
        kind=ObservationKind.COMMAND_RESULT,
        source="pytest",
        summary="all tests passed",
        success=True,
        exit_code=0,
    )
    line = ok.to_prompt_line()
    assert "[command_result]" in line and "[ok]" in line and "exit 0" in line


def test_observation_serializes():
    obs = Observation(
        kind=ObservationKind.SEARCH_RESULT,
        source="search_files",
        summary="2 matches",
        success=True,
    )
    restored = Observation.model_validate_json(obs.model_dump_json())
    assert restored.kind is ObservationKind.SEARCH_RESULT


# ---------------------------------------------------------------------------
# CommandResult parsing + capture
# ---------------------------------------------------------------------------


def test_parse_command_output_success():
    text = "Exit code: 0\nOutput:\nhello world\n[resources] wall 0.02s cpu 0.01s"
    result = parse_command_output(text)
    assert result.exit_code == 0
    assert result.stdout == "hello world"
    assert result.success is True
    assert result.duration_ms == 20.0


def test_parse_command_output_failure_with_stderr():
    text = "Exit code: 1\nOutput:\npart\nError Output:\nboom\n[resources] wall 0.10s"
    result = parse_command_output(text)
    assert result.exit_code == 1
    assert result.stdout == "part"
    assert result.stderr == "boom"
    assert result.success is False


def test_parse_command_output_timeout():
    result = parse_command_output("Error: command timed out after 15 seconds.\n[resources] timed out")
    assert result.timed_out is True
    assert result.success is False


def test_parse_command_output_never_raises():
    result = parse_command_output("")
    assert result.exit_code is None
    assert not result.success


def test_capture_command_runs_real_tool(sandbox):
    # capture_command shells out to the registered run_command tool, so the
    # output stays available to the agent — never hidden.
    result = capture_command("echo hello")
    assert result.exit_code == 0
    assert "hello" in result.stdout
    assert result.output  # the full formatted output is preserved


# ---------------------------------------------------------------------------
# CodeContext
# ---------------------------------------------------------------------------


def test_code_context_attach_task():
    task = TaskState(goal="Fix the failing tests", task_type=TaskType.DEBUGGING)
    ctx = CodeContext(workspace=discover_workspace("."))
    ctx.attach_task(task)
    assert ctx.task_goal == "Fix the failing tests"
    assert ctx.task_type == "debugging"


def test_code_context_records_observations_and_relevant_files():
    ctx = CodeContext()
    ctx.add_relevant_file("auth.py")
    ctx.add_relevant_file("auth.py")  # deduplicated
    assert ctx.relevant_files == ["auth.py"]
    ctx.record_observation(
        ObservationKind.COMMAND_RESULT, "run_command", "exit 0", "ok", success=True
    )
    ctx.record_error("pytest", "1 test failed")
    assert len(ctx.observations) == 2
    assert ctx.has_failures()
    assert "pytest" in ctx.summary()


def test_code_context_summary_is_bounded():
    ctx = CodeContext()
    for i in range(50):
        ctx.record_observation(ObservationKind.FILE_CONTENT, f"f{i}.py", f"summary {i}")
    summary = ctx.summary(max_observations=5)
    assert "summary 49" in summary  # most recent
    assert "summary 0" not in summary  # capped
    assert "CODING CONTEXT" in summary


def test_code_context_tracker_round_trips():
    ctx = CodeContext()
    ctx.tracker.record("a.txt", EditAction.CREATE, resulting_state="x")
    restored = CodeContext.model_validate_json(ctx.model_dump_json())
    assert len(restored.tracker.modifications) == 1
    assert restored.tracker.modifications[0].path == "a.txt"


# ---------------------------------------------------------------------------
# TaskState integration
# ---------------------------------------------------------------------------


def test_task_state_carries_code_context(sandbox):
    task = TaskState(goal="Add a health endpoint", task_type=TaskType.SOFTWARE_ENGINEERING)
    task.code_context = CodeContext(workspace=discover_workspace(str(sandbox)))
    task.code_context.attach_task(task)
    restored = TaskState.model_validate_json(task.model_dump_json())
    assert restored.code_context is not None
    assert restored.code_context.task_goal == "Add a health endpoint"
    assert restored.code_context.workspace is not None


def test_prepared_task_gets_coding_context(sandbox):
    # prepare_task_for_execution attaches a CodeContext with the discovered
    # workspace for every planned task.
    from ultron.core.intelligence.task_planning import prepare_task_for_execution

    class FakeEngine:
        def __init__(self, responses):
            self._responses = list(responses)
            self.calls = []

        async def generate(self, messages, **kwargs):
            self.calls.append(messages)
            return self._responses.pop(0) if self._responses else ""

    plan_payload = json.dumps(
        {
            "steps": [
                {
                    "id": 1,
                    "description": "Inspect the repository",
                    "purpose": "Understand the project",
                    "expected_outcome": "Repository understood",
                    "completion_criteria": ["Repository inspected"],
                    "dependencies": [],
                    "failure_strategy": "stop",
                    "retry_policy": 0,
                },
                {
                    "id": 2,
                    "description": "Verify the final user goal",
                    "purpose": "Confirm the request is satisfied",
                    "expected_outcome": "Goal satisfied",
                    "completion_criteria": ["Goal verified"],
                    "dependencies": [1],
                    "failure_strategy": "stop",
                    "retry_policy": 0,
                },
            ],
            "completion_criteria": ["Feature implemented"],
            "verification_requirements": ["The feature works"],
        }
    )
    engine = FakeEngine([plan_payload])
    task = _run(
        prepare_task_for_execution(
            "Add a health endpoint", engine, cwd=str(sandbox)
        )
    )
    assert task is not None
    assert task.plan is not None
    assert task.code_context is not None
    assert task.code_context.workspace is not None
    assert task.code_context.workspace.project_root == str(sandbox)


# ---------------------------------------------------------------------------
# Confirmation preservation (the context lives on the TaskState)
# ---------------------------------------------------------------------------


def test_code_context_survives_confirmation_round_trip(sandbox):
    task = TaskState(goal="create a report", task_type=TaskType.MULTI_STEP)
    task.code_context = CodeContext(workspace=discover_workspace(str(sandbox)))
    task.code_context.attach_task(task)
    task.code_context.add_relevant_file("data.txt")
    task.code_context.record_observation(ObservationKind.SEARCH_RESULT, "search_files", "found data.txt", success=True)

    # Simulate the confirmation lifecycle: park the task, then resume it —
    # exactly what main.py does through continue_task_after_confirmation.
    task.wait_for_confirmation()
    task.resume()

    assert task.code_context is not None
    assert "data.txt" in task.code_context.relevant_files
    assert len(task.code_context.observations) == 1


def test_pending_action_accepts_new_action_types(sandbox):
    # The new coding edit actions are first-class PendingAction types.
    action = PendingAction(action_type="replace_in_file", target="app.py", content=json.dumps({"old": "a", "new": "b"}))
    assert action.action_type == "replace_in_file"
    restored = PendingAction.model_validate_json(action.model_dump_json())
    assert restored.target == "app.py"


# ---------------------------------------------------------------------------
# Security gating: the new coding tools route through the boundary
# ---------------------------------------------------------------------------


def test_coding_tools_classified_by_boundary():
    from ultron.security import SecurityBoundary

    boundary = SecurityBoundary(mode="interactive")
    # Read-only inspection is LOW → allow.
    assert boundary.check("list_directory", ".").decision.value == "allow"
    assert boundary.check("search_files", "query").decision.value == "allow"
    assert boundary.check("discover_workspace_summary", "").decision.value == "allow"
    # State-changing edits are HIGH → confirm.
    assert boundary.check("create_file", "new.txt", "content").decision.value == "confirm"
    assert boundary.check("replace_in_file", "app.py", "new").decision.value == "confirm"
    assert boundary.check("append_to_file", "app.py", "more").decision.value == "confirm"
    assert boundary.check("delete_file", "app.py").decision.value == "confirm"
    assert boundary.check("rename_file", "a.txt", "b.txt").decision.value == "confirm"
    # Protected paths escalate to CRITICAL (still confirm, never silent).
    assert boundary.check("delete_file", ".env").tier.value == "critical"


def test_guardrails_deny_path_escape_for_new_tools(sandbox):
    from ultron.security.guardrails import GuardrailsEngine

    engine = GuardrailsEngine()
    # /etc is outside ALLOWED_BASE_DIR → path escape is a hard deny for
    # EVERY coding file op, not just delete_file.
    for action_type in (
        "list_directory",
        "search_files",
        "create_file",
        "replace_file",
        "replace_in_file",
        "append_to_file",
        "delete_file",
        "rename_file",
    ):
        result = engine.evaluate(action_type=action_type, target="/etc/passwd")
        assert result.blocked, action_type


def test_boundary_critical_escalation_for_all_mutating_ops(sandbox):
    from ultron.security import SecurityBoundary

    boundary = SecurityBoundary(mode="interactive")
    # .env is a protected path — every state-changing op on it escalates to
    # CRITICAL (never silent), and read-only inspection stays low/medium.
    for action_type in (
        "create_file",
        "replace_file",
        "replace_in_file",
        "append_to_file",
        "delete_file",
        "rename_file",
    ):
        verdict = boundary.check(action_type, ".env", "secret")
        assert verdict.tier.value == "critical", action_type
    assert boundary.check("list_directory", ".").tier.value == "low"


def test_react_route_coding_file_op_pending_action(sandbox):
    # A ReAct agent routing a targeted edit must produce a PendingAction
    # (confirm), not execute it silently.
    class FakeEngine:
        async def generate(self, messages, **kwargs):
            return ""

    agent = ReActAgent(FakeEngine())
    outcome = agent._route_coding_file_op(
        "replace_in_file",
        {"file_path": "app.py", "old": "a", "new": "b"},
    )
    assert isinstance(outcome, ChatMessage)
    assert outcome.pending_action is not None
    assert outcome.pending_action.action_type == "replace_in_file"
    assert outcome.pending_action.target == "app.py"
    payload = json.loads(outcome.pending_action.content or "{}")
    assert payload == {"old": "a", "new": "b"}


def test_react_generic_path_still_routes_read_only(sandbox):
    # list_directory is read-only (LOW) → the generic path executes it and
    # returns a string observation, never a PendingAction.
    class FakeEngine:
        async def generate(self, messages, **kwargs):
            return ""

    (sandbox / "x.txt").write_text("x", encoding="utf-8")
    agent = ReActAgent(FakeEngine())
    outcome = agent._route_tool("list_directory", {"path": "."}, "list files")
    assert isinstance(outcome, str)
    assert "x.txt" in outcome


def test_execute_pending_action_runs_confirmed_edits(sandbox):
    from ultron.main import execute_pending_action

    # Targeted edit through the confirmed-action executor.
    (sandbox / "app.py").write_text("x = 1\nprint(x)\n", encoding="utf-8")
    action = PendingAction(
        action_type="replace_in_file",
        target="app.py",
        content=json.dumps({"old": "print(x)", "new": "print(x * 2)"}),
    )
    result = _run(execute_pending_action(action))
    assert "Replaced" in result
    assert "print(x * 2)" in (sandbox / "app.py").read_text(encoding="utf-8")

    # Append through the confirmed-action executor.
    action2 = PendingAction(action_type="append_to_file", target="app.py", content="\n# done")
    result2 = _run(execute_pending_action(action2))
    assert "Appended" in result2

    # Delete through the confirmed-action executor.
    (sandbox / "tmp.txt").write_text("x", encoding="utf-8")
    action3 = PendingAction(action_type="delete_file", target="tmp.txt")
    result3 = _run(execute_pending_action(action3))
    assert "Deleted" in result3
    assert not (sandbox / "tmp.txt").exists()


def test_registry_exposes_coding_tools(sandbox):
    from ultron.core.tools.registry import TOOLS, get_tools_schema

    for name in (
        "list_directory",
        "search_files",
        "discover_workspace_summary",
        "create_file",
        "replace_file",
        "replace_in_file",
        "append_to_file",
        "delete_file",
        "rename_file",
    ):
        assert name in TOOLS, name
    schema_names = {entry["name"] for entry in get_tools_schema()}
    assert "replace_in_file" in schema_names
    assert "list_directory" in schema_names


# ---------------------------------------------------------------------------
# End-to-end: plan-aware agent uses coding tools through the real loop
# ---------------------------------------------------------------------------


def test_confirmed_edit_is_recorded_in_task_tracker(sandbox):
    # The reviewer-flagged gap: a confirmed coding edit must land in the
    # task's modification tracker (path, action, step, success) — proving
    # modification tracking is behavior, not just an abstraction.
    from ultron.main import (
        continue_task_after_confirmation,
        execute_pending_action,
    )

    (sandbox / "app.py").write_text("def run():\n    return 'old'\n", encoding="utf-8")

    class FakeEngine:
        def __init__(self, responses):
            self._responses = list(responses)

        async def generate(self, messages, **kwargs):
            return self._responses.pop(0) if self._responses else ""

    edit_call = _tool_call(
        "replace_in_file",
        file_path="app.py",
        old="return 'old'",
        new="return 'new'",
    )
    engine = FakeEngine([edit_call, "Done.", _all_done()])
    agent = ReActAgent(engine, max_iterations=5)

    msg1 = _run(agent.run("update the run function", []))
    assert msg1.pending_action is not None
    assert msg1.pending_action.action_type == "replace_in_file"
    task = msg1.task_state
    assert task is not None
    assert task.code_context is not None  # coding edits get workspace context
    assert task.code_context.tracker.modifications == []  # not yet executed

    # Confirmation → execution → continuation → verification → completion.
    result = _run(execute_pending_action(msg1.pending_action))
    assert "Replaced 1 occurrence" in result
    final = _run(continue_task_after_confirmation(agent, task, result, []))
    assert final.pending_action is None
    assert task.is_complete() is True

    # The confirmed edit is now recorded in the modification tracker.
    mods = task.code_context.tracker.modifications
    assert len(mods) == 1
    assert mods[0].path == "app.py"
    assert mods[0].action.value == "targeted_edit"
    assert mods[0].success is True
    assert task.code_context.workspace is not None
    # And it survived the confirmation boundary (goal intact).
    assert task.code_context.task_goal == "update the run function"


def test_mixed_task_gets_code_context_for_later_coding_edit(sandbox):
    # A task created by a NON-coding confirmation (run_command) must still
    # gain workspace context when a coding edit is confirmed later — the
    # attach must not be limited to fresh tasks.
    from ultron.main import (
        continue_task_after_confirmation,
        execute_pending_action,
    )

    class FakeEngine:
        def __init__(self, responses):
            self._responses = list(responses)

        async def generate(self, messages, **kwargs):
            return self._responses.pop(0) if self._responses else ""

    (sandbox / "app.py").write_text("old", encoding="utf-8")
    mkdir_call = _tool_call("run_command", command="mkdir app")
    edit_call = _tool_call(
        "replace_in_file",
        file_path="app.py",
        old="old",
        new="new",
    )
    engine = FakeEngine([mkdir_call, edit_call, "Done.", _all_done()])
    agent = ReActAgent(engine, max_iterations=6)

    msg1 = _run(agent.run("fix the app", []))
    assert msg1.pending_action.action_type == "run_command"
    task = msg1.task_state
    assert task.code_context is None  # non-coding action, no context yet

    result1 = _run(execute_pending_action(msg1.pending_action))
    msg2 = _run(continue_task_after_confirmation(agent, task, result1, []))
    assert msg2.pending_action.action_type == "replace_in_file"
    # The coding edit attached workspace context to the existing task.
    assert task.code_context is not None
    assert task.code_context.workspace is not None
    assert task.code_context.tracker.modifications == []

    result2 = _run(execute_pending_action(msg2.pending_action))
    final = _run(continue_task_after_confirmation(agent, task, result2, []))
    assert final.pending_action is None
    assert task.is_complete() is True
    assert len(task.code_context.tracker.modifications) == 1
    assert task.code_context.tracker.modifications[0].success is True


def test_cancelled_edit_recorded_as_failure_in_tracker(sandbox):
    # A user-cancelled confirmation is a FAILED observation — the tracker must
    # not record it as a successful modification (it must agree with
    # execution_history).
    from ultron.main import continue_task_after_confirmation

    class FakeEngine:
        def __init__(self, responses):
            self._responses = list(responses)

        async def generate(self, messages, **kwargs):
            return self._responses.pop(0) if self._responses else ""

    edit_call = _tool_call(
        "replace_in_file",
        file_path="app.py",
        old="old",
        new="new",
    )
    engine = FakeEngine([edit_call, "Done.", _all_done()])
    agent = ReActAgent(engine, max_iterations=5)

    msg1 = _run(agent.run("update the app", []))
    task = msg1.task_state
    assert task.code_context is not None

    # User denies the edit → cancelled observation is fed back.
    final = _run(
        continue_task_after_confirmation(agent, task, "Action cancelled by user.", [])
    )
    assert final.pending_action is None
    assert task.is_complete() is True  # verification accepted the (empty) work
    assert task.execution_history[0].success is False
    mods = task.code_context.tracker.modifications
    assert len(mods) == 1
    assert mods[0].path == "app.py"
    assert mods[0].success is False
    assert "cancelled" in (mods[0].error or "")


def test_react_agent_reaches_completion_with_coding_tools(sandbox):
    # Full loop: read-only search observation feeds a planned step, then a
    # confirmation-gated targeted edit is requested — proving the coding
    # tools integrate with the Fix #1/#2 lifecycle.
    (sandbox / "app.py").write_text("def run():\n    return 'old'\n", encoding="utf-8")

    class FakeEngine:
        def __init__(self, responses):
            self._responses = list(responses)

        async def generate(self, messages, **kwargs):
            return self._responses.pop(0) if self._responses else ""

    tool_call = (
        'Thought: search for the run function.\n'
        '```json\n{"tool": "search_files", "arguments": {"query": "def run", "path": "."}}\n```'
    )
    engine = FakeEngine([tool_call, "Done — I found it."])
    agent = ReActAgent(engine, max_iterations=5)
    message = _run(agent.run("Find the run function", []))
    assert message.pending_action is None
    assert "found" in message.content.lower()
