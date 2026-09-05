"""
Phase 3.2: Autonomous Repair & Self-Correction Contract Tests.

Tests:
1. Grader verification enforcement (Test E)
2. Failure classification and localization (Failure analysis)
3. Repair budget exhaustion and gating (Test C)
4. Cancellation propagation in repair loop (Test D)
5. End-to-end repair cycle (Test B - deterministic mock engine)
"""

import asyncio
import json
import sys
from typing import Any

import pytest

from tests.model_in_loop.harness.grader import GradingReport
from ultron.core.agents.react import ReActAgent
from ultron.core.coding.context import CodeContext
from ultron.core.coding.executor import (
    CodingExecutor,
    FailureCategory,
    RepairBudget,
    classify_failure,
    localize_failure,
)
from ultron.core.coding.workspace import discover_workspace
from ultron.core.context import RepositoryContextManager
from ultron.core.runtime import (
    AgentRuntime,
    CancellationToken,
    EventBus,
    RuntimeEventType,
    RuntimeStatus,
)
from ultron.core.tools import paths as tools_paths
from ultron.core.types import (
    ChatMessage,
    TaskState,
    TaskType,
)
from ultron.main import (
    continue_task_after_confirmation,
    execute_pending_action,
)

PYTHON = sys.executable


class ScriptedEngine:
    """Deterministic scripted engine returning predefined responses."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[Any] = []

    async def generate(self, messages, **kwargs) -> str:
        self.calls.append(messages)
        return self._responses.pop(0) if self._responses else ""

    async def stream(self, messages, **kwargs):
        yield ""


def _tool_call(tool: str, **arguments) -> str:
    return f"```json\n{json.dumps({'tool': tool, 'arguments': arguments})}\n```"


# ===========================================================================
# 1. Test E: Grader Verification Enforcement
# ===========================================================================


def test_grader_verification_executed_enforced():
    """
    Regression test: proves verification_executed == False causes is_success
    to be False even if every other criterion passes.
    """
    report = GradingReport(
        scenario_name="test_scenario",
        correct_file_modified=True,
        expected_implementation=True,
        tests_pass=True,
        no_unrelated_changes=True,
        non_empty_diff=True,
        verification_executed=False,  # <--- FALSE
        budget_respected=True,
    )
    assert not report.is_success, (
        "GradingReport must NOT succeed when verification_executed is False"
    )

    # When verification_executed is True, all pass
    report.verification_executed = True
    assert report.is_success, (
        "GradingReport must succeed when all criteria including verification_executed are True"
    )


# ===========================================================================
# 2. Failure Classification & Localization
# ===========================================================================


def test_failure_classification_and_localization():
    """
    Verifies that failure outputs are classified into exact categories
    and file/line/test locations are correctly extracted.
    """
    pytest_failure_output = """
============================= test session starts ==============================
collected 2 items

test_calculator.py .F                                                    [100%]

=================================== FAILURES ===================================
___________________________________ test_add ___________________________________

    def test_add():
>       assert add(2, 3) == 5
E       assert -1 == 5
E        +  where -1 = add(2, 3)

calculator.py:3: AssertionError
=========================== short test summary info ============================
FAILED test_calculator.py::test_add - assert -1 == 5
1 failed, 1 passed in 0.05s
"""
    analysis = classify_failure(
        command="pytest",
        exit_code=1,
        stdout=pytest_failure_output,
    )
    assert analysis.category == FailureCategory.TEST_ASSERTION
    assert analysis.file in ("calculator.py", "test_calculator.py")
    assert analysis.line == 3 or analysis.test_name == "test_calculator.py::test_add"

    location = localize_failure(stdout=pytest_failure_output)
    assert location.file in ("calculator.py", "test_calculator.py")
    assert location.test_name == "test_calculator.py::test_add"


# ===========================================================================
# 3. Test C: Repair Budget Exhaustion
# ===========================================================================


def test_repair_budget_exhaustion_blocks_repeated_failures(tmp_path, monkeypatch):
    """
    Verifies that when a repair action repeatedly fails, the repair budget
    exhausts and blocks repeated identical failures from executing.
    """
    monkeypatch.setattr(tools_paths, "ALLOWED_BASE_DIR", tmp_path)
    monkeypatch.chdir(tmp_path)

    budget = RepairBudget(max_repair_attempts=3, max_identical_actions=2)
    executor = CodingExecutor(budget=budget)

    tool_name = "replace_in_file"
    args = {"file_path": "calc.py", "target": "old", "replacement": "new"}

    # Action hasn't failed yet
    assert executor.gate_action(tool_name, args) is None

    # First failure
    executor.record_observation(tool_name, args, "Error: target not found", succeeded=False)
    assert budget.failure_count == 1
    assert not budget.repeat_blocked(tool_name, args)
    assert executor.gate_action(tool_name, args) is None

    # Second failure (identical action)
    executor.record_observation(tool_name, args, "Error: target not found", succeeded=False)
    assert budget.failure_count == 2
    assert budget.repeat_blocked(tool_name, args)

    # Now the action is gated/blocked before running
    block_msg = executor.gate_action(tool_name, args)
    assert block_msg is not None
    assert "This exact action has already failed" in block_msg

    # Third failure exhausts total repair budget
    executor.record_observation("write_file", {"file_path": "foo.py"}, "Error: denied", succeeded=False)
    assert budget.exhausted()
    exhausted_block = executor.gate_new_action_with_exhausted_budget("write_file")
    assert exhausted_block is not None
    assert "The repair budget for this task is exhausted" in exhausted_block


# ===========================================================================
# 4. Test D: Cancellation in Repair Loop
# ===========================================================================


@pytest.mark.anyio
async def test_repair_cancellation_stops_execution(tmp_path, monkeypatch):
    """
    Verifies that CancellationToken cancellation immediately halts the
    agent execution loop and raises CancelledError.
    """
    monkeypatch.setattr(tools_paths, "ALLOWED_BASE_DIR", tmp_path)
    monkeypatch.chdir(tmp_path)

    cancel_token = CancellationToken()
    cancel_token.cancel()  # Pre-cancelled

    engine = ScriptedEngine([
        _tool_call("read_file", file_path="calc.py"),
    ])
    agent = ReActAgent(engine=engine, max_iterations=5)
    runtime = AgentRuntime()

    # 1. AgentRuntime returns structured CANCELLED result
    result = await runtime.execute(
        agent=agent,
        user_input="Fix the calculator bug",
        cancellation_token=cancel_token,
    )
    assert result.status == RuntimeStatus.CANCELLED
    assert result.is_terminal

    # 2. Direct agent execution raises asyncio.CancelledError on check
    with pytest.raises(asyncio.CancelledError):
        await agent.run("Fix the calculator bug", [], cancellation_token=cancel_token)


# ===========================================================================
# 5. Test B: Deterministic Repair Cycle
# ===========================================================================


@pytest.mark.anyio
async def test_deterministic_repair_cycle(tmp_path, monkeypatch):
    """
    Simulates an autonomous repair cycle:
    1. Agent reads file
    2. Agent makes initial (flawed) edit
    3. Agent runs pytest -> fails
    4. Agent analyzes failure observation
    5. Agent applies repaired edit
    6. Agent runs pytest -> passes
    7. Agent finishes
    """
    monkeypatch.setattr(tools_paths, "ALLOWED_BASE_DIR", tmp_path)
    monkeypatch.chdir(tmp_path)

    calc_file = tmp_path / "calculator.py"
    test_file = tmp_path / "test_calculator.py"

    calc_file.write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    test_file.write_text("from calculator import add\ndef test_add():\n    assert add(2, 3) == 5\n", encoding="utf-8")

    # Scripted sequence:
    # 1. read_file
    # 2. replace_in_file with wrong edit (a * b)
    # 3. run_command pytest -> fail
    # 4. replace_in_file with repaired edit (a + b)
    # 5. run_command pytest -> pass
    # 6. Final answer
    responses = [
        _tool_call("read_file", file_path="calculator.py"),
        _tool_call("replace_in_file", file_path="calculator.py", old="return a - b", new="return a * b"),
        _tool_call("run_command", command=f"{PYTHON} -m pytest -q"),
        _tool_call("replace_in_file", file_path="calculator.py", old="return a * b", new="return a + b"),
        _tool_call("run_command", command=f"{PYTHON} -m pytest -q"),
        "The calculator bug has been identified, repaired, and verified via pytest.",
    ]

    engine = ScriptedEngine(responses)
    ws = discover_workspace(str(tmp_path))
    cm = RepositoryContextManager(workspace=ws)
    event_bus = EventBus()
    events_recorded: list[RuntimeEventType] = []
    event_bus.subscribe(lambda e: events_recorded.append(e.event_type))

    runtime = AgentRuntime(context_manager=cm, event_bus=event_bus)
    agent = ReActAgent(engine=engine, max_iterations=10)

    task = TaskState(goal="Fix failing test in calculator.py", task_type=TaskType.DEBUGGING)
    code_ctx = CodeContext(workspace=ws)
    code_ctx.attach_task(task)
    task.code_context = code_ctx

    history: list[ChatMessage] = []

    # Run initial turn
    result = await runtime.execute(agent=agent, user_input=task.goal, history=history, task=task)
    msg = result.message

    # Step through pending actions
    for _ in range(8):
        if msg is None or msg.pending_action is None:
            break
        action_res = await execute_pending_action(msg.pending_action)
        msg = await continue_task_after_confirmation(agent, task, action_res, history)

    # Verification:
    # 1. calculator.py contains a + b
    assert "return a + b" in calc_file.read_text(encoding="utf-8")
    # 2. Task recorded tool executions and observations
    assert len(task.execution_history) >= 4
    # 3. Tool events were emitted
    assert RuntimeEventType.TOOL_STARTED in events_recorded
    assert RuntimeEventType.TOOL_COMPLETED in events_recorded
