"""
Tests for Planning Robustness, Observable Recovery Pipeline, and Verification Invariants.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from ultron.core.agents.react import ReActAgent
from ultron.core.intelligence.plan_validation import validate_plan
from ultron.core.intelligence.task_planning import (
    build_debugging_plan,
    build_research_plan,
    build_software_engineering_plan,
    build_system_operation_plan,
    fallback_plan,
    generate_task_plan,
    parse_plan_json,
    prepare_task_for_execution,
)
from ultron.core.types import (
    AcceptanceCriterion,
    AcceptanceCriterionStatus,
    EvidenceLevel,
    StepStatus,
    TaskState,
    TaskStatus,
    TaskType,
    WorkspaceKind,
)


class ScriptedEngine:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls = []

    async def generate(self, messages) -> str:
        self.calls.append(messages)
        return self.responses.pop(0) if self.responses else ""


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# TEST A — Valid complex plan generation
# ---------------------------------------------------------------------------
def test_a_valid_complex_plan_generation(tmp_path: Path):
    valid_json = """
    {
        "steps": [
            {
                "id": 1,
                "description": "Scaffold project and create source files",
                "purpose": "Create backend application structure",
                "dependencies": [],
                "expected_outcome": "Project files exist on disk",
                "completion_criteria": ["Source files created"],
                "failure_strategy": "retry",
                "retry_policy": 2
            },
            {
                "id": 2,
                "description": "Run tests and verify functionality",
                "purpose": "Verify backend works as expected",
                "dependencies": [1],
                "expected_outcome": "All tests pass",
                "completion_criteria": ["Tests pass"],
                "failure_strategy": "stop"
            }
        ],
        "completion_criteria": ["Backend implemented and tested"],
        "verification_requirements": ["Tests pass cleanly"]
    }
    """
    engine = ScriptedEngine([valid_json])
    plan = _run(
        generate_task_plan(
            "Build an expense tracker",
            TaskType.SOFTWARE_ENGINEERING,
            engine,
            cwd=str(tmp_path),
        )
    )
    assert plan is not None
    assert len(plan.steps) == 2
    assert validate_plan(plan).valid
    assert plan.steps[0].description == "Scaffold project and create source files"
    assert plan.steps[1].dependencies == [1]


# ---------------------------------------------------------------------------
# TEST B — Malformed planner JSON recovery
# ---------------------------------------------------------------------------
def test_b_malformed_planner_json_recovery(tmp_path: Path):
    # LLM returns malformed JSON with markdown thoughts and trailing commas
    malformed_json_with_recovery = """
    Thinking: Here is the plan.
    ```json
    {
        "steps": [
            {
                "id": 1,
                "description": "Scaffold project",
                "purpose": "Initialize files",
                "dependencies": [],
                "expected_outcome": "Files exist",
                "completion_criteria": ["Files exist"],
                "failure_strategy": "retry",
            },
        ],
        "completion_criteria": ["Project created"],
        "verification_requirements": ["Files verified"],
    }
    ```
    """
    plan = parse_plan_json(
        malformed_json_with_recovery,
        "Create an app",
        TaskType.SOFTWARE_ENGINEERING,
        WorkspaceKind.NEW_WORKSPACE,
        "",
    )
    assert plan is not None
    assert len(plan.steps) == 1
    assert plan.steps[0].description == "Scaffold project"


# ---------------------------------------------------------------------------
# TEST C — Invalid plan structure recovery
# ---------------------------------------------------------------------------
def test_c_invalid_plan_structure_recovery(tmp_path: Path):
    # Cyclic dependency in planner response; 1st call fails, 2nd call (recovery) provides valid plan
    cyclic_json = """
    {
        "steps": [
            {"id": 1, "description": "A", "expected_outcome": "A done", "completion_criteria": ["A ok"], "dependencies": [2]},
            {"id": 2, "description": "B", "expected_outcome": "B done", "completion_criteria": ["B ok"], "dependencies": [1]}
        ],
        "completion_criteria": ["Done"],
        "verification_requirements": ["Verified"]
    }
    """
    valid_corrected_json = """
    {
        "steps": [
            {"id": 1, "description": "A", "expected_outcome": "A done", "completion_criteria": ["A ok"], "dependencies": []},
            {"id": 2, "description": "B", "expected_outcome": "B done", "completion_criteria": ["B ok"], "dependencies": [1]}
        ],
        "completion_criteria": ["Done"],
        "verification_requirements": ["Verified"]
    }
    """
    engine = ScriptedEngine([cyclic_json, valid_corrected_json])
    plan = _run(
        generate_task_plan(
            "Build service",
            TaskType.SOFTWARE_ENGINEERING,
            engine,
            cwd=str(tmp_path),
        )
    )
    assert plan is not None
    assert validate_plan(plan).valid
    assert len(engine.calls) == 2  # Attempted recovery with validation feedback


# ---------------------------------------------------------------------------
# TEST D — Empty plan recovery
# ---------------------------------------------------------------------------
def test_d_empty_plan_recovery(tmp_path: Path):
    # Empty steps array returned by LLM
    empty_json = '{"steps": [], "completion_criteria": [], "verification_requirements": []}'
    engine = ScriptedEngine([empty_json, empty_json])
    task = _run(
        prepare_task_for_execution(
            "Build a personal expense manager",
            engine,
            cwd=str(tmp_path),
        )
    )
    assert task is not None
    assert task.plan is not None
    assert validate_plan(task.plan).valid
    assert len(task.plan.steps) >= 4  # recovered to deterministic fallback plan
    assert task.plan.steps[0].id == 1
    assert "implement" in task.plan.steps[0].description.lower() or "scaffold" in task.plan.steps[0].description.lower()
    assert task.plan.failure_recovery is not None


# ---------------------------------------------------------------------------
# TEST E — Engine failure / timeout recovery
# ---------------------------------------------------------------------------
def test_e_engine_failure_recovery(tmp_path: Path):
    class BrokenEngine:
        async def generate(self, messages) -> str:
            raise RuntimeError("Connection timed out to LLM backend")

    task = _run(
        prepare_task_for_execution(
            "Build a complete desktop personal expense manager",
            BrokenEngine(),
            cwd=str(tmp_path),
        )
    )
    assert task is not None
    assert task.plan is not None
    assert validate_plan(task.plan).valid
    assert len(task.plan.steps) == 4
    assert task.status == TaskStatus.TASK_STARTED


# ---------------------------------------------------------------------------
# TEST F — Fallback plan structural validation across all task types
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "plan_fn, task_type",
    [
        (build_software_engineering_plan, TaskType.SOFTWARE_ENGINEERING),
        (build_debugging_plan, TaskType.DEBUGGING),
        (build_research_plan, TaskType.RESEARCH),
        (build_system_operation_plan, TaskType.SYSTEM_OPERATION),
    ],
)
def test_f_fallback_plan_structural_validation(plan_fn, task_type):
    plan = plan_fn("Accomplish task goal", WorkspaceKind.NEW_WORKSPACE)
    report = validate_plan(plan)
    assert report.valid is True
    assert len(report.issues) == 0
    assert len(plan.steps) >= 3
    # Step 1 must be an action/investigation step, never solitary goal verification
    assert plan.steps[0].id == 1
    assert plan.steps[0].dependencies == []
    # Final step must be verification
    assert "verify" in plan.steps[-1].description.lower()


# ---------------------------------------------------------------------------
# TEST G — Final verification ordering
# ---------------------------------------------------------------------------
def test_g_final_verification_ordering():
    plan = fallback_plan("Build an app", TaskType.SOFTWARE_ENGINEERING, WorkspaceKind.NEW_WORKSPACE)
    task = TaskState(goal="Build an app", task_type=TaskType.SOFTWARE_ENGINEERING)
    task.attach_plan(plan)

    # Step 1 is active; final step (step 4) cannot run or succeed before dependencies 1..3
    step1 = task.current_plan_step()
    assert step1 is not None
    assert step1.id == 1
    assert not task.is_complete()


# ---------------------------------------------------------------------------
# TEST H — Goal preservation during recovery
# ---------------------------------------------------------------------------
def test_h_goal_preservation_during_recovery(tmp_path: Path):
    user_goal = "Build me a complete desktop personal expense manager for macOS."
    engine = ScriptedEngine(["invalid json response"])
    task = _run(prepare_task_for_execution(user_goal, engine, cwd=str(tmp_path)))

    assert task is not None
    assert task.goal == user_goal
    assert task.plan is not None
    assert task.plan.goal == user_goal
    assert task.user_intent is not None
    assert task.user_intent.goal == user_goal


# ---------------------------------------------------------------------------
# TEST I — Recovery does not equal task failure
# ---------------------------------------------------------------------------
def test_i_recovery_does_not_equal_task_failure(tmp_path: Path):
    engine = ScriptedEngine(["invalid json"])
    task = _run(prepare_task_for_execution("Create a FastAPI backend", engine, cwd=str(tmp_path)))

    assert task is not None
    assert task.status == TaskStatus.TASK_STARTED
    assert task.is_complete() is False
    assert task.plan is not None
    assert task.plan.step(1).status == StepStatus.PENDING
    assert task.plan.failure_recovery is not None


# ---------------------------------------------------------------------------
# TEST J — Genuine verification failure after work
# ---------------------------------------------------------------------------
def test_j_genuine_verification_failure_after_work():
    task = TaskState(goal="Build weather app", task_type=TaskType.SOFTWARE_ENGINEERING)
    plan = build_software_engineering_plan("Build weather app")
    task.attach_plan(plan)

    # Mark steps 1-3 succeeded
    task.plan.step(1).status = StepStatus.SUCCEEDED
    task.plan.step(2).status = StepStatus.SUCCEEDED
    task.plan.step(3).status = StepStatus.SUCCEEDED
    task.plan.step(4).status = StepStatus.RUNNING

    # Record some work history
    task.record_tool_execution(tool_name="write_file", target="app.py", success=True, detail="Wrote app.py")
    task.record_tool_execution(tool_name="run_command", target="python app.py", success=True, detail="Started app")

    # One acceptance criterion remains unmet
    task.acceptance_criteria = [
        AcceptanceCriterion(
            id="behavior_interaction",
            description="Weather data displayed in GUI",
            verification_method="execution",
            required=True,
            status=AcceptanceCriterionStatus.FAILED,
            evidence="Window opened but was completely blank",
            evidence_level=EvidenceLevel.LEVEL_3_LAUNCHED,
        )
    ]

    engine = ScriptedEngine(['{"plan_criteria": [{"description": "Build weather app", "satisfied": true}, {"description": "All application files created, launched, interacted with, and verified", "satisfied": true}, {"description": "The user goal is satisfied: Build weather app", "satisfied": true}], "step_criteria": [{"description": "Build weather app", "satisfied": true}, {"description": "Verified completion with empirical evidence", "satisfied": true}]}'])
    agent = ReActAgent(engine)

    accepted, _ = _run(agent._verify_plan_task(task, "Build weather app", "Done.", []))
    # Verification must reject completion because acceptance criterion is unsatisfied
    assert accepted is False
    assert task.is_complete() is False
    assert any("task incomplete" in m.content.lower() for m in task.context if m.name == "task_verification")


# ---------------------------------------------------------------------------
# TEST K — Successful completion after work
# ---------------------------------------------------------------------------
def test_k_successful_completion_after_work():
    task = TaskState(goal="Build weather app", task_type=TaskType.SOFTWARE_ENGINEERING)
    plan = build_software_engineering_plan("Build weather app")
    task.attach_plan(plan)

    # Mark all steps succeeded
    for s in task.plan.steps:
        s.status = StepStatus.SUCCEEDED

    task.record_tool_execution(tool_name="write_file", target="app.py", success=True, detail="Wrote app.py")
    task.record_tool_execution(tool_name="run_command", target="python app.py", success=True, detail="Verified app")

    for c in task.acceptance_criteria:
        c.status = AcceptanceCriterionStatus.VERIFIED
        c.evidence = "Empirically verified"
        c.evidence_level = EvidenceLevel.LEVEL_5_VERIFIED

    engine = ScriptedEngine(['{"plan_criteria": [{"description": "Build weather app", "satisfied": true}, {"description": "All application files created, launched, interacted with, and verified", "satisfied": true}, {"description": "The user goal is satisfied: Build weather app", "satisfied": true}]}'])
    agent = ReActAgent(engine)

    msg = _run(agent._verify_plan_task(task, "Build weather app", "All features verified and working.", []))
    accepted, _ = msg
    assert accepted is True
    assert task.is_complete() is True
    assert task.status == TaskStatus.TASK_COMPLETED


# ---------------------------------------------------------------------------
# TEST L — Execution history invariant rejects premature completion
# ---------------------------------------------------------------------------
def test_l_execution_history_invariant_rejects_premature_completion():
    task = TaskState(goal="Build desktop application", task_type=TaskType.SOFTWARE_ENGINEERING)
    # No tool execution history recorded
    assert len(task.execution_history) == 0

    engine = ScriptedEngine(['{"requirements": [{"description": "Build desktop application", "satisfied": true}]}'])
    agent = ReActAgent(engine)

    accepted, _ = _run(agent._verify_task(task, "Build desktop application", "I have finished.", []))
    assert accepted is False
    assert task.is_complete() is False
    assert any("no tool executions" in m.content.lower() for m in task.context if m.name == "task_verification")
