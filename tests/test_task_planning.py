"""Tests for the structured planning layer (Fix #2)."""

from __future__ import annotations

import asyncio

import pytest

from ultron.core.intelligence.plan_validation import validate_plan
from ultron.core.intelligence.task_planning import (
    detect_workspace_kind,
    fallback_plan,
    generate_task_plan,
    parse_plan_json,
    probe_working_context,
)
from ultron.core.types import PlanStep, StepStatus, TaskPlan, TaskType, WorkspaceKind


class FakeEngine:
    """Minimal fake LLM engine returning canned responses."""

    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls = 0

    async def generate(self, messages) -> str:
        self.calls += 1
        return self.responses.pop(0)


def _run(coro):
    return asyncio.run(coro)


VALID_PLAN_JSON = """
{
  "goal": "ignored - caller decides the goal",
  "assumptions": ["Python 3.11 is available"],
  "constraints": ["Do not touch unrelated code"],
  "steps": [
    {"id": 1, "description": "Establish the project workspace",
     "purpose": "Create the working directory",
     "expected_outcome": "Working directory exists",
     "completion_criteria": ["Directory exists"], "dependencies": [],
     "failure_strategy": "stop", "retry_policy": 0},
    {"id": 2, "description": "Implement the requested functionality",
     "purpose": "Implement the feature",
     "expected_outcome": "Feature exists",
     "completion_criteria": ["Feature implemented"], "dependencies": [1],
     "failure_strategy": "retry", "retry_policy": 2},
    {"id": 3, "description": "Validate implementation",
     "purpose": "Run tests and checks",
     "expected_outcome": "Tests pass",
     "completion_criteria": ["Tests pass"], "dependencies": [2],
     "failure_strategy": "stop", "retry_policy": 0},
    {"id": 4, "description": "Verify final user goal",
     "purpose": "Confirm the original request is satisfied",
     "expected_outcome": "Goal verified",
     "completion_criteria": ["User goal satisfied"], "dependencies": [3],
     "failure_strategy": "stop", "retry_policy": 0}
  ],
  "completion_criteria": ["Application implemented", "Tests pass"],
  "verification_requirements": ["Run the application and confirm it starts"],
  "failure_recovery": "Stop on first unresolvable error and report remaining steps"
}
"""


# ---------------------------------------------------------------------------
# Workspace awareness
# ---------------------------------------------------------------------------


def test_detect_new_workspace(tmp_path):
    assert detect_workspace_kind(str(tmp_path)) is WorkspaceKind.NEW_WORKSPACE


def test_detect_existing_project(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname = \"x\"\n")
    (tmp_path / ".git").mkdir()
    assert detect_workspace_kind(str(tmp_path)) is WorkspaceKind.EXISTING_PROJECT


def test_detect_unknown_when_dir_missing(tmp_path):
    assert (
        detect_workspace_kind(str(tmp_path / "does-not-exist"))
        is WorkspaceKind.UNKNOWN
    )


def test_probe_working_context(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\n")
    (tmp_path / "src").mkdir()
    (tmp_path / "README.md").write_text("# hi")
    context = probe_working_context(str(tmp_path))
    assert "pyproject.toml" in context
    assert "src" in context
    assert "README.md" in context


# ---------------------------------------------------------------------------
# Plan generation
# ---------------------------------------------------------------------------


def test_generate_task_plan_valid(tmp_path):
    engine = FakeEngine([VALID_PLAN_JSON])
    goal = "Create a TodoList application"
    plan = _run(
        generate_task_plan(
            goal,
            TaskType.SOFTWARE_ENGINEERING,
            engine,
            cwd=str(tmp_path),
        )
    )
    assert plan is not None
    assert isinstance(plan, TaskPlan)
    # Caller-provided fields are authoritative — the model cannot override them.
    assert plan.goal == goal
    assert plan.task_type is TaskType.SOFTWARE_ENGINEERING
    assert plan.workspace is WorkspaceKind.NEW_WORKSPACE
    assert len(plan.steps) == 4
    assert plan.steps[1].dependencies == [1]
    assert plan.steps[1].failure_strategy.value == "retry"
    assert plan.steps[1].retry_policy == 2
    assert plan.completion_criteria == ["Application implemented", "Tests pass"]
    assert plan.verification_requirements == [
        "Run the application and confirm it starts"
    ]
    assert validate_plan(plan).valid
    assert engine.calls == 1


def test_generate_task_plan_parse_failure(tmp_path):
    engine = FakeEngine(["this is definitely not json"])
    plan = _run(
        generate_task_plan(
            "Do a thing",
            TaskType.SOFTWARE_ENGINEERING,
            engine,
            cwd=str(tmp_path),
        )
    )
    assert plan is None


def test_generate_task_plan_rejects_invalid_plan(tmp_path):
    circular = (
        '{"steps": [{"id": 1, "description": "a", "expected_outcome": "o", '
        '"completion_criteria": ["c"], "dependencies": [2]}, '
        '{"id": 2, "description": "b", "expected_outcome": "o", '
        '"completion_criteria": ["c"], "dependencies": [1]}], '
        '"completion_criteria": ["x"], "verification_requirements": ["y"]}'
    )
    engine = FakeEngine([circular])
    plan = _run(
        generate_task_plan(
            "Do a thing",
            TaskType.SOFTWARE_ENGINEERING,
            engine,
            cwd=str(tmp_path),
        )
    )
    assert plan is None


def test_generate_task_plan_informational_never_plans():
    engine = FakeEngine([])
    plan = _run(
        generate_task_plan("What is the capital of France?", TaskType.INFORMATIONAL, engine)
    )
    assert plan is None
    assert engine.calls == 0


def test_parse_plan_json_requires_steps_list():
    plan = parse_plan_json(
        '{"steps": "nope"}',
        "goal",
        TaskType.SIMPLE_ACTION,
        WorkspaceKind.NEW_WORKSPACE,
        "",
    )
    assert plan is None


# ---------------------------------------------------------------------------
# Fallback plan
# ---------------------------------------------------------------------------


def test_fallback_plan_is_valid_and_verifies_goal():
    plan = fallback_plan(
        "Make the tests pass",
        TaskType.DEBUGGING,
        WorkspaceKind.EXISTING_PROJECT,
    )
    assert validate_plan(plan).valid
    assert len(plan.steps) == 1
    assert "verify" in plan.steps[0].description.lower()
    assert plan.completion_criteria == ["Make the tests pass"]
    assert plan.workspace is WorkspaceKind.EXISTING_PROJECT


# ---------------------------------------------------------------------------
# Plan model behavior
# ---------------------------------------------------------------------------


def _plan_with_steps() -> TaskPlan:
    return TaskPlan(
        goal="goal",
        task_type=TaskType.SOFTWARE_ENGINEERING,
        steps=[
            PlanStep(id=1, description="one", expected_outcome="o1", completion_criteria=["c1"]),
            PlanStep(
                id=2,
                description="two",
                expected_outcome="o2",
                completion_criteria=["c2"],
                dependencies=[1],
            ),
            PlanStep(
                id=3,
                description="three",
                expected_outcome="o3",
                completion_criteria=["c3"],
                dependencies=[2],
            ),
        ],
        completion_criteria=["done"],
        verification_requirements=["verify"],
    )


def test_next_step_respects_dependencies():
    plan = _plan_with_steps()
    assert plan.next_step().id == 1
    plan.set_step_status(1, StepStatus.SUCCEEDED, result="ok")
    assert plan.next_step().id == 2
    plan.set_step_status(2, StepStatus.SUCCEEDED)
    assert plan.next_step().id == 3


def test_step_status_accessors():
    plan = _plan_with_steps()
    plan.set_step_status(1, StepStatus.SUCCEEDED)
    plan.set_step_status(2, StepStatus.FAILED)
    plan.set_step_status(3, StepStatus.BLOCKED)
    assert [s.id for s in plan.completed_steps()] == [1]
    assert [s.id for s in plan.failed_steps()] == [2]
    assert [s.id for s in plan.blocked_steps()] == [3]
    assert [s.id for s in plan.remaining_steps()] == [3]


def test_is_satisfied_requires_all_steps_done():
    plan = _plan_with_steps()
    assert plan.is_satisfied() is False
    plan.set_step_status(1, StepStatus.SUCCEEDED)
    plan.set_step_status(2, StepStatus.SUCCEEDED)
    plan.set_step_status(3, StepStatus.SUCCEEDED)
    assert plan.is_satisfied() is True


def test_set_step_status_unknown_raises():
    plan = _plan_with_steps()
    with pytest.raises(ValueError):
        plan.set_step_status(99, StepStatus.SUCCEEDED)


def test_plan_json_roundtrip():
    plan = _plan_with_steps()
    restored = TaskPlan.model_validate_json(plan.model_dump_json())
    assert restored.goal == plan.goal
    assert [s.id for s in restored.steps] == [1, 2, 3]
    assert restored.steps[1].dependencies == [1]
    assert restored.completion_criteria == ["done"]
