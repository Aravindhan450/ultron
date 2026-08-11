"""Tests for TaskState <-> structured plan integration (Fix #2)."""

from __future__ import annotations

import pytest

from ultron.core.types import (
    ChatMessage,
    PlanStep,
    Role,
    StepStatus,
    TaskPlan,
    TaskState,
    TaskStatus,
    TaskType,
    WorkspaceKind,
)


def make_plan(**overrides) -> TaskPlan:
    plan = TaskPlan(
        goal="Create a TodoList application",
        task_type=TaskType.SOFTWARE_ENGINEERING,
        workspace=WorkspaceKind.NEW_WORKSPACE,
        steps=[
            PlanStep(
                id=1,
                description="Establish the project workspace",
                expected_outcome="Working directory exists",
                completion_criteria=["TodoList directory exists"],
            ),
            PlanStep(
                id=2,
                description="Implement the requested functionality",
                expected_outcome="The app exists and satisfies the request",
                completion_criteria=["Application implemented"],
                dependencies=[1],
            ),
            PlanStep(
                id=3,
                description="Verify final user goal",
                expected_outcome="Original request satisfied",
                completion_criteria=["User goal verified"],
                dependencies=[2],
            ),
        ],
        completion_criteria=["TodoList application implemented"],
        verification_requirements=["Application runs without errors"],
    )
    return plan.model_copy(update=overrides)


# ---------------------------------------------------------------------------
# attach_plan
# ---------------------------------------------------------------------------


def test_attach_plan_seeds_task_state():
    task = TaskState(goal="Create a TodoList application")
    plan = make_plan()
    task.attach_plan(plan)

    assert task.task_type is TaskType.SOFTWARE_ENGINEERING
    assert task.plan is plan
    assert task.total_steps == 3
    assert {r.description for r in task.requirements} == {
        "TodoList application implemented",
        "Application runs without errors",
    }
    assert len(task.remaining_requirements()) == 2


def test_attach_plan_deduplicates_criteria():
    task = TaskState(goal="g")
    plan = make_plan(completion_criteria=["X", "X"], verification_requirements=["X"])
    task.attach_plan(plan)
    assert len(task.requirements) == 1
    assert task.requirements[0].description == "X"


def test_attach_plan_never_completes_task():
    task = TaskState(goal="g")
    task.attach_plan(make_plan())
    assert task.status is TaskStatus.TASK_STARTED
    assert task.is_complete() is False


# ---------------------------------------------------------------------------
# Clarification
# ---------------------------------------------------------------------------


def test_require_clarification_blocks_task():
    task = TaskState(goal="Deploy this.")
    task.require_clarification(["Where should this be deployed?"])
    assert task.status is TaskStatus.TASK_BLOCKED
    assert task.clarification_required is True
    assert task.clarification_questions == ["Where should this be deployed?"]
    assert task.is_complete() is False


def test_require_clarification_deduplicates_questions():
    task = TaskState(goal="g")
    task.require_clarification(["q1", "q1", "q2", ""])
    assert task.clarification_questions == ["q1", "q2"]


def test_attach_plan_with_clarification_flag_blocks():
    plan = make_plan(
        needs_clarification=True,
        clarification_questions=["Which environment?"],
    )
    task = TaskState(goal="Deploy this.")
    task.attach_plan(plan)
    assert task.status is TaskStatus.TASK_BLOCKED
    assert task.clarification_required is True


# ---------------------------------------------------------------------------
# Plan persistence (survives LLM turns / confirmations / serialization)
# ---------------------------------------------------------------------------


def test_plan_survives_task_state_json_roundtrip():
    task = TaskState(goal="g")
    task.attach_plan(make_plan())
    restored = TaskState.model_validate_json(task.model_dump_json())

    assert restored.plan is not None
    assert restored.plan.goal == task.plan.goal
    assert [s.id for s in restored.plan.steps] == [1, 2, 3]
    assert restored.plan.steps[1].dependencies == [1]
    assert {r.description for r in restored.requirements} == {
        r.description for r in task.requirements
    }
    assert restored.task_type is TaskType.SOFTWARE_ENGINEERING


def test_plan_survives_confirmation_channel_roundtrip():
    """The plan rides ChatMessage.task_state across the confirmation boundary."""
    task = TaskState(goal="g")
    task.attach_plan(make_plan())
    message = ChatMessage(role=Role.ASSISTANT, content="needs approval", task_state=task)

    restored = ChatMessage.model_validate_json(message.model_dump_json())

    assert restored.task_state is not None
    assert restored.task_state.plan is not None
    assert restored.task_state.plan.steps[2].description == "Verify final user goal"
    assert restored.task_state.total_steps == 3


# ---------------------------------------------------------------------------
# Step accessors through TaskState
# ---------------------------------------------------------------------------


def test_task_state_step_accessors():
    task = TaskState(goal="g")
    plan = make_plan()
    task.attach_plan(plan)

    plan.set_step_status(1, StepStatus.SUCCEEDED, result="created")
    plan.set_step_status(2, StepStatus.FAILED, error="boom")

    assert [s.id for s in task.completed_steps()] == [1]
    assert [s.id for s in task.failed_steps()] == [2]
    # Step 3 is still pending work, but it cannot run until step 2 succeeds.
    assert [s.id for s in task.remaining_steps()] == [3]
    assert task.current_plan_step() is None  # step 3 blocked on failed step 2


def test_current_plan_step_respects_dependencies():
    task = TaskState(goal="g")
    task.attach_plan(make_plan())
    assert task.current_plan_step().id == 1

    task.plan.set_step_status(1, StepStatus.SUCCEEDED)
    assert task.current_plan_step().id == 2


def test_step_accessors_without_plan_are_empty():
    task = TaskState(goal="g")
    assert task.remaining_steps() == []
    assert task.completed_steps() == []
    assert task.failed_steps() == []
    assert task.blocked_steps() == []
    assert task.current_plan_step() is None


# ---------------------------------------------------------------------------
# Completion enforcement with a plan
# ---------------------------------------------------------------------------


def test_cannot_complete_while_plan_criteria_unmet():
    task = TaskState(goal="g")
    task.attach_plan(make_plan())
    # The plan guard fires first: PENDING steps block completion.
    with pytest.raises(ValueError, match="plan steps are unfinished"):
        task.mark_complete()
    # Even with the plan fully satisfied, unmet criteria still block.
    for step_id in (1, 2, 3):
        task.plan.set_step_status(step_id, StepStatus.SUCCEEDED)
    with pytest.raises(ValueError, match="incomplete requirements"):
        task.mark_complete()


def test_completes_only_after_criteria_satisfied_and_marked():
    task = TaskState(goal="g")
    task.attach_plan(make_plan())
    for requirement in task.remaining_requirements():
        task.mark_requirement_complete(requirement.description)
    assert task.is_complete() is False  # explicit mark_complete still required
    with pytest.raises(ValueError):  # plan steps still pending
        task.mark_complete()
    for step_id in (1, 2, 3):
        task.plan.set_step_status(step_id, StepStatus.SUCCEEDED)
    task.mark_complete()
    assert task.is_complete() is True
    assert task.status is TaskStatus.TASK_COMPLETED


def test_completion_enforced_even_with_all_steps_succeeded():
    """Plan steps succeeding is not enough — criteria must be satisfied."""
    task = TaskState(goal="g")
    task.attach_plan(make_plan())
    for step_id in (1, 2, 3):
        task.plan.set_step_status(step_id, StepStatus.SUCCEEDED)
    assert task.plan.is_satisfied()
    assert task.is_complete() is False
    with pytest.raises(ValueError):
        task.mark_complete()


def test_summary_includes_plan_context():
    task = TaskState(goal="g")
    task.attach_plan(make_plan())
    summary = task.summary()
    assert "requirements=0/2" in summary
    assert "step=0/3" in summary
