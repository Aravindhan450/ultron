"""Tests for deterministic plan validation (Fix #2)."""

from __future__ import annotations

from ultron.core.intelligence.plan_validation import validate_plan
from ultron.core.types import PlanStep, TaskPlan, TaskType


def step(
    step_id: int,
    deps: tuple[int, ...] = (),
    criteria: tuple[str, ...] = ("done",),
) -> PlanStep:
    return PlanStep(
        id=step_id,
        description=f"step {step_id}",
        expected_outcome=f"outcome {step_id}",
        completion_criteria=list(criteria),
        dependencies=list(deps),
    )


def make_plan(
    steps: list[PlanStep],
    task_type: TaskType = TaskType.SOFTWARE_ENGINEERING,
    criteria: tuple[str, ...] = ("goal done",),
    verification: tuple[str, ...] = ("verify",),
) -> TaskPlan:
    return TaskPlan(
        goal="goal",
        task_type=task_type,
        steps=steps,
        completion_criteria=list(criteria),
        verification_requirements=list(verification),
    )


def test_valid_plan_passes():
    report = validate_plan(make_plan([step(1), step(2, deps=(1,))]))
    assert report.valid
    assert report.issues == []
    assert report.circular_dependencies == []
    assert report.unreachable_steps == []


def test_duplicate_step_ids_rejected():
    report = validate_plan(make_plan([step(1), step(1)]))
    assert not report.valid
    assert any(i.code == "duplicate_step_id" for i in report.issues)


def test_self_dependency_rejected():
    report = validate_plan(make_plan([step(1, deps=(1,))]))
    assert not report.valid
    assert any(i.code == "self_dependency" for i in report.issues)


def test_unknown_dependency_rejected():
    report = validate_plan(make_plan([step(1), step(2, deps=(7,))]))
    assert not report.valid
    assert any(i.code == "unknown_dependency" for i in report.issues)


def test_circular_dependency_rejected():
    report = validate_plan(make_plan([step(1, deps=(2,)), step(2, deps=(1,))]))
    assert not report.valid
    assert report.circular_dependencies == [[1, 2]]
    assert any(i.code == "circular_dependency" for i in report.issues)


def test_nested_circular_dependency_rejected():
    # 1 -> 2 -> 3 -> 2 : cycle on {2, 3}
    report = validate_plan(
        make_plan([step(1, deps=(2,)), step(2, deps=(3,)), step(3, deps=(2,))])
    )
    assert not report.valid
    assert [2, 3] in report.circular_dependencies


def test_unreachable_step_reported():
    report = validate_plan(make_plan([step(1), step(2, deps=(1,)), step(3, deps=(99,))]))
    assert not report.valid
    assert report.unreachable_steps == [3]


def test_missing_step_criteria_rejected():
    report = validate_plan(
        make_plan(
            [
                PlanStep(id=1, description="s", expected_outcome="o"),
                step(2, deps=(1,)),
            ]
        )
    )
    assert not report.valid
    assert any(i.code == "missing_step_criteria" for i in report.issues)


def test_missing_plan_criteria_rejected_for_complex_type():
    report = validate_plan(
        make_plan([step(1)], criteria=(), verification=("verify",))
    )
    assert not report.valid
    assert any(i.code == "missing_plan_criteria" for i in report.issues)


def test_missing_verification_rejected_for_complex_type():
    report = validate_plan(
        make_plan([step(1)], criteria=("done",), verification=())
    )
    assert not report.valid
    assert any(i.code == "missing_verification" for i in report.issues)


def test_empty_steps_rejected_for_action_type():
    report = validate_plan(make_plan([]))
    assert not report.valid
    assert any(i.code == "no_steps" for i in report.issues)


def test_informational_plan_needs_no_criteria():
    report = validate_plan(
        TaskPlan(goal="what is x", task_type=TaskType.INFORMATIONAL, steps=[])
    )
    assert report.valid


def test_simple_action_plan_without_steps_rejected():
    report = validate_plan(
        TaskPlan(
            goal="list files",
            task_type=TaskType.SIMPLE_ACTION,
            steps=[],
            completion_criteria=["done"],
            verification_requirements=["verify"],
        )
    )
    assert not report.valid
    assert any(i.code == "no_steps" for i in report.issues)
