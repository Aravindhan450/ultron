"""tests.test_autonomous_coding_agent_v1
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Deterministic tests for Ultron Autonomous Coding Agent v1:
- Acceptance criteria lifecycle & completion gating
- Evidence level requirements (Level 0 through Level 6)
- Application lifecycle state transitions
- Failure classification & repair hint generation
- Ineffective repetition detection & strategy change
- TaskPlan and TaskState integration
"""

import pytest

from ultron.core.coding.executor import (
    CodingExecutor,
    FailureCategory,
    RepairBudget,
    classify_failure,
)
from ultron.core.intelligence.task_planning import build_artifact_plan
from ultron.core.types import (
    AcceptanceCriterion,
    AcceptanceCriterionStatus,
    ApplicationLifecycleState,
    EvidenceLevel,
    TaskState,
)


def test_acceptance_criterion_creation_and_defaults():
    crit = AcceptanceCriterion(
        id="app_launch",
        description="Application launches and GUI window appears",
        verification_method="process_check",
        required=True,
    )
    assert crit.id == "app_launch"
    assert crit.status == AcceptanceCriterionStatus.PENDING
    assert crit.evidence_level == EvidenceLevel.LEVEL_0_GENERATED
    assert crit.required is True


def test_task_state_acceptance_criteria_management():
    task = TaskState(goal="Build a Tamil Nadu weather desktop app")
    crit = task.add_acceptance_criterion(
        id="app_launch",
        description="Application launches and GUI window appears",
        verification_method="process_check",
    )
    assert crit.id == "app_launch"
    assert len(task.acceptance_criteria) == 1
    assert not task.all_required_criteria_satisfied()
    assert len(task.remaining_acceptance_criteria()) == 1

    # Verify criterion with Level 3 evidence
    task.verify_acceptance_criterion(
        id="app_launch",
        evidence="Process PID 48291 active and responding",
        level=EvidenceLevel.LEVEL_3_LAUNCHED,
    )
    assert task.all_required_criteria_satisfied()
    assert task.acceptance_criteria[0].status == AcceptanceCriterionStatus.VERIFIED
    assert task.acceptance_criteria[0].evidence_level == EvidenceLevel.LEVEL_3_LAUNCHED


def test_completion_blocked_without_acceptance_criteria():
    task = TaskState(goal="Build an expense tracker")
    task.add_acceptance_criterion(
        id="db_persistence",
        description="Expenses saved to SQLite database",
        verification_method="db_query",
        required=True,
    )
    # Attempting to mark complete without verifying criterion raises ValueError
    with pytest.raises(ValueError, match="unsatisfied acceptance criteria"):
        task.mark_complete()

    assert not task.is_complete()

    # Once verified, mark_complete succeeds
    task.verify_acceptance_criterion(
        id="db_persistence",
        evidence="SELECT count(*) FROM expenses -> 3",
        level=EvidenceLevel.LEVEL_4_INTERACTED,
    )
    task.mark_complete()
    assert task.is_complete()


def test_application_lifecycle_state_tracking():
    task = TaskState(goal="Build web weather app")
    assert task.app_lifecycle_state is None

    task.app_lifecycle_state = ApplicationLifecycleState.CREATED
    assert task.app_lifecycle_state == ApplicationLifecycleState.CREATED

    task.app_lifecycle_state = ApplicationLifecycleState.STARTED
    assert task.app_lifecycle_state == ApplicationLifecycleState.STARTED

    task.app_lifecycle_state = ApplicationLifecycleState.INTERACTED
    assert task.app_lifecycle_state == ApplicationLifecycleState.INTERACTED

    task.app_lifecycle_state = ApplicationLifecycleState.VERIFIED
    assert task.app_lifecycle_state == ApplicationLifecycleState.VERIFIED


def test_failure_classification_categories():
    # Syntax
    f_syntax = classify_failure(
        command="python app.py",
        stderr="SyntaxError: invalid syntax in line 4",
    )
    assert f_syntax.category == FailureCategory.SYNTAX

    # Environment - macOS Tkinter
    f_env = classify_failure(
        command="python app.py",
        stderr="ModuleNotFoundError: No module named '_tkinter'",
    )
    assert f_env.category == FailureCategory.ENVIRONMENT
    assert "/usr/bin/python3" in f_env.repair_hint

    # Network / API
    f_net = classify_failure(
        command="curl http://localhost:8080",
        stderr="ConnectionRefusedError: [Errno 61] Connection refused",
    )
    assert f_net.category == FailureCategory.NETWORK_API

    # Data
    f_data = classify_failure(
        command="python query.py",
        stderr="sqlite3.OperationalError: no such table: expenses",
    )
    assert f_data.category == FailureCategory.DATA

    # UI
    f_ui = classify_failure(
        command="python app.py",
        stderr="_tkinter.TclError: cannot connect to X server",
    )
    assert f_ui.category == FailureCategory.UI


def test_repetition_detector_and_strategy_shift():
    executor = CodingExecutor(budget=RepairBudget(max_identical_actions=2))

    # First attempt fails
    executor.record_observation(
        tool_name="run_command",
        arguments={"command": "python app.py"},
        observation="Exit code: 1\nModuleNotFoundError: No module named '_tkinter'",
        succeeded=False,
    )
    # Gating should allow a second try
    assert executor.gate_action("run_command", {"command": "python app.py"}) is None

    # Second identical failure
    executor.record_observation(
        tool_name="run_command",
        arguments={"command": "python app.py"},
        observation="Exit code: 1\nModuleNotFoundError: No module named '_tkinter'",
        succeeded=False,
    )
    # Third attempt should be BLOCKED and provide strategy shift feedback
    blocked_msg = executor.gate_action("run_command", {"command": "python app.py"})
    assert blocked_msg is not None
    assert "failed 2 time(s)" in blocked_msg
    assert "change your strategy" in blocked_msg
    assert "Diagnosis: environment" in blocked_msg


def test_build_artifact_plan_has_acceptance_criteria(tmp_path, monkeypatch):
    monkeypatch.setenv("ULTRON_WORKSPACE", str(tmp_path))
    plan = build_artifact_plan("Build Tamil Nadu weather app")

    assert len(plan.steps) == 4
    assert len(plan.acceptance_criteria) >= 4
    assert any(c.id == "workspace_isolation" for c in plan.acceptance_criteria)
    assert any(c.id == "application_execution" for c in plan.acceptance_criteria)
    assert any(c.id == "behavior_interaction" for c in plan.acceptance_criteria)
    assert any(c.id == "verified_completion" for c in plan.acceptance_criteria)

    # Attach to TaskState and verify inheritance
    task = TaskState(goal="Build Tamil Nadu weather app")
    task.attach_plan(plan)
    assert len(task.acceptance_criteria) == len(plan.acceptance_criteria)
    assert not task.all_required_criteria_satisfied()
