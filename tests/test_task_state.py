"""
Unit tests for TaskState — the explicit task + completion-state abstraction.

Covers lifecycle states (started / running / waiting / failed / blocked /
completed), generic requirement-based completion criteria, execution-history
recording, and JSON serialization.

The central invariant under test: an intermediate tool success never
completes a task — completion is always explicit.
"""

import json

import pytest

from ultron.core.types import (
    TaskError,
    TaskRequirement,
    TaskState,
    TaskStatus,
    ToolExecution,
)


def test_new_task_starts_correctly():
    task = TaskState(goal="Create a TodoList application")
    assert task.goal == "Create a TodoList application"
    assert task.status == TaskStatus.TASK_STARTED
    assert task.requirements == []
    assert task.current_step == 0
    assert task.total_steps == 0
    assert task.execution_history == []
    assert task.errors == []
    assert task.is_complete() is False
    assert task.is_blocked is False
    assert task.is_waiting_confirmation is False
    assert task.started_at is not None
    assert task.updated_at is not None


def test_requirements_can_be_added():
    task = TaskState(goal="build an app")
    added = task.add_requirement("TodoList directory exists")
    task.add_requirements(["application files exist", "application can run"])
    assert isinstance(added, TaskRequirement)
    assert len(task.requirements) == 3
    assert task.remaining_requirements() == task.requirements
    assert task.completed_requirements == []
    assert all(r.completed is False for r in task.requirements)
    # requirements are typed models, not bare strings
    assert all(isinstance(r, TaskRequirement) for r in task.requirements)


def test_requirements_can_be_completed():
    task = TaskState(goal="build an app")
    task.add_requirements(["dir exists", "app runs"])
    task.mark_requirement_complete("dir exists")
    assert task.requirements[0].completed is True
    assert len(task.completed_requirements) == 1
    assert [r.description for r in task.remaining_requirements()] == ["app runs"]


def test_mark_requirement_incomplete_reverts():
    task = TaskState(goal="build an app")
    task.add_requirement("app runs")
    task.mark_requirement_complete("app runs")
    assert task.is_complete() is False  # explicit completion still required
    task.mark_requirement_incomplete("app runs")
    assert task.requirements[0].completed is False
    assert len(task.remaining_requirements()) == 1


def test_unknown_requirement_raises():
    task = TaskState(goal="x")
    task.add_requirement("known")
    with pytest.raises(ValueError):
        task.mark_requirement_complete("unknown")
    with pytest.raises(ValueError):
        task.mark_requirement_incomplete("unknown")


def test_duplicate_requirement_raises():
    task = TaskState(goal="g")
    task.add_requirement("dir exists")
    with pytest.raises(ValueError, match="already exists"):
        task.add_requirement("dir exists")
    with pytest.raises(ValueError):
        task.add_requirements(["a", "a"])


def test_incomplete_requirements_prevent_completion():
    task = TaskState(goal="Create a TodoList application")
    task.add_requirements(["TodoList directory exists", "application can run"])
    task.mark_requirement_complete("TodoList directory exists")
    with pytest.raises(ValueError, match="incomplete requirements"):
        task.mark_complete()
    assert task.is_complete() is False


def test_all_requirements_completed_results_in_completion():
    task = TaskState(goal="Create a TodoList application")
    task.add_requirements(
        [
            "TodoList directory exists",
            "application files exist",
            "application can run",
        ]
    )
    for req in task.remaining_requirements():
        task.mark_requirement_complete(req.description)
    assert task.is_complete() is False  # explicit mark still required
    task.mark_complete()
    assert task.is_complete() is True
    assert task.status == TaskStatus.TASK_COMPLETED
    assert task.remaining_requirements() == []


def test_task_with_no_requirements_can_complete_explicitly():
    task = TaskState(goal="hello")
    assert task.is_complete() is False  # never implicit
    task.mark_complete()
    assert task.is_complete() is True


def test_tool_success_alone_never_completes_task():
    # Requirement 3: a tool succeeding / exit code 0 must not complete a task.
    task = TaskState(goal="Create a TodoList application")
    task.add_requirement("TodoList directory exists")
    task.record_tool_execution("run_command", target="mkdir TodoList", success=True)
    task.record_tool_execution("write_file", target="TodoList/index.html", success=True)
    assert task.is_complete() is False
    assert task.status == TaskStatus.TASK_STARTED  # unchanged by tool results


def test_tool_execution_history_is_recorded():
    task = TaskState(goal="g")
    entry = task.record_tool_execution(
        "run_command", target="mkdir TodoList", success=True, detail="Exit code: 0"
    )
    assert len(task.execution_history) == 1
    assert isinstance(entry, ToolExecution)
    assert entry.tool_name == "run_command"
    assert entry.target == "mkdir TodoList"
    assert entry.success is True
    assert entry.detail == "Exit code: 0"
    assert entry.timestamp is not None

    failed = task.record_tool_execution(
        "write_file", target="x.txt", success=False, detail="Error"
    )
    assert failed.success is False
    assert len(task.execution_history) == 2
    # even a failed tool entry does not flip the status on its own
    assert task.status == TaskStatus.TASK_STARTED


def test_record_failure_marks_task_failed():
    task = TaskState(goal="g")
    task.status = TaskStatus.TASK_RUNNING
    error = task.record_failure("step 2 blew up", step=2)
    assert isinstance(error, TaskError)
    assert error.message == "step 2 blew up"
    assert error.step == 2
    assert error.timestamp is not None
    assert len(task.errors) == 1
    assert task.status == TaskStatus.TASK_FAILED


def test_failed_task_cannot_report_completed():
    task = TaskState(goal="g")
    task.record_failure("boom")
    assert task.is_complete() is False
    with pytest.raises(ValueError):
        task.mark_complete()
    # a failed task cannot be resumed or wait for confirmation either
    with pytest.raises(ValueError):
        task.resume()
    with pytest.raises(ValueError):
        task.wait_for_confirmation()


def test_blocked_task_cannot_report_completed():
    task = TaskState(goal="g")
    task.block("blocked by security")
    assert task.status == TaskStatus.TASK_BLOCKED
    assert task.is_blocked is True
    assert task.is_complete() is False
    assert task.errors[0].message == "blocked by security"
    with pytest.raises(ValueError):
        task.mark_complete()
    with pytest.raises(ValueError):
        task.resume()


def test_waiting_confirmation_state_can_resume():
    task = TaskState(goal="g")
    task.status = TaskStatus.TASK_RUNNING
    task.wait_for_confirmation()
    assert task.status == TaskStatus.WAITING_CONFIRMATION
    assert task.is_waiting_confirmation is True
    task.resume()
    assert task.status == TaskStatus.TASK_RUNNING
    assert task.is_waiting_confirmation is False


def test_resume_from_started_starts_running():
    task = TaskState(goal="g")
    task.resume()
    assert task.status == TaskStatus.TASK_RUNNING


def test_terminal_states_reject_confirmation_wait():
    task = TaskState(goal="g")
    task.mark_complete()
    with pytest.raises(ValueError):
        task.wait_for_confirmation()
    with pytest.raises(ValueError):
        task.resume()
    with pytest.raises(ValueError):
        task.block("nope")  # cannot block a completed task


def test_set_current_step_and_total_steps():
    task = TaskState(goal="g")
    task.set_current_step(3)
    assert task.current_step == 3
    task.set_total_steps(5)
    assert task.total_steps == 5
    with pytest.raises(ValueError):
        task.set_current_step(-1)
    with pytest.raises(ValueError):
        task.set_total_steps(-1)


def test_record_failure_does_not_downgrade_blocked():
    task = TaskState(goal="g")
    task.block("blocked by security")
    task.record_failure("late failure")
    assert task.status == TaskStatus.TASK_BLOCKED
    assert len(task.errors) == 2  # block message + failure


def test_record_failure_does_not_downgrade_completed():
    task = TaskState(goal="g")
    task.mark_complete()
    task.record_failure("late failure")
    assert task.status == TaskStatus.TASK_COMPLETED
    assert task.is_complete() is True


def test_resume_is_idempotent_from_running():
    task = TaskState(goal="g")
    task.status = TaskStatus.TASK_RUNNING
    task.resume()
    assert task.status == TaskStatus.TASK_RUNNING


def test_deserialized_incomplete_status_cannot_report_complete():
    # Defense-in-depth: even a corrupt payload claiming completion while
    # requirements remain incomplete must not report complete.
    task = TaskState(goal="g")
    task.add_requirement("unfinished")
    task.status = TaskStatus.TASK_COMPLETED
    assert task.is_complete() is False


def test_state_serialization_roundtrip():
    task = TaskState(goal="Create a TodoList application")
    task.add_requirements(["dir exists", "app runs"])
    task.mark_requirement_complete("dir exists")
    task.set_current_step(2)
    task.total_steps = 4
    task.record_tool_execution("run_command", target="mkdir TodoList", success=True)
    task.wait_for_confirmation()
    task.resume()

    restored = TaskState.model_validate_json(task.model_dump_json())

    assert restored.goal == task.goal
    assert restored.status == task.status
    assert [r.description for r in restored.requirements] == [
        r.description for r in task.requirements
    ]
    assert [r.completed for r in restored.requirements] == [
        r.completed for r in task.requirements
    ]
    assert restored.current_step == task.current_step
    assert restored.total_steps == task.total_steps
    assert len(restored.execution_history) == len(task.execution_history)
    assert restored.execution_history[0].tool_name == "run_command"
    assert restored.execution_history[0].target == "mkdir TodoList"
    assert restored.is_complete() is False

    # model_dump(mode="json") is also valid for logging and matches the JSON
    dumped = task.model_dump(mode="json")
    assert json.loads(task.model_dump_json()) == dumped


def test_json_roundtrip_preserves_terminal_state():
    task = TaskState(goal="g")
    task.add_requirement("r")
    task.mark_requirement_complete("r")
    task.mark_complete()
    restored = TaskState.model_validate_json(task.model_dump_json())
    assert restored.is_complete() is True


def test_summary_is_debuggable():
    task = TaskState(goal="Create a TodoList application")
    task.add_requirements(["dir exists", "app runs"])
    task.mark_requirement_complete("dir exists")
    task.record_tool_execution("run_command", target="mkdir TodoList")
    text = task.summary()
    assert "Create a TodoList application" in text
    assert "task_started" in text
    assert "requirements=1/2" in text
    assert "tools=1" in text
    assert "errors=0" in text
