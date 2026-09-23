"""
ultron.core.runtime.projection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Deterministic state projector / reducer that folds a stream of TaskEvents
into a canonical TaskState.

Guarantees:
- Pure reducer semantics: given the exact same sequence of events, produces
  an identical TaskState.
- Rebuilds lifecycle status, requirements, acceptance criteria, plan, execution
  history, errors, routing, and recovery references.
"""

from __future__ import annotations

from collections.abc import Sequence

from ultron.core.runtime.events import TaskEvent, TaskEventType
from ultron.core.types import (
    TaskLifecycleStatus,
    TaskPlan,
    TaskState,
)


def project_task_state(
    events: Sequence[TaskEvent], initial: TaskState | None = None
) -> TaskState:
    """
    Folds an ordered sequence of TaskEvents into a TaskState.
    """
    sorted_events = sorted(events, key=lambda e: e.sequence)

    task: TaskState | None = initial

    for event in sorted_events:
        payload = event.payload or {}
        etype = event.event_type

        # Task creation
        if etype == TaskEventType.TASK_CREATED:
            goal = payload.get("goal", "")
            task = TaskState(
                task_id=event.task_id,
                goal=goal,
                parent_task_id=payload.get("parent_task_id"),
                session_id=payload.get("session_id"),
                workspace_root=payload.get("workspace_root"),
                created_at=event.timestamp,
                updated_at=event.timestamp,
            )
            continue

        if task is None:
            # If the event stream didn't start with TASK_CREATED, synthesize initial TaskState
            task = TaskState(
                task_id=event.task_id,
                goal=payload.get("goal", "Synthetic task"),
                created_at=event.timestamp,
                updated_at=event.timestamp,
            )

        task.updated_at = event.timestamp

        # Task state transitions
        if etype == TaskEventType.TASK_STATE_CHANGED:
            target_str = payload.get("to_state") or payload.get("target") or payload.get("status")
            reason = payload.get("reason", "")
            if target_str:
                try:
                    target_status = TaskLifecycleStatus(target_str)
                    task.transition_to(target_status, reason=reason)
                except ValueError:
                    pass

        elif etype == TaskEventType.TASK_COMPLETED:
            if task.lifecycle_status != TaskLifecycleStatus.COMPLETED:
                task.transition_to(TaskLifecycleStatus.COMPLETED, reason="completed event")

        elif etype == TaskEventType.TASK_FAILED:
            error_msg = payload.get("error") or payload.get("message") or "Task failed"
            step = payload.get("step")
            task.record_failure(error_msg, step=step)

        elif etype == TaskEventType.TASK_BLOCKED:
            msg = payload.get("reason") or payload.get("message") or "Task blocked"
            task.block(msg)

        elif etype == TaskEventType.TASK_CANCELLED:
            if task.lifecycle_status != TaskLifecycleStatus.CANCELLED:
                task.transition_to(TaskLifecycleStatus.CANCELLED, reason="cancelled event")

        elif etype == TaskEventType.CONFIRMATION_REQUESTED:
            task.wait_for_confirmation()

        elif etype == TaskEventType.CONFIRMATION_RESOLVED:
            decision = payload.get("decision", "approve")
            if decision == "approve":
                task.resume()
            else:
                task.block(f"Confirmation rejected: {decision}")

        # Planning
        elif etype == TaskEventType.PLAN_PROPOSED:
            plan_data = payload.get("plan")
            if isinstance(plan_data, dict):
                try:
                    plan = TaskPlan.model_validate(plan_data)
                    task.attach_plan(plan)
                except Exception:  # noqa: BLE001, S110
                    pass
            elif isinstance(plan_data, TaskPlan):
                task.attach_plan(plan_data)

            if "requirements" in payload:
                for req_desc in payload["requirements"]:
                    try:
                        task.add_requirement(req_desc)
                    except ValueError:
                        pass

            if "acceptance_criteria" in payload:
                for crit_data in payload["acceptance_criteria"]:
                    if isinstance(crit_data, dict):
                        task.add_acceptance_criterion(
                            id=crit_data.get("id", ""),
                            description=crit_data.get("description", ""),
                            verification_method=crit_data.get("verification_method", "execution"),
                            required=crit_data.get("required", True),
                        )

        elif etype == TaskEventType.PLAN_REVISED:
            revision_note = payload.get("note") or payload.get("revision", "Plan revised")
            task.record_plan_revision(revision_note)

        # Model Routing
        elif etype == TaskEventType.MODEL_SELECTED:
            model_name = payload.get("model") or payload.get("selected_model")
            task.selected_model = model_name
            task.routing_decision = payload

        # Execution
        elif etype == TaskEventType.TOOL_STARTED:
            tool_name = payload.get("tool_name", "")
            action_id = payload.get("action_id", "")
            if action_id and action_id not in task.active_action_ids:
                task.active_action_ids.append(action_id)

        elif etype == TaskEventType.TOOL_COMPLETED:
            tool_name = payload.get("tool_name", "")
            action_id = payload.get("action_id", "")
            success = payload.get("success", True)
            target = payload.get("target", "")
            detail = payload.get("detail", "")

            if action_id:
                if action_id in task.active_action_ids:
                    task.active_action_ids.remove(action_id)
                if success:
                    task.completed_action_ids.append(action_id)
                else:
                    task.failed_action_ids.append(action_id)

            task.record_tool_execution(
                tool_name=tool_name,
                target=target,
                success=success,
                detail=detail,
            )

        # Verification
        elif etype == TaskEventType.VERIFICATION_STARTED:
            task.requires_verification = True
            task.verification_status = "started"
            if task.lifecycle_status in (
                TaskLifecycleStatus.EXECUTING,
                TaskLifecycleStatus.REPAIRING,
            ):
                task.transition_to(TaskLifecycleStatus.VERIFYING, reason="verification started")

        elif etype == TaskEventType.VERIFICATION_COMPLETED:
            passed = payload.get("passed", False)
            task.verification_status = "passed" if passed else "failed"
            task.last_verification_result = payload

            # Verify acceptance criteria mentioned
            for crit_id in payload.get("verified_criteria", []):
                evidence = payload.get("evidence", "Verified")
                task.verify_acceptance_criterion(crit_id, evidence=evidence)

            for crit_id in payload.get("failed_criteria", []):
                err = payload.get("error", "Verification failed")
                task.fail_acceptance_criterion(crit_id, error=err)

        # Repair
        elif etype == TaskEventType.REPAIR_ATTEMPTED:
            if task.lifecycle_status in (
                TaskLifecycleStatus.VERIFYING,
                TaskLifecycleStatus.EXECUTING,
                TaskLifecycleStatus.FAILED,
            ):
                task.transition_to_repair()

    if task is None:
        raise ValueError("Cannot project TaskState from empty event stream with no initial state.")

    return task
