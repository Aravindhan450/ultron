"""
ultron.core.runtime.state
~~~~~~~~~~~~~~~~~~~~~~~~~

Authoritative state machine and runtime tracking model for AgentRuntime.

Lifecycle states:
    CREATED -> INITIALIZING -> RUNNING -> VERIFYING -> COMPLETED
    Terminal failures / stops:
    FAILED, TIMED_OUT, BUDGET_EXCEEDED, CANCELLED
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from ultron.core.runtime.budget import RuntimeBudget


class RuntimeStatus(str, Enum):
    """Lifecycle status of an AgentRuntime execution."""

    CREATED = "created"
    INITIALIZING = "initializing"
    RUNNING = "running"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    BUDGET_EXCEEDED = "budget_exceeded"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        """True if the status represents an end-state with no outgoing transitions."""
        return self in TERMINAL_STATUSES

    @property
    def is_active(self) -> bool:
        """True if the status represents an actively progressing or waiting state."""
        return self in ACTIVE_STATUSES


ACTIVE_STATUSES: frozenset[RuntimeStatus] = frozenset(
    {
        RuntimeStatus.CREATED,
        RuntimeStatus.INITIALIZING,
        RuntimeStatus.RUNNING,
        RuntimeStatus.VERIFYING,
    }
)

TERMINAL_STATUSES: frozenset[RuntimeStatus] = frozenset(
    {
        RuntimeStatus.COMPLETED,
        RuntimeStatus.FAILED,
        RuntimeStatus.TIMED_OUT,
        RuntimeStatus.BUDGET_EXCEEDED,
        RuntimeStatus.CANCELLED,
    }
)

# Deterministic transition table
RUNTIME_TRANSITIONS: dict[RuntimeStatus, frozenset[RuntimeStatus]] = {
    RuntimeStatus.CREATED: frozenset(
        {
            RuntimeStatus.INITIALIZING,
            RuntimeStatus.RUNNING,
            RuntimeStatus.FAILED,
            RuntimeStatus.CANCELLED,
        }
    ),
    RuntimeStatus.INITIALIZING: frozenset(
        {
            RuntimeStatus.RUNNING,
            RuntimeStatus.FAILED,
            RuntimeStatus.CANCELLED,
            RuntimeStatus.TIMED_OUT,
        }
    ),
    RuntimeStatus.RUNNING: frozenset(
        {
            RuntimeStatus.VERIFYING,
            RuntimeStatus.COMPLETED,
            RuntimeStatus.FAILED,
            RuntimeStatus.TIMED_OUT,
            RuntimeStatus.BUDGET_EXCEEDED,
            RuntimeStatus.CANCELLED,
        }
    ),
    RuntimeStatus.VERIFYING: frozenset(
        {
            RuntimeStatus.RUNNING,
            RuntimeStatus.COMPLETED,
            RuntimeStatus.FAILED,
            RuntimeStatus.TIMED_OUT,
            RuntimeStatus.BUDGET_EXCEEDED,
            RuntimeStatus.CANCELLED,
        }
    ),
    # Terminal states have NO outgoing edges
    RuntimeStatus.COMPLETED: frozenset(),
    RuntimeStatus.FAILED: frozenset(),
    RuntimeStatus.TIMED_OUT: frozenset(),
    RuntimeStatus.BUDGET_EXCEEDED: frozenset(),
    RuntimeStatus.CANCELLED: frozenset(),
}


def assert_runtime_transition(
    current: RuntimeStatus, target: RuntimeStatus
) -> None:
    """
    Validates a state transition. Raises ValueError on illegal transitions.
    """
    allowed = RUNTIME_TRANSITIONS.get(current, frozenset())
    if target not in allowed:
        raise ValueError(
            f"Illegal runtime transition from {current.value} to {target.value}. "
            f"Allowed targets: {[s.value for s in allowed]}"
        )


class RunState(BaseModel):
    """
    Authoritative state tracked during one AgentRuntime execution run.
    Integrates with TaskState rather than duplicating task domain entities.
    """

    run_id: str
    task_id: str | None = None
    status: RuntimeStatus = RuntimeStatus.CREATED
    started_at: datetime | None = None
    finished_at: datetime | None = None
    budget: RuntimeBudget = Field(default_factory=RuntimeBudget)
    error: str | None = None
    cancellation_requested: bool = False
    cancellation_reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def transition_to(self, new_status: RuntimeStatus, error: str | None = None) -> None:
        """
        Safely transitions the run to new_status or raises ValueError.
        """
        assert_runtime_transition(self.status, new_status)
        now = datetime.now(UTC)

        if (
            self.status is RuntimeStatus.CREATED
            and new_status in (RuntimeStatus.INITIALIZING, RuntimeStatus.RUNNING)
            and self.started_at is None
        ):
            self.started_at = now
            self.budget.started_at = now

        self.status = new_status
        if error is not None:
            self.error = error

        if new_status.is_terminal and self.finished_at is None:
            self.finished_at = now

    def request_cancellation(self, reason: str = "User cancelled run") -> None:
        """Flags cancellation cooperatively."""
        self.cancellation_requested = True
        self.cancellation_reason = reason
