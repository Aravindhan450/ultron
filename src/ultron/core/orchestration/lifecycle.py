"""
ultron.core.orchestration.lifecycle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Agent lifecycle state machine (Fix #7, section 7.1).

An orchestrated agent's lifecycle is explicit and transition-validated:

    PENDING -> ASSIGNED -> RUNNING -> WAITING -> RUNNING -> COMPLETED

Failure states (FAILED / BLOCKED / CANCELLED) are reachable from ANY
active state (PENDING / ASSIGNED / RUNNING / WAITING); COMPLETED is only
reachable from RUNNING.

Lifecycle states:

- PENDING    — created, not yet assigned work
- ASSIGNED   — has an objective + execution context, not yet started
- RUNNING    — actively executing
- WAITING    — paused awaiting input (confirmation / clarification)
- COMPLETED  — finished successfully (terminal)
- FAILED     — failed (terminal)
- BLOCKED    — hard-stopped (security block / policy) (terminal)
- CANCELLED  — cancelled by the controller (terminal)

Terminal states accept NO further transitions. Invalid transitions raise
``ValueError`` — the lifecycle is never left to convention or to the LLM.

This module is pure and deterministic: it knows nothing about engines,
tools, TaskState, or the LLM. TaskState remains the authoritative state for
the overall *task*; this state machine governs the *agent run* that works
on it.
"""

from __future__ import annotations

from enum import Enum


class AgentStatus(str, Enum):
    """Lifecycle state of an orchestrated agent run."""

    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        """True for terminal states (no further transitions allowed)."""
        return self in TERMINAL_STATUSES

    @property
    def is_active(self) -> bool:
        """True for non-terminal, working states."""
        return self in ACTIVE_STATUSES


# The only legal transitions, keyed by source state. Anything not listed is
# invalid: PENDING may only be assigned or cancelled; only ASSIGNED may start;
# only RUNNING may complete or wait; WAITING may only resume or fail/block/
# cancel; terminal states have no outgoing edges at all.
TRANSITIONS: dict[AgentStatus, frozenset[AgentStatus]] = {
    AgentStatus.PENDING: frozenset(
        {
            AgentStatus.ASSIGNED,
            AgentStatus.FAILED,
            AgentStatus.BLOCKED,
            AgentStatus.CANCELLED,
        }
    ),
    AgentStatus.ASSIGNED: frozenset(
        {
            AgentStatus.RUNNING,
            AgentStatus.FAILED,
            AgentStatus.BLOCKED,
            AgentStatus.CANCELLED,
        }
    ),
    AgentStatus.RUNNING: frozenset(
        {
            AgentStatus.WAITING,
            AgentStatus.COMPLETED,
            AgentStatus.FAILED,
            AgentStatus.BLOCKED,
            AgentStatus.CANCELLED,
        }
    ),
    AgentStatus.WAITING: frozenset(
        {
            AgentStatus.RUNNING,
            AgentStatus.FAILED,
            AgentStatus.BLOCKED,
            AgentStatus.CANCELLED,
        }
    ),
    AgentStatus.COMPLETED: frozenset(),
    AgentStatus.FAILED: frozenset(),
    AgentStatus.BLOCKED: frozenset(),
    AgentStatus.CANCELLED: frozenset(),
}

TERMINAL_STATUSES: frozenset[AgentStatus] = frozenset(
    {
        AgentStatus.COMPLETED,
        AgentStatus.FAILED,
        AgentStatus.BLOCKED,
        AgentStatus.CANCELLED,
    }
)

ACTIVE_STATUSES: frozenset[AgentStatus] = frozenset(
    {
        AgentStatus.PENDING,
        AgentStatus.ASSIGNED,
        AgentStatus.RUNNING,
        AgentStatus.WAITING,
    }
)


def can_transition(current: AgentStatus, target: AgentStatus) -> bool:
    """True when ``current -> target`` is a legal lifecycle transition."""
    return target in TRANSITIONS.get(current, frozenset())


def assert_transition(
    current: AgentStatus,
    target: AgentStatus,
    context: str = "agent state",
) -> None:
    """
    Raises ValueError when ``current -> target`` is not a legal transition.

    ``context`` names the record being transitioned for a useful error
    message (e.g. the agent id).
    """
    if current not in TRANSITIONS:
        raise ValueError(f"Unknown agent status '{current}'")
    if target not in TRANSITIONS[current]:
        raise ValueError(
            f"Invalid {context} lifecycle transition: "
            f"'{current.value}' -> '{target.value}'"
        )


def transitions_from(status: AgentStatus) -> frozenset[AgentStatus]:
    """The set of states reachable from ``status`` (empty when terminal)."""
    return TRANSITIONS.get(status, frozenset())
