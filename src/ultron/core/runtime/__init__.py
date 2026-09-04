"""
ultron.core.runtime
~~~~~~~~~~~~~~~~~~~

Phase 1 AgentRuntime Package.

Provides lifecycle management, budget enforcement, cooperative cancellation,
and structured event observability for Ultron agents.
"""

from ultron.core.runtime.budget import RuntimeBudget
from ultron.core.runtime.cancellation import CancellationToken
from ultron.core.runtime.events import (
    EventBus,
    EventListener,
    RuntimeEvent,
    RuntimeEventType,
)
from ultron.core.runtime.result import RunResult
from ultron.core.runtime.runtime import AgentRuntime
from ultron.core.runtime.state import (
    ACTIVE_STATUSES,
    RUNTIME_TRANSITIONS,
    TERMINAL_STATUSES,
    RunState,
    RuntimeStatus,
    assert_runtime_transition,
)

__all__ = [
    "ACTIVE_STATUSES",
    "RUNTIME_TRANSITIONS",
    "TERMINAL_STATUSES",
    "AgentRuntime",
    "CancellationToken",
    "EventBus",
    "EventListener",
    "RunResult",
    "RunState",
    "RuntimeBudget",
    "RuntimeEvent",
    "RuntimeEventType",
    "RuntimeStatus",
    "assert_runtime_transition",
]
