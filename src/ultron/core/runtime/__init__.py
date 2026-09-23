"""
ultron.core.runtime
~~~~~~~~~~~~~~~~~~~

Phase 1 AgentRuntime Package.

Provides lifecycle management, budget enforcement, cooperative cancellation,
and structured event observability for Ultron agents.
"""

from ultron.core.runtime.budget import BudgetExceededError, RuntimeBudget
from ultron.core.runtime.cancellation import CancellationToken
from ultron.core.runtime.event_store import (
    EventStore,
    InMemoryEventStore,
    JsonlEventStore,
    SequenceViolationError,
    get_default_event_store,
    reset_default_event_store,
    set_default_event_store,
)
from ultron.core.runtime.events import (
    EventBus,
    EventListener,
    RuntimeEvent,
    RuntimeEventType,
    TaskEvent,
    TaskEventType,
    sanitize_event_payload,
)
from ultron.core.runtime.projection import project_task_state
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
from ultron.core.runtime.task_state import (
    CanonicalTaskState,
    InvalidStateTransitionError,
    TaskLifecycleStatus,
    TaskState,
    assert_task_transition,
)

__all__ = [
    "ACTIVE_STATUSES",
    "RUNTIME_TRANSITIONS",
    "TERMINAL_STATUSES",
    "AgentRuntime",
    "BudgetExceededError",
    "CancellationToken",
    "CanonicalTaskState",
    "EventBus",
    "EventListener",
    "EventStore",
    "InMemoryEventStore",
    "InvalidStateTransitionError",
    "JsonlEventStore",
    "RunResult",
    "RunState",
    "RuntimeBudget",
    "RuntimeEvent",
    "RuntimeEventType",
    "RuntimeStatus",
    "SequenceViolationError",
    "TaskEvent",
    "TaskEventType",
    "TaskLifecycleStatus",
    "TaskState",
    "assert_runtime_transition",
    "assert_task_transition",
    "get_default_event_store",
    "project_task_state",
    "reset_default_event_store",
    "sanitize_event_payload",
    "set_default_event_store",
]
