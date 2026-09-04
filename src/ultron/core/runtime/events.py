"""
ultron.core.runtime.events
~~~~~~~~~~~~~~~~~~~~~~~~~~

Lightweight, in-process, typed event system for AgentRuntime executions.

Provides structured visibility across lifecycle stages:
- RUN_STARTED, STEP_STARTED, MODEL_CALLED
- TOOL_STARTED, TOOL_COMPLETED
- DELEGATION_STARTED, DELEGATION_COMPLETED
- VERIFICATION_STARTED, VERIFICATION_COMPLETED
- RUN_COMPLETED, RUN_FAILED, RUN_CANCELLED
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Coroutine
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from ultron.core.logging import get_logger

logger = get_logger("ultron.runtime.events")


class RuntimeEventType(str, Enum):
    """Structured event types emitted across the agent runtime lifecycle."""

    RUN_STARTED = "run_started"
    STEP_STARTED = "step_started"
    MODEL_CALLED = "model_called"
    TOOL_STARTED = "tool_started"
    TOOL_COMPLETED = "tool_completed"
    DELEGATION_STARTED = "delegation_started"
    DELEGATION_COMPLETED = "delegation_completed"
    VERIFICATION_STARTED = "verification_started"
    VERIFICATION_COMPLETED = "verification_completed"
    RUN_COMPLETED = "run_completed"
    RUN_FAILED = "run_failed"
    RUN_CANCELLED = "run_cancelled"


class RuntimeEvent(BaseModel):
    """
    Immutable representation of an event emitted during an AgentRuntime run.
    """

    event_type: RuntimeEventType
    run_id: str
    task_id: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    payload: dict[str, Any] = Field(default_factory=dict)


# Type aliases for sync or async listeners
EventListener = Callable[[RuntimeEvent], Any | Coroutine[Any, Any, Any]]


class EventBus:
    """
    In-process publisher/subscriber event bus for AgentRuntime executions.
    """

    def __init__(self) -> None:
        self._listeners: dict[RuntimeEventType, list[EventListener]] = defaultdict(list)
        self._global_listeners: list[EventListener] = []
        self._history: list[RuntimeEvent] = []

    def subscribe(
        self,
        listener: EventListener,
        event_types: list[RuntimeEventType] | RuntimeEventType | None = None,
    ) -> None:
        """
        Subscribe a listener to specific event types or all events (if event_types is None).
        """
        if event_types is None:
            if listener not in self._global_listeners:
                self._global_listeners.append(listener)
            return

        if isinstance(event_types, RuntimeEventType):
            event_types = [event_types]

        for et in event_types:
            if listener not in self._listeners[et]:
                self._listeners[et].append(listener)

    def unsubscribe(
        self,
        listener: EventListener,
        event_types: list[RuntimeEventType] | RuntimeEventType | None = None,
    ) -> None:
        """
        Unsubscribe a listener from specific event types or global listener list.
        """
        if event_types is None:
            if listener in self._global_listeners:
                self._global_listeners.remove(listener)
            for listeners_for_type in self._listeners.values():
                if listener in listeners_for_type:
                    listeners_for_type.remove(listener)
            return

        if isinstance(event_types, RuntimeEventType):
            event_types = [event_types]

        for et in event_types:
            if listener in self._listeners[et]:
                self._listeners[et].remove(listener)

    async def emit(self, event: RuntimeEvent) -> None:
        """
        Emits an event to all matching listeners and records it to history.
        """
        self._history.append(event)
        targets = list(self._global_listeners) + list(self._listeners.get(event.event_type, []))

        for listener in targets:
            try:
                res = listener(event)
                if hasattr(res, "__await__"):
                    await res
            except Exception as exc:  # noqa: BLE001 — event listeners must never crash runtime
                logger.debug(f"Event listener failed on {event.event_type}: {exc}")

    def emit_sync(self, event: RuntimeEvent) -> None:
        """
        Synchronous emission helper for non-async handlers (schedules or ignores coroutines).
        """
        self._history.append(event)
        targets = list(self._global_listeners) + list(self._listeners.get(event.event_type, []))

        for listener in targets:
            try:
                listener(event)
            except Exception as exc:  # noqa: BLE001 — event listeners must never crash runtime
                logger.debug(f"Sync event listener failed on {event.event_type}: {exc}")

    @property
    def history(self) -> list[RuntimeEvent]:
        """Returns recorded events in chronologic order."""
        return list(self._history)

    def clear(self) -> None:
        """Clears listener lists and event history."""
        self._listeners.clear()
        self._global_listeners.clear()
        self._history.clear()
