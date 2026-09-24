"""
ultron.core.runtime.events
~~~~~~~~~~~~~~~~~~~~~~~~~~

Durable, typed, append-only event system for Ultron tasks and AgentRuntime executions.

Provides:
- TaskEventType: Authoritative enum of 20 domain events across the task lifecycle.
- TaskEvent: Immutable, frozen, sequence-ordered event model with secret redaction.
- RuntimeEventType / RuntimeEvent: Backwards-compatible aliases and models.
- EventBus: In-process pub/sub event bus with optional persistent EventStore backing.
- sanitize_event_payload: Deep credential & secret scrubber.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable, Coroutine
from datetime import UTC, datetime
from enum import Enum, unique
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from ultron.core.logging import get_logger
from ultron.security.scanners.secret import mask_matches, scan_secrets

if TYPE_CHECKING:
    from ultron.core.runtime.event_store import EventStore

logger = get_logger("ultron.runtime.events")


# ---------------------------------------------------------------------------
# Task Event Types (Phase 1 Taxonomy)
# ---------------------------------------------------------------------------


@unique
class TaskEventType(str, Enum):
    """
    Authoritative taxonomy of task events across the Ultron runtime lifecycle.
    """

    # Lifecycle & Identity
    TASK_CREATED = "task_created"
    TASK_STATE_CHANGED = "task_state_changed"
    TASK_BLOCKED = "task_blocked"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    TASK_CANCELLED = "task_cancelled"

    # Context & Planning
    CONTEXT_RESOLVED = "context_resolved"
    CONTEXT_BUILT = "context_built"
    PLAN_PROPOSED = "plan_proposed"
    PLAN_VALIDATED = "plan_validated"
    PLAN_REVISED = "plan_revised"

    # Execution & Routing
    MODEL_SELECTED = "model_selected"
    ACTION_PROPOSED = "action_proposed"
    SECURITY_EVALUATED = "security_evaluated"
    CONFIRMATION_REQUESTED = "confirmation_requested"
    CONFIRMATION_RESOLVED = "confirmation_resolved"
    TOOL_STARTED = "tool_started"
    TOOL_COMPLETED = "tool_completed"

    # Verification & Repair
    VERIFICATION_STARTED = "verification_started"
    VERIFICATION_COMPLETED = "verification_completed"
    REPAIR_ATTEMPTED = "repair_attempted"

    # Legacy Runtime Aliases (backwards compatibility)
    RUN_STARTED = "run_started"
    STEP_STARTED = "step_started"
    MODEL_CALLED = "model_called"
    DELEGATION_STARTED = "delegation_started"
    DELEGATION_COMPLETED = "delegation_completed"
    RUN_COMPLETED = "run_completed"
    RUN_FAILED = "run_failed"
    RUN_CANCELLED = "run_cancelled"


# Backwards compatibility alias
RuntimeEventType = TaskEventType


# ---------------------------------------------------------------------------
# Secret & PII Scrubbing
# ---------------------------------------------------------------------------

_SENSITIVE_KEY_RE = re.compile(
    r"(?i)(password|passwd|secret|api[_-]?key|token|auth|credential|private[_-]?key|jwt)"
)


def sanitize_event_payload(payload: Any) -> Any:
    """
    Recursively scrubs secrets, credentials, API keys, and sensitive tokens
    from event payloads before serialization or persistence.
    """
    if isinstance(payload, dict):
        sanitized: dict[str, Any] = {}
        for key, value in payload.items():
            str_key = str(key)
            if _SENSITIVE_KEY_RE.search(str_key) and isinstance(value, (str, bytes)):
                sanitized[str_key] = "********"
            else:
                sanitized[str_key] = sanitize_event_payload(value)
        return sanitized
    elif isinstance(payload, list):
        return [sanitize_event_payload(item) for item in payload]
    elif isinstance(payload, tuple):
        return tuple(sanitize_event_payload(item) for item in payload)
    elif isinstance(payload, str):
        findings = scan_secrets(payload)
        if findings:
            return mask_matches(payload, findings)
        return payload
    return payload


# ---------------------------------------------------------------------------
# Immutable Task Event Model
# ---------------------------------------------------------------------------


class TaskEvent(BaseModel):
    """
    Durable, immutable representation of a single lifecycle event.

    Guarantees:
    - sequence numbering >= 1 per task
    - frozen immutability
    - secret sanitization on creation
    - correlation and causation tracking
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    task_id: str
    sequence: int = Field(default=1, ge=1)
    event_type: TaskEventType
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    payload: dict[str, Any] = Field(default_factory=dict)
    source: str = "runtime"
    correlation_id: str | None = None
    causation_id: str | None = None

    def model_post_init(self, __context: Any, /) -> None:
        """Sanitizes payload immediately upon instantiation."""
        if self.payload:
            sanitized = sanitize_event_payload(self.payload)
            object.__setattr__(self, "payload", sanitized)


# Backwards compatibility wrapper for RuntimeEvent
class RuntimeEvent(BaseModel):
    """
    Backwards-compatible wrapper around runtime events.
    Can be converted to/from TaskEvent.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    event_type: TaskEventType
    run_id: str
    task_id: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    payload: dict[str, Any] = Field(default_factory=dict)

    def to_task_event(self, sequence: int = 1) -> TaskEvent:
        tid = self.task_id or self.run_id
        return TaskEvent(
            task_id=tid,
            sequence=sequence,
            event_type=self.event_type,
            timestamp=self.timestamp,
            payload=self.payload,
            source=f"run_{self.run_id}",
            correlation_id=self.run_id,
        )


# Type aliases for sync or async listeners
EventListener = Callable[[TaskEvent | RuntimeEvent], Any | Coroutine[Any, Any, Any]]


# ---------------------------------------------------------------------------
# EventBus
# ---------------------------------------------------------------------------


class EventBus:
    """
    In-process publisher/subscriber event bus for Ultron tasks and AgentRuntime executions.
    Optionally persists events to an append-only EventStore before dispatching.
    """

    def __init__(self, store: EventStore | None = None) -> None:
        if store is None:
            from ultron.core.runtime.event_store import get_default_event_store

            store = get_default_event_store()
        self._listeners: dict[TaskEventType, list[EventListener]] = defaultdict(list)
        self._global_listeners: list[EventListener] = []
        self._history: list[RuntimeEvent] = []
        self._store: EventStore | None = store
        self._task_sequences: dict[str, int] = defaultdict(int)

    def set_store(self, store: EventStore) -> None:
        """Attaches an EventStore backend for durable event logging."""
        self._store = store

    def subscribe(
        self,
        listener: EventListener,
        event_types: list[TaskEventType] | TaskEventType | None = None,
    ) -> None:
        """
        Subscribe a listener to specific event types or all events (if event_types is None).
        """
        if event_types is None:
            if listener not in self._global_listeners:
                self._global_listeners.append(listener)
            return

        if isinstance(event_types, TaskEventType):
            event_types = [event_types]

        for et in event_types:
            if listener not in self._listeners[et]:
                self._listeners[et].append(listener)

    def unsubscribe(
        self,
        listener: EventListener,
        event_types: list[TaskEventType] | TaskEventType | None = None,
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

        if isinstance(event_types, TaskEventType):
            event_types = [event_types]

        for et in event_types:
            if listener in self._listeners[et]:
                self._listeners[et].remove(listener)

    async def emit(self, event: TaskEvent | RuntimeEvent) -> None:
        """
        Emits an event to all matching listeners, persists it if a store is attached,
        and records it to history.
        """
        self._persist_event(event)

        # Record in legacy history for compatibility
        if isinstance(event, RuntimeEvent):
            self._history.append(event)
        else:
            self._history.append(
                RuntimeEvent(
                    event_type=event.event_type,
                    run_id=event.correlation_id or event.task_id,
                    task_id=event.task_id,
                    timestamp=event.timestamp,
                    payload=event.payload,
                )
            )

        targets = list(self._global_listeners) + list(self._listeners.get(event.event_type, []))

        for listener in targets:
            try:
                res = listener(event)
                if hasattr(res, "__await__"):
                    await res
            except Exception as exc:  # noqa: BLE001 — event listeners must never crash runtime
                logger.debug(f"Event listener failed on {event.event_type}: {exc}")

    def emit_sync(self, event: TaskEvent | RuntimeEvent) -> None:
        """
        Synchronous emission helper for non-async handlers.
        """
        self._persist_event(event)

        if isinstance(event, RuntimeEvent):
            self._history.append(event)
        else:
            self._history.append(
                RuntimeEvent(
                    event_type=event.event_type,
                    run_id=event.correlation_id or event.task_id,
                    task_id=event.task_id,
                    timestamp=event.timestamp,
                    payload=event.payload,
                )
            )

        targets = list(self._global_listeners) + list(self._listeners.get(event.event_type, []))

        for listener in targets:
            try:
                listener(event)
            except Exception as exc:  # noqa: BLE001 — event listeners must never crash runtime
                logger.debug(f"Sync event listener failed on {event.event_type}: {exc}")

    def _persist_event(self, event: TaskEvent | RuntimeEvent) -> None:
        if self._store is None:
            return

        task_event: TaskEvent
        if isinstance(event, TaskEvent):
            tid = event.task_id
            last_seq = self._store.last_sequence(tid)
            if event.sequence <= last_seq:
                # Re-sequence to next monotonic number
                next_seq = max(last_seq + 1, self._task_sequences[tid] + 1)
                self._task_sequences[tid] = next_seq
                task_event = TaskEvent(
                    task_id=event.task_id,
                    sequence=next_seq,
                    event_type=event.event_type,
                    timestamp=event.timestamp,
                    payload=event.payload,
                    source=event.source,
                    correlation_id=event.correlation_id,
                    causation_id=event.causation_id,
                )
            else:
                self._task_sequences[tid] = event.sequence
                task_event = event
        else:
            tid = event.task_id or event.run_id
            last_seq = self._store.last_sequence(tid)
            next_seq = max(last_seq + 1, self._task_sequences[tid] + 1)
            self._task_sequences[tid] = next_seq
            task_event = event.to_task_event(sequence=next_seq)

        try:
            self._store.append(task_event)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Failed to persist event {task_event.event_type} to store: {exc}")

    @property
    def history(self) -> list[RuntimeEvent]:
        """Returns recorded events in chronologic order."""
        return list(self._history)

    def clear(self) -> None:
        """Clears listener lists and event history."""
        self._listeners.clear()
        self._global_listeners.clear()
        self._history.clear()
        self._task_sequences.clear()
