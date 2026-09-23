"""
ultron.core.runtime.event_store
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Durable, append-only event store abstractions and storage backends.

Provides:
- EventStore: Abstract base class for persisting and querying TaskEvents.
- JsonlEventStore: File-backed JSONL store (~/.ultron/events/<task_id>.jsonl) with mutex locks,
  monotonic sequence verification, and corruption tolerance.
- InMemoryEventStore: In-memory store for unit testing.
"""

from __future__ import annotations

import json
import threading
from abc import ABC, abstractmethod
from pathlib import Path

from ultron.core.logging import get_logger
from ultron.core.runtime.events import TaskEvent

logger = get_logger("ultron.runtime.event_store")

DEFAULT_EVENT_STORE_DIR = Path.home() / ".ultron" / "events"


class SequenceViolationError(ValueError):
    """Raised when an event sequence is not strictly greater than previous sequence."""


class EventStore(ABC):
    """Abstract interface for task event persistence."""

    @abstractmethod
    def append(self, event: TaskEvent) -> None:
        """Persists a single event atomically."""

    @abstractmethod
    def append_many(self, events: list[TaskEvent]) -> None:
        """Persists multiple events atomically."""

    @abstractmethod
    def get(self, task_id: str) -> list[TaskEvent]:
        """Returns all events for task_id ordered by sequence."""

    @abstractmethod
    def get_since(self, task_id: str, sequence: int) -> list[TaskEvent]:
        """Returns events for task_id with sequence > sequence."""

    @abstractmethod
    def get_event(self, task_id: str, sequence: int) -> TaskEvent | None:
        """Returns a specific event by task_id and sequence."""

    @abstractmethod
    def last_sequence(self, task_id: str) -> int:
        """Returns highest sequence number for task_id, or 0 if none exist."""

    @abstractmethod
    def list_tasks(self) -> list[str]:
        """Returns all known task_ids in the store."""


class InMemoryEventStore(EventStore):
    """Thread-safe in-memory event store for testing."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: dict[str, list[TaskEvent]] = {}

    def append(self, event: TaskEvent) -> None:
        with self._lock:
            task_events = self._events.setdefault(event.task_id, [])
            last_seq = task_events[-1].sequence if task_events else 0
            if event.sequence <= last_seq:
                raise SequenceViolationError(
                    f"Sequence violation for task {event.task_id}: "
                    f"event.sequence {event.sequence} <= last sequence {last_seq}"
                )
            task_events.append(event)

    def append_many(self, events: list[TaskEvent]) -> None:
        for event in events:
            self.append(event)

    def get(self, task_id: str) -> list[TaskEvent]:
        with self._lock:
            return list(self._events.get(task_id, []))

    def get_since(self, task_id: str, sequence: int) -> list[TaskEvent]:
        with self._lock:
            return [e for e in self._events.get(task_id, []) if e.sequence > sequence]

    def get_event(self, task_id: str, sequence: int) -> TaskEvent | None:
        with self._lock:
            for e in self._events.get(task_id, []):
                if e.sequence == sequence:
                    return e
            return None

    def last_sequence(self, task_id: str) -> int:
        with self._lock:
            evs = self._events.get(task_id, [])
            return evs[-1].sequence if evs else 0

    def list_tasks(self) -> list[str]:
        with self._lock:
            return list(self._events.keys())


class JsonlEventStore(EventStore):
    """
    Append-only persistent event store storing events in JSON Lines format:
    `<base_dir>/<task_id>.jsonl`.

    Guarantees:
    - Thread-safety via per-task file locking.
    - Monotonic sequence verification.
    - Corrupt line handling without crashing valid events.
    """

    def __init__(self, base_dir: Path | str | None = None) -> None:
        self.base_dir = Path(base_dir) if base_dir else DEFAULT_EVENT_STORE_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._task_locks: dict[str, threading.Lock] = {}

    def _get_task_lock(self, task_id: str) -> threading.Lock:
        with self._lock:
            if task_id not in self._task_locks:
                self._task_locks[task_id] = threading.Lock()
            return self._task_locks[task_id]

    def _task_path(self, task_id: str) -> Path:
        safe_id = "".join(c for c in task_id if c.isalnum() or c in ("-", "_"))
        return self.base_dir / f"{safe_id}.jsonl"

    def append(self, event: TaskEvent) -> None:
        task_lock = self._get_task_lock(event.task_id)
        with task_lock:
            path = self._task_path(event.task_id)
            last_seq = self._read_last_sequence_locked(path)
            if event.sequence <= last_seq:
                raise SequenceViolationError(
                    f"Sequence violation for task {event.task_id}: "
                    f"event.sequence {event.sequence} <= last sequence {last_seq}"
                )

            line = event.model_dump_json() + "\n"
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()

    def append_many(self, events: list[TaskEvent]) -> None:
        for event in events:
            self.append(event)

    def get(self, task_id: str) -> list[TaskEvent]:
        task_lock = self._get_task_lock(task_id)
        with task_lock:
            path = self._task_path(task_id)
            return self._read_events_locked(path)

    def get_since(self, task_id: str, sequence: int) -> list[TaskEvent]:
        events = self.get(task_id)
        return [e for e in events if e.sequence > sequence]

    def get_event(self, task_id: str, sequence: int) -> TaskEvent | None:
        events = self.get(task_id)
        for e in events:
            if e.sequence == sequence:
                return e
        return None

    def last_sequence(self, task_id: str) -> int:
        task_lock = self._get_task_lock(task_id)
        with task_lock:
            path = self._task_path(task_id)
            return self._read_last_sequence_locked(path)

    def list_tasks(self) -> list[str]:
        with self._lock:
            tasks: list[str] = []
            for path in self.base_dir.glob("*.jsonl"):
                tasks.append(path.stem)
            return sorted(tasks)

    def _read_last_sequence_locked(self, path: Path) -> int:
        if not path.exists():
            return 0
        events = self._read_events_locked(path)
        return events[-1].sequence if events else 0

    def _read_events_locked(self, path: Path) -> list[TaskEvent]:
        if not path.exists():
            return []

        events: list[TaskEvent] = []
        with open(path, encoding="utf-8") as f:
            for line_no, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    ev = TaskEvent.model_validate(data)
                    events.append(ev)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        f"Skipping corrupt event line {line_no} in {path}: {exc}"
                    )
        return sorted(events, key=lambda e: e.sequence)


_default_event_store: EventStore | None = None


def get_default_event_store() -> EventStore:
    """
    Returns the process-wide default EventStore for production task execution.
    Defaults to a durable JsonlEventStore located at ~/.ultron/events/.
    """
    global _default_event_store
    if _default_event_store is None:
        _default_event_store = JsonlEventStore()
    return _default_event_store


def set_default_event_store(store: EventStore | None) -> None:
    """Overrides the default event store (e.g. for testing with InMemoryEventStore)."""
    global _default_event_store
    _default_event_store = store


def reset_default_event_store() -> None:
    """Resets the process-wide default event store to None (will re-instantiate JsonlEventStore)."""
    global _default_event_store
    _default_event_store = None
