"""
ultron.core.runtime.task_state
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Authoritative task state model and deterministic finite lifecycle for Ultron.

All primary definitions reside canonically in `ultron.core.types` to guarantee
zero circular imports across runtime, intelligence, and agent subsystems.
This module re-exports the canonical types for clean package organization under runtime.
"""

from __future__ import annotations

from ultron.core.types import (
    TASK_TRANSITIONS,
    CanonicalTaskState,
    InvalidStateTransitionError,
    TaskLifecycleStatus,
    TaskState,
    assert_task_transition,
)

__all__ = [
    "TASK_TRANSITIONS",
    "CanonicalTaskState",
    "InvalidStateTransitionError",
    "TaskLifecycleStatus",
    "TaskState",
    "assert_task_transition",
]
