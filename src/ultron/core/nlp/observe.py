"""ultron.core.nlp.observe
~~~~~~~~~~~~~~~~~~~~~~~~~~

Structured observability for the natural-language → tool pipeline.

Every routed action can be recorded as an :class:`ActionRecord`:

    user_intent -> selected_tool -> extracted_arguments
    -> normalized_arguments -> security_decision
    -> confirmation_required -> execution_result -> follow_up_action

Records live in a bounded in-process ring buffer (no disk writes, no secrets
— the record only carries argument *metadata*, never command output text).
Internal chain-of-thought is deliberately not recorded; only the action
summary is.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

MAX_RECORDS = 200


@dataclass
class ActionRecord:
    """One routed tool action, for diagnostics and tests."""

    user_intent: str
    selected_tool: str
    extracted_arguments: dict[str, Any] = field(default_factory=dict)
    normalized_arguments: dict[str, Any] = field(default_factory=dict)
    security_decision: str = "pending"  # allow | confirm | deny | pending
    confirmation_required: bool = False
    execution_result: str = "not_run"  # success | failed | cancelled | not_run
    follow_up_action: str | None = None
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# In-process ring buffer
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_records: list[ActionRecord] = []


def record_action(
    user_intent: str,
    selected_tool: str,
    extracted_arguments: dict[str, Any] | None = None,
    normalized_arguments: dict[str, Any] | None = None,
    security_decision: str = "pending",
    confirmation_required: bool = False,
    execution_result: str = "not_run",
    follow_up_action: str | None = None,
) -> ActionRecord:
    """Appends one action record to the ring buffer and returns it."""
    rec = ActionRecord(
        user_intent=user_intent,
        selected_tool=selected_tool,
        extracted_arguments=dict(extracted_arguments or {}),
        normalized_arguments=dict(normalized_arguments or {}),
        security_decision=security_decision,
        confirmation_required=confirmation_required,
        execution_result=execution_result,
        follow_up_action=follow_up_action,
    )
    with _lock:
        _records.append(rec)
        if len(_records) > MAX_RECORDS:
            del _records[: len(_records) - MAX_RECORDS]
    return rec


def recent_actions(n: int = 10) -> list[ActionRecord]:
    """Returns the most recent *n* action records (newest first)."""
    with _lock:
        return list(reversed(_records[-n:]))


def clear_action_records() -> None:
    """Empties the ring buffer (used by tests and by nothing else)."""
    with _lock:
        _records.clear()
