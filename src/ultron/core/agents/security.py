"""
ultron.core.agents.security
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Shared security gate for the agent tool-call paths.

Every tool call the agents execute is routed through :func:`check_action`,
which delegates to the ``SecurityBoundary`` (risk classifier + guardrails)
from ``ultron.security``. The verdict decides what happens:

- ``deny``    → the action must never execute; callers return
                :func:`blocked_message` instead of running the tool.
- ``confirm`` → the action needs interactive user approval; agents emit a
                ``PendingAction`` so the CLI can prompt before executing.
- ``allow``   → the action may execute directly (auto-allowed risk tier,
                or a permissive security mode).

The boundary instance is created lazily from the active settings so the
security mode (``ULTRON_SECURITY_MODE``) takes effect automatically. Tests
can replace ``_boundary`` to exercise a specific mode deterministically.
"""

from ultron.core.logging import get_logger
from ultron.security import Decision, get_boundary

logger = get_logger("ultron.agents.security")

# Shared boundary, created once from the active settings. Tests may swap this
# for a SecurityBoundary with a specific mode.
_boundary = None


def get_security():
    """
    Returns the shared SecurityBoundary, creating it from the active settings
    on first use.
    """
    global _boundary
    if _boundary is None:
        _boundary = get_boundary()
    return _boundary


def check_action(action_type: str, target: str = "", content: str | None = None):
    """
    Routes one tool call through the security boundary and returns its verdict
    (a ``BoundaryResult`` with ``decision`` and ``tier``).
    """
    return get_security().check(action_type, target, content)


def security_mode() -> str:
    """Returns the active security mode (permissive / interactive / strict)."""
    return get_security().mode


def is_denied(result) -> bool:
    """True when the verdict is a hard block — the action must not execute."""
    return result.decision == Decision.DENY


def is_confirm(result) -> bool:
    """True when the verdict requires interactive user approval first."""
    return result.decision == Decision.CONFIRM


def is_allow(result) -> bool:
    """True when the verdict allows the action to execute directly."""
    return result.decision == Decision.ALLOW


def blocked_message(result) -> str:
    """User/model-facing message for a denied action."""
    return f"Blocked by security: {result.reason}"
