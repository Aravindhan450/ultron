"""
ultron.security
~~~~~~~~~~~~~~~

The security boundary: risk classification, guardrails, and the
allow/confirm/deny gate that protects every tool action.

Public API:

    from ultron.security import (
        SecurityBoundary, GuardrailsEngine,
        RiskTier, Decision, BoundaryResult, GuardrailsResult,
        get_boundary,
    )

    verdict = get_boundary().check("run_command", "rm -rf /")
    verdict.decision   # Decision.DENY / CONFIRM / ALLOW
    verdict.tier       # RiskTier
"""

from ultron.security.boundary import SecurityBoundary, get_boundary
from ultron.security.guardrails import GuardrailsEngine
from ultron.security.models import (
    BoundaryResult,
    Decision,
    GuardrailFinding,
    GuardrailsResult,
    RiskTier,
)

__all__ = [
    "BoundaryResult",
    "Decision",
    "GuardrailFinding",
    "GuardrailsEngine",
    "GuardrailsResult",
    "RiskTier",
    "SecurityBoundary",
    "get_boundary",
]
