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

Every verdict is also appended to a JSON-lines audit trail
(``~/.ultron/security_audit.jsonl`` by default) — see
:mod:`ultron.security.audit`.
"""

from ultron.security.audit import AuditLog, get_audit_log, record_verdict
from ultron.security.boundary import SecurityBoundary, get_boundary
from ultron.security.file_policy import (
    SYSTEM_PATH_MARKERS,
    FilePolicy,
    FilePolicyResult,
    check_file_access,
    get_file_policy,
    is_protected,
)
from ultron.security.guardrails import GuardrailsEngine
from ultron.security.models import (
    BoundaryResult,
    Decision,
    GuardrailFinding,
    GuardrailsResult,
    RiskTier,
)

__all__ = [
    "SYSTEM_PATH_MARKERS",
    "AuditLog",
    "BoundaryResult",
    "Decision",
    "FilePolicy",
    "FilePolicyResult",
    "GuardrailFinding",
    "GuardrailsEngine",
    "GuardrailsResult",
    "RiskTier",
    "SecurityBoundary",
    "check_file_access",
    "get_audit_log",
    "get_boundary",
    "get_file_policy",
    "is_protected",
    "record_verdict",
]
