"""
ultron.security.models
~~~~~~~~~~~~~~~~~~~~~~

Shared data models for the security boundary and guardrails modules.

These mirror the risk model documented in docs/user.md:

    Low       → auto-allowed        (read public files, search web)
    Medium    → soft confirm        (open apps, non-sensitive messages)
    High      → explicit confirm    (delete files, send emails)
    Critical  → explicit + extra verification (system changes, financial actions)
"""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class RiskTier(str, Enum):
    """Risk level of a tool action, from auto-allowed to extra-verified."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Decision(str, Enum):
    """What the boundary decides should happen for an action."""

    ALLOW = "allow"          # execute without confirmation
    CONFIRM = "confirm"      # require explicit user approval first
    DENY = "deny"            # hard-block: must never execute


class GuardrailFinding(BaseModel):
    """
    A single guardrail hit: which rule fired, where, and a redacted snippet.

    Attributes:
        rule: Machine-readable rule name (e.g. ``aws_access_key``).
        severity: ``info``, ``warning``, or ``critical``.
        location: Where the finding applies — ``content``, ``url``, ``path``,
            or ``command``.
        snippet: Short, truncated view of the matched text.
        message: Human-readable explanation.
    """

    rule: str
    severity: Literal["info", "warning", "critical"] = "warning"
    location: str = "content"
    snippet: str = ""
    message: str = ""


class GuardrailsResult(BaseModel):
    """
    Result of running the guardrail checks against one action.

    Attributes:
        passed: True when no hard block fired (findings may still exist).
        blocked: True when the action must be denied outright.
        findings: All rule hits, ordered by discovery.
        sanitized_text: Redacted copy of the scanned content when secrets or
            PII were found and could be masked; None otherwise.
        block_reason: Message of the first critical finding that caused a block.
    """

    passed: bool = True
    blocked: bool = False
    findings: list[GuardrailFinding] = Field(default_factory=list)
    sanitized_text: str | None = None
    block_reason: str | None = None


class BoundaryResult(BaseModel):
    """
    The security boundary's verdict for one tool action.

    Attributes:
        action_type: The tool action (e.g. ``run_command``, ``write_file``).
        target: The command / filename / URL / SQL the action applies to.
        tier: The classified risk tier.
        decision: What to do — allow, confirm, or deny.
        reason: Human-readable justification for the decision.
        guardrails: The guardrail scan results that fed into the decision.
    """

    action_type: str
    target: str
    tier: RiskTier
    decision: Decision
    reason: str = ""
    guardrails: GuardrailsResult | None = None
