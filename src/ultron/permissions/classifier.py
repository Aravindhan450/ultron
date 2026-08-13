"""
ultron.permissions.classifier
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Permission classifier for tool actions.

The permission layer sits *on top of* the security boundary: the boundary
classifies risk (``RiskTier``) and decides ``allow`` / ``confirm`` / ``deny``;
this module turns that verdict into a concrete permission requirement the
interactive confirmation flow can act on — including the extra-verification
step the docs promise for CRITICAL actions.

    action ──► SecurityBoundary.check() ──► RiskTier + Decision
                     │
                     ▼
             PermissionClassifier.classify()
                     │
                     ▼
        PermissionRequest (level: AUTO / CONFIRM / CONFIRM_CRITICAL / DENY)

Nothing here prompts the user — that is ``ultron.permissions.confirm``'s job.
This module only decides *whether* permission is needed and how strong it
must be.
"""

from dataclasses import dataclass
from enum import Enum

from ultron.core.types import PendingAction
from ultron.security import Decision, RiskTier, SecurityBoundary


class PermissionLevel(str, Enum):
    """What the permission system should do with a tool action."""

    AUTO = "auto"                  # no permission needed — run automatically
    CONFIRM = "confirm"            # one interactive confirmation
    CONFIRM_CRITICAL = "confirm_critical"  # confirmation + typed verification
    DENY = "deny"                  # hard block — never run, never ask


# Friendly action labels for the confirmation cards.  Registered tools get
# their labels from the canonical definitions table (single source of truth);
# only the non-registered PendingAction type is mapped here as UI policy.
_PENDING_ONLY_LABELS = {
    "overwrite_file": "Overwrite existing file",
}


@dataclass
class PermissionRequest:
    """
    One tool action and what the permission system decided about it.

    Attributes:
        action_type: The tool action (e.g. ``run_command``, ``write_file``).
        target: The command / filename / URL / SQL the action applies to.
        content: Optional payload (file content, HTTP body, …).
        tier: Risk tier from the security boundary (``RiskTier``).
        decision: The boundary's verdict (``Decision``).
        permission: What the permission system should do (``PermissionLevel``).
        reason: Human-readable justification from the boundary.
    """

    action_type: str
    target: str
    tier: RiskTier
    decision: Decision
    permission: PermissionLevel
    reason: str = ""
    content: str | None = None

    @property
    def needs_confirmation(self) -> bool:
        """True when the action must go through an interactive prompt."""
        return self.permission in (PermissionLevel.CONFIRM, PermissionLevel.CONFIRM_CRITICAL)

    @property
    def is_critical(self) -> bool:
        """True when the action is classified CRITICAL (extra verification)."""
        return self.tier == RiskTier.CRITICAL

    @property
    def action_label(self) -> str:
        """Friendly human label for the confirmation card.

        Labels for registered tools (and their aliases) come from the
        canonical definitions table; a tiny policy map covers the one
        non-registered PendingAction type.
        """
        from ultron.core.tools.definitions import action_label_for

        return (
            action_label_for(self.action_type)
            or _PENDING_ONLY_LABELS.get(self.action_type)
            or self.action_type
        )

    @property
    def prompt_title(self) -> str:
        """Title for the confirmation card, escalated for CRITICAL actions."""
        if self.permission == PermissionLevel.CONFIRM_CRITICAL:
            return "⚠ Critical Action — Confirmation Required"
        return "Confirmation Required"


class PermissionClassifier:
    """
    Maps tool actions to permission requirements by reusing the security
    boundary's risk tiers and verdicts.
    """

    def __init__(self, boundary: SecurityBoundary | None = None) -> None:
        self.boundary = boundary or SecurityBoundary()

    def classify(self, action_type: str, target: str = "", content: str | None = None) -> PermissionRequest:
        """
        Classifies one tool action.

        Runs the action through the security boundary (risk classification +
        guardrails) and maps the verdict to a PermissionLevel:

        - ``deny``                 → ``DENY`` (hard block, never offer)
        - ``allow``                → ``AUTO`` (no confirmation needed)
        - ``confirm`` + CRITICAL   → ``CONFIRM_CRITICAL`` (typed verification)
        - ``confirm`` otherwise    → ``CONFIRM`` (one interactive prompt)
        """
        verdict = self.boundary.check(action_type, target, content)
        return PermissionRequest(
            action_type=action_type,
            target=target,
            content=content,
            tier=verdict.tier,
            decision=verdict.decision,
            permission=self._level_for(verdict.decision, verdict.tier),
            reason=verdict.reason,
        )

    def classify_pending(self, pending: PendingAction) -> PermissionRequest:
        """
        Classifies a ``PendingAction`` from the agent layer (the object the
        CLI receives when an agent requests interactive confirmation).

        Alias spellings (``fetch_page``, ``db_query``, ``web_search``) are
        resolved to their registered tool through the canonical definitions
        table — the single source of truth for action aliases.
        """
        from ultron.core.tools.definitions import canonical_action_name

        action = canonical_action_name(pending.action_type)
        target = pending.target or ""

        # main.py encodes HTTP requests as ``run_command`` with an
        # ``http_request:METHOD:url[:body]`` target. Reclassify those through
        # the HTTP rules so GET stays auto-allowed (LOW) and POST/PUT/DELETE
        # are confirmed — otherwise they would all land in the generic
        # run_command branch and be mis-tiered.
        if pending.action_type == "run_command" and target.startswith("http_request:"):
            return self.classify("make_http_request", target, pending.content)

        return self.classify(action, target, pending.content)

    @staticmethod
    def _level_for(decision: Decision, tier: RiskTier) -> PermissionLevel:
        if decision == Decision.DENY:
            return PermissionLevel.DENY
        if decision == Decision.ALLOW:
            return PermissionLevel.AUTO
        if tier == RiskTier.CRITICAL:
            return PermissionLevel.CONFIRM_CRITICAL
        return PermissionLevel.CONFIRM
