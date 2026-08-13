"""
ultron.security.boundary
~~~~~~~~~~~~~~~~~~~~~~~~

SecurityBoundary — the risk classifier + decision gate described in
docs/agents.md:

    Every tool call goes through:
      Risk Classifier → Permission system → GuardrailsEngine

It answers two questions for any tool action:

1. **How risky is it?** (``classify_action``) — maps an action to a RiskTier
   using the four-tier model from docs/user.md (low / medium / high /
   critical).
2. **What should we do about it?** (``check``) — combines the tier with the
   configured security mode and the guardrail scan into a single verdict:

   - ``allow``    → run automatically
   - ``confirm``  → ask the user first
   - ``deny``     → hard-block (guardrail violation: secret exfiltration,
                   unsafe URL, or path escape)

Security mode policy (``ULTRON_SECURITY_MODE``):

    permissive   → confirm only critical
    interactive  → confirm high + critical          (default)
    strict       → confirm medium + high + critical (only low is automatic)

The boundary never executes anything itself — it only *decides*. Execution
and interactive confirmation stay in the CLI/agent layer, so the module can
be dropped into any call path without changing how tools run.
"""

import re

from ultron.core.config import settings
from ultron.core.logging import get_logger
from ultron.core.tools.builtin.database import is_readonly_query
from ultron.security.audit import AuditLog, get_audit_log
from ultron.security.file_policy import is_protected
from ultron.security.guardrails import DANGEROUS_COMMAND_PATTERNS, GuardrailsEngine
from ultron.security.models import BoundaryResult, Decision, RiskTier

logger = get_logger("ultron.security")

# Commands considered read-only (informational). Everything else that reaches
# the shell is treated as state-changing and therefore higher risk.
_READONLY_COMMAND = re.compile(
    r"^\s*(?:"
    r"ls|cat|pwd|echo|head|tail|grep|find|which|whereis|env|whoami|date|uptime|df|du|"
    r"git\s+(?:status|diff|log|show)|"
    r"pytest|ruff|"
    r"(?:[\w./-]*/)?(?:\\.venv|venv)?/?bin/python(?:3)?\s+-m\s+pytest|"
    r"python(?:3)?\s+-m\s+pytest|"
    r"pip\s+(?:list|show)|"
    r"sqlite3\s+.*\s+'(?:select|with)"
    r")\b",
    re.IGNORECASE,
)

# Shell metacharacters that turn an otherwise read-only command into a
# state-changing one: redirection, pipes, chaining, or command substitution.
_SHELL_SIDE_EFFECTS = re.compile(r"[>|;&`]|\$\(")


# Protected system/credential markers now live in
# ``ultron.security.file_policy`` (SYSTEM_PATH_MARKERS) — the single source
# of truth for the path policy.

# State-changing HTTP methods (POST/PUT/DELETE/PATCH) — require confirmation.
_STATE_CHANGING_HTTP = {"POST", "PUT", "DELETE", "PATCH"}

# SQL verbs that can destroy or restructure data.
_DESTRUCTIVE_SQL = re.compile(r"\b(?:drop|truncate|alter|grant|revoke)\b", re.IGNORECASE)

# Ordering of the risk tiers, used to combine per-command verdicts into a
# batch verdict (the batch takes the worst tier).
_TIER_RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3}


class SecurityBoundary:
    """
    Classifies tool actions and decides allow/confirm/deny.

    Attributes:
        mode: Security mode — ``permissive``, ``interactive`` (default),
            or ``strict``. Falls back to ``settings.security_mode``.
        engine: The GuardrailsEngine used by :meth:`check`.
        audit_log: The :class:`~ultron.security.audit.AuditLog` every verdict
            is recorded to (JSON-lines file). Defaults to the shared log.
    """

    def __init__(
        self,
        mode: str | None = None,
        engine: GuardrailsEngine | None = None,
        audit_log: AuditLog | None = None,
    ) -> None:
        self.mode = mode or settings.security_mode
        self.engine = engine or GuardrailsEngine()
        self.audit_log = audit_log or get_audit_log()

    # ------------------------------------------------------------------
    # Risk classification
    # ------------------------------------------------------------------

    def classify_action(self, action_type: str, target: str = "", content: str | None = None) -> RiskTier:
        """
        Maps a tool action to a risk tier.

        The rules are intentionally conservative: when unsure, an action is
        classified HIGH rather than LOW so the user gets asked.
        """
        # Alias spellings (db_query, fetch_page, web_search) resolve to their
        # registered tool up front so policy branches and the canonical lookup
        # below see one spelling. Imported lazily (see the note on
        # get_tool_definition).
        from ultron.core.tools.definitions import canonical_action_name

        action = canonical_action_name(action_type or "")

        # Policy-sensitive actions whose tier depends on the ACTUAL target
        # (HTTP method / SQL verb / shell metacharacters). These are security
        # policy refinements over the tool's declared canonical risk.
        if action == "make_http_request":
            method = self._http_method(target, content)
            if method in _STATE_CHANGING_HTTP:
                return RiskTier.HIGH
            return RiskTier.LOW

        if action == "run_query":
            sql = target or content or ""
            if not is_readonly_query(sql):
                if _DESTRUCTIVE_SQL.search(sql):
                    return RiskTier.CRITICAL
                return RiskTier.HIGH
            return RiskTier.LOW

        # Shell commands. "Read-only" is only trusted when there is no shell
        # metacharacter (redirection, pipe, chaining, substitution) that could
        # turn e.g. `echo` or `cat` into a state-changing write.
        if action == "run_command":
            return self._classify_command(target)

        # Parallel command batches are at least as dangerous as their most
        # dangerous command: classify each command individually (newline-
        # separated in the target) and take the worst tier. A single
        # state-changing or destructive command escalates the whole batch.
        if action == "run_parallel":
            worst = RiskTier.LOW
            for cmd in (target or "").splitlines():
                if not cmd.strip():
                    continue
                tier = self._classify_command(cmd)
                if _TIER_RANK[tier.value] > _TIER_RANK[worst.value]:
                    worst = tier
            return worst

        # Registered tools: the risk tier comes from the canonical definitions
        # table (the single source of truth for tool metadata). Imported
        # lazily: definitions pulls in coding modules that transitively import
        # this package, so a module-level import would be circular.
        from ultron.core.tools.definitions import get_tool_definition

        definition = get_tool_definition(action)
        if definition is not None:
            tier = RiskTier(definition.risk.value)
            # Policy: state-changing file tools touching system/credential
            # paths escalate to CRITICAL regardless of the declared tier.
            if (
                not definition.read_only
                and tier == RiskTier.HIGH
                and self._touches_system_path(target)
            ):
                return RiskTier.CRITICAL
            return tier

        # Internal non-registered actions (security policy; no registry tool).
        if action == "query_chain":
            # Knowledge-graph reasoning over local memory — read-only.
            return RiskTier.LOW
        if action == "overwrite_file":
            # Pending-action type for overwriting an existing file — gated
            # exactly like the registered write tools.
            if self._touches_system_path(target):
                return RiskTier.CRITICAL
            return RiskTier.HIGH

        # Unknown actions default to HIGH.
        return RiskTier.HIGH

    # ------------------------------------------------------------------
    # Decision policy
    # ------------------------------------------------------------------

    def decide(self, tier: RiskTier, mode: str | None = None) -> Decision:
        """
        Maps a risk tier to allow/confirm under the active security mode.
        """
        mode = mode or self.mode
        if mode == "permissive":
            return Decision.CONFIRM if tier == RiskTier.CRITICAL else Decision.ALLOW
        if mode == "strict":
            return Decision.ALLOW if tier == RiskTier.LOW else Decision.CONFIRM
        # interactive (default): confirm high + critical, auto-allow the rest.
        return Decision.CONFIRM if tier in {RiskTier.HIGH, RiskTier.CRITICAL} else Decision.ALLOW

    # ------------------------------------------------------------------
    # The gate
    # ------------------------------------------------------------------

    def check(
        self,
        action_type: str,
        target: str = "",
        content: str | None = None,
        mode: str | None = None,
    ) -> BoundaryResult:
        """
        Classifies an action, runs guardrails, and returns a final verdict.

        Guardrail violations always produce ``deny``; otherwise the tier and
        security mode determine ``allow`` vs ``confirm``.
        """
        tier = self.classify_action(action_type, target, content)
        guardrails = self.engine.evaluate(action_type=action_type, target=target, content=content)

        if guardrails.blocked:
            result = BoundaryResult(
                action_type=action_type,
                target=target,
                tier=RiskTier.CRITICAL,
                decision=Decision.DENY,
                reason=guardrails.block_reason or "Blocked by guardrails",
                guardrails=guardrails,
            )
            self._audit(result)
            return result

        decision = self.decide(tier, mode)
        finding_count = len(guardrails.findings)
        reason = (
            f"Risk tier '{tier.value}' → '{decision.value}' "
            f"under security mode '{mode or self.mode}'"
        )
        if finding_count:
            reason += f" ({finding_count} guardrail finding{'s' if finding_count != 1 else ''})"
        result = BoundaryResult(
            action_type=action_type,
            target=target,
            tier=tier,
            decision=decision,
            reason=reason,
            guardrails=guardrails,
        )
        self._audit(result)
        return result

    def _audit(self, result: BoundaryResult) -> None:
        """
        Records every verdict to the application log and the JSON-lines audit
        trail (``~/.ultron/security_audit.jsonl`` by default).

        Denials are logged as warnings, confirmations as info, and automatic
        allowances at debug level to avoid log spam. The audit file records
        all three without filtering.
        """
        line = f"action={result.action_type} tier={result.tier.value} decision={result.decision.value}"
        if result.decision == Decision.DENY:
            logger.warning("%s reason=%s", line, result.reason)
        elif result.decision == Decision.CONFIRM:
            logger.info("%s", line)
        else:
            logger.debug("%s", line)

        try:
            self.audit_log.record(result, mode=self.mode)
        except Exception:
            # Auditing must never break the gate — log and move on.
            logger.exception("audit: failed to record verdict")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_command(command: str) -> RiskTier:
        """
        Classifies a single shell command by the shared command rules.

        Dangerous patterns (rm -rf /, curl | sh, …) are CRITICAL; read-only
        commands without shell metacharacters are LOW; everything else is
        treated as state-changing and therefore HIGH.
        """
        if any(pattern.search(command) for _rule, pattern in DANGEROUS_COMMAND_PATTERNS):
            return RiskTier.CRITICAL
        if _READONLY_COMMAND.match(command or "") and not _SHELL_SIDE_EFFECTS.search(command or ""):
            return RiskTier.LOW
        return RiskTier.HIGH

    @staticmethod
    def _http_method(target: str, content: str | None) -> str:
        """Infers the HTTP method from the pending-action encoding or content."""
        text = target or ""
        if text.startswith("http_request:"):
            parts = text.split(":", 3)
            if len(parts) >= 2 and parts[1].upper() in _STATE_CHANGING_HTTP | {"GET"}:
                return parts[1].upper()
        for word in (content or "").upper().split():
            if word in _STATE_CHANGING_HTTP:
                return word
        return "GET"

    @staticmethod
    def _touches_system_path(target: str) -> bool:
        """
        True when a file target references a system/credential path.

        Delegates to the shared file policy so the marker list lives in one
        place (``ultron.security.file_policy``).
        """
        return is_protected(target)


def get_boundary() -> SecurityBoundary:
    """
    Returns a shared SecurityBoundary configured from the active settings.
    """
    return SecurityBoundary()
