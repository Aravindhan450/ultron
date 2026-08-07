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
    r"pip\s+(?:list|show)|"
    r"sqlite3\s+.*\s+'(?:select|with)"
    r")\b",
    re.IGNORECASE,
)

# Shell metacharacters that turn an otherwise read-only command into a
# state-changing one: redirection, pipes, chaining, or command substitution.
_SHELL_SIDE_EFFECTS = re.compile(r"[>|;&`]|\$\(")


# Path markers that make a write/overwrite CRITICAL — touching system
# configuration, credential stores, or boot/security files.
_SYSTEM_PATH_MARKERS = (
    "/etc/",
    "/etc/passwd",
    "/etc/shadow",
    "/boot/",
    "/System/",
    "/Library/LaunchDaemons",
    ".ssh/",
    "id_rsa",
    "id_ed25519",
    "id_dsa",
    ".aws/",
    ".git-credentials",
    ".htpasswd",
    "authorized_keys",
    "key.pem",
    "wallet",
    "secrets.json",
    ".env",
)

# State-changing HTTP methods (POST/PUT/DELETE/PATCH) — require confirmation.
_STATE_CHANGING_HTTP = {"POST", "PUT", "DELETE", "PATCH"}

# SQL verbs that can destroy or restructure data.
_DESTRUCTIVE_SQL = re.compile(r"\b(?:drop|truncate|alter|grant|revoke)\b", re.IGNORECASE)


class SecurityBoundary:
    """
    Classifies tool actions and decides allow/confirm/deny.

    Attributes:
        mode: Security mode — ``permissive``, ``interactive`` (default),
            or ``strict``. Falls back to ``settings.security_mode``.
        engine: The GuardrailsEngine used by :meth:`check`.
    """

    def __init__(self, mode: str | None = None, engine: GuardrailsEngine | None = None) -> None:
        self.mode = mode or settings.security_mode
        self.engine = engine or GuardrailsEngine()

    # ------------------------------------------------------------------
    # Risk classification
    # ------------------------------------------------------------------

    def classify_action(self, action_type: str, target: str = "", content: str | None = None) -> RiskTier:
        """
        Maps a tool action to a risk tier.

        The rules are intentionally conservative: when unsure, an action is
        classified HIGH rather than LOW so the user gets asked.
        """
        action = (action_type or "").lower()

        # Read-only, low-risk operations.
        if action in {
            "read_file",
            "web_search",
            "fetch_page_text",
            "get_all_memories",
            "search_memories",
            "add_memory",
        }:
            return RiskTier.LOW

        # HTTP requests: GET is read-only; state-changing methods are HIGH.
        if action == "make_http_request":
            method = self._http_method(target, content)
            if method in _STATE_CHANGING_HTTP:
                return RiskTier.HIGH
            return RiskTier.LOW

        # Database queries: SELECT/WITH are read-only; destructive verbs are CRITICAL.
        if action == "run_query":
            sql = target or content or ""
            if not is_readonly_query(sql):
                if _DESTRUCTIVE_SQL.search(sql):
                    return RiskTier.CRITICAL
                return RiskTier.HIGH
            return RiskTier.LOW

        # File writes: HIGH, escalated to CRITICAL when touching system paths.
        if action in {"write_file", "overwrite_file"}:
            if self._touches_system_path(target):
                return RiskTier.CRITICAL
            return RiskTier.HIGH

        # Shell commands. "Read-only" is only trusted when there is no shell
        # metacharacter (redirection, pipe, chaining, substitution) that could
        # turn e.g. `echo` or `cat` into a state-changing write.
        if action == "run_command":
            if any(pattern.search(target) for _rule, pattern in DANGEROUS_COMMAND_PATTERNS):
                return RiskTier.CRITICAL
            if _READONLY_COMMAND.match(target or "") and not _SHELL_SIDE_EFFECTS.search(target or ""):
                return RiskTier.LOW
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

    @staticmethod
    def _audit(result: BoundaryResult) -> None:
        """
        Records every verdict to the application log.

        Denials are logged as warnings, confirmations as info, and automatic
        allowances at debug level to avoid log spam.
        """
        line = f"action={result.action_type} tier={result.tier.value} decision={result.decision.value}"
        if result.decision == Decision.DENY:
            logger.warning("%s reason=%s", line, result.reason)
        elif result.decision == Decision.CONFIRM:
            logger.info("%s", line)
        else:
            logger.debug("%s", line)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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
        """True when a file target references a system/credential path."""
        text = (target or "").lower()
        return any(marker in text for marker in _SYSTEM_PATH_MARKERS)


def get_boundary() -> SecurityBoundary:
    """
    Returns a shared SecurityBoundary configured from the active settings.
    """
    return SecurityBoundary()
