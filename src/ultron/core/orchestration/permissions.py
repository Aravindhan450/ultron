"""
ultron.core.orchestration.permissions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Agent permissions for the orchestration layer (Fix #7, section 7.2).

Every agent TYPE carries a frozen :class:`AgentPermissions` profile:

- ``read`` / ``write`` / ``test`` / ``shell`` / ``network`` — permission
  levels per action category, expressed with the SAME ``Decision`` enum the
  security boundary uses (ALLOW / CONFIRM / DENY). CONFIRM is not a
  shortcut: it delegates to the real :class:`SecurityBoundary`, so risk
  classification, guardrails and the active security mode still apply.
- ``allowed_tools`` — an explicit tool whitelist. **Deny-by-default**: a
  tool not listed is DENIED, even when its category level is ALLOW.
- ``other`` — default CONFIRM for whitelisted tools with no explicit
  category (they go through the security boundary).

Tool classification (:func:`classify_tool`) is deterministic and
category-based (READ / WRITE / TEST / SHELL / NETWORK / OTHER), with special
handling for ``run_command``: test commands (pytest, npm test, cargo test,
...) classify as TEST, read-only commands (ls, cat, git diff, ...) as READ,
and anything else as SHELL.

CRITICAL REQUIREMENT: the LLM can NEVER modify its own permissions. The
profile is **frozen** (``frozen=True`` — attribute assignment raises) and is
constructed ONLY by the runtime (:mod:`ultron.core.orchestration.registry`).
Agents receive a scoped :class:`ExecutionContext` that carries the profile
as read-only data; there is no API on the agent contract to change it, and
the registry is never handed to an agent.
"""

from __future__ import annotations

import re
from enum import Enum

from pydantic import BaseModel, ConfigDict

from ultron.core.tools.definitions import TOOL_DEFINITIONS, ToolDomain
from ultron.security import Decision, SecurityBoundary, get_boundary

# ---------------------------------------------------------------------------
# Action categories
# ---------------------------------------------------------------------------


class PermissionCategory(str, Enum):
    """Deterministic action category a tool call belongs to."""

    READ = "read"
    WRITE = "write"
    TEST = "test"
    SHELL = "shell"
    NETWORK = "network"
    OTHER = "other"  # whitelisted but unclassified -> security evaluation


# Tool -> category classification is DERIVED from the canonical definitions
# table (STEP 2A) at call time, so a canonical metadata change propagates
# immediately: read-only non-network tools classify READ, state-changing
# file/memory/learning tools classify WRITE, network-domain tools classify
# NETWORK, and shell tools classify SHELL.  Only deliberate policy choices
# are listed here explicitly: ``retrieve`` is read-only research that
# read-only agents rely on (READ, not NETWORK), plus the non-registered
# action names (future git tools, the overwrite_file pending-action type,
# alias spellings) that have no canonical definition to derive from.
_POLICY_READ = frozenset({"git_status", "git_diff"})
_POLICY_WRITE = frozenset({"overwrite_file"})
_POLICY_NETWORK = frozenset({"fetch_page", "web_search"})

# Explicitly shell-driven tools (run_command is classified separately below).
_SHELL_TOOLS = frozenset({"run_command", "run_parallel"})

# Test-command prefixes: running the project's tests (still subject to the
# tester/coder level; a test run is never a blind shell grant).
_TEST_COMMAND_RE = re.compile(
    r"^\s*(?:"
    r"pytest|py\.test|python\s+-m\s+pytest|unittest|"
    r"npm\s+test|yarn\s+test|pnpm\s+test|bun\s+test|"
    r"cargo\s+test|go\s+test|mvn\s+test|gradle\s+test|"
    r"make\s+test|jest|vitest|ctest"
    r")\b",
    re.IGNORECASE,
)

# Read-only command prefixes (mirrors the security boundary's own rule).
_READONLY_COMMAND_RE = re.compile(
    r"^\s*(?:"
    r"ls|cat|pwd|echo|head|tail|grep|find|which|whereis|env|whoami|date|"
    r"uptime|df|du|git\s+(?:status|diff|log|show)"
    r")\b",
    re.IGNORECASE,
)

# Shell metacharacters that turn a read-only-looking command into a
# state-changing one (redirection, pipes, chaining, substitution).
_SHELL_SIDE_EFFECTS = re.compile(r"[>|;&`]|\$\(")


def _category_for_tool(action: str) -> PermissionCategory | None:
    """Canonical-derived category for a registered tool, or None (OTHER).

    Precedence mirrors the historical sets: READ first (read-only
    non-network tools, plus the ``retrieve`` policy carve-out), then WRITE
    (state-changing file/memory/learning), then NETWORK (remaining network
    domain tools).  Reads the canonical table at call time so metadata
    changes propagate immediately.
    """
    definition = TOOL_DEFINITIONS.get(action)
    if definition is None:
        return None
    if definition.read_only and (
        definition.domain is not ToolDomain.NETWORK or action == "retrieve"
    ):
        return PermissionCategory.READ
    if (
        not definition.read_only
        and definition.domain
        in (ToolDomain.FILESYSTEM, ToolDomain.MEMORY, ToolDomain.LEARNING)
    ):
        return PermissionCategory.WRITE
    if definition.domain is ToolDomain.NETWORK:
        return PermissionCategory.NETWORK
    return None


def classify_tool(action_type: str, target: str = "") -> PermissionCategory:
    """
    Classifies a tool call into a :class:`PermissionCategory`.

    ``run_command`` is target-sensitive: test commands are TEST, read-only
    commands (no shell metacharacters) are READ, everything else is SHELL.
    Registered tools classify from canonical metadata (call-time derived);
    unlisted tools are OTHER (whitelisted tools with no explicit category
    default to security evaluation). Never raises.
    """
    action = (action_type or "").lower()
    if action == "run_command":
        command = target or ""
        if _TEST_COMMAND_RE.match(command) and not _SHELL_SIDE_EFFECTS.search(command):
            return PermissionCategory.TEST
        if _READONLY_COMMAND_RE.match(command) and not _SHELL_SIDE_EFFECTS.search(
            command
        ):
            return PermissionCategory.READ
        return PermissionCategory.SHELL
    category = _category_for_tool(action)
    if category is not None:
        return category
    if action in _POLICY_READ:
        return PermissionCategory.READ
    if action in _POLICY_WRITE:
        return PermissionCategory.WRITE
    if action in _POLICY_NETWORK:
        return PermissionCategory.NETWORK
    if action in _SHELL_TOOLS:
        return PermissionCategory.SHELL
    return PermissionCategory.OTHER


class PermissionCheck(BaseModel):
    """The verdict of one permission check (decision + why)."""

    decision: Decision
    category: PermissionCategory
    reason: str = ""
    delegated: bool = False  # True when the security boundary made the final call

    def to_prompt_line(self, max_len: int = 200) -> str:
        head = f"[{self.decision.value}] category={self.category.value}"
        if self.delegated:
            head = f"{head} (delegated to security boundary)"
        if self.reason:
            head = f"{head}: {self.reason}"
        return head[:max_len]


# ---------------------------------------------------------------------------
# The frozen permission profile
# ---------------------------------------------------------------------------


class AgentPermissions(BaseModel):
    """
    Runtime-controlled permission profile for one agent type.

    Frozen: no code (and certainly no LLM) can reassign fields after
    construction — permissions are fixed by the runtime when the profile is
    built (see registry.py). Levels use ``ultron.security.Decision``.

    ``allowed_tools`` is the tool whitelist (deny-by-default). Category
    levels decide what happens to WHITELISTED tools:

    - ALLOW   -> the action may execute directly
    - DENY    -> the action is hard-blocked
    - CONFIRM -> the action is evaluated by the SecurityBoundary (risk
                 classification + guardrails + security mode); the boundary
                 is the final authority and is never bypassed.

    ``other`` defaults to CONFIRM so any whitelisted-but-unclassified tool
    still passes through the security boundary instead of running blindly.
    """

    model_config = ConfigDict(frozen=True)

    read: Decision = Decision.ALLOW
    write: Decision = Decision.DENY
    test: Decision = Decision.DENY
    shell: Decision = Decision.DENY
    network: Decision = Decision.DENY
    other: Decision = Decision.CONFIRM  # whitelisted, unclassified -> boundary
    # A tuple, not a list: frozen=True blocks field reassignment, and a tuple
    # additionally blocks in-place mutation — the whitelist truly cannot be
    # changed after construction, not even by an agent holding the object.
    allowed_tools: tuple[str, ...] = ()

    def level_for(self, category: PermissionCategory) -> Decision:
        """The permission level governing one action category."""
        return {
            PermissionCategory.READ: self.read,
            PermissionCategory.WRITE: self.write,
            PermissionCategory.TEST: self.test,
            PermissionCategory.SHELL: self.shell,
            PermissionCategory.NETWORK: self.network,
            PermissionCategory.OTHER: self.other,
        }[category]

    def check_action(
        self,
        action_type: str,
        target: str = "",
        content: str | None = None,
        boundary: SecurityBoundary | None = None,
    ) -> PermissionCheck:
        """
        Decides what may happen for one tool call.

        1. A tool NOT in ``allowed_tools`` is denied outright (deny-by-
           default), whatever its category level.
        2. A whitelisted tool whose category level is ALLOW/DENY is decided
           deterministically.
        3. A CONFIRM-level tool is delegated to the SecurityBoundary —
           risk tier, guardrails and the active security mode decide, and
           the boundary's verdict (allow/confirm/deny) is returned as-is.
        """
        if action_type not in self.allowed_tools:
            return PermissionCheck(
                decision=Decision.DENY,
                category=PermissionCategory.OTHER,
                reason=f"tool '{action_type}' is not in the agent's allowed_tools",
            )
        category = classify_tool(action_type, target)
        level = self.level_for(category)
        if level is Decision.DENY:
            return PermissionCheck(
                decision=Decision.DENY,
                category=category,
                reason=f"agent permission '{category.value}' is DENY",
            )
        if level is Decision.ALLOW:
            return PermissionCheck(
                decision=Decision.ALLOW,
                category=category,
                reason=f"agent permission '{category.value}' is ALLOW",
            )
        # CONFIRM -> real security evaluation. The boundary is the final
        # authority: guardrails can still deny, high/critical tiers confirm.
        boundary = boundary or get_boundary()
        verdict = boundary.check(action_type, target, content)
        return PermissionCheck(
            decision=verdict.decision,
            category=category,
            reason=verdict.reason,
            delegated=True,
        )

    def describe(self) -> str:
        """Compact human-readable profile for logs and prompts."""
        bits = [
            f"read={self.read.value}",
            f"write={self.write.value}",
            f"test={self.test.value}",
            f"shell={self.shell.value}",
            f"network={self.network.value}",
            f"tools={len(self.allowed_tools)}",
        ]
        return "AgentPermissions(" + ", ".join(bits) + ")"
