"""
ultron.core.orchestration.registry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Agent registry + agent-type metadata (Fix #7, section 7.2).

Every orchestrated agent type is described by an :class:`AgentSpec`:

- ``name`` / ``agent_type`` — identity
- ``capabilities`` — what the type can do (read, write, test, ...)
- ``permissions`` — the frozen :class:`AgentPermissions` profile (read /
  write / test / shell / network levels + tool whitelist)
- ``max_budget`` — the :class:`ExecutionBudget` template for runs of this
  type (survives instantiation into a per-run copy)
- ``risk_level`` — the type's standing risk tier (metadata)

The :class:`AgentRegistry` provides register / retrieve / validate / list /
capability-check / instantiate, rejects duplicate registrations, and fails
safely on unknown types.

CRITICAL REQUIREMENT: permissions are controlled by the RUNTIME, never by
an LLM or an agent. The baseline specs below are constructed here at import
time (``DEFAULT_REGISTRY``), the permission profile is frozen, and the
registry object itself is never handed to an agent — agents only ever see
the scoped :class:`ExecutionContext` built by :meth:`AgentRegistry.
instantiate`.
"""

from __future__ import annotations

from ultron.core.logging import get_logger
from ultron.core.orchestration.models import (
    AgentIdentity,
    AgentState,
    AgentType,
    ExecutionBudget,
    ExecutionContext,
)
from ultron.core.orchestration.permissions import (
    AgentPermissions,
    PermissionCategory,
    PermissionCheck,
    classify_tool,
)
from ultron.core.tools.definitions import TOOL_DEFINITIONS, ToolDomain
from ultron.security import BoundaryResult, Decision, RiskTier, SecurityBoundary

logger = get_logger("ultron.orchestration.registry")

# ---------------------------------------------------------------------------
# AgentSpec — one agent type's full metadata
# ---------------------------------------------------------------------------


class AgentSpec:
    """
    Full metadata for one orchestrated agent type.

    Attributes:
        agent_type: The :class:`AgentType` this spec describes.
        name: Human-readable role name (e.g. "coder").
        capabilities: List of capability labels (read, search, write, test,
            shell, network, code_intelligence, ...).
        permissions: Frozen :class:`AgentPermissions` — the runtime profile.
        max_budget: :class:`ExecutionBudget` template for this type.
        risk_level: Standing risk tier of the type (metadata; the actual
            per-action risk is decided by the security boundary).
        description: One-line role description.
    """

    def __init__(
        self,
        agent_type: AgentType,
        name: str,
        capabilities: list[str],
        permissions: AgentPermissions,
        max_budget: ExecutionBudget | None = None,
        risk_level: RiskTier = RiskTier.LOW,
        description: str = "",
    ) -> None:
        self.agent_type = agent_type
        self.name = name
        self.capabilities = list(capabilities)
        self.permissions = permissions
        self.max_budget = max_budget or ExecutionBudget()
        self.risk_level = risk_level
        self.description = description

    def has_capability(self, capability: str) -> bool:
        return capability in self.capabilities

    def check_action(
        self,
        action_type: str,
        target: str = "",
        content: str | None = None,
        boundary: SecurityBoundary | None = None,
    ) -> PermissionCheck:
        """Delegates one action to this type's frozen permission profile."""
        return self.permissions.check_action(action_type, target, content, boundary)

    def describe(self) -> str:
        return (
            f"AgentSpec({self.agent_type.value}/{self.name}, "
            f"risk={self.risk_level.value}, caps={','.join(self.capabilities)}, "
            f"{self.permissions.describe()})"
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class AgentRegistry:
    """
    Runtime registry of agent types.

    - :meth:`register` — add a spec; duplicate agent_type OR name is rejected.
    - :meth:`get` — retrieve by :class:`AgentType` or by name/alias string;
      unknown types raise ``KeyError`` (fail safely).
    - :meth:`list` / :meth:`names` — inventory.
    - :meth:`validate` — every registered spec is internally consistent.
    - :meth:`has_capability` / :meth:`capabilities` — capability lookup.
    - :meth:`check_action` — permission lookup for one action.
    - :meth:`instantiate` — build a scoped :class:`AgentState` (identity +
      ExecutionContext with this type's allowed tools, frozen permissions
      and a per-run budget copy) ready for an agent to run.
    """

    def __init__(self) -> None:
        self._specs: dict[AgentType, AgentSpec] = {}
        self._by_name: dict[str, AgentType] = {}

    # -- registration --------------------------------------------------------

    def register(self, spec: AgentSpec) -> None:
        """Registers one agent type; rejects duplicate type OR name."""
        if spec.agent_type in self._specs:
            raise ValueError(
                f"Duplicate agent registration: '{spec.agent_type.value}' already "
                f"registered as '{self._specs[spec.agent_type].name}'"
            )
        if spec.name in self._by_name:
            raise ValueError(
                f"Duplicate agent name: '{spec.name}' already registered for "
                f"'{self._by_name[spec.name].value}'"
            )
        self._specs[spec.agent_type] = spec
        self._by_name[spec.name] = spec.agent_type

    # -- retrieval -----------------------------------------------------------

    def get(self, agent_type: AgentType | str) -> AgentSpec:
        """
        Retrieves a spec by :class:`AgentType`, its enum value, or its
        registered name. Unknown types raise ``KeyError`` (fail safely).
        """
        key = self._resolve(agent_type)
        if key is None:
            raise KeyError(f"Unknown agent type: '{agent_type}'")
        return self._specs[key]

    def _resolve(self, agent_type: AgentType | str) -> AgentType | None:
        if isinstance(agent_type, AgentType):
            return agent_type if agent_type in self._specs else None
        value = str(agent_type).lower()
        for member in AgentType:
            if member.value == value:
                return member if member in self._specs else None
        return self._by_name.get(value)

    def list(self) -> list[AgentSpec]:
        """All registered specs, in registration order."""
        return [self._specs[t] for t in AgentType if t in self._specs]

    def names(self) -> list[str]:
        """Registered role names, in registration order."""
        return [spec.name for spec in self.list()]

    def __len__(self) -> int:
        return len(self._specs)

    def __contains__(self, agent_type: AgentType | str) -> bool:
        return self._resolve(agent_type) is not None

    # -- validation ----------------------------------------------------------

    def validate(self) -> list[str]:
        """
        Returns a list of problems across all registered specs (empty = ok).

        Checks: capabilities reference the standard vocabulary, the
        permission profile is frozen, and every listed allowed tool has a
        deterministic category.
        """
        problems: list[str] = []
        for spec in self.list():
            if not spec.capabilities:
                problems.append(f"{spec.name}: no capabilities")
            for tool in spec.permissions.allowed_tools:
                if classify_tool(tool) is PermissionCategory.OTHER and tool != "run_command":
                    problems.append(
                        f"{spec.name}: tool '{tool}' has no deterministic category"
                    )
        return problems

    # -- capability + permission lookup ---------------------------------------

    def capabilities(self, agent_type: AgentType | str) -> list[str]:
        return list(self.get(agent_type).capabilities)

    def has_capability(self, agent_type: AgentType | str, capability: str) -> bool:
        return self.get(agent_type).has_capability(capability)

    def permissions_for(self, agent_type: AgentType | str) -> AgentPermissions:
        return self.get(agent_type).permissions

    def check_action(
        self,
        agent_type: AgentType | str,
        action_type: str,
        target: str = "",
        content: str | None = None,
        boundary: SecurityBoundary | None = None,
    ) -> Decision:
        """
        Permission verdict for one action under an agent type's profile.

        When a ``boundary`` is supplied, the verdict is ALSO recorded to its
        audit log — the same JSON-lines security trail the boundary itself
        writes — so permission-layer denials/decisions are inspectable
        alongside boundary verdicts. (CONFIRM-level checks already delegate
        to the boundary, which records them itself; this records the
        deterministic ALLOW/DENY decisions the boundary never sees.)
        """
        spec = self.get(agent_type)
        check = spec.check_action(action_type, target, content, boundary)
        if boundary is not None and not check.delegated:
            # The tier is the ACTION's classified risk (not the agent type's
            # standing risk), so the audit record reads like a boundary
            # verdict: a secret-embedding write denial is classified
            # CRITICAL, not "medium because the coder is medium".
            tier = boundary.classify_action(action_type, target, content)
            record = BoundaryResult(
                action_type=action_type,
                target=target,
                tier=tier,
                decision=check.decision,
                reason=f"agent permission [{check.category.value}]: {check.reason}",
            )
            try:
                boundary.audit_log.record(record, mode=boundary.mode)
            except Exception:  # noqa: BLE001 — auditing must never break the gate
                logger.warning(
                    "registry: failed to record permission verdict for "
                    "action_type=%s",
                    action_type,
                )
        return check.decision

    # -- instantiate ----------------------------------------------------------

    def instantiate(
        self,
        agent_type: AgentType | str,
        agent_id: str,
        task_id: str,
        objective: str,
        workspace: str = "",
        current_plan_step: int | None = None,
    ) -> AgentState:
        """
        Builds a scoped :class:`AgentState` for one agent run.

        The state's ExecutionContext carries THIS type's allowed tools, the
        frozen permission profile, and a per-run copy of the type's budget —
        the runtime-controlled shell the agent executes inside. The registry
        itself is NOT attached; no agent can reach it and mutate permissions.
        """
        spec = self.get(agent_type)
        budget = spec.max_budget.model_copy(deep=True)
        context = ExecutionContext(
            task_id=task_id,
            agent_id=agent_id,
            workspace=workspace,
            allowed_tools=list(spec.permissions.allowed_tools),
            permissions={"agent_type": spec.agent_type.value, "risk_level": spec.risk_level.value},
            agent_permissions=spec.permissions,
            budget=budget,
            current_plan_step=current_plan_step,
        )
        return AgentState(
            task_id=task_id,
            identity=AgentIdentity(
                agent_id=agent_id,
                agent_type=spec.agent_type,
                display_name=spec.name,
            ),
            objective=objective,
            context=context,
        )


# ---------------------------------------------------------------------------
# Baseline registry — the six agent types, controlled by the runtime
# ---------------------------------------------------------------------------

# Shared read/search/code-intelligence whitelist used by the read-only roles.
# DERIVED from the canonical definitions table (STEP 2A) — the read-only
# filesystem + code-intelligence tools, minus ``code_investigation``
# (deliberate policy: repository investigation stays with the LLM-driven
# agents, not the read-only specialist roles).
_READ_SEARCH_TOOLS = [
    name
    for name, definition in TOOL_DEFINITIONS.items()
    if definition.read_only
    and definition.domain
    in (ToolDomain.FILESYSTEM, ToolDomain.CODE_INTELLIGENCE)
    and name != "code_investigation"
]

# State-changing file tools the coder may use (policy), derived from the
# canonical filesystem-write domain + the overwrite_file pending-action type.
_WRITE_TOOLS = [
    name
    for name, definition in TOOL_DEFINITIONS.items()
    if not definition.read_only and definition.domain is ToolDomain.FILESYSTEM
] + ["overwrite_file"]


def _baseline_specs() -> list[AgentSpec]:
    """The six standard agent types with their baseline permissions.

    Baselines (section 7.2):

    - supervisor: read/search only; no direct code modification
    - researcher: read/search/LSP; no writes
    - coder: read/search/write/test/shell, all state-change subject to the
      security boundary (CONFIRM)
    - tester: read/search/test; no application-code writes
    - reviewer: read/search/git diff; no writes
    - security: read/search/security analysis; no writes
    """
    return [
        AgentSpec(
            agent_type=AgentType.SUPERVISOR,
            name="supervisor",
            capabilities=["read", "search", "coordinate"],
            permissions=AgentPermissions(
                read=Decision.ALLOW,
                write=Decision.DENY,
                test=Decision.DENY,
                shell=Decision.DENY,
                network=Decision.DENY,
                allowed_tools=list(_READ_SEARCH_TOOLS),
            ),
            max_budget=ExecutionBudget(max_steps=30, max_tool_calls=100, timeout_seconds=900),
            risk_level=RiskTier.LOW,
            description="Coordinates other agents; read/search only, no direct code modification.",
        ),
        AgentSpec(
            agent_type=AgentType.RESEARCH,
            name="researcher",
            capabilities=["read", "search", "code_intelligence", "lsp"],
            permissions=AgentPermissions(
                read=Decision.ALLOW,
                write=Decision.DENY,
                test=Decision.DENY,
                shell=Decision.DENY,
                network=Decision.ALLOW,  # web research is read-only intent
                allowed_tools=list(_READ_SEARCH_TOOLS) + ["web_search", "fetch_page_text"],
            ),
            max_budget=ExecutionBudget(max_steps=25, max_tool_calls=80, timeout_seconds=900),
            risk_level=RiskTier.LOW,
            description="Read/search/code intelligence only; no writes.",
        ),
        AgentSpec(
            agent_type=AgentType.CODING,
            name="coder",
            capabilities=["read", "search", "write", "test", "shell", "network", "code_intelligence"],
            permissions=AgentPermissions(
                read=Decision.ALLOW,
                write=Decision.CONFIRM,  # every state-change goes through the boundary
                test=Decision.CONFIRM,
                shell=Decision.CONFIRM,
                network=Decision.CONFIRM,
                allowed_tools=(
                    list(_READ_SEARCH_TOOLS)
                    + list(_WRITE_TOOLS)
                    + ["run_command", "run_parallel", "web_search", "fetch_page_text", "make_http_request"]
                ),
            ),
            max_budget=ExecutionBudget(max_steps=50, max_tool_calls=200, timeout_seconds=1800),
            risk_level=RiskTier.MEDIUM,
            description="Writes, tests and runs shell — all state-change subject to security.",
        ),
        AgentSpec(
            agent_type=AgentType.TEST_QA,
            name="tester",
            capabilities=["read", "search", "test"],
            permissions=AgentPermissions(
                read=Decision.ALLOW,
                write=Decision.DENY,  # no application-code writes
                test=Decision.ALLOW,  # test commands may run directly
                shell=Decision.DENY,  # only TEST-classified commands are permitted
                network=Decision.DENY,
                allowed_tools=list(_READ_SEARCH_TOOLS) + ["run_command"],
            ),
            max_budget=ExecutionBudget(max_steps=25, max_tool_calls=80, timeout_seconds=900),
            risk_level=RiskTier.LOW,
            description="Runs and analyzes tests; no application-code writes.",
        ),
        AgentSpec(
            agent_type=AgentType.REVIEWER,
            name="reviewer",
            capabilities=["read", "search", "code_intelligence", "git_diff"],
            permissions=AgentPermissions(
                read=Decision.ALLOW,
                write=Decision.DENY,
                test=Decision.DENY,
                shell=Decision.DENY,  # read-only git commands classify as READ
                network=Decision.DENY,
                allowed_tools=list(_READ_SEARCH_TOOLS) + ["run_command"],
            ),
            max_budget=ExecutionBudget(max_steps=25, max_tool_calls=80, timeout_seconds=900),
            risk_level=RiskTier.LOW,
            description="Reviews code and git diffs; no writes.",
        ),
        AgentSpec(
            agent_type=AgentType.SECURITY,
            name="security",
            capabilities=["read", "search", "security_analysis"],
            permissions=AgentPermissions(
                read=Decision.ALLOW,
                write=Decision.DENY,
                test=Decision.DENY,
                shell=Decision.DENY,
                network=Decision.ALLOW,  # advisory lookups
                allowed_tools=list(_READ_SEARCH_TOOLS) + ["web_search", "fetch_page_text"],
            ),
            max_budget=ExecutionBudget(max_steps=25, max_tool_calls=80, timeout_seconds=900),
            risk_level=RiskTier.MEDIUM,
            description="Security analysis; read-only, never writes.",
        ),
    ]


# The runtime-controlled singleton. Never handed to an agent.
DEFAULT_REGISTRY: AgentRegistry = AgentRegistry()
for _spec in _baseline_specs():
    DEFAULT_REGISTRY.register(_spec)
