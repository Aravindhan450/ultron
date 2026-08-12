"""
FIX #7 section 7.2 — Agent registry + agent permissions: deterministic tests.

Covers every required area without an LLM:

- all six agent types registered with full metadata
- registry lookup (by enum / value / name), listing, validation
- duplicate registration rejected
- unknown agent type fails safely
- capability lookup
- permission lookup
- unauthorized tool / unauthorized write / unauthorized shell
- security enforcement (CONFIRM levels delegate to the real boundary;
  guardrails still deny secrets)
- the LLM can never modify its own permissions (frozen profile)
- instantiate() builds a scoped, runtime-controlled ExecutionContext

Uses only in-memory / temp objects — never touches the real audit trail
except via an explicitly-supplied SecurityBoundary pointed at a temp log.
"""

import pytest

from ultron.core.orchestration import (
    DEFAULT_REGISTRY,
    AgentPermissions,
    AgentRegistry,
    AgentSpec,
    AgentState,
    AgentType,
    PermissionCategory,
    classify_tool,
)
from ultron.core.orchestration.registry import _baseline_specs
from ultron.security import Decision, RiskTier, SecurityBoundary
from ultron.security.audit import AuditLog

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_boundary(tmp_path):
    """Interactive boundary pointed at a throwaway audit file."""
    return SecurityBoundary(mode="interactive", audit_log=AuditLog(tmp_path / "audit.jsonl"))


# ---------------------------------------------------------------------------
# The six agent types
# ---------------------------------------------------------------------------


def test_all_six_agent_types_registered():
    registry = DEFAULT_REGISTRY
    assert len(registry) == 6
    names = set(registry.names())
    assert names == {
        "supervisor",
        "researcher",
        "coder",
        "tester",
        "reviewer",
        "security",
    }
    # Every section-7.1 enum value is registered.
    for member in AgentType:
        assert member in registry


def test_every_spec_has_full_metadata():
    from pydantic import ValidationError

    for spec in DEFAULT_REGISTRY.list():
        assert spec.name
        assert spec.capabilities, f"{spec.name}: capabilities missing"
        assert spec.permissions.allowed_tools, f"{spec.name}: no allowed tools"
        assert spec.max_budget is not None
        assert spec.max_budget.max_steps > 0
        assert spec.risk_level in RiskTier
        assert spec.description
        # The profile is frozen — the runtime cannot be overridden post-hoc.
        with pytest.raises(ValidationError):
            spec.permissions.read = Decision.DENY


def test_baseline_permission_shapes():
    by_name = {s.name: s for s in DEFAULT_REGISTRY.list()}

    # Supervisor: read/search only, no writes/shell/network.
    supervisor = by_name["supervisor"]
    assert supervisor.permissions.read is Decision.ALLOW
    assert supervisor.permissions.write is Decision.DENY
    assert supervisor.permissions.shell is Decision.DENY
    assert supervisor.permissions.network is Decision.DENY
    assert "write_file" not in supervisor.permissions.allowed_tools
    assert "run_command" not in supervisor.permissions.allowed_tools

    # Researcher: read/search/LSP; no writes.
    researcher = by_name["researcher"]
    assert researcher.permissions.read is Decision.ALLOW
    assert researcher.permissions.write is Decision.DENY
    assert researcher.permissions.shell is Decision.DENY
    assert "find_definition" in researcher.permissions.allowed_tools

    # Coder: write/test/shell/network subject to security (CONFIRM).
    coder = by_name["coder"]
    assert coder.permissions.write is Decision.CONFIRM
    assert coder.permissions.test is Decision.CONFIRM
    assert coder.permissions.shell is Decision.CONFIRM
    assert coder.permissions.network is Decision.CONFIRM
    assert "write_file" in coder.permissions.allowed_tools
    assert "run_command" in coder.permissions.allowed_tools

    # Tester: read/search/test; no application-code writes.
    tester = by_name["tester"]
    assert tester.permissions.read is Decision.ALLOW
    assert tester.permissions.test is Decision.ALLOW
    assert tester.permissions.write is Decision.DENY
    assert tester.permissions.shell is Decision.DENY
    assert "write_file" not in tester.permissions.allowed_tools
    assert "run_command" in tester.permissions.allowed_tools  # for test commands

    # Reviewer: read/search/git diff; no writes.
    reviewer = by_name["reviewer"]
    assert reviewer.permissions.read is Decision.ALLOW
    assert reviewer.permissions.write is Decision.DENY
    assert reviewer.permissions.shell is Decision.DENY
    assert "write_file" not in reviewer.permissions.allowed_tools

    # Security: read/search/analysis; no writes.
    security = by_name["security"]
    assert security.permissions.read is Decision.ALLOW
    assert security.permissions.write is Decision.DENY
    assert security.permissions.shell is Decision.DENY
    assert "write_file" not in security.permissions.allowed_tools


# ---------------------------------------------------------------------------
# Registry: lookup / listing / validation / duplicates / unknown
# ---------------------------------------------------------------------------


def test_registry_lookup_by_enum_value_and_name():
    spec = DEFAULT_REGISTRY.get(AgentType.CODING)
    assert spec.name == "coder"
    assert DEFAULT_REGISTRY.get("coding") is spec  # enum value
    assert DEFAULT_REGISTRY.get("coder") is spec  # role name
    assert DEFAULT_REGISTRY.get("CODING") is spec  # case-insensitive


def test_registry_list_and_contains():
    assert len(DEFAULT_REGISTRY.list()) == 6
    assert "coder" in DEFAULT_REGISTRY
    assert AgentType.CODING in DEFAULT_REGISTRY
    assert "not_an_agent" not in DEFAULT_REGISTRY


def test_registry_validate_clean():
    assert DEFAULT_REGISTRY.validate() == []


def test_duplicate_registration_rejected():
    registry = AgentRegistry()
    for spec in _baseline_specs():
        registry.register(spec)
    duplicate = _baseline_specs()[2]  # coder
    with pytest.raises(ValueError, match="Duplicate agent registration"):
        registry.register(duplicate)
    # Duplicate NAME (different, unregistered type) is also rejected.
    fresh = AgentRegistry()
    fresh.register(_baseline_specs()[2])  # only coder registered
    impostor = AgentSpec(
        agent_type=AgentType.RESEARCH,  # not yet registered in `fresh`
        name="coder",  # collides with the existing coder name
        capabilities=["read"],
        permissions=AgentPermissions(allowed_tools=["read_file"]),
    )
    with pytest.raises(ValueError, match="Duplicate agent name"):
        fresh.register(impostor)


def test_unknown_agent_type_fails_safely():
    with pytest.raises(KeyError, match="Unknown agent type"):
        DEFAULT_REGISTRY.get("the-llm-agent")
    with pytest.raises(KeyError, match="Unknown agent type"):
        DEFAULT_REGISTRY.get("coder-but-typo")


# ---------------------------------------------------------------------------
# Capabilities + permission lookup
# ---------------------------------------------------------------------------


def test_capability_lookup():
    assert DEFAULT_REGISTRY.has_capability("coder", "write")
    assert DEFAULT_REGISTRY.has_capability("coder", "test")
    assert DEFAULT_REGISTRY.has_capability("researcher", "code_intelligence")
    assert DEFAULT_REGISTRY.has_capability("reviewer", "git_diff")
    assert not DEFAULT_REGISTRY.has_capability("researcher", "write")
    assert not DEFAULT_REGISTRY.has_capability("supervisor", "write")
    assert DEFAULT_REGISTRY.capabilities("coder") is not None


def test_permission_lookup():
    perms = DEFAULT_REGISTRY.permissions_for(AgentType.TEST_QA)
    assert perms.write is Decision.DENY
    assert perms.test is Decision.ALLOW


def test_tool_classification_deterministic():
    assert classify_tool("read_file") is PermissionCategory.READ
    assert classify_tool("write_file") is PermissionCategory.WRITE
    assert classify_tool("web_search") is PermissionCategory.NETWORK
    assert classify_tool("run_parallel") is PermissionCategory.SHELL
    assert classify_tool("run_command", "pytest tests/") is PermissionCategory.TEST
    assert classify_tool("run_command", "npm test") is PermissionCategory.TEST
    assert classify_tool("run_command", "git diff") is PermissionCategory.READ
    assert classify_tool("run_command", "ls -la") is PermissionCategory.READ
    assert classify_tool("run_command", "git commit -m x") is PermissionCategory.SHELL
    assert classify_tool("run_command", "echo hi > file") is PermissionCategory.SHELL
    assert classify_tool("unknown_tool") is PermissionCategory.OTHER
    # check_connectivity is a NETWORK action — never misclassified as READ.
    assert classify_tool("check_connectivity") is PermissionCategory.NETWORK


# ---------------------------------------------------------------------------
# Permission enforcement (deterministic decisions, no boundary needed)
# ---------------------------------------------------------------------------


def test_unauthorized_tool_denied(temp_boundary):
    # A tool not in the agent's whitelist is denied regardless of category.
    assert DEFAULT_REGISTRY.check_action(
        "coder", "read_file", boundary=temp_boundary
    ) is Decision.ALLOW
    # Coder write is CONFIRM-level; with a real target the interactive
    # boundary confirms (HIGH tier) rather than guardrail-denying an empty
    # target.
    assert DEFAULT_REGISTRY.check_action(
        "coder", "write_file", "src/app.py", boundary=temp_boundary
    ) is Decision.CONFIRM
    # Supervisor cannot run commands at all (not whitelisted).
    assert DEFAULT_REGISTRY.check_action(
        "supervisor", "run_command", "ls", boundary=temp_boundary
    ) is Decision.DENY
    # Coder CAN run commands (whitelisted) at CONFIRM level; without a
    # boundary the deterministic LEVEL is what we assert (pytest classifies
    # as TEST, coder.test is CONFIRM).
    assert (
        DEFAULT_REGISTRY.permissions_for("coder").level_for(
            classify_tool("run_command", "pytest")
        )
        is Decision.CONFIRM
    )


def test_unauthorized_write_blocked():
    assert DEFAULT_REGISTRY.check_action("researcher", "write_file", "x.py") is Decision.DENY
    assert DEFAULT_REGISTRY.check_action("tester", "write_file", "app.py") is Decision.DENY
    assert DEFAULT_REGISTRY.check_action("reviewer", "replace_in_file", "app.py") is Decision.DENY
    assert DEFAULT_REGISTRY.check_action("security", "write_file", "x.py") is Decision.DENY
    assert DEFAULT_REGISTRY.check_action("supervisor", "delete_file", "x.py") is Decision.DENY


def test_unauthorized_shell_blocked():
    # Tester: test commands allowed, arbitrary shell denied.
    assert DEFAULT_REGISTRY.check_action("tester", "run_command", "pytest tests/") is Decision.ALLOW
    assert DEFAULT_REGISTRY.check_action("tester", "run_command", "rm -rf src/") is Decision.DENY
    # Researcher: no shell at all.
    assert DEFAULT_REGISTRY.check_action("researcher", "run_command", "ls") is Decision.DENY
    # Reviewer: read-only git is READ; state-changing git is SHELL -> denied.
    assert DEFAULT_REGISTRY.check_action("reviewer", "run_command", "git diff") is Decision.ALLOW
    assert DEFAULT_REGISTRY.check_action("reviewer", "run_command", "git commit -m x") is Decision.DENY


def test_unauthorized_network_blocked():
    assert DEFAULT_REGISTRY.check_action("supervisor", "web_search", "x") is Decision.DENY
    assert DEFAULT_REGISTRY.check_action("tester", "fetch_page_text", "https://x") is Decision.DENY
    # Researcher may research the web (network ALLOW).
    assert DEFAULT_REGISTRY.check_action("researcher", "web_search", "x") is Decision.ALLOW


# ---------------------------------------------------------------------------
# Security enforcement: CONFIRM levels delegate to the real boundary
# ---------------------------------------------------------------------------


def test_coder_write_goes_through_security_boundary(temp_boundary):
    # Coder write is CONFIRM -> the security boundary decides. A normal write
    # is HIGH risk -> confirm under interactive mode.
    assert (
        DEFAULT_REGISTRY.check_action(
            "coder", "write_file", "src/app.py", boundary=temp_boundary
        )
        is Decision.CONFIRM
    )
    # A secret-embedding write is guardrail-DENIED even for the coder.
    assert (
        DEFAULT_REGISTRY.check_action(
            "coder", "write_file", "config.py", "api_key=sk-1234567890abcdef",
            boundary=temp_boundary,
        )
        is Decision.DENY
    )


def test_coder_read_allowed_directly(temp_boundary):
    assert (
        DEFAULT_REGISTRY.check_action("coder", "read_file", "src/app.py", boundary=temp_boundary)
        is Decision.ALLOW
    )


def test_tester_test_command_allowed_but_writes_blocked(temp_boundary):
    assert (
        DEFAULT_REGISTRY.check_action("tester", "run_command", "pytest tests/", boundary=temp_boundary)
        is Decision.ALLOW
    )
    assert (
        DEFAULT_REGISTRY.check_action("tester", "write_file", "src/app.py", boundary=temp_boundary)
        is Decision.DENY
    )


def test_permission_decisions_recorded_to_audit_log(temp_boundary):
    # A permission-level DENY that the boundary never sees must still land in
    # the audit trail so security is fully inspectable.
    DEFAULT_REGISTRY.check_action(
        "researcher", "write_file", "secret.py", boundary=temp_boundary
    )
    records = temp_boundary.audit_log.read()
    assert any(
        r["action_type"] == "write_file" and r["decision"] == "deny"
        for r in records
    )


def test_delegated_verdicts_not_double_recorded(temp_boundary):
    # A CONFIRM-level check delegates to the boundary, which records its own
    # verdict; the registry must NOT add a second record for the same action.
    DEFAULT_REGISTRY.check_action(
        "coder", "write_file", "src/app.py", boundary=temp_boundary
    )
    records = temp_boundary.audit_log.read()
    write_records = [r for r in records if r["action_type"] == "write_file"]
    assert len(write_records) == 1  # exactly one: the boundary's own verdict


def test_audit_record_uses_action_risk_tier(temp_boundary):
    # The recorded tier reflects the ACTION's classified risk, not the agent
    # type's standing risk — a secret-embedding write denial is CRITICAL.
    DEFAULT_REGISTRY.check_action(
        "coder", "write_file", "config.py", "api_key=sk-1234567890abcdef",
        boundary=temp_boundary,
    )
    records = temp_boundary.audit_log.read()
    secret_deny = [
        r for r in records
        if r["action_type"] == "write_file" and r["decision"] == "deny"
    ]
    assert secret_deny and secret_deny[0]["tier"] == "critical"


# ---------------------------------------------------------------------------
# The LLM can never modify its own permissions
# ---------------------------------------------------------------------------


def test_permissions_are_frozen_immutable():
    from pydantic import ValidationError

    perms = AgentPermissions(allowed_tools=["read_file"])
    original = perms.read
    with pytest.raises(ValidationError):
        perms.read = Decision.DENY  # frozen: assignment must fail
    assert perms.read is original
    # allowed_tools is a tuple, so even in-place mutation is impossible.
    with pytest.raises(AttributeError):
        perms.allowed_tools.append("run_command")  # type: ignore[attr-defined]
    assert perms.allowed_tools == ("read_file",)


def test_context_check_action_enforces_profile():
    state = DEFAULT_REGISTRY.instantiate(
        "researcher", "r1", "task-1", "research auth", workspace="/tmp/proj"
    )
    ctx = state.context
    assert ctx.check_action("read_file") is Decision.ALLOW
    assert ctx.check_action("write_file", "x.py") is Decision.DENY
    assert ctx.check_action("run_command", "ls") is Decision.DENY  # no shell
    # Without a profile (hand-built context) the whitelist fallback applies.
    from ultron.core.orchestration import ExecutionContext

    bare = ExecutionContext(task_id="t", agent_id="a", allowed_tools=["read_file"])
    assert bare.check_action("read_file") is Decision.ALLOW
    assert bare.check_action("run_command") is Decision.DENY


def test_context_cannot_rebind_permission_profile():
    state = DEFAULT_REGISTRY.instantiate(
        "coder", "c1", "task-1", "write code", workspace="/tmp/proj"
    )
    ctx = state.context
    assert ctx.agent_permissions is not None
    original = ctx.agent_permissions
    # Rebinding the runtime profile must be rejected — an agent cannot swap
    # in a permissive profile.
    from ultron.core.orchestration import AgentPermissions

    with pytest.raises(ValueError, match="cannot be reassigned"):
        ctx.agent_permissions = AgentPermissions(
            read=Decision.ALLOW, write=Decision.ALLOW, allowed_tools=["run_command"]
        )
    assert ctx.agent_permissions is original
    # The runtime can still assign it once (from PENDING/construction).
    from ultron.core.orchestration import ExecutionContext

    fresh = ExecutionContext(task_id="t", agent_id="a")
    profile = DEFAULT_REGISTRY.permissions_for("reviewer")
    fresh.agent_permissions = profile
    assert fresh.agent_permissions is profile


# ---------------------------------------------------------------------------
# instantiate(): scoped, runtime-controlled execution context
# ---------------------------------------------------------------------------


def test_instantiate_builds_scoped_state():
    state = DEFAULT_REGISTRY.instantiate(
        "coder",
        "c1",
        "task-9",
        "add health endpoint",
        workspace="/tmp/proj",
        current_plan_step=2,
    )
    assert isinstance(state, AgentState)
    assert state.task_id == "task-9"
    assert state.objective == "add health endpoint"
    assert state.identity.agent_type is AgentType.CODING
    ctx = state.context
    assert ctx.agent_permissions is not None
    assert ctx.agent_permissions.write is Decision.CONFIRM
    assert "write_file" in ctx.allowed_tools
    assert ctx.workspace == "/tmp/proj"
    assert ctx.current_plan_step == 2
    # The per-run budget is a copy of the type's template.
    assert ctx.budget.max_steps == DEFAULT_REGISTRY.get("coder").max_budget.max_steps
    assert ctx.budget.steps_used == 0


def test_instantiate_fails_safely_for_unknown_type():
    with pytest.raises(KeyError, match="Unknown agent type"):
        DEFAULT_REGISTRY.instantiate("llm-agent", "x", "t", "o")


def test_baseline_specs_are_reusable_for_custom_registries():
    registry = AgentRegistry()
    for spec in _baseline_specs():
        registry.register(spec)
    assert registry.validate() == []
    assert len(registry) == 6
