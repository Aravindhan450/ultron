"""STEP 2A — single-source-of-truth validation for tool/capability metadata.

Proves the canonical definitions table (``ultron.core.tools.definitions``)
is the ONE authoritative metadata source and that every consumer derives
from it:

- Phase 7: coverage + runtime consistency (no registered tool unclassified,
  no consumer keeping an independent authoritative table, boundary tiers
  equal the declared canonical risk);
- Phase 8: mutation proof — changing ONE canonical definition propagates to
  schema, capability lookup, security classification and orchestration
  classification;
- Phase 6: guardrails still override declared metadata;
- Phase 9: all 58 registered tools have complete canonical metadata;
- Phase 10: no exact-match/hardcoded routing for historical diagnostic
  symbols remains in the routing layer.
"""

from __future__ import annotations

import importlib
import importlib.util
import pathlib

import pytest

from ultron.core.tools.definitions import (
    TOOL_DEFINITIONS,
    ToolCapability,
    ToolDomain,
    ToolRisk,
    action_label_for,
    canonical_action_name,
    code_intel_tool_names,
    generic_code_tool_names,
    get_tool_definition,
    preferred_tool_for,
    tool_aliases,
    web_tool_names,
)
from ultron.core.tools.registry import TOOLS, get_tools_schema
from ultron.security import Decision, RiskTier
from ultron.security.boundary import SecurityBoundary

# ---------------------------------------------------------------------------
# Phase 9 — coverage: every registered tool has complete canonical metadata
# ---------------------------------------------------------------------------


def test_every_registered_tool_has_canonical_metadata():
    assert set(TOOLS) == set(TOOL_DEFINITIONS)
    assert len(TOOLS) == 58
    assert len(TOOL_DEFINITIONS) == 58


def test_canonical_definitions_are_complete():
    for name, definition in TOOL_DEFINITIONS.items():
        assert definition.name == name
        assert definition.capabilities, f"{name}: no capabilities declared"
        assert definition.domain is not None
        assert definition.risk in ToolRisk
        assert definition.func is not None
        assert definition.resolved_description.strip(), f"{name}: no description"
        assert definition.read_only in (True, False)


def test_canonical_definitions_are_unique():
    # The dict itself cannot hold duplicates, but prove the model field and
    # the key agree and no alias collides with a registered name.
    for name, definition in TOOL_DEFINITIONS.items():
        assert definition.name == name
        for alias in definition.aliases:
            assert alias != name
            assert alias not in TOOL_DEFINITIONS


def test_schema_generation_consumes_canonical():
    schema = get_tools_schema()
    names = {entry["name"] for entry in schema}
    assert names == set(TOOL_DEFINITIONS)
    for entry in schema:
        assert entry["description"].strip()


# ---------------------------------------------------------------------------
# Phase 7 — no independent authoritative tables remain
# ---------------------------------------------------------------------------


def test_old_duplicate_structures_are_removed():
    from ultron.core import nlp
    from ultron.core.intelligence import parallel_tools as pt
    from ultron.permissions import classifier

    assert importlib.util.find_spec("ultron.core.nlp.capabilities") is None
    assert not hasattr(nlp, "TOOL_CAPABILITIES")
    assert not hasattr(nlp, "select_tool")
    assert not hasattr(classifier, "_ACTION_LABELS")
    assert not hasattr(classifier, "_ACTION_ALIASES")
    assert not hasattr(pt, "TOOL_TO_ACTION")
    assert not hasattr(pt, "ACTION_TO_TOOL")


def test_react_redirect_sets_are_derived_from_canonical():
    from ultron.core.agents.react import (
        _CODE_INTEL_TOOLS,
        _SPECIFIC_SYMBOL_TOOLS,
        _TURN_CORRECTABLE_TOOLS,
    )

    assert _CODE_INTEL_TOOLS == code_intel_tool_names()
    assert _TURN_CORRECTABLE_TOOLS == frozenset(
        generic_code_tool_names() | web_tool_names()
    )
    specific = {
        t
        for t in (
            preferred_tool_for(ToolCapability.DEFINITION_LOOKUP),
            preferred_tool_for(ToolCapability.REFERENCE_LOOKUP),
        )
        if t
    }
    assert _SPECIFIC_SYMBOL_TOOLS == specific
    # The redirect universe stays read-only: code-intel tools are all
    # declared read-only in the canonical table.
    assert code_intel_tool_names() <= {n for n, d in TOOL_DEFINITIONS.items() if d.read_only}


def test_boundary_tiers_match_canonical_declared_risk():
    """The security boundary must classify every registered tool at its
    declared canonical risk (policy-refined tools are exempt: their tier is
    target-sensitive by design)."""
    policy_refined = {"make_http_request", "run_query", "run_command", "run_parallel"}
    boundary = SecurityBoundary()
    for name, definition in TOOL_DEFINITIONS.items():
        if name in policy_refined:
            continue
        tier = boundary.classify_action(name, "target-value")
        assert tier == RiskTier(definition.risk.value), (
            f"{name}: boundary {tier.value} != canonical {definition.risk.value}"
        )


def test_boundary_resolves_aliases():
    boundary = SecurityBoundary()
    for name, definition in TOOL_DEFINITIONS.items():
        for alias in definition.aliases:
            assert boundary.classify_action(alias, "x") == boundary.classify_action(
                name, "x"
            ), f"alias {alias} gates differently from {name}"


def test_capability_preference_maps_routing_categories():
    expected = {
        ToolCapability.DEFINITION_LOOKUP: "find_definition",
        ToolCapability.REFERENCE_LOOKUP: "find_references",
        ToolCapability.SYMBOL_SEARCH: "find_symbol",
        ToolCapability.SYMBOL_INSPECTION: "report_symbol",
        ToolCapability.CODE_SEARCH: "code_search",
        ToolCapability.SEMANTIC_SEARCH: "semantic_search",
        ToolCapability.REPOSITORY_INVESTIGATION: "code_investigation",
        ToolCapability.TERMINAL_EXECUTION: "run_command",
        ToolCapability.TEST_EXECUTION: "run_command",
        ToolCapability.FILE_READ: "read_file",
        ToolCapability.FILE_WRITE: "write_file",
        ToolCapability.DIRECTORY_LIST: "list_directory",
        ToolCapability.FILE_SEARCH: "search_files",
    }
    for capability, tool in expected.items():
        assert preferred_tool_for(capability) == tool, f"{capability} -> {tool}"


def test_aliases_resolve_to_canonical_tools():
    aliases = tool_aliases()
    assert aliases == {
        "search_web": "web_search",
        "fetch_page_text": "fetch_page",
        "run_query": "db_query",
    }
    for name, alias in aliases.items():
        assert canonical_action_name(alias) == name
        assert get_tool_definition(alias).name == name


def test_action_labels_come_from_canonical():
    assert action_label_for("run_command") == "Execute terminal command"
    assert action_label_for("web_search") == "Search the web"
    assert action_label_for("db_query") == "Execute database query"
    assert action_label_for("search_web") == "Search the web"
    assert action_label_for("get_debug_context") is None


def test_classifier_consumes_canonical_labels_and_aliases():
    from ultron.permissions.classifier import PermissionClassifier, PermissionLevel

    classifier = PermissionClassifier()
    request = classifier.classify("fetch_page", "https://example.com")
    assert request.permission == PermissionLevel.AUTO
    assert request.action_label == "Fetch web page"
    request = classifier.classify("overwrite_file", "src/a.py")
    assert request.action_label == "Overwrite existing file"


def test_orchestration_classification_derives_from_canonical():
    from ultron.core.orchestration.permissions import (
        PermissionCategory,
        classify_tool,
    )

    assert classify_tool("read_file") is PermissionCategory.READ
    assert classify_tool("write_file") is PermissionCategory.WRITE
    assert classify_tool("check_connectivity") is PermissionCategory.NETWORK
    assert classify_tool("run_parallel") is PermissionCategory.SHELL
    # read-only code intelligence never classifies as WRITE.
    for name in ("code_search", "find_definition", "semantic_search"):
        assert classify_tool(name) is PermissionCategory.READ


# ---------------------------------------------------------------------------
# Phase 8 — mutation proof: one canonical change propagates everywhere
# ---------------------------------------------------------------------------


def test_mutation_propagates_to_schema_capability_security_and_orchestration(monkeypatch):
    from ultron.core.orchestration.permissions import (
        PermissionCategory,
        classify_tool,
    )

    definition = TOOL_DEFINITIONS["read_file"].model_copy(
        deep=True, update={
            "name": "__synthetic_meta_tool__",
            "func": lambda file_path: "ok",
            "capabilities": (ToolCapability.CODING_REQUEST,),
            "read_only": False,
            "risk": ToolRisk.HIGH,
            "domain": ToolDomain.FILESYSTEM,
            "aliases": ("syn_alias",),
        }
    )
    monkeypatch.setitem(TOOL_DEFINITIONS, definition.name, definition)

    # 1. Tool schema generation sees the new tool.
    schema_names = {entry["name"] for entry in get_tools_schema()}
    assert definition.name in schema_names

    # 2. Capability lookup sees it (CODING_REQUEST is unclaimed otherwise).
    assert preferred_tool_for(ToolCapability.CODING_REQUEST) == definition.name
    assert canonical_action_name("syn_alias") == definition.name

    # 3. Security classification sees the declared risk + alias.
    boundary = SecurityBoundary()
    assert boundary.classify_action(definition.name, "x") == RiskTier.HIGH
    assert boundary.classify_action("syn_alias", "x") == RiskTier.HIGH

    # 4. Orchestration classification sees read_only/domain.
    assert classify_tool(definition.name) is PermissionCategory.WRITE

    # 5. Changing ONE field (risk + read_only + domain) propagates again.
    monkeypatch.setitem(
        TOOL_DEFINITIONS,
        definition.name,
        definition.model_copy(
            deep=True,
            update={"read_only": True, "risk": ToolRisk.LOW, "domain": ToolDomain.NETWORK},
        ),
    )
    assert boundary.classify_action(definition.name, "x") == RiskTier.LOW
    assert classify_tool(definition.name) is PermissionCategory.NETWORK


# ---------------------------------------------------------------------------
# Phase 6 — guardrails override declared metadata
# ---------------------------------------------------------------------------


def test_guardrails_override_canonical_low_risk():
    """A LOW-risk registered tool whose payload carries a credential must be
    DENIED by guardrails — metadata never overrides the security gate."""
    boundary = SecurityBoundary(mode="interactive")
    assert boundary.classify_action("add_memory", "", None) == RiskTier.LOW
    verdict = boundary.check(
        "add_memory",
        "",
        content="the api key is sk-ABCDEF0123456789abcdef",
        mode="interactive",
    )
    assert verdict.decision == Decision.DENY
    assert verdict.tier == RiskTier.CRITICAL


def test_guardrails_override_system_path_escalation():
    """Metadata says a write is HIGH; a system/credential path escalates it to
    CRITICAL (security policy wins over declared risk)."""
    boundary = SecurityBoundary()
    assert boundary.classify_action("write_file", "src/example.py") == RiskTier.HIGH
    protected = boundary.classify_action("write_file", "/etc/crontab")
    assert protected in (RiskTier.CRITICAL,)


# ---------------------------------------------------------------------------
# Phase 10 — anti-hardcoding: no exact-match routing on diagnostic symbols
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module_path",
    [
        "src/ultron/core/nlp/intent.py",
        "src/ultron/core/agents/react.py",
        "src/ultron/core/agents/simple.py",
        "src/ultron/core/coding/intelligence/resolve.py",
    ],
)
def test_no_exact_match_routing_on_historical_symbols(module_path):
    text = pathlib.Path(module_path).read_text()
    # The historical diagnostic symbols must never appear as exact-match
    # routing conditions (docstring examples are fine — code is not).
    for pattern in (
        '== "taskstate"',
        '"taskstate" ==',
        '== "TaskState"',
        'if "TaskState" in',
        'if "taskstate" in',
        'if "codingexecutor" in',
        'if "supervisor" in user_input',
    ):
        assert pattern not in text, f"{module_path} contains hardcoded match: {pattern}"


def test_generic_routing_works_for_arbitrary_symbols():
    from ultron.core.nlp.intent import route_request

    # Arbitrary, never-seen entity names must route generically.
    for phrase, tool in (
        ("Find references to ZzqxVendorSymbol", "find_references"),
        ("Where is ZzqxVendorSymbol defined?", "find_definition"),
        ("How does the ZzqxVendorComponent work?", "code_investigation"),
    ):
        intent = route_request(phrase)
        assert intent is not None
        assert intent.tool == tool, f"{phrase!r} routed to {intent.tool}"


def test_remaining_policy_sets_reference_known_canonical_tools():
    """Policy sets may remain (execution ordering / concurrency policy), but
    every tool name they reference must resolve to a canonical tool or alias
    so they can never silently drift from the single source of truth."""
    from ultron.core.coding.executor import _STATE_CHANGING_TOOLS, _inspection_tools
    from ultron.core.intelligence.parallel_tools import BATCH_READONLY_TOOLS
    from ultron.core.intelligence.planning import PLAN_ACTION_SPECS
    from ultron.core.tools.definitions import (
        TOOL_DEFINITIONS,
        canonical_action_name,
        tool_aliases,
    )

    # Registered tools + their alias spellings (aliases map canonical -> alias,
    # so the values are the alternative spellings). Internal non-tool actions
    # such as ``query_chain`` are documented exceptions.
    internal_actions = frozenset({"query_chain"})
    known = set(TOOL_DEFINITIONS) | set(tool_aliases().values())

    policy_tool_names: set[str] = set()
    policy_tool_names.update(tool for _action, (tool, _label, _req) in PLAN_ACTION_SPECS.items())
    policy_tool_names.update(BATCH_READONLY_TOOLS)
    policy_tool_names.update(_STATE_CHANGING_TOOLS)

    for name in sorted(policy_tool_names):
        assert name in known or name in internal_actions, (
            f"policy set references unknown tool {name!r}; add it to the "
            "canonical tool definitions"
        )
        # Every tool reference must canonicalize to a registered tool; internal
        # actions are exempt because they never map to an executable tool.
        if name not in internal_actions:
            assert canonical_action_name(name) in TOOL_DEFINITIONS, name

    # The executor's derived inspection set is a superset of read_file.
    assert "read_file" in _inspection_tools()
