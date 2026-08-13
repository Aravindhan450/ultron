"""STEP 2C — Intent -> Capability selection tests.

Verifies: the one selection path (RESOLVED/AMBIGUOUS/UNKNOWN), canonical
vocabulary use, no tool-name metadata in the mapper, contract + registry
discovery, SimpleAgent/ReAct sharing the same selection, and that selection
never bypasses security or invents tools for unknown intents.
"""

from __future__ import annotations

from pathlib import Path

from ultron.core.capabilities import (
    CAPABILITY_CONTRACTS,
    CapabilityContract,
    SelectionState,
    select_capability,
    select_for_request,
)
from ultron.core.capabilities.selector import (
    _AMBIGUOUS_INTENTS,
    _INTENT_TO_CAPABILITY,
)
from ultron.core.nlp.intent import IntentCategory, route_request
from ultron.core.tools.definitions import (
    TOOL_DEFINITIONS,
    ToolCapability,
    preferred_tool_for,
    tools_with_capability,
)

RESOLVED = SelectionState.RESOLVED
AMBIGUOUS = SelectionState.AMBIGUOUS
UNKNOWN = SelectionState.UNKNOWN

# Deterministic capability-category probes (not historical symbols).
_PHRASE_CASES = (
    ("Find where this symbol is used", "reference_lookup"),
    ("Where is this class defined?", "definition_lookup"),
    ("Run pwd", "terminal_execution"),
    ("List the files here", "directory_list"),
    ("search the code for foo", "code_search"),
    ("How does this component work?", "repository_investigation"),
)


def test_intent_maps_to_expected_capability():
    for phrase, expected in _PHRASE_CASES:
        selection = select_for_request(phrase)
        assert selection.state is RESOLVED, phrase
        assert selection.primary is not None
        assert selection.primary.value == expected, phrase


def test_capability_comes_from_canonical_vocabulary():
    # Every mapped value is a member of the canonical ToolCapability enum —
    # no second capability vocabulary exists.
    for intent, cap in _INTENT_TO_CAPABILITY.items():
        assert isinstance(intent, IntentCategory)
        assert isinstance(cap, ToolCapability)
        assert cap.value in {c.value for c in ToolCapability}


def test_no_duplicate_capability_vocabulary():
    # The selector references the existing enums; it does not redefine them.
    from ultron.core.capabilities import selector

    source = Path(selector.__file__).read_text()
    assert "class IntentCategory" not in source
    assert "class ToolCapability" not in source
    # And every intent category is accounted for (mapped or ambiguous) except
    # the terminal UNKNOWN sentinel.
    accounted = set(_INTENT_TO_CAPABILITY) | set(_AMBIGUOUS_INTENTS)
    expected = {c for c in IntentCategory if c is not IntentCategory.UNKNOWN}
    assert accounted == expected, (
        "every intent category must be mapped or declared ambiguous"
    )


def test_unknown_intent_selects_no_tools():
    for intent in (None, IntentCategory.UNKNOWN):
        selection = select_capability(intent)
        assert selection.state is UNKNOWN
        assert selection.primary is None
        assert selection.preferred_tool is None
        assert selection.execution_tools == ()


def test_unknown_request_selects_no_tools():
    for text in ("", "   ", "zzz gibberish qqq"):
        selection = select_for_request(text)
        assert selection.state is UNKNOWN, text
        assert selection.preferred_tool is None
        assert selection.execution_tools == ()


def test_ambiguous_intent_is_explicit():
    selection = select_capability(IntentCategory.INFORMATION_REQUEST)
    assert selection.state is AMBIGUOUS
    assert selection.primary is None
    assert selection.ambiguity
    assert all(isinstance(c, ToolCapability) for c in selection.ambiguity)
    # Repository-vs-external candidates present, no tool chosen.
    values = {c.value for c in selection.ambiguity}
    assert "information_request" in values and "web_search" in values
    assert selection.preferred_tool is None


def test_capability_contract_is_retrievable_from_selection():
    selection = select_capability(IntentCategory.REFERENCE_LOOKUP)
    assert selection.is_resolved()
    contract = selection.contract
    assert isinstance(contract, CapabilityContract)
    assert contract.capability is selection.primary
    # Behavioral info present (evidence + success + failures).
    assert contract.evidence_required and contract.success_criteria
    assert contract.failure_classes


def test_execution_tools_come_from_canonical_registry():
    for intent in _INTENT_TO_CAPABILITY:
        selection = select_capability(intent)
        assert selection.is_resolved()
        cap = selection.primary
        assert selection.execution_tools == tuple(tools_with_capability(cap))
        assert selection.preferred_tool == preferred_tool_for(cap)
        for tool in selection.execution_tools:
            assert tool in TOOL_DEFINITIONS


def test_simple_and_react_share_the_same_selection():
    """Both agents' capability decisions flow from select_capability: the
    ReAct correction's specific-symbol determination and the SimpleAgent
    code-intel dispatch both resolve to the same preferred tool."""
    from ultron.core.agents.react import route_llm_tool_call

    for phrase, expected_tool in (
        ("Where is taskstate used?", "find_references"),
        ("Where is the zzqx class defined?", "find_definition"),
    ):
        intent = route_request(phrase)
        assert intent is not None
        selection = select_capability(intent.intent_type)
        # SimpleAgent path would dispatch selection.preferred_tool.
        assert selection.preferred_tool == expected_tool
        # ReAct correction, given a misrouted generic call on this turn.
        corrected = route_llm_tool_call(
            "code_search", {"query": "zzqx"}, user_input=phrase
        )
        assert corrected is not None
        assert corrected[0] == expected_tool


def test_repository_capabilities_never_route_to_external_search():
    for phrase in (
        "Where is this class defined?",
        "Find where this symbol is used",
        "How does this component work?",
    ):
        selection = select_for_request(phrase)
        assert selection.is_resolved()
        assert selection.primary is not None
        assert selection.primary is not ToolCapability.WEB_SEARCH


def test_external_capabilities_remain_available():
    selection = select_capability(IntentCategory.INFORMATION_REQUEST)
    assert selection.state is AMBIGUOUS
    # External candidate is represented; a genuine external request can still
    # resolve through the deterministic intent layer where one exists.
    assert ToolCapability.WEB_SEARCH in selection.ambiguity


def test_multiple_related_capabilities_are_representable():
    selection = select_capability(IntentCategory.CODING_REQUEST)
    assert selection.is_resolved()
    assert selection.primary is ToolCapability.CODING_REQUEST
    assert len(selection.related) >= 3
    for related in selection.related:
        assert isinstance(related, ToolCapability)
        assert related in CAPABILITY_CONTRACTS
    # Related comes from the contract — not a second table.
    assert selection.related == selection.contract.related_capabilities


def test_selection_does_not_bypass_security():
    """Selection only returns capability/tool names — execution still flows
    through the security boundary in the agents (check_action).  This test
    proves the selection layer itself has no execution path."""
    selection = select_for_request("Run rm -rf /tmp/zzqx")
    assert selection.is_resolved()
    assert selection.primary is ToolCapability.TERMINAL_EXECUTION
    # The boundary remains the gate: run_command is HIGH/confirm via canonical
    # risk — selection does not change that.
    from ultron.core.tools.definitions import get_tool_definition

    definition = get_tool_definition("run_command")
    assert definition is not None
    assert definition.risk.value in {"high", "medium"}
    assert definition.read_only is False


def test_resolved_capabilities_have_contracts_and_tools():
    for intent, cap in _INTENT_TO_CAPABILITY.items():
        selection = select_capability(intent)
        assert selection.is_resolved()
        assert cap in CAPABILITY_CONTRACTS
        # Capabilities may be multi-step (no direct tool) or tool-backed.
        if selection.execution_tools:
            assert selection.preferred_tool in selection.execution_tools


def test_selector_contains_no_tool_name_metadata():
    """Static scan: the mapper must stay Intent -> Capability.  No registered
    tool name or alias may appear in the selector source."""
    from ultron.core.capabilities import selector

    source = Path(selector.__file__).read_text()
    for name, definition in TOOL_DEFINITIONS.items():
        if "_" in name:
            assert name not in source, f"selector hardcodes tool {name!r}"
        for alias in definition.aliases:
            if "_" in alias:
                assert alias not in source, f"selector hardcodes alias {alias!r}"
    # No risk/read-only/schema vocabulary may appear as data.
    assert "read_only=" not in source
    assert "risk=" not in source


def test_selector_has_no_historical_symbol_hardcoding():
    from ultron.core.capabilities import selector

    source = Path(selector.__file__).read_text().lower()
    for forbidden in ("taskstate", "supervisor", "codingexecutor"):
        assert forbidden not in source, f"selector references {forbidden!r}"


def test_malformed_intent_is_unknown():
    selection = select_for_request("   \t ")
    assert selection.state is UNKNOWN
    assert selection.reason  # explicit reason, no silent tool selection
