"""STEP 2B — generic capability contract tests.

Verifies the contracts are: complete (one per canonical capability),
behavioral (purpose/inputs/evidence/success/failure), free of tool metadata,
capable of discovering execution mechanisms through the canonical registry,
generic (no project-specific symbols), and non-duplicating (mutation-proof).
"""

from __future__ import annotations

import re
from enum import Enum
from pathlib import Path

from ultron.core.capabilities import (
    CAPABILITY_CONTRACTS,
    CapabilityContract,
    CapabilityFailure,
    EvidenceKind,
    contract_for,
)
from ultron.core.tools.definitions import (
    TOOL_DEFINITIONS,
    ToolCapability,
    tools_with_capability,
)

# Historical diagnostic symbols — contracts must never mention them.
_FORBIDDEN_NAMES = (
    "taskstate",
    "supervisor",
    "codingexecutor",
    "orchestrationvalidator",
)

# Success-criteria language must ground itself in evidence/result descriptors.
_SUCCESS_WORDS = (
    "result", "evidence", "verified", "captured", "outcome", "executed",
    "occurrence", "explicit", "exists", "confirm", "list", "fetch", "return",
    "explains", "ran",
)


def _flatten(contract: CapabilityContract) -> list[str]:
    """All free-text content of a contract, for metadata-leak scanning."""
    parts = [contract.purpose, contract.user_intent]
    parts.extend(contract.required_inputs)
    parts.extend(contract.required_context)
    parts.extend(contract.success_criteria)
    return [p.lower() for p in parts]


def test_every_canonical_capability_has_a_contract():
    assert set(CAPABILITY_CONTRACTS) == set(ToolCapability)
    assert len(CAPABILITY_CONTRACTS) == len(ToolCapability) == 44


def test_capability_identifiers_are_unique_and_registered_identically():
    names = [c.capability for c in CAPABILITY_CONTRACTS.values()]
    assert len(names) == len(set(names))
    for cap, contract in CAPABILITY_CONTRACTS.items():
        assert contract.capability is cap


def test_contracts_contain_required_behavioral_information():
    for cap, c in CAPABILITY_CONTRACTS.items():
        assert c.purpose.strip(), cap
        assert c.user_intent.strip(), cap
        assert c.required_inputs or c.required_context or c.may_require_investigation, cap
        assert c.evidence_required, cap
        assert c.success_criteria, cap
        assert c.failure_classes, cap


def test_contracts_do_not_duplicate_tool_metadata():
    """No contract may define tool names/aliases — those live in TOOL_DEFINITIONS.

    Only underscore-bearing identifiers are checked as whole tokens: they are
    unambiguous code identifiers and can never legitimately appear in the
    natural-language contract text (plain-English words like "retrieve" or
    "build" that happen to name tools are not definitions).
    """
    # The model itself carries no tool-metadata fields.
    fields = set(CapabilityContract.__dataclass_fields__)
    for meta in ("risk", "read_only", "requires_confirmation", "aliases", "domain",
                 "schema", "tool"):
        assert meta not in fields, f"contract model leaks tool metadata field {meta!r}"

    identifiers: set[str] = set()
    for definition in TOOL_DEFINITIONS.values():
        if "_" in definition.name:
            identifiers.add(definition.name)
        for alias in definition.aliases:
            if "_" in alias:
                identifiers.add(alias)

    for cap, c in CAPABILITY_CONTRACTS.items():
        for fragment in _flatten(c):
            for ident in identifiers:
                assert ident.lower() not in fragment, (
                    f"contract {cap.value} mentions tool identifier {ident!r}"
                )


def test_contracts_do_not_duplicate_risk_or_read_only_sets():
    """Declared risk/read-only membership must come from canonical, not from
    contracts.  The only failure-like data in a contract is failure classes."""
    for c in CAPABILITY_CONTRACTS.values():
        assert not any(f.value == "risk" for f in c.failure_classes)
        assert not any(f.value == "read_only" for f in c.failure_classes)


def test_capabilities_discover_tools_through_canonical_registry():
    for cap, c in CAPABILITY_CONTRACTS.items():
        assert c.execution_tools() == tools_with_capability(cap)
        for tool in c.execution_tools():
            assert tool in TOOL_DEFINITIONS
        if c.execution_tools():
            assert c.preferred_tool() == c.execution_tools()[0]
        else:
            assert c.preferred_tool() is None


def test_multi_tool_capabilities_exist():
    multi = [c for c in CAPABILITY_CONTRACTS.values() if len(c.execution_tools()) > 1]
    assert multi, "expected at least one capability served by multiple tools"
    served = {c.capability.value for c in multi}
    assert {"file_write", "memory_query", "information_request"} & served


def test_contracts_are_generic_no_project_symbols():
    for cap, c in CAPABILITY_CONTRACTS.items():
        for fragment in _flatten(c):
            for forbidden in _FORBIDDEN_NAMES:
                assert forbidden not in fragment, (
                    f"contract {cap.value} references historical symbol {forbidden!r}"
                )


def test_success_criteria_are_evidence_based():
    for cap, c in CAPABILITY_CONTRACTS.items():
        for criterion in c.success_criteria:
            lowered = criterion.lower()
            assert any(word in lowered for word in _SUCCESS_WORDS), (
                f"{cap.value}: {criterion!r} is not evidence-grounded"
            )


def test_failure_states_are_explicit():
    for cap, c in CAPABILITY_CONTRACTS.items():
        assert c.failure_classes, cap
        assert all(isinstance(f, CapabilityFailure) for f in c.failure_classes)
        # No generic "operation failed" placeholder.
        assert not any(f.value == "operation_failed" for f in c.failure_classes)


def test_multi_step_capabilities_are_represented():
    for cap in (
        ToolCapability.REPOSITORY_INVESTIGATION,
        ToolCapability.CODING_REQUEST,
        ToolCapability.TEST_EXECUTION,
    ):
        c = contract_for(cap)
        assert c is not None
        assert c.may_require_investigation, cap
        assert c.may_require_multiple_calls, cap
        assert c.related_capabilities, cap
        for related in c.related_capabilities:
            assert related in CAPABILITY_CONTRACTS, f"{cap} -> {related}"


def test_related_capabilities_are_capabilities_not_tools():
    for cap, c in CAPABILITY_CONTRACTS.items():
        for related in c.related_capabilities:
            # Capability identifiers may coincide with tool names for the
            # 1:1 cases (code_search, semantic_search) — that is canonical
            # vocabulary, not a second definition.  The real check: related
            # entries are capability enum members with contracts.
            assert isinstance(related, ToolCapability)
            assert related in CAPABILITY_CONTRACTS, f"{cap.value} -> {related}"


def test_single_step_capabilities_default_to_no_investigation():
    single = (
        ToolCapability.FILE_READ,
        ToolCapability.DIRECTORY_LIST,
        ToolCapability.TERMINAL_EXECUTION,
        ToolCapability.DEFINITION_LOOKUP,
    )
    for cap in single:
        c = contract_for(cap)
        assert c is not None
        assert not c.may_require_investigation
        assert not c.may_require_multiple_calls


def test_mutation_add_tool_propagates_to_contract_discovery(monkeypatch):
    """Adding a tool that serves an existing capability must be visible to the
    contract with zero contract changes (single source of truth)."""
    from ultron.core.tools.definitions import ToolDefinition, ToolDomain, ToolRisk

    synthetic = ToolDefinition(
        name="zzqxtest_lookup",
        func=lambda: "ok",
        capabilities=(ToolCapability.DEFINITION_LOOKUP,),
        read_only=True,
        risk=ToolRisk.LOW,
        domain=ToolDomain.CODE_INTELLIGENCE,
    )
    monkeypatch.setitem(TOOL_DEFINITIONS, "zzqxtest_lookup", synthetic)
    contract = contract_for(ToolCapability.DEFINITION_LOOKUP)
    assert contract is not None
    assert "zzqxtest_lookup" in contract.execution_tools()
    # Insertion order: the canonical table decides the preferred tool.
    assert contract.preferred_tool() == "find_definition"


def test_mutation_remove_tool_propagates_to_contract_discovery(monkeypatch):
    """Removing a tool must immediately be reflected in contract discovery —
    no stale capability->tool mapping can survive."""
    from ultron.core.tools.definitions import (
        ToolDefinition,
        ToolDomain,
        ToolRisk,
    )

    synthetic = ToolDefinition(
        name="zzqx_only_refs",
        func=lambda: "ok",
        capabilities=(ToolCapability.REFERENCE_LOOKUP,),
        read_only=True,
        risk=ToolRisk.LOW,
        domain=ToolDomain.CODE_INTELLIGENCE,
    )
    monkeypatch.setitem(TOOL_DEFINITIONS, "zzqx_only_refs", synthetic)
    contract = contract_for(ToolCapability.REFERENCE_LOOKUP)
    assert contract is not None
    assert "zzqx_only_refs" in contract.execution_tools()
    # Remove it: discovery updates immediately.
    monkeypatch.delitem(TOOL_DEFINITIONS, "zzqx_only_refs")
    assert "zzqx_only_refs" not in contract.execution_tools()


def test_new_capability_needs_no_tool_metadata_changes(monkeypatch):
    """A new capability is added purely at the contract layer; existing tool
    metadata is untouched."""
    from ultron.core.capabilities import contracts as contracts_mod

    # Python enums cannot gain members via subclassing (3.12); build an
    # extended enum functionally that includes a synthetic capability.
    members = {member.name: member.value for member in ToolCapability}
    members["ZZQX_NEW_CAP"] = "zzqx_new_cap"
    Extended = Enum("ToolCapabilityExtended", members)
    synthetic_cap = Extended.ZZQX_NEW_CAP

    new_contract = CapabilityContract(
        capability=synthetic_cap,
        purpose="Test-only capability.",
        user_intent="test",
        required_inputs=("input",),
        required_context=(),
        evidence_required=(EvidenceKind.VERIFIED,),
        success_criteria=("result contains verified evidence",),
        failure_classes=(CapabilityFailure.NO_EVIDENCE,),
        may_require_investigation=False,
        may_require_multiple_calls=False,
        related_capabilities=(),
    )
    monkeypatch.setitem(contracts_mod.CAPABILITY_CONTRACTS, synthetic_cap, new_contract)
    assert contract_for(synthetic_cap) is new_contract
    # No tools serve it yet -> explicit "unavailable" condition, no invention.
    assert new_contract.execution_tools() == []
    monkeypatch.delitem(contracts_mod.CAPABILITY_CONTRACTS, synthetic_cap)


def test_contract_source_has_no_tool_metadata_tables():
    """Static scan: the contracts module must not define any tool-name set or
    tool->capability mapping (no second registry in disguise)."""
    source = Path(
        Path(__file__).resolve().parent.parent / "src/ultron/core/capabilities/contracts.py"
    ).read_text()
    assert "TOOL_DEFINITIONS = {" not in source
    assert "CAPABILITY_TOOLS" not in source
    assert "tool_names =" not in source
    assert not re.search(r'^\s*"[a-z_]+":\s*\[', source, re.MULTILINE), (
        "contracts.py contains a literal tool list"
    )
    # The only imports from definitions are the capability vocabulary + queries.
    assert "from ultron.core.tools.definitions import" in source
