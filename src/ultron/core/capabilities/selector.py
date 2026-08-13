"""Generic Intent -> Capability selection (STEP 2C).

Connects the existing NLP intent system (``nlp/intent.py``) to the existing
capability contracts (``capabilities/contracts.py``) through ONE path:

    user request
        -> IntentCategory (nlp/intent.py)          [what the user wants]
        -> ToolCapability (canonical vocabulary)   [what ability satisfies it]
        -> CapabilityContract                      [behavior: inputs/evidence]
        -> TOOL_DEFINITIONS query                  [available tools]
        -> security -> execution

This module defines the mapping *between* the two existing vocabularies — it
creates no new intent vocabulary and no new capability vocabulary, and it owns
no tool metadata.  Tool choice is never hardcoded here: a selection exposes
``execution_tools()`` / ``preferred_tool()`` that query ``TOOL_DEFINITIONS``.

States (Phase 5/12):

    RESOLVED  — one capability clearly satisfies the intent
    AMBIGUOUS — several plausible capabilities; context/clarification needed
    UNKNOWN   — no deterministic intent, or no capability mapping

Unknown selections never pick a tool.  Security remains the final authority
before execution (this layer is selection only).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ultron.core.capabilities.contracts import (
    CapabilityContract,
    contract_for,
)
from ultron.core.nlp.intent import IntentCategory, route_request
from ultron.core.tools.definitions import (
    ToolCapability,
    preferred_tool_for,
    tools_with_capability,
)


class SelectionState(str, Enum):
    """Outcome of selecting a capability for an intent."""

    RESOLVED = "resolved"
    AMBIGUOUS = "ambiguous"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class CapabilitySelection:
    """One intent's capability selection result.

    ``primary`` is the capability that satisfies the intent; ``related`` are
    capabilities that may additionally be needed (from the contract, not a
    second table).  ``ambiguity`` lists candidate capabilities when the intent
    is context-dependent.
    """

    state: SelectionState
    intent: IntentCategory | None
    primary: ToolCapability | None = None
    related: tuple[ToolCapability, ...] = ()
    ambiguity: tuple[ToolCapability, ...] = ()
    reason: str | None = None

    # -- Contract + canonical-registry discovery (never local metadata) ------

    @property
    def contract(self) -> CapabilityContract | None:
        """Behavioral contract for the selected capability, if resolved."""
        if self.primary is None:
            return None
        return contract_for(self.primary)

    @property
    def execution_tools(self) -> tuple[str, ...]:
        """Registered tools that can execute the selected capability."""
        if self.primary is None:
            return ()
        return tuple(tools_with_capability(self.primary))

    @property
    def preferred_tool(self) -> str | None:
        """Preferred registered tool, or None (no capability / no tool)."""
        if self.primary is None:
            return None
        return preferred_tool_for(self.primary)

    def is_resolved(self) -> bool:
        return self.state is SelectionState.RESOLVED and self.primary is not None


# ---------------------------------------------------------------------------
# Intent -> capability mapping.
#
# Keyed by the NLP vocabulary (IntentCategory); values are capabilities from
# the canonical ToolCapability vocabulary.  IntentCategory values deliberately
# match ToolCapability values, but the table is written out explicitly so the
# layer reads as a single authoritative connection and survives vocabulary
# drift.  This is NOT tool selection: tools are discovered from
# TOOL_DEFINITIONS after selection.
# ---------------------------------------------------------------------------

_INTENT_TO_CAPABILITY: dict[IntentCategory, ToolCapability] = {
    # INFORMATION_REQUEST is deliberately absent: it is context-dependent
    # (repository vs external) and declared AMBIGUOUS below — never guessed.
    IntentCategory.REPOSITORY_INSPECTION: ToolCapability.REPOSITORY_INSPECTION,
    IntentCategory.FILE_READ: ToolCapability.FILE_READ,
    IntentCategory.FILE_WRITE: ToolCapability.FILE_WRITE,
    IntentCategory.FILE_CREATE: ToolCapability.FILE_CREATE,
    IntentCategory.FILE_DELETE: ToolCapability.FILE_DELETE,
    IntentCategory.FILE_RENAME: ToolCapability.FILE_RENAME,
    IntentCategory.DIRECTORY_LIST: ToolCapability.DIRECTORY_LIST,
    IntentCategory.DIRECTORY_CREATE: ToolCapability.DIRECTORY_CREATE,
    IntentCategory.FILE_SEARCH: ToolCapability.FILE_SEARCH,
    IntentCategory.CODE_SEARCH: ToolCapability.CODE_SEARCH,
    IntentCategory.SYMBOL_SEARCH: ToolCapability.SYMBOL_SEARCH,
    IntentCategory.DEFINITION_LOOKUP: ToolCapability.DEFINITION_LOOKUP,
    IntentCategory.REFERENCE_LOOKUP: ToolCapability.REFERENCE_LOOKUP,
    IntentCategory.SYMBOL_INSPECTION: ToolCapability.SYMBOL_INSPECTION,
    IntentCategory.SEMANTIC_SEARCH: ToolCapability.SEMANTIC_SEARCH,
    IntentCategory.REPOSITORY_INVESTIGATION: ToolCapability.REPOSITORY_INVESTIGATION,
    IntentCategory.TERMINAL_EXECUTION: ToolCapability.TERMINAL_EXECUTION,
    IntentCategory.TEST_EXECUTION: ToolCapability.TEST_EXECUTION,
    IntentCategory.APPLICATION_START: ToolCapability.APPLICATION_START,
    IntentCategory.APPLICATION_STOP: ToolCapability.APPLICATION_STOP,
    IntentCategory.BUILD: ToolCapability.BUILD,
    IntentCategory.INSTALL: ToolCapability.INSTALL,
    IntentCategory.GIT_OPERATION: ToolCapability.GIT_OPERATION,
    IntentCategory.LINT: ToolCapability.LINT,
    IntentCategory.TYPECHECK: ToolCapability.TYPECHECK,
    IntentCategory.FORMAT: ToolCapability.FORMAT,
    IntentCategory.CODING_REQUEST: ToolCapability.CODING_REQUEST,
    IntentCategory.MEMORY_QUERY: ToolCapability.MEMORY_QUERY,
    IntentCategory.MEMORY_UPDATE: ToolCapability.MEMORY_UPDATE,
}

# Intents whose satisfaction genuinely depends on context the NLP layer alone
# cannot decide.  Selection reports AMBIGUOUS with these candidates; later
# routing may inspect context or ask the user.  (The current detectors
# disambiguate most requests before reaching this layer; this mechanism keeps
# the remaining context-dependent cases explicit instead of guessing.)
_AMBIGUOUS_INTENTS: dict[IntentCategory, tuple[ToolCapability, ...]] = {
    # Repository-vs-external depends on the topic/context, not phrasing alone.
    IntentCategory.INFORMATION_REQUEST: (
        ToolCapability.INFORMATION_REQUEST,
        ToolCapability.WEB_SEARCH,
    ),
}


def select_capability(intent: IntentCategory | None) -> CapabilitySelection:
    """Maps one intent category to a capability selection (the ONE path)."""
    if intent is None or intent is IntentCategory.UNKNOWN:
        return CapabilitySelection(
            SelectionState.UNKNOWN,
            intent,
            reason="unknown intent: no capability selected, no tool chosen",
        )
    if intent in _AMBIGUOUS_INTENTS:
        return CapabilitySelection(
            SelectionState.AMBIGUOUS,
            intent,
            ambiguity=_AMBIGUOUS_INTENTS[intent],
            reason="intent is context-dependent; multiple plausible capabilities",
        )
    primary = _INTENT_TO_CAPABILITY.get(intent)
    if primary is None:
        return CapabilitySelection(
            SelectionState.UNKNOWN,
            intent,
            reason=f"no capability mapping for intent {intent.value}",
        )
    contract = contract_for(primary)
    related: tuple[ToolCapability, ...] = (
        contract.related_capabilities if contract is not None else ()
    )
    return CapabilitySelection(
        SelectionState.RESOLVED,
        intent,
        primary=primary,
        related=related,
        reason="resolved to canonical capability",
    )


def select_for_request(text: str | None) -> CapabilitySelection:
    """NLP entry point: route a natural-language request to a capability.

    Uses the existing deterministic intent layer; never an LLM classifier.
    """
    if not text or not text.strip():
        return CapabilitySelection(
            SelectionState.UNKNOWN,
            None,
            reason="malformed/empty request: no intent, no capability",
        )
    intent = route_request(text.strip())
    if intent is None:
        return CapabilitySelection(
            SelectionState.UNKNOWN,
            None,
            reason="no deterministic intent detected",
        )
    return select_capability(intent.intent_type)
