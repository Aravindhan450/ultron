"""Generic behavioral capability contracts + intent->capability selection.

Contracts describe what capabilities *mean* — purpose, inputs, evidence,
success, and failure.  The selector connects the NLP intent system to the
contracts through one path.  Tool metadata remains exclusively in
``ultron.core.tools.definitions.TOOL_DEFINITIONS``; both layers discover
execution mechanisms by querying it.
"""

from ultron.core.capabilities.contracts import (
    CAPABILITY_CONTRACTS,
    CapabilityContract,
    CapabilityFailure,
    EvidenceKind,
    capability_names,
    contract_for,
)
from ultron.core.capabilities.selector import (
    CapabilitySelection,
    SelectionState,
    select_capability,
    select_for_request,
)

__all__ = [
    "CAPABILITY_CONTRACTS",
    "CapabilityContract",
    "CapabilityFailure",
    "CapabilitySelection",
    "EvidenceKind",
    "SelectionState",
    "capability_names",
    "contract_for",
    "select_capability",
    "select_for_request",
]
