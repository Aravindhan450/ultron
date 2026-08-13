"""Generic behavioral capability contracts (STEP 2B).

Contracts describe what capabilities *mean* — purpose, inputs, evidence,
success, and failure.  Tool metadata remains exclusively in
``ultron.core.tools.definitions.TOOL_DEFINITIONS``; contracts discover their
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

__all__ = [
    "CAPABILITY_CONTRACTS",
    "CapabilityContract",
    "CapabilityFailure",
    "EvidenceKind",
    "capability_names",
    "contract_for",
]
