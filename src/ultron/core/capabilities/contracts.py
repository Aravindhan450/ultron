"""Generic behavioral capability contracts.

A *capability contract* describes what a capability means: the goal it
accomplishes, the user intent it serves, the inputs and context it needs,
the evidence that establishes success, and the failure conditions the agent
must be able to interpret.  Contracts are behavior, not metadata.

Architecture (STEP 2B):

    TOOL_DEFINITIONS (canonical tool metadata)
            |
            v
    CAPABILITY_CONTRACTS (behavioral contracts, keyed by ToolCapability)
            |
            v
    routing / validation / execution

Contracts deliberately contain **no** tool metadata: no tool names, aliases,
risk, read-only status, confirmation requirements, schemas, descriptions, or
domains.  A contract discovers its execution mechanisms live by querying the
canonical registry (``tools_with_capability`` / ``preferred_tool_for``), so a
tool can be added, re-tagged, or removed without touching any contract.

``coding_request`` is the one capability with no direct tool: it is a
multi-step ability (investigate -> modify -> test) expressed through
``related_capabilities`` and the multi-step flags.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ultron.core.tools.definitions import (
    ToolCapability,
    preferred_tool_for,
    tools_with_capability,
)


class EvidenceKind(str, Enum):
    """How strongly a result is grounded in the repository/external source.

    Preserves the existing evidence semantics used by the code-intelligence
    resolver; contracts only *require* specific kinds, they never weaken them.
    """

    VERIFIED = "verified"  # directly supported by source/index evidence
    INFERRED = "inferred"  # reasonable conclusion from multiple findings
    SOURCE_MATCH = "source_match"  # lexical/source occurrence, not indexed
    SEMANTIC_MATCH = "semantic_match"  # conceptual match, not exact symbol
    UNKNOWN = "unknown"  # cannot be established


class CapabilityFailure(str, Enum):
    """Meaningful failure classes a capability can report.

    Never collapse these into a generic "operation failed": the agent must be
    able to distinguish and respond to each.
    """

    INVALID_INPUT = "invalid_input"
    UNRESOLVED_ENTITY = "unresolved_entity"
    NO_EVIDENCE = "no_evidence"
    TOOL_UNAVAILABLE = "tool_unavailable"
    PERMISSION_DENIED = "permission_denied"
    TIMEOUT = "timeout"
    EXECUTION_FAILURE = "execution_failure"
    INSUFFICIENT_CONTEXT = "insufficient_context"


@dataclass(frozen=True)
class CapabilityContract:
    """Behavioral contract for one capability in the canonical vocabulary."""

    capability: ToolCapability
    # Human-readable: what the agent is trying to accomplish.
    purpose: str
    # Semantic intent this capability serves (descriptive, never detection).
    user_intent: str
    # Information the capability needs to run.
    required_inputs: tuple[str, ...]
    # Context the capability needs beyond its explicit inputs.
    required_context: tuple[str, ...]
    # Evidence kinds that establish success for this capability.
    evidence_required: tuple[EvidenceKind, ...]
    # Evidence-based success criteria (no exact-answer strings).
    success_criteria: tuple[str, ...]
    # Failure classes the agent must be able to interpret.
    failure_classes: tuple[CapabilityFailure, ...]
    # Additional investigation may be required (multi-step, ordered).
    may_require_investigation: bool
    # The capability may require more than one tool call.
    may_require_multiple_calls: bool
    # Sub-capabilities that may be needed to fulfil this one (capabilities,
    # NOT tools — execution mechanisms come from the canonical registry).
    related_capabilities: tuple[ToolCapability, ...]

    # -- Canonical-registry discovery (never a local metadata table) --------

    def execution_tools(self) -> list[str]:
        """Registered tools that can execute this capability (live query)."""
        return tools_with_capability(self.capability)

    def preferred_tool(self) -> str | None:
        """Preferred registered tool for this capability, or None."""
        return preferred_tool_for(self.capability)

    def has_execution_tools(self) -> bool:
        """True when at least one registered tool serves this capability."""
        return bool(self.execution_tools())


CAPABILITY_CONTRACTS: dict[ToolCapability, CapabilityContract] = {}


def _contract(
    capability: ToolCapability,
    purpose: str,
    *,
    user_intent: str = "",
    required_inputs: tuple[str, ...] = (),
    required_context: tuple[str, ...] = (),
    evidence: tuple[EvidenceKind, ...] = (EvidenceKind.VERIFIED,),
    success: tuple[str, ...] = (),
    failures: tuple[CapabilityFailure, ...] = (),
    may_investigate: bool = False,
    may_multi: bool = False,
    related: tuple[ToolCapability, ...] = (),
) -> None:
    """Registers one contract.  Pure behavior — no tool metadata allowed."""
    if capability in CAPABILITY_CONTRACTS:
        raise ValueError(f"duplicate capability contract: {capability.value}")
    CAPABILITY_CONTRACTS[capability] = CapabilityContract(
        capability=capability,
        purpose=purpose,
        user_intent=user_intent or purpose,
        required_inputs=required_inputs,
        required_context=required_context,
        evidence_required=evidence,
        success_criteria=success,
        failure_classes=failures,
        may_require_investigation=may_investigate,
        may_require_multiple_calls=may_multi,
        related_capabilities=related,
    )


def contract_for(capability: ToolCapability) -> CapabilityContract | None:
    """Returns the behavioral contract for a capability, or None."""
    return CAPABILITY_CONTRACTS.get(capability)


# Common failure bundles -----------------------------------------------------

_INVALID = (CapabilityFailure.INVALID_INPUT,)
_NO_EVIDENCE = (CapabilityFailure.UNRESOLVED_ENTITY, CapabilityFailure.NO_EVIDENCE)
_MUTATION = (
    CapabilityFailure.INVALID_INPUT,
    CapabilityFailure.PERMISSION_DENIED,
    CapabilityFailure.EXECUTION_FAILURE,
    CapabilityFailure.TIMEOUT,
)
_EXECUTION = (
    CapabilityFailure.INVALID_INPUT,
    CapabilityFailure.PERMISSION_DENIED,
    CapabilityFailure.EXECUTION_FAILURE,
    CapabilityFailure.TIMEOUT,
)
_READ_FAIL = _INVALID + (CapabilityFailure.PERMISSION_DENIED, CapabilityFailure.TIMEOUT)

# ---------------------------------------------------------------------------
# Repository intelligence
# ---------------------------------------------------------------------------

_contract(
    ToolCapability.DEFINITION_LOOKUP,
    "Locate a verified definition of an arbitrary repository entity.",
    user_intent="The user wants to know where something is defined.",
    required_inputs=("repository entity",),
    required_context=("project root",),
    evidence=(EvidenceKind.VERIFIED,),
    success=("result contains a verified definition with file location",),
    failures=_NO_EVIDENCE + _READ_FAIL + (CapabilityFailure.TOOL_UNAVAILABLE,),
    related=(ToolCapability.SYMBOL_SEARCH, ToolCapability.CODE_SEARCH),
)

_contract(
    ToolCapability.REFERENCE_LOOKUP,
    "Find where an arbitrary repository entity is referenced or used.",
    user_intent="The user wants to know where something is used or referenced.",
    required_inputs=("repository entity",),
    required_context=("project root",),
    evidence=(EvidenceKind.VERIFIED, EvidenceKind.SOURCE_MATCH),
    success=(
        "result contains verified references, or an explicit evidence-grounded no-references answer",
    ),
    failures=_NO_EVIDENCE + _READ_FAIL + (CapabilityFailure.TOOL_UNAVAILABLE,),
    related=(
        ToolCapability.SYMBOL_SEARCH,
        ToolCapability.CODE_SEARCH,
        ToolCapability.DEPENDENCY_ANALYSIS,
    ),
)

_contract(
    ToolCapability.SYMBOL_SEARCH,
    "Find a named symbol anywhere in the repository.",
    user_intent="The user wants to locate a symbol by name.",
    required_inputs=("symbol",),
    required_context=("project root",),
    evidence=(EvidenceKind.VERIFIED,),
    success=("result contains located symbols with file locations",),
    failures=_NO_EVIDENCE + _READ_FAIL,
)

_contract(
    ToolCapability.SYMBOL_INSPECTION,
    "Report the structure of a repository symbol (type, signature, members).",
    user_intent="The user wants to understand what a symbol is and contains.",
    required_inputs=("symbol",),
    required_context=("project root",),
    evidence=(EvidenceKind.VERIFIED,),
    success=("result describes the symbol from indexed source",),
    failures=_NO_EVIDENCE + _READ_FAIL,
)

_contract(
    ToolCapability.CODE_SEARCH,
    "Search repository source for lexical occurrences of text.",
    user_intent="The user wants every textual occurrence of a phrase.",
    required_inputs=("query",),
    required_context=("project root",),
    evidence=(EvidenceKind.SOURCE_MATCH,),
    success=("result lists source occurrences with file locations",),
    failures=_NO_EVIDENCE + _READ_FAIL,
)

_contract(
    ToolCapability.SEMANTIC_SEARCH,
    "Find repository code conceptually related to a description.",
    user_intent="The user wants to find where something is implemented/handled.",
    required_inputs=("query",),
    required_context=("project root",),
    evidence=(EvidenceKind.SEMANTIC_MATCH, EvidenceKind.SOURCE_MATCH),
    success=("result identifies relevant files/symbols with evidence",),
    failures=_NO_EVIDENCE + _READ_FAIL,
)

_contract(
    ToolCapability.REPOSITORY_INVESTIGATION,
    "Explain how something works in this repository using source evidence.",
    user_intent="The user asks how a project component works or fits together.",
    required_inputs=("entity or topic",),
    required_context=("project root", "related source"),
    evidence=(
        EvidenceKind.VERIFIED,
        EvidenceKind.INFERRED,
        EvidenceKind.SOURCE_MATCH,
        EvidenceKind.SEMANTIC_MATCH,
    ),
    success=(
        "result explains the requested behavior using relevant repository evidence, distinguishing verified facts from inferences",
    ),
    failures=(
        CapabilityFailure.UNRESOLVED_ENTITY,
        CapabilityFailure.NO_EVIDENCE,
        CapabilityFailure.INSUFFICIENT_CONTEXT,
        CapabilityFailure.TOOL_UNAVAILABLE,
    ),
    may_investigate=True,
    may_multi=True,
    related=(
        ToolCapability.DEFINITION_LOOKUP,
        ToolCapability.REFERENCE_LOOKUP,
        ToolCapability.SYMBOL_SEARCH,
        ToolCapability.SYMBOL_INSPECTION,
        ToolCapability.CODE_SEARCH,
        ToolCapability.SEMANTIC_SEARCH,
        ToolCapability.DEPENDENCY_ANALYSIS,
        ToolCapability.FILE_READ,
        ToolCapability.FILE_SEARCH,
    ),
)

_contract(
    ToolCapability.DEPENDENCY_ANALYSIS,
    "Report a symbol's imports and dependents (dependency relationships).",
    user_intent="The user wants to know what depends on something.",
    required_inputs=("symbol or file",),
    required_context=("project root",),
    evidence=(EvidenceKind.VERIFIED,),
    success=("result lists import/dependency relationships with locations",),
    failures=_NO_EVIDENCE + _READ_FAIL,
)

_contract(
    ToolCapability.REPOSITORY_INSPECTION,
    "Provide a high-level overview of the repository (summary, index state).",
    user_intent="The user wants to understand the project at a glance.",
    required_inputs=(),
    required_context=("project root",),
    evidence=(EvidenceKind.VERIFIED,),
    success=("result describes the workspace and code-index state",),
    failures=(CapabilityFailure.NO_EVIDENCE, CapabilityFailure.TOOL_UNAVAILABLE),
)

_contract(
    ToolCapability.FILE_INSPECTION,
    "Report the structure/contents of a specific file in the repository.",
    user_intent="The user wants a structural overview of a file.",
    required_inputs=("file path",),
    required_context=("project root",),
    evidence=(EvidenceKind.VERIFIED,),
    success=("result describes the file from actual source",),
    failures=_READ_FAIL + _NO_EVIDENCE,
)

# ---------------------------------------------------------------------------
# Filesystem
# ---------------------------------------------------------------------------

_contract(
    ToolCapability.FILE_READ,
    "Read the contents of a file.",
    user_intent="The user wants to see what is in a file.",
    required_inputs=("file path",),
    required_context=("workspace root",),
    evidence=(EvidenceKind.VERIFIED,),
    success=("result contains the file content for the resolved path",),
    failures=_READ_FAIL + (CapabilityFailure.UNRESOLVED_ENTITY,),
)

_contract(
    ToolCapability.FILE_SEARCH,
    "Find files whose content matches a query.",
    user_intent="The user wants to find files containing something.",
    required_inputs=("query",),
    required_context=("workspace root",),
    evidence=(EvidenceKind.SOURCE_MATCH,),
    success=("result lists matching files with locations",),
    failures=_NO_EVIDENCE + _READ_FAIL,
)

_contract(
    ToolCapability.FILE_WRITE,
    "Create or modify file content.",
    user_intent="The user wants to change or add content to a file.",
    required_inputs=("file path", "content"),
    required_context=("workspace root",),
    evidence=(EvidenceKind.VERIFIED,),
    success=("result confirms the file was written as requested",),
    failures=_MUTATION,
)

_contract(
    ToolCapability.FILE_CREATE,
    "Create a new file.",
    user_intent="The user wants a new file to exist.",
    required_inputs=("file path", "content"),
    required_context=("workspace root",),
    evidence=(EvidenceKind.VERIFIED,),
    success=("result confirms the new file exists",),
    failures=_MUTATION,
)

_contract(
    ToolCapability.FILE_DELETE,
    "Delete a file.",
    user_intent="The user wants a file removed.",
    required_inputs=("file path",),
    required_context=("workspace root",),
    evidence=(EvidenceKind.VERIFIED,),
    success=("result confirms the file was deleted",),
    failures=_MUTATION,
)

_contract(
    ToolCapability.FILE_RENAME,
    "Rename or move a file.",
    user_intent="The user wants a file renamed or moved.",
    required_inputs=("source path", "destination path"),
    required_context=("workspace root",),
    evidence=(EvidenceKind.VERIFIED,),
    success=("result confirms the file now exists at the destination",),
    failures=_MUTATION,
)

_contract(
    ToolCapability.DIRECTORY_LIST,
    "List the entries of a directory.",
    user_intent="The user wants to see what is in a directory.",
    required_inputs=("directory path",),
    required_context=("workspace root",),
    evidence=(EvidenceKind.VERIFIED,),
    success=("result lists the directory entries for the resolved path",),
    failures=_READ_FAIL + (CapabilityFailure.UNRESOLVED_ENTITY,),
)

_contract(
    ToolCapability.DIRECTORY_CREATE,
    "Create a directory.",
    user_intent="The user wants a new directory to exist.",
    required_inputs=("directory path",),
    required_context=("workspace root",),
    evidence=(EvidenceKind.VERIFIED,),
    success=("result confirms the directory exists",),
    failures=_MUTATION,
)

# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

_contract(
    ToolCapability.TERMINAL_EXECUTION,
    "Execute a command and capture its outcome.",
    user_intent="The user explicitly wants a command run.",
    required_inputs=("command",),
    required_context=("working directory",),
    evidence=(EvidenceKind.VERIFIED,),
    success=("command outcome captured with exit status and output",),
    failures=_EXECUTION,
)

_contract(
    ToolCapability.TEST_EXECUTION,
    "Discover and run tests, capturing results.",
    user_intent="The user wants tests run or test failures understood.",
    required_inputs=("test scope or target",),
    required_context=("project root", "test environment"),
    evidence=(EvidenceKind.VERIFIED,),
    success=("tests actually executed and results (pass/fail) captured",),
    failures=_EXECUTION + (CapabilityFailure.INSUFFICIENT_CONTEXT,),
    may_investigate=True,
    may_multi=True,
    related=(
        ToolCapability.REPOSITORY_INVESTIGATION,
        ToolCapability.FILE_SEARCH,
        ToolCapability.TERMINAL_EXECUTION,
    ),
)

_contract(
    ToolCapability.BUILD,
    "Build the project using its configured build command.",
    user_intent="The user wants the project built.",
    required_inputs=("build target",),
    required_context=("project root", "build configuration"),
    evidence=(EvidenceKind.VERIFIED,),
    success=("build ran and its outcome was captured",),
    failures=_EXECUTION + (CapabilityFailure.INSUFFICIENT_CONTEXT,),
)

_contract(
    ToolCapability.INSTALL,
    "Install project dependencies.",
    user_intent="The user wants dependencies installed.",
    required_inputs=("package or target",),
    required_context=("project root", "environment configuration"),
    evidence=(EvidenceKind.VERIFIED,),
    success=("installation ran and its outcome was captured",),
    failures=_EXECUTION,
)

_contract(
    ToolCapability.LINT,
    "Run the project linter.",
    user_intent="The user wants the code linted.",
    required_inputs=("lint scope",),
    required_context=("project root", "lint configuration"),
    evidence=(EvidenceKind.VERIFIED,),
    success=("lint ran and its findings were captured",),
    failures=_EXECUTION + (CapabilityFailure.INSUFFICIENT_CONTEXT,),
)

_contract(
    ToolCapability.TYPECHECK,
    "Run the project type checker.",
    user_intent="The user wants the code type-checked.",
    required_inputs=("check scope",),
    required_context=("project root", "typecheck configuration"),
    evidence=(EvidenceKind.VERIFIED,),
    success=("typecheck ran and its findings were captured",),
    failures=_EXECUTION + (CapabilityFailure.INSUFFICIENT_CONTEXT,),
)

_contract(
    ToolCapability.FORMAT,
    "Format the project code.",
    user_intent="The user wants the code formatted.",
    required_inputs=("format scope",),
    required_context=("project root", "format configuration"),
    evidence=(EvidenceKind.VERIFIED,),
    success=("formatting ran and its outcome was captured",),
    failures=_EXECUTION + (CapabilityFailure.INSUFFICIENT_CONTEXT,),
)

_contract(
    ToolCapability.GIT_OPERATION,
    "Inspect or mutate repository git state.",
    user_intent="The user asks about or changes git state.",
    required_inputs=("git operation", "target"),
    required_context=("git root",),
    evidence=(EvidenceKind.VERIFIED,),
    success=("git command outcome captured (status/diff/commit result)",),
    failures=_EXECUTION,
)

_contract(
    ToolCapability.APPLICATION_START,
    "Start the application or a development server.",
    user_intent="The user wants the app started.",
    required_inputs=("app or service",),
    required_context=("project root", "run configuration"),
    evidence=(EvidenceKind.VERIFIED,),
    success=("start command ran and its outcome was captured",),
    failures=_EXECUTION + (CapabilityFailure.INSUFFICIENT_CONTEXT,),
)

_contract(
    ToolCapability.APPLICATION_STOP,
    "Stop a running application or server.",
    user_intent="The user wants the app stopped.",
    required_inputs=("app or service",),
    required_context=("project root", "run configuration"),
    evidence=(EvidenceKind.VERIFIED,),
    success=("stop command ran and its outcome was captured",),
    failures=_EXECUTION + (CapabilityFailure.INSUFFICIENT_CONTEXT,),
)

# ---------------------------------------------------------------------------
# External research / network
# ---------------------------------------------------------------------------

_contract(
    ToolCapability.WEB_SEARCH,
    "Search external sources for information not in the repository.",
    user_intent="The user wants external or current information.",
    required_inputs=("query",),
    required_context=(),
    evidence=(EvidenceKind.SOURCE_MATCH, EvidenceKind.SEMANTIC_MATCH),
    success=(
        "external evidence is returned and clearly distinguished from repository evidence",
    ),
    failures=_EXECUTION + _NO_EVIDENCE,
    may_multi=True,
    related=(ToolCapability.PAGE_FETCH,),
)

_contract(
    ToolCapability.PAGE_FETCH,
    "Fetch and extract the readable content of a web page.",
    user_intent="The user wants a page's content read.",
    required_inputs=("url",),
    required_context=(),
    evidence=(EvidenceKind.VERIFIED,),
    success=("page content was fetched and extracted",),
    failures=_EXECUTION + (CapabilityFailure.UNRESOLVED_ENTITY,),
)

_contract(
    ToolCapability.HTTP_REQUEST,
    "Make an HTTP request and capture the response.",
    user_intent="The user wants an API/endpoint contacted.",
    required_inputs=("url", "method"),
    required_context=(),
    evidence=(EvidenceKind.VERIFIED,),
    success=("response captured with status and body",),
    failures=_EXECUTION,
)

_contract(
    ToolCapability.INFORMATION_REQUEST,
    "Answer a question from repository, memory, or external evidence.",
    user_intent="The user asks a general question answerable by the system.",
    required_inputs=("question",),
    required_context=("project root",),
    evidence=(
        EvidenceKind.VERIFIED,
        EvidenceKind.SOURCE_MATCH,
        EvidenceKind.SEMANTIC_MATCH,
    ),
    success=("answer grounded in evidence with its source indicated",),
    failures=(CapabilityFailure.NO_EVIDENCE, CapabilityFailure.INSUFFICIENT_CONTEXT),
    may_investigate=True,
)

# ---------------------------------------------------------------------------
# Memory / knowledge
# ---------------------------------------------------------------------------

_contract(
    ToolCapability.MEMORY_QUERY,
    "Retrieve stored facts or memories matching criteria.",
    user_intent="The user wants remembered information recalled.",
    required_inputs=("query",),
    required_context=(),
    evidence=(EvidenceKind.VERIFIED,),
    success=("result contains matching stored facts or an explicit empty result",),
    failures=(CapabilityFailure.NO_EVIDENCE, CapabilityFailure.INVALID_INPUT),
)

_contract(
    ToolCapability.MEMORY_UPDATE,
    "Store a new fact or relationship in memory.",
    user_intent="The user wants something remembered.",
    required_inputs=("fact or relationship",),
    required_context=(),
    evidence=(EvidenceKind.VERIFIED,),
    success=("result confirms the fact was stored",),
    failures=_MUTATION,
)

_contract(
    ToolCapability.GRAPH_REASONING,
    "Query the knowledge-graph triples for relationships.",
    user_intent="The user wants knowledge-graph facts or relationships.",
    required_inputs=("query", "subject/object"),
    required_context=(),
    evidence=(EvidenceKind.VERIFIED,),
    success=("result contains matching graph triples",),
    failures=(CapabilityFailure.NO_EVIDENCE, CapabilityFailure.INVALID_INPUT),
)

_contract(
    ToolCapability.MEMORY_ASSOCIATION,
    "Discover or explain connections between stored facts.",
    user_intent="The user wants to understand how remembered things relate.",
    required_inputs=("topic or entities",),
    required_context=(),
    evidence=(EvidenceKind.VERIFIED, EvidenceKind.INFERRED),
    success=("result identifies associations with evidence",),
    failures=(CapabilityFailure.NO_EVIDENCE, CapabilityFailure.INSUFFICIENT_CONTEXT),
    may_investigate=True,
)

# ---------------------------------------------------------------------------
# Other system abilities
# ---------------------------------------------------------------------------

_contract(
    ToolCapability.DATABASE_QUERY,
    "Execute a database query and capture results.",
    user_intent="The user wants data queried from a database.",
    required_inputs=("sql",),
    required_context=("database connection",),
    evidence=(EvidenceKind.VERIFIED,),
    success=("query outcome captured (rows or explicit error)",),
    failures=_EXECUTION,
)

_contract(
    ToolCapability.API_SCHEMA_LEARNING,
    "Learn or query the schema of an external API.",
    user_intent="The user wants to understand or record an API's schema.",
    required_inputs=("api or endpoint",),
    required_context=(),
    evidence=(EvidenceKind.VERIFIED,),
    success=("schema learned/queried and evidence returned",),
    failures=_EXECUTION + _NO_EVIDENCE,
)

_contract(
    ToolCapability.RESOURCE_MONITORING,
    "Report system resource usage or forecasts.",
    user_intent="The user wants to know resource state.",
    required_inputs=("resource type",),
    required_context=(),
    evidence=(EvidenceKind.VERIFIED,),
    success=("resource measurements returned",),
    failures=(CapabilityFailure.NO_EVIDENCE, CapabilityFailure.EXECUTION_FAILURE),
)

_contract(
    ToolCapability.STRUCTURED_OUTPUT,
    "Validate or enforce a structured output against a schema.",
    user_intent="The user wants structured/validated output.",
    required_inputs=("data", "schema"),
    required_context=(),
    evidence=(EvidenceKind.VERIFIED,),
    success=("validation outcome returned with reasons",),
    failures=(CapabilityFailure.INVALID_INPUT, CapabilityFailure.EXECUTION_FAILURE),
)

_contract(
    ToolCapability.PLAN_MANAGEMENT,
    "Create, inspect, or preflight a plan.",
    user_intent="The user wants planning or plan inspection.",
    required_inputs=("plan or step",),
    required_context=("task state",),
    evidence=(EvidenceKind.VERIFIED,),
    success=("plan validated/inspected and outcome returned",),
    failures=(CapabilityFailure.INVALID_INPUT, CapabilityFailure.EXECUTION_FAILURE),
)

_contract(
    ToolCapability.DEBUG_ENVIRONMENT,
    "Inspect the runtime environment to diagnose failures.",
    user_intent="The user wants environment state for debugging.",
    required_inputs=("diagnostic scope",),
    required_context=("workspace",),
    evidence=(EvidenceKind.VERIFIED,),
    success=("diagnostic information returned",),
    failures=(CapabilityFailure.NO_EVIDENCE, CapabilityFailure.EXECUTION_FAILURE),
)

_contract(
    ToolCapability.PARALLEL_BATCH,
    "Run several independent tool calls concurrently.",
    user_intent="The user wants multiple operations run together.",
    required_inputs=("batch of calls",),
    required_context=("workspace",),
    evidence=(EvidenceKind.VERIFIED,),
    success=("each call's outcome captured with isolation of failures",),
    failures=_EXECUTION,
    may_multi=True,
)

_contract(
    ToolCapability.CODING_REQUEST,
    "Implement or modify code end-to-end (investigate, edit, verify).",
    user_intent="The user wants code changed or implemented.",
    required_inputs=("task description",),
    required_context=("project root", "task state", "workspace"),
    evidence=(
        EvidenceKind.VERIFIED,
        EvidenceKind.SOURCE_MATCH,
        EvidenceKind.SEMANTIC_MATCH,
    ),
    success=(
        "change made against repository evidence and verification ran (tests/typecheck where configured)",
    ),
    failures=(
        CapabilityFailure.INVALID_INPUT,
        CapabilityFailure.NO_EVIDENCE,
        CapabilityFailure.EXECUTION_FAILURE,
        CapabilityFailure.INSUFFICIENT_CONTEXT,
    ),
    may_investigate=True,
    may_multi=True,
    related=(
        ToolCapability.REPOSITORY_INVESTIGATION,
        ToolCapability.FILE_READ,
        ToolCapability.FILE_SEARCH,
        ToolCapability.FILE_WRITE,
        ToolCapability.FILE_CREATE,
        ToolCapability.TEST_EXECUTION,
        ToolCapability.TERMINAL_EXECUTION,
    ),
)


def capability_names() -> list[str]:
    """Sorted capability identifiers with a contract."""
    return sorted(c.value for c in CAPABILITY_CONTRACTS)
