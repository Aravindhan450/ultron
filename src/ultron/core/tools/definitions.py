"""ultron.core.tools.definitions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

THE ONE AUTHORITATIVE SOURCE OF TRUTH for tool/capability metadata.

Every registered tool is defined exactly once here as a
:class:`ToolDefinition`: its canonical name, the executable function it
binds to, its capability roles, read-only status, declared risk, aliases,
domain, and the argument roles the security layer scans (target/content).

Consumers derive their views from this module — they never maintain
independent tool-name tables:

- ``registry.TOOLS`` and ``get_tools_schema()`` derive identity + schemas.
- ``nlp`` routing, ``agents/react.route_llm_tool_call`` redirect sets,
  ``agents/simple`` target/content extraction and capability→tool mapping.
- ``security.boundary`` risk tiers for registered tools.
- ``permissions.classifier`` confirmation labels + action aliases.
- ``intelligence.parallel_tools`` action-name canonicalization.
- ``orchestration.permissions`` read/write/network/shell classification.

Security policy (HTTP method / SQL verb / shell metacharacter refinement,
system-path escalation, guardrails) stays in the security layer — the
declared risk here is the *baseline*; guardrails always override it.

CAPABILITY vs TOOL (STEP 2A spec): a *capability* is what the agent wants
to accomplish; a *tool* is the executable mechanism. The mapping is
many-to-many: one tool may serve several capabilities (``run_command``
serves TERMINAL_EXECUTION, TEST_EXECUTION, BUILD, …) and one capability may
be served by several tools (TERMINAL_EXECUTION by run_command/run_parallel).
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from enum import Enum
from typing import Any

from pydantic import BaseModel

# --- The executable functions (bound once, here, to the canonical names) ---
from ultron.core.coding.edits import (
    append_to_file,
    create_file,
    delete_file,
    rename_file,
    replace_file,
    replace_in_file,
)
from ultron.core.coding.intelligence.tools import (
    code_index_status,
    code_investigation,
    code_search,
    find_definition,
    find_references,
    find_symbol,
    get_dependents,
    get_imports,
    report_file,
    report_symbol,
    semantic_search,
)
from ultron.core.coding.workspace import (
    discover_workspace_summary,
    list_directory,
    search_files,
)
from ultron.core.intelligence.debug_context import (
    check_dependency,
    diagnose_failure,
    get_debug_context,
)
from ultron.core.intelligence.parallel_tools import (
    run_tool_batch,
    synthesize_analysis,
)
from ultron.core.intelligence.planning import (
    analyze_dependencies,
    list_plan_actions,
    preflight_plan_tool,
)
from ultron.core.intelligence.structured_output import (
    enforce_schema,
    list_schemas,
    schema_validate,
)
from ultron.core.learning.api_schema import (
    api_usage_hint,
    forget_api,
    get_api_knowledge,
    learn_api_schema,
)
from ultron.core.learning.associations import (
    discover_connections,
    explain_relation,
    memory_connections,
    related_facts,
)
from ultron.core.tools.builtin.command_runner import run_command, run_parallel
from ultron.core.tools.builtin.database import run_query
from ultron.core.tools.builtin.file_reader import read_file
from ultron.core.tools.builtin.file_writer import write_file
from ultron.core.tools.builtin.http_client import make_http_request
from ultron.core.tools.builtin.retrieval import check_connectivity, retrieve
from ultron.core.tools.builtin.web_search import fetch_page_text, search_web
from ultron.core.tools.memory.graph import (
    add_triple,
    get_all_triples,
    query_triples,
    search_triples,
    store_memory_text,
)
from ultron.core.tools.memory.sqlite import (
    get_all_memories,
    search_memories,
)
from ultron.core.tools.resource_monitor import check_resources, resource_forecast


class ToolCapability(str, Enum):
    """The capability vocabulary (what an agent wants to accomplish).

    Value strings deliberately match the NL intent categories they serve
    (``definition_lookup``, ``reference_lookup``, …) so routing code can
    map a capability to its tool without a second vocabulary.  The NL
    intent layer is a *consumer* of this vocabulary, never an owner.
    """

    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    FILE_CREATE = "file_create"
    FILE_DELETE = "file_delete"
    FILE_RENAME = "file_rename"
    DIRECTORY_LIST = "directory_list"
    DIRECTORY_CREATE = "directory_create"
    FILE_SEARCH = "file_search"
    REPOSITORY_INSPECTION = "repository_inspection"
    CODE_SEARCH = "code_search"
    SYMBOL_SEARCH = "symbol_search"
    DEFINITION_LOOKUP = "definition_lookup"
    REFERENCE_LOOKUP = "reference_lookup"
    SYMBOL_INSPECTION = "symbol_inspection"
    FILE_INSPECTION = "file_inspection"
    SEMANTIC_SEARCH = "semantic_search"
    REPOSITORY_INVESTIGATION = "repository_investigation"
    DEPENDENCY_ANALYSIS = "dependency_analysis"
    TERMINAL_EXECUTION = "terminal_execution"
    TEST_EXECUTION = "test_execution"
    BUILD = "build"
    INSTALL = "install"
    GIT_OPERATION = "git_operation"
    LINT = "lint"
    TYPECHECK = "typecheck"
    FORMAT = "format"
    APPLICATION_START = "application_start"
    APPLICATION_STOP = "application_stop"
    HTTP_REQUEST = "http_request"
    WEB_SEARCH = "web_search"
    PAGE_FETCH = "page_fetch"
    INFORMATION_REQUEST = "information_request"
    DATABASE_QUERY = "database_query"
    MEMORY_UPDATE = "memory_update"
    MEMORY_QUERY = "memory_query"
    GRAPH_REASONING = "graph_reasoning"
    API_SCHEMA_LEARNING = "api_schema_learning"
    RESOURCE_MONITORING = "resource_monitoring"
    STRUCTURED_OUTPUT = "structured_output"
    PLAN_MANAGEMENT = "plan_management"
    DEBUG_ENVIRONMENT = "debug_environment"
    PARALLEL_BATCH = "parallel_batch"
    MEMORY_ASSOCIATION = "memory_association"
    CODING_REQUEST = "coding_request"


class ToolDomain(str, Enum):
    """Which subsystem a tool belongs to (used for capability queries)."""

    FILESYSTEM = "filesystem"
    CODE_INTELLIGENCE = "code_intelligence"
    EXECUTION = "execution"
    NETWORK = "network"
    MEMORY = "memory"
    LEARNING = "learning"
    SYSTEM = "system"
    PLANNING = "planning"
    DEBUG = "debug"
    OUTPUT = "output"
    DATABASE = "database"
    PARALLEL = "parallel"


class ToolRisk(str, Enum):
    """Declared baseline risk, mirroring ``security.RiskTier`` values.

    The security boundary converts this to its own tier and may refine it
    by policy (method / verb / shell metacharacters / system paths);
    guardrails always override it.  Never a second security system.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ToolDefinition(BaseModel):
    """One canonical definition of a registered tool.

    Everything the rest of the system knows about a tool is declared here
    once.  Consumers derive their views; nothing may re-declare this data.
    """

    name: str
    func: Callable[..., Any]
    capabilities: tuple[ToolCapability, ...]
    read_only: bool
    risk: ToolRisk
    domain: ToolDomain
    # Explicit confirmation override. None = the boundary decides from risk.
    requires_confirmation: bool | None = None
    # Alternative action spellings (e.g. "web_search" for search_web).
    aliases: tuple[str, ...] = ()
    # Argument role metadata for the security boundary's (target, content)
    # scan — which argument is the audited/gated target and which carries
    # content that is scanned for secrets/URLs/paths.
    target_arg: str | None = None
    content_arg: str | None = None
    target_default: str = ""
    # User-facing confirmation-card label (falls back to the name).
    action_label: str | None = None
    # Override for the schema description; defaults to the function docstring.
    description: str | None = None
    # Advisory: may be suggested by the LLM batch planner.
    planner_friendly: bool = False

    @property
    def resolved_description(self) -> str:
        """The description used in tool schemas (override or docstring)."""
        if self.description:
            return self.description
        doc = inspect.getdoc(self.func) or f"Tool function {self.name}"
        return doc.split("\n\n")[0].strip()


# ---------------------------------------------------------------------------
# THE canonical table — one entry per registered tool, declared exactly once.
# Insertion order matters for ``preferred_tool_for`` (first tool wins).
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS: dict[str, ToolDefinition] = {}


def _define(
    name: str,
    func: Callable[..., Any],
    *,
    capabilities: tuple[ToolCapability, ...],
    read_only: bool,
    risk: ToolRisk,
    domain: ToolDomain,
    requires_confirmation: bool | None = None,
    aliases: tuple[str, ...] = (),
    target_arg: str | None = None,
    content_arg: str | None = None,
    target_default: str = "",
    action_label: str | None = None,
    description: str | None = None,
    planner_friendly: bool = False,
) -> None:
    TOOL_DEFINITIONS[name] = ToolDefinition(
        name=name,
        func=func,
        capabilities=capabilities,
        read_only=read_only,
        risk=risk,
        domain=domain,
        requires_confirmation=requires_confirmation,
        aliases=aliases,
        target_arg=target_arg,
        content_arg=content_arg,
        target_default=target_default,
        action_label=action_label,
        description=description,
        planner_friendly=planner_friendly,
    )


def _reg(name: str, func: Callable[..., Any], **kwargs: Any) -> None:
    """Short alias for :func:`_define` used by the table below."""
    _define(name, func, **kwargs)


# --- Filesystem -----------------------------------------------------------
_reg(
    "read_file", read_file,
    capabilities=(ToolCapability.FILE_READ,),
    read_only=True, risk=ToolRisk.LOW, domain=ToolDomain.FILESYSTEM,
    target_arg="file_path", action_label="Read file",
)
_reg(
    "write_file", write_file,
    capabilities=(ToolCapability.FILE_WRITE,),
    read_only=False, risk=ToolRisk.HIGH, domain=ToolDomain.FILESYSTEM,
    requires_confirmation=True,
    target_arg="file_path", content_arg="content", action_label="Create new file",
)
_reg(
    "list_directory", list_directory,
    capabilities=(ToolCapability.DIRECTORY_LIST,),
    read_only=True, risk=ToolRisk.LOW, domain=ToolDomain.FILESYSTEM,
    target_arg="path", target_default=".", action_label="List directory",
)
_reg(
    "search_files", search_files,
    capabilities=(ToolCapability.FILE_SEARCH,),
    read_only=True, risk=ToolRisk.LOW, domain=ToolDomain.FILESYSTEM,
    target_arg="query", action_label="Search files",
)
_reg(
    "discover_workspace_summary", discover_workspace_summary,
    capabilities=(ToolCapability.REPOSITORY_INSPECTION,),
    read_only=True, risk=ToolRisk.LOW, domain=ToolDomain.FILESYSTEM,
    action_label="Inspect workspace",
)
_reg(
    "create_file", create_file,
    capabilities=(ToolCapability.FILE_CREATE,),
    read_only=False, risk=ToolRisk.HIGH, domain=ToolDomain.FILESYSTEM,
    requires_confirmation=True,
    target_arg="file_path", content_arg="content", action_label="Create new file",
)
_reg(
    "replace_file", replace_file,
    capabilities=(ToolCapability.FILE_WRITE,),
    read_only=False, risk=ToolRisk.HIGH, domain=ToolDomain.FILESYSTEM,
    requires_confirmation=True,
    target_arg="file_path", content_arg="content", action_label="Replace file",
)
_reg(
    "replace_in_file", replace_in_file,
    capabilities=(ToolCapability.FILE_WRITE,),
    read_only=False, risk=ToolRisk.HIGH, domain=ToolDomain.FILESYSTEM,
    requires_confirmation=True,
    target_arg="file_path", content_arg="new",
    action_label="Edit file (targeted replace)",
)
_reg(
    "append_to_file", append_to_file,
    capabilities=(ToolCapability.FILE_WRITE,),
    read_only=False, risk=ToolRisk.HIGH, domain=ToolDomain.FILESYSTEM,
    requires_confirmation=True,
    target_arg="file_path", content_arg="content", action_label="Append to file",
)
_reg(
    "delete_file", delete_file,
    capabilities=(ToolCapability.FILE_DELETE,),
    read_only=False, risk=ToolRisk.HIGH, domain=ToolDomain.FILESYSTEM,
    requires_confirmation=True,
    target_arg="file_path", action_label="Delete file",
)
_reg(
    "rename_file", rename_file,
    capabilities=(ToolCapability.FILE_RENAME,),
    read_only=False, risk=ToolRisk.HIGH, domain=ToolDomain.FILESYSTEM,
    requires_confirmation=True,
    target_arg="file_path", content_arg="new_path", action_label="Rename/move file",
)

# --- Code intelligence (read-only) ----------------------------------------
_reg(
    "code_search", code_search,
    capabilities=(ToolCapability.CODE_SEARCH,),
    read_only=True, risk=ToolRisk.LOW, domain=ToolDomain.CODE_INTELLIGENCE,
    target_arg="path", content_arg="query", target_default=".",
    action_label="Search code",
)
_reg(
    "code_investigation", code_investigation,
    capabilities=(ToolCapability.REPOSITORY_INVESTIGATION,),
    read_only=True, risk=ToolRisk.LOW, domain=ToolDomain.CODE_INTELLIGENCE,
    target_arg="path", content_arg="query", target_default=".",
    action_label="Investigate repository code",
)
_reg(
    "find_symbol", find_symbol,
    capabilities=(ToolCapability.SYMBOL_SEARCH,),
    read_only=True, risk=ToolRisk.LOW, domain=ToolDomain.CODE_INTELLIGENCE,
    target_arg="path", content_arg="name", target_default=".",
    action_label="Find symbol",
)
_reg(
    "find_definition", find_definition,
    capabilities=(ToolCapability.DEFINITION_LOOKUP,),
    read_only=True, risk=ToolRisk.LOW, domain=ToolDomain.CODE_INTELLIGENCE,
    target_arg="path", content_arg="name", target_default=".",
    action_label="Find definition",
)
_reg(
    "find_references", find_references,
    capabilities=(ToolCapability.REFERENCE_LOOKUP,),
    read_only=True, risk=ToolRisk.LOW, domain=ToolDomain.CODE_INTELLIGENCE,
    target_arg="path", content_arg="name", target_default=".",
    action_label="Find references",
)
_reg(
    "get_imports", get_imports,
    capabilities=(ToolCapability.DEPENDENCY_ANALYSIS,),
    read_only=True, risk=ToolRisk.LOW, domain=ToolDomain.CODE_INTELLIGENCE,
    target_arg="file_path", content_arg="path", action_label="Get imports",
)
_reg(
    "get_dependents", get_dependents,
    capabilities=(ToolCapability.DEPENDENCY_ANALYSIS,),
    read_only=True, risk=ToolRisk.LOW, domain=ToolDomain.CODE_INTELLIGENCE,
    target_arg="file_path", content_arg="path", action_label="Get dependents",
)
_reg(
    "semantic_search", semantic_search,
    capabilities=(ToolCapability.SEMANTIC_SEARCH,),
    read_only=True, risk=ToolRisk.LOW, domain=ToolDomain.CODE_INTELLIGENCE,
    target_arg="path", content_arg="query", target_default=".",
    action_label="Semantic code search",
)
_reg(
    "code_index_status", code_index_status,
    capabilities=(ToolCapability.REPOSITORY_INSPECTION,),
    read_only=True, risk=ToolRisk.LOW, domain=ToolDomain.CODE_INTELLIGENCE,
    target_arg="path", target_default=".", action_label="Code index status",
)
_reg(
    "report_file", report_file,
    capabilities=(ToolCapability.FILE_INSPECTION,),
    read_only=True, risk=ToolRisk.LOW, domain=ToolDomain.CODE_INTELLIGENCE,
    target_arg="file_path", content_arg="path", action_label="Report file symbols",
)
_reg(
    "report_symbol", report_symbol,
    capabilities=(ToolCapability.SYMBOL_INSPECTION,),
    read_only=True, risk=ToolRisk.LOW, domain=ToolDomain.CODE_INTELLIGENCE,
    target_arg="path", content_arg="name", target_default=".",
    action_label="Report symbol",
)

# --- Execution ------------------------------------------------------------
_reg(
    "run_command", run_command,
    capabilities=(
        ToolCapability.TERMINAL_EXECUTION,
        ToolCapability.DIRECTORY_CREATE,
        ToolCapability.TEST_EXECUTION,
        ToolCapability.BUILD,
        ToolCapability.INSTALL,
        ToolCapability.GIT_OPERATION,
        ToolCapability.LINT,
        ToolCapability.TYPECHECK,
        ToolCapability.FORMAT,
        ToolCapability.APPLICATION_START,
        ToolCapability.APPLICATION_STOP,
    ),
    read_only=False, risk=ToolRisk.HIGH, domain=ToolDomain.EXECUTION,
    requires_confirmation=True, action_label="Execute terminal command",
)
_reg(
    "run_parallel", run_parallel,
    capabilities=(ToolCapability.TERMINAL_EXECUTION,),
    read_only=False, risk=ToolRisk.HIGH, domain=ToolDomain.EXECUTION,
    requires_confirmation=True,
)

# --- Network --------------------------------------------------------------
_reg(
    "make_http_request", make_http_request,
    capabilities=(ToolCapability.HTTP_REQUEST, ToolCapability.INFORMATION_REQUEST),
    read_only=False, risk=ToolRisk.MEDIUM, domain=ToolDomain.NETWORK,
    action_label="Make HTTP request",
)
_reg(
    "retrieve", retrieve,
    capabilities=(ToolCapability.INFORMATION_REQUEST, ToolCapability.PAGE_FETCH),
    read_only=True, risk=ToolRisk.LOW, domain=ToolDomain.NETWORK,
)
_reg(
    "check_connectivity", check_connectivity,
    capabilities=(ToolCapability.INFORMATION_REQUEST,),
    read_only=True, risk=ToolRisk.LOW, domain=ToolDomain.NETWORK,
    target_arg="url",
)
_reg(
    "search_web", search_web,
    capabilities=(ToolCapability.WEB_SEARCH, ToolCapability.INFORMATION_REQUEST),
    read_only=True, risk=ToolRisk.LOW, domain=ToolDomain.NETWORK,
    aliases=("web_search",), target_arg="query", action_label="Search the web",
)
_reg(
    "fetch_page_text", fetch_page_text,
    capabilities=(ToolCapability.PAGE_FETCH, ToolCapability.INFORMATION_REQUEST),
    read_only=True, risk=ToolRisk.LOW, domain=ToolDomain.NETWORK,
    aliases=("fetch_page",), target_arg="url", action_label="Fetch web page",
)

# --- Database -------------------------------------------------------------
_reg(
    "run_query", run_query,
    capabilities=(ToolCapability.DATABASE_QUERY,),
    read_only=False, risk=ToolRisk.MEDIUM, domain=ToolDomain.DATABASE,
    aliases=("db_query",), action_label="Execute database query",
)

# --- Memory ---------------------------------------------------------------
_reg(
    "add_memory", store_memory_text,
    capabilities=(ToolCapability.MEMORY_UPDATE,),
    read_only=False, risk=ToolRisk.LOW, domain=ToolDomain.MEMORY,
    content_arg="fact", action_label="Save memory",
)
_reg(
    "add_triple", add_triple,
    capabilities=(ToolCapability.MEMORY_UPDATE,),
    read_only=False, risk=ToolRisk.LOW, domain=ToolDomain.MEMORY,
)
_reg(
    "get_all_memories", get_all_memories,
    capabilities=(ToolCapability.MEMORY_QUERY,),
    read_only=True, risk=ToolRisk.LOW, domain=ToolDomain.MEMORY,
    action_label="Read memories",
)
_reg(
    "search_memories", search_memories,
    capabilities=(ToolCapability.MEMORY_QUERY,),
    read_only=True, risk=ToolRisk.LOW, domain=ToolDomain.MEMORY,
    target_arg="keyword", action_label="Search memories",
)
_reg(
    "query_triples", query_triples,
    capabilities=(ToolCapability.MEMORY_QUERY, ToolCapability.GRAPH_REASONING),
    read_only=True, risk=ToolRisk.LOW, domain=ToolDomain.MEMORY,
)
_reg(
    "search_triples", search_triples,
    capabilities=(ToolCapability.MEMORY_QUERY,),
    read_only=True, risk=ToolRisk.LOW, domain=ToolDomain.MEMORY,
    target_arg="keyword",
)
_reg(
    "get_all_triples", get_all_triples,
    capabilities=(ToolCapability.MEMORY_QUERY,),
    read_only=True, risk=ToolRisk.LOW, domain=ToolDomain.MEMORY,
)
_reg(
    "memory_connections", memory_connections,
    capabilities=(ToolCapability.MEMORY_QUERY, ToolCapability.GRAPH_REASONING),
    read_only=True, risk=ToolRisk.LOW, domain=ToolDomain.MEMORY,
    target_arg="topic",
)
_reg(
    "related_facts", related_facts,
    capabilities=(ToolCapability.MEMORY_QUERY,),
    read_only=True, risk=ToolRisk.LOW, domain=ToolDomain.MEMORY,
)
_reg(
    "discover_connections", discover_connections,
    capabilities=(ToolCapability.GRAPH_REASONING, ToolCapability.MEMORY_ASSOCIATION),
    read_only=True, risk=ToolRisk.LOW, domain=ToolDomain.MEMORY,
)
_reg(
    "explain_relation", explain_relation,
    capabilities=(ToolCapability.GRAPH_REASONING, ToolCapability.MEMORY_ASSOCIATION),
    read_only=True, risk=ToolRisk.LOW, domain=ToolDomain.MEMORY,
)

# --- Learning -------------------------------------------------------------
_reg(
    "learn_api_schema", learn_api_schema,
    capabilities=(ToolCapability.API_SCHEMA_LEARNING,),
    read_only=False, risk=ToolRisk.LOW, domain=ToolDomain.LEARNING,
    target_arg="base_url",
)
_reg(
    "api_usage_hint", api_usage_hint,
    capabilities=(ToolCapability.API_SCHEMA_LEARNING,),
    read_only=True, risk=ToolRisk.LOW, domain=ToolDomain.LEARNING,
    target_arg="url", content_arg="body",
)
_reg(
    "get_api_knowledge", get_api_knowledge,
    capabilities=(ToolCapability.API_SCHEMA_LEARNING,),
    read_only=True, risk=ToolRisk.LOW, domain=ToolDomain.LEARNING,
    target_arg="base_url",
)
_reg(
    "forget_api", forget_api,
    capabilities=(ToolCapability.API_SCHEMA_LEARNING,),
    read_only=False, risk=ToolRisk.LOW, domain=ToolDomain.LEARNING,
    target_arg="base_url",
)

# --- System / resources ---------------------------------------------------
_reg(
    "check_resources", check_resources,
    capabilities=(ToolCapability.RESOURCE_MONITORING,),
    read_only=True, risk=ToolRisk.LOW, domain=ToolDomain.SYSTEM,
)
_reg(
    "resource_forecast", resource_forecast,
    capabilities=(ToolCapability.RESOURCE_MONITORING,),
    read_only=True, risk=ToolRisk.LOW, domain=ToolDomain.SYSTEM,
    target_arg="command",
)

# --- Structured output ----------------------------------------------------
_reg(
    "enforce_schema", enforce_schema,
    capabilities=(ToolCapability.STRUCTURED_OUTPUT,),
    read_only=True, risk=ToolRisk.LOW, domain=ToolDomain.OUTPUT,
    target_arg="format", content_arg="text",
)
_reg(
    "schema_validate", schema_validate,
    capabilities=(ToolCapability.STRUCTURED_OUTPUT,),
    read_only=True, risk=ToolRisk.LOW, domain=ToolDomain.OUTPUT,
    target_arg="format", content_arg="text",
)
_reg(
    "list_schemas", list_schemas,
    capabilities=(ToolCapability.STRUCTURED_OUTPUT,),
    read_only=True, risk=ToolRisk.LOW, domain=ToolDomain.OUTPUT,
)

# --- Planning -------------------------------------------------------------
_reg(
    "preflight_plan", preflight_plan_tool,
    capabilities=(ToolCapability.PLAN_MANAGEMENT,),
    read_only=True, risk=ToolRisk.LOW, domain=ToolDomain.PLANNING,
    content_arg="steps_json",
)
_reg(
    "analyze_dependencies", analyze_dependencies,
    capabilities=(ToolCapability.PLAN_MANAGEMENT,),
    read_only=True, risk=ToolRisk.LOW, domain=ToolDomain.PLANNING,
    content_arg="steps_json",
)
_reg(
    "list_plan_actions", list_plan_actions,
    capabilities=(ToolCapability.PLAN_MANAGEMENT,),
    read_only=True, risk=ToolRisk.LOW, domain=ToolDomain.PLANNING,
)

# --- Debug / environment --------------------------------------------------
_reg(
    "get_debug_context", get_debug_context,
    capabilities=(ToolCapability.DEBUG_ENVIRONMENT,),
    read_only=True, risk=ToolRisk.LOW, domain=ToolDomain.DEBUG,
)
_reg(
    "diagnose_failure", diagnose_failure,
    capabilities=(ToolCapability.DEBUG_ENVIRONMENT,),
    read_only=True, risk=ToolRisk.LOW, domain=ToolDomain.DEBUG,
    target_arg="command", content_arg="text",
)
_reg(
    "check_dependency", check_dependency,
    capabilities=(ToolCapability.DEBUG_ENVIRONMENT,),
    read_only=True, risk=ToolRisk.LOW, domain=ToolDomain.DEBUG,
    target_arg="name",
)

# --- Parallel batches -----------------------------------------------------
_reg(
    "run_tool_batch", run_tool_batch,
    capabilities=(ToolCapability.PARALLEL_BATCH,),
    read_only=False, risk=ToolRisk.LOW, domain=ToolDomain.PARALLEL,
    content_arg="calls_json",
)
_reg(
    "synthesize_analysis", synthesize_analysis,
    capabilities=(ToolCapability.PARALLEL_BATCH,),
    read_only=True, risk=ToolRisk.LOW, domain=ToolDomain.PARALLEL,
    content_arg="results_json",
)


# ---------------------------------------------------------------------------
# Derived queries — the only way consumers should read tool metadata.
# ---------------------------------------------------------------------------


def canonical_action_name(action: str) -> str:
    """Maps any alias spelling to the canonical registered tool name.

    ``web_search`` -> ``search_web``; ``fetch_page`` -> ``fetch_page_text``;
    ``db_query`` -> ``run_query``; already-canonical names are unchanged.
    """
    action = (action or "").lower()
    for name, definition in TOOL_DEFINITIONS.items():
        if action == name or action in definition.aliases:
            return name
    return action


def get_tool_definition(name: str) -> ToolDefinition | None:
    """Returns the canonical definition for a tool (alias-aware), or None."""
    return TOOL_DEFINITIONS.get(canonical_action_name(name))


def tools_with_capability(capability: ToolCapability) -> list[str]:
    """Registered tool names serving *capability* (insertion order)."""
    return [n for n, d in TOOL_DEFINITIONS.items() if capability in d.capabilities]


def preferred_tool_for(capability: ToolCapability) -> str | None:
    """The first registered tool serving *capability* (insertion order)."""
    tools = tools_with_capability(capability)
    return tools[0] if tools else None


def tools_with_any_capability(capabilities: set[ToolCapability]) -> frozenset[str]:
    """Registered tool names serving at least one of *capabilities*."""
    wanted = set(capabilities)
    return frozenset(n for n, d in TOOL_DEFINITIONS.items() if wanted & set(d.capabilities))


def readonly_tool_names() -> frozenset[str]:
    """All registered tools declared read-only (canonical names only)."""
    return frozenset(n for n, d in TOOL_DEFINITIONS.items() if d.read_only)


def code_intel_tool_names() -> frozenset[str]:
    """Registered code-intelligence tools (the ReAct redirect universe)."""
    return frozenset(
        n for n, d in TOOL_DEFINITIONS.items() if d.domain == ToolDomain.CODE_INTELLIGENCE
    )


def generic_code_tool_names() -> frozenset[str]:
    """Generic lexical/semantic code tools the model may over-use."""
    wanted = {ToolCapability.CODE_SEARCH, ToolCapability.SEMANTIC_SEARCH}
    return frozenset(n for n, d in TOOL_DEFINITIONS.items() if wanted & set(d.capabilities))


def web_tool_names() -> frozenset[str]:
    """Web-search tools and their alias spellings (never repo questions)."""
    names: set[str] = set()
    for n, d in TOOL_DEFINITIONS.items():
        if ToolCapability.WEB_SEARCH in d.capabilities:
            names.add(n)
            names.update(d.aliases)
    return frozenset(names)


def action_label_for(action: str) -> str | None:
    """User-facing confirmation label for a tool/action, or None."""
    definition = get_tool_definition(action)
    if definition is not None:
        return definition.action_label
    return None


def tool_aliases() -> dict[str, str]:
    """Canonical name -> first alias, for action-name canonicalization."""
    return {
        name: definition.aliases[0]
        for name, definition in TOOL_DEFINITIONS.items()
        if definition.aliases
    }
