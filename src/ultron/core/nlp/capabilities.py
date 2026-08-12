"""ultron.core.nlp.capabilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Machine-readable capability metadata for the registered tools.

This is *advisory metadata for tool selection only* — it does not replace the
security boundary, which remains the authority on whether an action may run.
Every tool listed here maps to a name in ``ultron.core.tools.registry``.

:func:`select_tool` is the schema-aware selection helper the routing layer
uses to prefer the most precise tool for a category (never the terminal when
a dedicated tool exists).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ultron.core.nlp.intent import IntentCategory


@dataclass(frozen=True)
class ToolCapability:
    """Metadata describing what one registered tool can do."""

    name: str
    category: IntentCategory
    read_only: bool
    risk_level: str  # "low" | "medium" | "high" | "critical"
    description: str = ""
    requires_confirmation: bool = False
    argument_names: tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Capability table — kept in sync with tools/registry.TOOLS
# ---------------------------------------------------------------------------

TOOL_CAPABILITIES: dict[str, ToolCapability] = {
    # Filesystem (read-only)
    "read_file": ToolCapability(
        "read_file", IntentCategory.FILE_READ, True, "low",
        "Read a local file", argument_names=("file_path",),
    ),
    "list_directory": ToolCapability(
        "list_directory", IntentCategory.DIRECTORY_LIST, True, "low",
        "List a directory tree", argument_names=("path",),
    ),
    "search_files": ToolCapability(
        "search_files", IntentCategory.FILE_SEARCH, True, "low",
        "Search file names and content", argument_names=("query", "path"),
    ),
    # Filesystem (state-changing)
    "write_file": ToolCapability(
        "write_file", IntentCategory.FILE_WRITE, False, "high",
        "Write content to a file", requires_confirmation=True,
        argument_names=("file_path", "content"),
    ),
    "create_file": ToolCapability(
        "create_file", IntentCategory.FILE_CREATE, False, "high",
        "Create a new file", requires_confirmation=True,
        argument_names=("file_path", "content"),
    ),
    "delete_file": ToolCapability(
        "delete_file", IntentCategory.FILE_DELETE, False, "high",
        "Delete a file", requires_confirmation=True,
        argument_names=("file_path",),
    ),
    "rename_file": ToolCapability(
        "rename_file", IntentCategory.FILE_RENAME, False, "high",
        "Rename/move a file", requires_confirmation=True,
        argument_names=("file_path", "new_path"),
    ),
    # Code intelligence (read-only)
    "code_search": ToolCapability(
        "code_search", IntentCategory.CODE_SEARCH, True, "low",
        "Lexical/regex search over the repository",
        argument_names=("query", "path"),
    ),
    "find_symbol": ToolCapability(
        "find_symbol", IntentCategory.SYMBOL_SEARCH, True, "low",
        "Find every symbol with a given name", argument_names=("name",),
    ),
    "find_definition": ToolCapability(
        "find_definition", IntentCategory.DEFINITION_LOOKUP, True, "low",
        "Find where a symbol is defined", argument_names=("name",),
    ),
    "find_references": ToolCapability(
        "find_references", IntentCategory.REFERENCE_LOOKUP, True, "low",
        "Find all references to a symbol", argument_names=("name",),
    ),
    "semantic_search": ToolCapability(
        "semantic_search", IntentCategory.SEMANTIC_SEARCH, True, "low",
        "Embedding-based semantic code search",
        argument_names=("query", "path"),
    ),
    # Execution
    "run_command": ToolCapability(
        "run_command", IntentCategory.TERMINAL_EXECUTION, False, "high",
        "Run a shell command", requires_confirmation=True,
        argument_names=("command",),
    ),
    "run_parallel": ToolCapability(
        "run_parallel", IntentCategory.TERMINAL_EXECUTION, False, "high",
        "Run multiple shell commands concurrently", requires_confirmation=True,
        argument_names=("commands",),
    ),
    "make_http_request": ToolCapability(
        "make_http_request", IntentCategory.INFORMATION_REQUEST, False, "medium",
        "Send an HTTP request", argument_names=("method", "url", "body"),
    ),
    # Network (read-only)
    "web_search": ToolCapability(
        "web_search", IntentCategory.INFORMATION_REQUEST, True, "low",
        "Search the web", argument_names=("query",),
    ),
    "fetch_page_text": ToolCapability(
        "fetch_page_text", IntentCategory.INFORMATION_REQUEST, True, "low",
        "Fetch a web page as text", argument_names=("url",),
    ),
    # Memory
    "add_memory": ToolCapability(
        "add_memory", IntentCategory.MEMORY_UPDATE, False, "low",
        "Store a fact", argument_names=("fact",),
    ),
    "search_memories": ToolCapability(
        "search_memories", IntentCategory.MEMORY_QUERY, True, "low",
        "Recall stored facts", argument_names=("keyword",),
    ),
}

# Map a category to its preferred tool name (most precise first).
_CATEGORY_PREFERENCE: dict[IntentCategory, str] = {
    IntentCategory.FILE_READ: "read_file",
    IntentCategory.FILE_WRITE: "write_file",
    IntentCategory.FILE_CREATE: "create_file",
    IntentCategory.FILE_DELETE: "delete_file",
    IntentCategory.FILE_RENAME: "rename_file",
    IntentCategory.DIRECTORY_LIST: "list_directory",
    IntentCategory.DIRECTORY_CREATE: "run_command",  # mkdir -p via shell
    IntentCategory.FILE_SEARCH: "search_files",
    IntentCategory.CODE_SEARCH: "code_search",
    IntentCategory.SYMBOL_SEARCH: "find_symbol",
    IntentCategory.DEFINITION_LOOKUP: "find_definition",
    IntentCategory.REFERENCE_LOOKUP: "find_references",
    IntentCategory.SEMANTIC_SEARCH: "semantic_search",
    IntentCategory.TERMINAL_EXECUTION: "run_command",
    IntentCategory.TEST_EXECUTION: "run_command",
    IntentCategory.BUILD: "run_command",
    IntentCategory.LINT: "run_command",
    IntentCategory.TYPECHECK: "run_command",
    IntentCategory.FORMAT: "run_command",
    IntentCategory.APPLICATION_START: "run_command",
    IntentCategory.APPLICATION_STOP: "run_command",
    IntentCategory.INSTALL: "run_command",
    IntentCategory.GIT_OPERATION: "run_command",
}


def select_tool(category: IntentCategory) -> str | None:
    """
    Returns the preferred tool name for *category*, or None when no tool is
    registered for it.  Dedicated tools always win over the terminal.
    """
    return _CATEGORY_PREFERENCE.get(category)
