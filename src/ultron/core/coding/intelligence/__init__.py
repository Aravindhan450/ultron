"""ultron.core.coding.intelligence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Codebase intelligence (Fix #4) — the layered repository intelligence
system for the coding agent:

- :mod:`search` — L1/L2: gitignore-aware filesystem + lexical search.
- :mod:`symbols` — the symbol/reference/import data model.
- :mod:`parsers` — modular parser registry (Python AST + heuristic regex
  parsers for many languages).
- :mod:`index` — incremental SQLite repository index (L3).
- :mod:`lsp` — LSP client abstraction (L4, graceful degradation).
- :mod:`semantic` — metadata-rich semantic search foundation (L5).
- :mod:`dependencies` — dependency graph, EXACT vs INFERRED (L6).
- :mod:`facade` — CodeIntelligence: one coherent API over all layers.

Security: every capability here is READ-ONLY. The facade never executes
tools, never modifies files, and every path argument resolves through the
shared path-safety gate. Search/index respect ``.gitignore`` and the
standard ignored directories (``.git``, ``node_modules``, ``.venv``,
``__pycache__``, ``dist``, ``build``, ...).
"""

from ultron.core.coding.intelligence.dependencies import (
    DependencyEdge,
    DependencyGraph,
    EdgeConfidence,
)
from ultron.core.coding.intelligence.facade import CodeIntelligence
from ultron.core.coding.intelligence.index import (
    IndexSummary,
    RepositoryIndex,
)
from ultron.core.coding.intelligence.lsp import (
    LSPCapabilities,
    LSPClient,
    LSPFacade,
    LSPLocation,
    LSPOperation,
    LSPServerManager,
    LSPSymbol,
    LSPUnavailableError,
    NoLSPServers,
)
from ultron.core.coding.intelligence.parsers import (
    EXTENSION_LANGUAGES,
    PARSERS,
    PythonAstParser,
    RegexParser,
    SourceParser,
    get_parser,
    language_for_path,
    parse_source,
)
from ultron.core.coding.intelligence.search import (
    GitIgnoreRules,
    list_source_files,
    search_code,
)
from ultron.core.coding.intelligence.semantic import (
    CodeChunk,
    Embedder,
    SemanticHit,
    SemanticSearch,
)
from ultron.core.coding.intelligence.symbols import (
    ImportEdge,
    ParseResult,
    Symbol,
    SymbolKind,
    SymbolLocation,
    SymbolReference,
    SymbolRelationship,
)

__all__ = [
    "EXTENSION_LANGUAGES",
    "PARSERS",
    "CodeChunk",
    "CodeIntelligence",
    "DependencyEdge",
    "DependencyGraph",
    "EdgeConfidence",
    "Embedder",
    "GitIgnoreRules",
    "ImportEdge",
    "IndexSummary",
    "LSPCapabilities",
    "LSPClient",
    "LSPFacade",
    "LSPLocation",
    "LSPOperation",
    "LSPServerManager",
    "LSPSymbol",
    "LSPUnavailableError",
    "NoLSPServers",
    "ParseResult",
    "PythonAstParser",
    "RegexParser",
    "RepositoryIndex",
    "SemanticHit",
    "SemanticSearch",
    "SourceParser",
    "Symbol",
    "SymbolKind",
    "SymbolLocation",
    "SymbolReference",
    "SymbolRelationship",
    "get_parser",
    "language_for_path",
    "list_source_files",
    "parse_source",
    "search_code",
]
