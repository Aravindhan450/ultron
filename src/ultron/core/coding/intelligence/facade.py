"""ultron.core.coding.intelligence.facade
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CodeIntelligence — the layered facade over the repository intelligence
stack (Fix #4).

It presents ONE coherent API to the coding agent while keeping the layers
separated underneath:

    L1 Filesystem   -> list_source_files
    L2 Lexical      -> search_code
    L3 AST/Symbol   -> RepositoryIndex (find_symbol / find_definition /
                       find_references / imports / dependents)
    L4 LSP          -> LSPFacade (abstraction; degrades gracefully)
    L5 Semantic     -> SemanticSearch (embedder-pluggable, lexical fallback)
    L6 Dependencies -> DependencyGraph (EXACT imports, INFERRED refs)

The facade applies the "cheapest reliable layer first" rule: exact symbol
lookup (L3) is always preferred over semantic search (L5), and LSP (L4) is
only consulted when a server is actually available. Every method returns
plain strings (tool-friendly) or typed lists (programmatic use), and no
method ever executes or modifies anything — the whole layer is read-only.

The facade is workspace-scoped: index construction accepts any root (for
programmatic use), but :meth:`search` and :meth:`files` route through the
shared path-safety gate, so they return "access denied" for roots outside
``ALLOWED_BASE_DIR``. The registered tools enforce the same boundary via
``_resolve_safe_path`` before constructing a facade.
"""

from __future__ import annotations

from pathlib import Path
from typing import Self

from ultron.core.coding.intelligence.dependencies import (
    DependencyEdge,
    DependencyGraph,
)
from ultron.core.coding.intelligence.index import IndexSummary, RepositoryIndex
from ultron.core.coding.intelligence.lsp import (
    LSPFacade,
    LSPServerManager,
    NoLSPServers,
)
from ultron.core.coding.intelligence.search import (
    list_source_files,
    search_code,
)
from ultron.core.coding.intelligence.semantic import (
    Embedder,
    SemanticHit,
    SemanticSearch,
)
from ultron.core.coding.intelligence.symbols import (
    ImportEdge,
    Symbol,
    SymbolReference,
)


class CodeIntelligence:
    """
    Layered repository intelligence for one workspace root.

    Usage::

        ci = CodeIntelligence(root="/path/to/repo")
        summary = ci.refresh()                       # incremental index
        defs = ci.find_definition("login")
        refs = ci.find_references("UserService")
        report = ci.report_symbol("UserService")     # tool-friendly string
    """

    def __init__(
        self,
        root: str | Path | None = None,
        db_path: str | Path | None = None,
        embedder: Embedder | None = None,
        lsp_manager: LSPServerManager | None = None,
    ) -> None:
        self.root = Path(root).resolve() if root else Path.cwd().resolve()
        self.index = RepositoryIndex(self.root, db_path=db_path)
        self.graph = DependencyGraph(self.index)
        self.semantic = SemanticSearch(self.index, embedder=embedder)
        self.lsp = LSPFacade(manager=lsp_manager or NoLSPServers())

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def refresh(self) -> IndexSummary:
        """Incremental re-index (re-parses only changed files)."""
        summary = self.index.refresh()
        self.semantic.invalidate()
        return summary

    def index_status(self) -> IndexSummary:
        """Current index row counts (no rescan)."""
        return self.index.status()

    def index_status_line(self) -> str:
        """Tool-friendly one-line status."""
        return self.index_status().to_prompt_line()

    # ------------------------------------------------------------------
    # L1/L2 — filesystem + lexical search
    # ------------------------------------------------------------------

    def search(self, query: str, **kwargs) -> str:
        """Lexical search (see :func:`search_code`)."""
        return search_code(query, path=str(self.root), **kwargs)

    def files(self, file_pattern: str | None = None, max_files: int = 500) -> str:
        """Source-file listing under the root."""
        return list_source_files(
            str(self.root), file_pattern=file_pattern, max_files=max_files
        )

    # ------------------------------------------------------------------
    # L3 — symbol intelligence
    # ------------------------------------------------------------------

    def find_symbol(self, name: str, *, case_insensitive: bool = False) -> list[Symbol]:
        return self.index.find_symbol(name, case_insensitive=case_insensitive)

    def find_definition(
        self, name: str, *, case_insensitive: bool = False
    ) -> list[Symbol]:
        return self.index.find_definition(name, case_insensitive=case_insensitive)

    def find_references(
        self, name: str, *, case_insensitive: bool = False
    ) -> list[SymbolReference]:
        return self.index.find_references(name, case_insensitive=case_insensitive)

    def get_imports(self, rel_path: str) -> list[ImportEdge]:
        return self.index.get_imports(rel_path)

    def get_dependents(self, rel_path: str) -> list[str]:
        return self.index.get_dependents(rel_path)

    # ------------------------------------------------------------------
    # L4 — LSP (graceful)
    # ------------------------------------------------------------------

    def lsp_available(self) -> bool:
        return self.lsp.available

    def lsp_start(self, root: str | None = None, preferred: str | None = None) -> bool:
        return self.lsp.start(str(root or self.root), preferred=preferred)

    def lsp_stop(self) -> None:
        self.lsp.stop()

    # ------------------------------------------------------------------
    # L5 — semantic (metadata-rich, degrades to lexical)
    # ------------------------------------------------------------------

    def search_semantically(self, query: str, top_k: int = 5) -> list[SemanticHit]:
        return self.semantic.search(query, top_k=top_k)

    # ------------------------------------------------------------------
    # L6 — dependency graph
    # ------------------------------------------------------------------

    def dependency_imports(self, rel_path: str) -> list[DependencyEdge]:
        return self.graph.imports(rel_path)

    def dependency_dependents(self, rel_path: str) -> list[DependencyEdge]:
        return self.graph.dependents(rel_path)

    def dependency_references(self, symbol_name: str) -> list[DependencyEdge]:
        return self.graph.references_to(symbol_name)

    def dependency_callers(self, symbol_name: str) -> list[DependencyEdge]:
        """INFERRED candidate callers (a name( pattern in a referencing file)."""
        return self.graph.callers_of(symbol_name)

    def dependency_inheritors(self, symbol_name: str) -> list[DependencyEdge]:
        """INFERRED subclasses/interfaces extending *symbol_name*."""
        return self.graph.inheritors_of(symbol_name)

    # ------------------------------------------------------------------
    # Tool-friendly reports (single strings for the ReAct loop)
    # ------------------------------------------------------------------

    def report_symbol(self, name: str, max_lines: int = 20) -> str:
        """Combined definition + references report for *name*."""
        lines: list[str] = []
        definitions = self.find_definition(name)
        if definitions:
            lines.append(f"Definitions of '{name}':")
            for symbol in definitions[:5]:
                lines.append(f"  - {symbol.to_prompt_line()}")
        else:
            lines.append(f"No definition found for '{name}' in the index.")

        references = self.find_references(name)
        if references:
            lines.append(f"References ({len(references)} found, showing up to {max_lines}):")
            for ref in references[:max_lines]:
                lines.append(f"  - {ref.to_prompt_line()}")
        else:
            lines.append(f"No references found for '{name}'.")
        return "\n".join(lines)

    def report_file(self, rel_path: str) -> str:
        """Imports + symbols of one file (by relative path)."""
        lines: list[str] = []
        imports = self.get_imports(rel_path)
        if imports:
            lines.append(f"Imports of {rel_path}:")
            for edge in imports[:20]:
                lines.append(f"  - {edge.to_prompt_line()}")
        symbols = self.index.find_symbols_in_file(rel_path)
        if symbols:
            lines.append(f"Symbols in {rel_path}:")
            for symbol in symbols[:20]:
                lines.append(f"  - {symbol.to_prompt_line()}")
        dependents = self.get_dependents(rel_path)
        if dependents:
            lines.append(f"Files importing {rel_path}:")
            for dep in dependents[:10]:
                lines.append(f"  - {dep}")
        return "\n".join(lines) or f"No index data for {rel_path}."

    def close(self) -> None:
        self.index.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
