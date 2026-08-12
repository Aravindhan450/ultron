"""ultron.core.coding.intelligence.dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dependency graph over the repository index (Fix #4).

Relationships are explicitly classified so uncertain edges are never
presented as facts:

- ``EXACT`` — import edges parsed from the source (``file A imports file B``).
- ``INFERRED`` — structural guesses (class A references class B, function A
  calls function B) derived from reference hits in the index. These are
  best-effort and labelled as such.

Callers/callees at this layer are inferred from reference occurrences: a
function name appearing in another file's references is treated as a
candidate caller/callee, not a fact. Exact call graphs are the LSP layer's
job (see ``lsp.py``).
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel

from ultron.core.coding.intelligence.index import RepositoryIndex


class EdgeConfidence(str, Enum):
    """How certain a dependency edge is."""

    EXACT = "exact"
    INFERRED = "inferred"


class DependencyEdge(BaseModel):
    """One dependency relationship between two source locations."""

    source: str  # file (relative) or qualified symbol
    target: str  # file (relative) or qualified symbol
    kind: str = "import"  # "import" | "references" | "inherits" | "calls"
    confidence: EdgeConfidence = EdgeConfidence.INFERRED
    detail: str = ""

    def to_prompt_line(self) -> str:
        return f"{self.source} -> {self.target} [{self.kind}/{self.confidence.value}]"


class DependencyGraph:
    """
    Dependency relationships derived from a :class:`RepositoryIndex`.

    Imports are EXACT; cross-file symbol references are INFERRED. The graph
    never fabricates certainty.
    """

    def __init__(self, index: RepositoryIndex) -> None:
        self.index = index

    # ------------------------------------------------------------------
    # Exact edges (from parsed imports)
    # ------------------------------------------------------------------

    def imports(self, rel_path: str) -> list[DependencyEdge]:
        """EXACT import edges of one file."""
        edges = self.index.get_imports(rel_path)
        return [
            DependencyEdge(
                source=rel_path,
                target=edge.target,
                kind="import",
                confidence=EdgeConfidence.EXACT,
                detail=edge.to_prompt_line(),
            )
            for edge in edges
        ]

    def dependents(self, rel_path: str) -> list[DependencyEdge]:
        """EXACT reverse edges: files that import *rel_path*."""
        files = self.index.get_dependents(rel_path)
        return [
            DependencyEdge(
                source=other,
                target=rel_path,
                kind="import",
                confidence=EdgeConfidence.EXACT,
            )
            for other in files
        ]

    # ------------------------------------------------------------------
    # Inferred edges (from reference occurrences)
    # ------------------------------------------------------------------

    def references_to(self, symbol_name: str) -> list[DependencyEdge]:
        """INFERRED: files/locations that reference *symbol_name*."""
        refs = self.index.find_references(symbol_name)
        return [
            DependencyEdge(
                source=ref.location.file,
                target=symbol_name,
                kind="references",
                confidence=EdgeConfidence.INFERRED,
                detail=f"{ref.location.line}: {ref.context[:120]}",
            )
            for ref in refs
        ]

    def callers_of(self, symbol_name: str) -> list[DependencyEdge]:
        """
        INFERRED callers of a function: reference sites in other files whose
        context actually contains a call-looking pattern (``name(``). Never
        claims exactness.
        """
        refs = self.index.find_references(symbol_name)
        edges: list[DependencyEdge] = []
        for ref in refs:
            if f"{symbol_name}(" not in ref.context:
                continue
            edges.append(
                DependencyEdge(
                    source=ref.location.file,
                    target=symbol_name,
                    kind="calls",
                    confidence=EdgeConfidence.INFERRED,
                    detail=f"{ref.location.line}: {ref.context[:120]}",
                )
            )
        return edges

    def inheritors_of(self, symbol_name: str) -> list[DependencyEdge]:
        """
        INFERRED subclasses: symbols whose parsed ``bases`` list contains
        *symbol_name* (exact base names from the parser, but the resolution
        of "who implements what" is only as good as the parser's base data).
        """
        edges: list[DependencyEdge] = []
        for symbol in self.index.find_symbol(symbol_name):
            if symbol_name in symbol.bases:
                edges.append(
                    DependencyEdge(
                        source=symbol.qualified_name,
                        target=symbol_name,
                        kind="inherits",
                        confidence=EdgeConfidence.INFERRED,
                        detail=symbol.location.to_prompt_line(),
                    )
                )
        # Walk every symbol with a matching base via a full scan (bounded).
        for name in self.index.all_symbol_names():
            for symbol in self.index.find_symbol(name):
                if symbol.bases and symbol_name in symbol.bases:
                    edges.append(
                        DependencyEdge(
                            source=symbol.qualified_name,
                            target=symbol_name,
                            kind="inherits",
                            confidence=EdgeConfidence.INFERRED,
                            detail=symbol.location.to_prompt_line(),
                        )
                    )
        # Deduplicate by (source, target, kind).
        seen: set[tuple[str, str, str]] = set()
        unique: list[DependencyEdge] = []
        for edge in edges:
            key = (edge.source, edge.target, edge.kind)
            if key not in seen:
                seen.add(key)
                unique.append(edge)
        return unique
