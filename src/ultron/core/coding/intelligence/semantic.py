"""ultron.core.coding.intelligence.semantic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Semantic search foundation (Fix #4).

This layer provides the metadata-rich chunk representation and the
search contract, WITHOUT hard-coding any embedding provider. An
:class:`Embedder` protocol is the seam: implementations can plug in local
embeddings (Ollama, MLX, ...) later without touching the facade.

Key rules:

- **Metadata is the payload**: every result carries repository, file,
  symbol, language, line range and chunk text so the coding agent can
  locate the ACTUAL source. The file remains authoritative — embeddings are
  only a retrieval hint, never the source of truth.
- **Graceful degradation**: with no embedder configured,
  :meth:`SemanticSearch.search` falls back to deterministic lexical
  matching over the same metadata and labels the mode
  ``lexical_fallback`` so callers know it is not semantic.

Chunks are derived from the repository index's symbols (one chunk per
symbol: name, kind, location, signature) plus a bounded preview of the
defining line. No full-file embedding is performed at this stage.
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel

from ultron.core.coding.intelligence.index import RepositoryIndex
from ultron.core.coding.intelligence.symbols import SymbolKind


class Embedder(Protocol):
    """Any local embedding provider (Ollama / MLX / on-disk model)."""

    def available(self) -> bool: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class CodeChunk(BaseModel):
    """One retrievable unit of code with full source-locating metadata."""

    repository: str = ""
    file: str = ""
    symbol: str = ""
    symbol_kind: str = "unknown"
    language: str = ""
    line_start: int = 1
    line_end: int = 1
    chunk: str = ""

    def to_prompt_line(self) -> str:
        return f"{self.file}:{self.line_start}-{self.line_end} {self.symbol} ({self.symbol_kind})"


class SemanticHit(BaseModel):
    """One semantic (or lexical-fallback) retrieval result."""

    chunk: CodeChunk
    score: float = 0.0
    mode: str = "lexical_fallback"  # "semantic" when an embedder ran

    def to_prompt_line(self) -> str:
        return f"[{self.mode} {self.score:.2f}] {self.chunk.to_prompt_line()}"


class SemanticSearch:
    """
    Metadata-rich chunk search over a :class:`RepositoryIndex`.

    With an embedder configured (and available) ``search`` embeds the query
    and returns cosine-ranked hits labelled ``semantic``; otherwise it
    returns deterministic substring-ranked hits labelled
    ``lexical_fallback``. Every hit carries file/line metadata.
    """

    def __init__(
        self,
        index: RepositoryIndex,
        embedder: Embedder | None = None,
        *,
        max_chunks: int = 2000,
    ) -> None:
        self.index = index
        self.embedder = embedder
        self.max_chunks = max_chunks
        self._chunks: list[CodeChunk] | None = None

    # ------------------------------------------------------------------
    # Chunk materialization (from the index — never raw repo dumps)
    # ------------------------------------------------------------------

    def _materialize(self) -> list[CodeChunk]:
        if self._chunks is not None:
            return self._chunks
        chunks: list[CodeChunk] = []
        for name in self.index.all_symbol_names():
            for symbol in self.index.find_symbol(name):
                if symbol.kind in {
                    SymbolKind.IMPORT,
                    SymbolKind.VARIABLE,
                    SymbolKind.MODULE,
                }:
                    continue
                if len(chunks) >= self.max_chunks:
                    break
                chunks.append(
                    CodeChunk(
                        repository=str(self.index.root),
                        file=symbol.location.file,
                        symbol=symbol.qualified_name,
                        symbol_kind=symbol.kind.value,
                        language=symbol.language,
                        line_start=symbol.location.line,
                        line_end=symbol.location.end_line or symbol.location.line,
                        chunk=symbol.signature or symbol.name,
                    )
                )
            if len(chunks) >= self.max_chunks:
                break
        self._chunks = chunks
        return chunks

    def invalidate(self) -> None:
        """Forgets the cached chunks (call after index.refresh())."""
        self._chunks = None

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 5) -> list[SemanticHit]:
        """Returns the best-matching chunks with source-locating metadata."""
        query = (query or "").strip()
        if not query:
            return []
        chunks = self._materialize()

        if self.embedder is not None and self.embedder.available():
            try:
                return self._semantic_search(chunks, query, top_k)
            except Exception:  # noqa: BLE001 — embedder failure degrades
                return self._lexical_search(chunks, query, top_k)

        return self._lexical_search(chunks, query, top_k)

    # -- internals -----------------------------------------------------

    def _semantic_search(
        self, chunks: list[CodeChunk], query: str, top_k: int
    ) -> list[SemanticHit]:
        query_vec = self.embedder.embed([query])[0]  # type: ignore[union-attr]
        texts = [f"{c.symbol} {c.chunk}" for c in chunks]
        vectors = self.embedder.embed(texts)  # type: ignore[union-attr]
        scored = []
        for chunk, vector in zip(chunks, vectors):
            score = _cosine(query_vec, vector)
            scored.append(SemanticHit(chunk=chunk, score=score, mode="semantic"))
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:top_k]

    def _lexical_search(
        self, chunks: list[CodeChunk], query: str, top_k: int
    ) -> list[SemanticHit]:
        terms = [t for t in query.lower().split() if len(t) >= 3]
        if not terms:
            return []
        scored: list[SemanticHit] = []
        for chunk in chunks:
            haystack = f"{chunk.symbol} {chunk.chunk}".lower()
            score = sum(1 for term in terms if term in haystack)
            if score:
                scored.append(SemanticHit(chunk=chunk, score=score))
        scored.sort(key=lambda h: (h.score, h.chunk.file), reverse=True)
        return scored[:top_k]


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
