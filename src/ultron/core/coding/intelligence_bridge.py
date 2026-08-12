"""ultron.core.coding.intelligence_bridge
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The bridge between the CodingExecutor (Fix #3) and the CodeIntelligence
facade (Fix #4).

The facade is deliberately NOT a pydantic model (it owns a SQLite
connection, an LSP manager and an embedder). The bridge is: a small
serializable pydantic model that lives on :class:`CodeContext`, storing
only the root/db path and an observability log, and constructing the real
:class:`CodeIntelligence` LAZILY (and rebuilding it after a
``model_dump_json`` round-trip, e.g. across a confirmation boundary).

Responsibilities:

- **Tool selection strategy** — :meth:`resolve_symbol` applies the
  "cheapest reliable layer first" rule: exact definition (L3) -> references
  (L3) -> semantic (L5, degrades to lexical) -> lexical search (L2).
- **Targeted context retrieval** — :meth:`context_block` extracts candidate
  symbols from the current plan step/goal and returns a BOUNDED block of
  definitions/references so the model never receives a repository dump.
- **Observability** — every query is recorded as an :class:`IntelligenceQuery`
  (operation, query, layer, hit count, duration) and summarized by
  :meth:`usage_summary` for verification evidence.
- **Index staleness** — :meth:`mark_dirty` is called after edits; the next
  bridge query refreshes the (incremental) index first, so the executor
  never reasons from stale source information.

The bridge never executes tools and never bypasses security — it is a
read-only query layer. The registered tools remain the model's primary
interface; this bridge is the executor's deterministic access point.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, PrivateAttr

from ultron.core.coding.intelligence.facade import CodeIntelligence
from ultron.core.coding.workspace import _resolve_safe_path

# The Ultron repository root (src/ultron/__init__.py -> parents[0]=ultron,
# [1]=src, [2]=repo). The bridge refuses to auto-index Ultron's own repo —
# unit tests run from it and must stay fast and side-effect free.
_ULTRON_ROOT = Path(__file__).resolve().parents[4]

# Common capitalized words that are not code symbols (filter for the
# candidate extractor). Bounded; cheap false-positive reduction only.
_STOPWORDS = frozenset(
    {
        "the", "this", "that", "what", "when", "where", "which", "how",
        "fix", "add", "create", "find", "run", "make", "use", "our",
        "your", "its", "would", "should", "could", "with", "from", "into",
        "after", "before", "then", "than", "and", "for", "are", "was",
        "not", "but", "all", "any", "each", "every", "one", "two", "new",
        "existing", "ensure", "verify", "review", "inspect", "explain",
        "purpose", "outcome", "overview", "context", "detail", "section",
        "trace", "flow", "works", "step", "next", "current", "overall",
        "complete", "perform", "carry", "out", "via", "through",
        "have", "has", "been", "being", "does", "doing", "done",
    }
)

_PASCAL = re.compile(r"\b[A-Z][A-Za-z0-9_]{1,}\b")
_QUOTED = re.compile(r"['\"]([A-Za-z_][A-Za-z0-9_.]*)['\"]")
_LOWERCASE = re.compile(r"\b[a-z][a-z0-9_]{3,}\b")  # words + snake_case + digits


class IntelligenceQuery(BaseModel):
    """One recorded code-intelligence query (observability)."""

    operation: str
    query: str = ""
    layer: str = ""
    hits: int = 0
    duration_ms: float = 0.0

    def to_prompt_line(self) -> str:
        return (
            f"{self.operation}('{self.query or '-'}') -> {self.layer or '?'} "
            f"x{self.hits} ({self.duration_ms:.0f}ms)"
        )


def _candidate_symbols(task: Any) -> list[str]:
    """Extracts plausible symbol names from a task's goal + current plan step.

    Deterministic and bounded. Two passes over the goal/step text, in order
    of code-likeness:

    1. PascalCase identifiers (``UserService``), plus quoted identifiers
       (``'route_login'``);
    2. lowercase tokens (``route_login``, ``token_for``, ``authentication``,
       ``service``) so natural-language steps like "Trace the authentication
       flow" still yield resolvable candidates.

    Deduplicated case-insensitively, filtered against a small stopword list,
    capped at 6 candidates.
    """
    texts = [str(getattr(task, "goal", "") or "")]
    step = getattr(task, "current_plan_step", None)
    if callable(step):
        try:
            step = step()
        except (TypeError, ValueError):
            step = None
    if step is not None:
        for attr in ("description", "purpose", "expected_outcome"):
            texts.append(str(getattr(step, attr, "") or ""))

    candidates: list[str] = []
    seen: set[str] = set()

    def _add(match: str) -> None:
        key = match.lower()
        if key in _STOPWORDS or key in seen:
            return
        seen.add(key)
        candidates.append(match)

    for text in texts:
        for match in _PASCAL.findall(text):
            _add(match)
            if len(candidates) >= 6:
                return candidates
        for match in _QUOTED.findall(text):
            _add(match)
            if len(candidates) >= 6:
                return candidates
        for match in _LOWERCASE.findall(text):
            _add(match)
            if len(candidates) >= 6:
                return candidates
    return candidates


def _fuzzy_symbol_names(ci: CodeIntelligence, term: str, limit: int = 4) -> list[str]:
    """Symbol names related to *term*: substring or shared-prefix match.

    Connects natural-language terms to indexed definitions deterministically
    (no embeddings needed): ``authentication`` -> ``authenticate`` via a
    shared prefix (stemming-lite), ``service`` -> ``UserService`` via
    substring. Scans the index's definition-name table once — cheap for
    bounded candidate sets and only invoked from the targeted context path.
    """
    term = (term or "").strip().lower()
    if len(term) < 3:
        return []
    names = ci.index.all_symbol_names()
    scored: list[tuple[int, str]] = []
    for name in names:
        n_lower = name.lower()
        if term == n_lower:
            scored.append((0, name))
        elif term in n_lower or n_lower in term:
            scored.append((1, name))
        else:
            shared = 0
            for a, b in zip(term, n_lower):
                if a != b:
                    break
                shared += 1
            if shared >= 4:
                scored.append((2, name))
    scored.sort(key=lambda pair: (pair[0], len(pair[1])))
    seen: set[str] = set()
    out: list[str] = []
    for _, name in scored:
        if name in seen:
            continue
        seen.add(name)
        out.append(name)
        if len(out) >= limit:
            break
    return out


class CodeIntelligenceBridge(BaseModel):
    """Lazy, serializable bridge from the executor to CodeIntelligence."""

    root: str | None = None
    enabled: bool = False
    dirty: bool = False  # files changed since the last index refresh
    queries: list[IntelligenceQuery] = Field(default_factory=list)
    max_queries: int = 200

    _ci: CodeIntelligence | None = PrivateAttr(default=None)
    _indexed: bool = PrivateAttr(default=False)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def enable(self, root: str | Path | None) -> bool:
        """Enables the bridge for *root* (path-safe, never Ultron itself).

        Returns False (silently disabling the bridge — NOT an error) when the
        root is outside ``ALLOWED_BASE_DIR`` (e.g. ``discover_workspace``
        found a project root above the allowed base), is Ultron's own
        repository, or is otherwise unusable. The executor simply degrades to
        the plain exploration guidance in that case.
        """
        if not root:
            return False
        resolved = _resolve_safe_path(str(root))
        if resolved is None:
            return False
        resolved = resolved.resolve()
        if resolved == _ULTRON_ROOT or _ULTRON_ROOT in resolved.parents:
            return False
        self._close_ci()
        self.root = str(resolved)
        self.enabled = True
        self.dirty = False
        self._indexed = False
        return True

    def close(self) -> None:
        """Closes the underlying CodeIntelligence (sqlite connection)."""
        self._close_ci()

    def _close_ci(self) -> None:
        if self._ci is not None:
            try:
                self._ci.close()
            except Exception:  # noqa: BLE001, S110 — best-effort close
                pass
            self._ci = None

    # ------------------------------------------------------------------
    # Index access + staleness
    # ------------------------------------------------------------------

    def _build(self) -> CodeIntelligence | None:
        """Lazily constructs the facade (None when not enabled)."""
        if not self.enabled or not self.root:
            return None
        if self._ci is None:
            # db_path defaults to the shared per-workspace location OUTSIDE
            # the scanned repository (see RepositoryIndex default).
            self._ci = CodeIntelligence(root=self.root)
        return self._ci

    def _ensure_fresh(self) -> CodeIntelligence | None:
        """Builds the facade, refreshing when the index is dirty OR never built."""
        ci = self._build()
        if ci is None:
            return None
        if self.dirty or not self._indexed:
            try:
                ci.refresh()
                self._indexed = True
            finally:
                self.dirty = False
        return ci

    def refresh(self) -> str:
        """Forces an incremental index refresh; returns the summary line."""
        ci = self._build()
        if ci is None:
            return ""
        summary = ci.refresh()
        self._indexed = True
        self.dirty = False
        self._record("refresh", "", "index", 0, 0.0)
        return summary.to_prompt_line()

    def mark_dirty(self) -> None:
        """Flags the index stale (call after a file modification)."""
        if self.enabled:
            self.dirty = True

    def _record(
        self, operation: str, query: str, layer: str, hits: int, duration_ms: float
    ) -> None:
        self.queries.append(
            IntelligenceQuery(
                operation=operation,
                query=query,
                layer=layer,
                hits=hits,
                duration_ms=duration_ms,
            )
        )
        if len(self.queries) > self.max_queries:
            self.queries = self.queries[-self.max_queries :]

    # ------------------------------------------------------------------
    # Query dispatch (tool-friendly strings)
    # ------------------------------------------------------------------

    def query(self, operation: str, **kwargs: Any) -> str:
        """Dispatches one intelligence operation; returns a formatted string.

        Supported operations mirror the registered tools:
        ``find_definition``, ``find_symbol``, ``find_references``,
        ``semantic_search``, ``code_search``, ``get_imports``,
        ``get_dependents``, ``report_symbol``, ``report_file``.
        """
        ci = self._ensure_fresh()
        if ci is None:
            return "Error: code intelligence is not enabled for this workspace."
        started = time.monotonic()
        name = str(kwargs.get("name") or kwargs.get("query") or "").strip()
        layer = "symbol"
        hits = 0

        if operation == "find_definition":
            out = self._format_definitions(ci, name)
            hits = _count_result_lines(out, "Definitions of")
        elif operation == "find_symbol":
            out = self._format_symbols(ci, name)
            hits = _count_result_lines(out, "Symbols named")
        elif operation == "find_references":
            out = self._format_references(ci, name)
            hits = _count_result_lines(out, "References to")
        elif operation == "semantic_search":
            semantic_hits = ci.search_semantically(name, top_k=int(kwargs.get("top_k", 5)))
            layer = semantic_hits[0].mode if semantic_hits else "semantic"
            out = _format_hits("Semantic matches", name, semantic_hits)
            hits = len(semantic_hits)
        elif operation == "code_search":
            out = ci.search(
                name,
                max_results=int(kwargs.get("max_results", 30)),
                regex=bool(kwargs.get("regex", False)),
                file_pattern=kwargs.get("file_pattern"),
                case_sensitive=bool(kwargs.get("case_sensitive", False)),
            )
            layer = "lexical"
            hits = len(out.splitlines()) if out and not out.startswith(("No ", "Error:")) else 0
        elif operation == "get_imports":
            rel = str(kwargs.get("file_path", "")).lstrip("./")
            edges = ci.get_imports(rel)
            layer = "dependency"
            out = _format_edges("Imports", rel, edges) if edges else f"No imports recorded for '{rel}'."
            hits = len(edges)
        elif operation == "get_dependents":
            rel = str(kwargs.get("file_path", "")).lstrip("./")
            deps = ci.get_dependents(rel)
            layer = "dependency"
            out = (
                "\n".join([f"Files importing {rel}:"] + [f"  - {d}" for d in deps[:20]])
                if deps
                else f"No files import '{rel}'."
            )
            hits = len(deps)
        elif operation == "report_symbol":
            out = ci.report_symbol(name)
            hits = _count_result_lines(out, "References")
        elif operation == "report_file":
            out = ci.report_file(str(kwargs.get("file_path", "")).lstrip("./"))
            layer = "file"
            hits = len(out.splitlines()) if not out.startswith("No index data") else 0
        else:
            return f"Error: unknown intelligence operation '{operation}'."

        self._record(operation, name or str(kwargs.get("file_path", "")), layer, hits, (time.monotonic() - started) * 1000)
        return out

    # -- formatting helpers (mirror the registered tools) -------------

    def _format_definitions(self, ci: CodeIntelligence, name: str) -> str:
        if not name:
            return "Error: a symbol 'name' is required."
        definitions = ci.find_definition(name)
        if not definitions:
            return f"No definition found for '{name}' in the index."
        lines = [f"Definitions of '{name}':"]
        for symbol in definitions[:10]:
            lines.append(f"  - {symbol.to_prompt_line()}")
        return "\n".join(lines)

    def _format_symbols(self, ci: CodeIntelligence, name: str) -> str:
        if not name:
            return "Error: a symbol 'name' is required."
        symbols = ci.find_symbol(name)
        if not symbols:
            return f"No symbol named '{name}' found in the index."
        lines = [f"Symbols named '{name}':"]
        for symbol in symbols[:20]:
            lines.append(f"  - {symbol.to_prompt_line()}")
        return "\n".join(lines)

    def _format_references(self, ci: CodeIntelligence, name: str) -> str:
        if not name:
            return "Error: a symbol 'name' is required."
        references = ci.find_references(name)
        if not references:
            return f"No references found for '{name}'."
        lines = [f"References to '{name}' ({len(references)} found, showing up to 20):"]
        for ref in references[:20]:
            lines.append(f"  - {ref.to_prompt_line()}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Tool selection ladder + targeted retrieval
    # ------------------------------------------------------------------

    def resolve_symbol(self, name: str, max_lines: int = 8) -> str:
        """Most precise layer first: definition -> references -> semantic -> lexical."""
        ci = self._ensure_fresh()
        if ci is None:
            return ""
        started = time.monotonic()
        name = name.strip()

        definitions = ci.find_definition(name)
        if definitions:
            lines = [f"'{name}' defined:"]
            for symbol in definitions[:3]:
                lines.append(f"  - {symbol.to_prompt_line()}")
            references = ci.find_references(name)
            if references:
                lines.append(f"References ({len(references)} found, up to {max_lines}):")
                for ref in references[:max_lines]:
                    lines.append(f"  - {ref.to_prompt_line()}")
            self._record("resolve_symbol", name, "symbol", len(definitions) + len(references), (time.monotonic() - started) * 1000)
            return "\n".join(lines)

        references = ci.find_references(name)
        if references:
            lines = [f"No definition found; '{name}' referenced:"]
            for ref in references[:max_lines]:
                lines.append(f"  - {ref.to_prompt_line()}")
            self._record("resolve_symbol", name, "symbol", len(references), (time.monotonic() - started) * 1000)
            return "\n".join(lines)

        # Fuzzy symbol-name match (substring/shared-prefix) — connects
        # natural-language terms to indexed definitions deterministically
        # before paying for semantic/lexical scans. Up to 2 definitions per
        # matched name so duplicate-symbol ambiguity (auth UserService vs
        # billing UserService) stays surfaced, never silently collapsed.
        fuzzy = _fuzzy_symbol_names(ci, name, limit=3)
        if fuzzy:
            lines = [f"'{name}' (related symbols):"]
            for symbol_name in fuzzy:
                for symbol in ci.find_definition(symbol_name)[:2]:
                    lines.append(f"  - {symbol.to_prompt_line()}")
            self._record("resolve_symbol", name, "symbol", len(lines) - 1, (time.monotonic() - started) * 1000)
            return "\n".join(lines)

        hits = ci.search_semantically(name, top_k=3)
        if hits:
            lines = [f"'{name}' (semantic/lexical matches):"]
            for hit in hits:
                lines.append(f"  - {hit.to_prompt_line()}")
            self._record("resolve_symbol", name, hits[0].mode, len(hits), (time.monotonic() - started) * 1000)
            return "\n".join(lines)

        lexical = ci.search(name, max_results=4)
        if lexical and not lexical.startswith("No matches"):
            indented = "\n".join("  " + line for line in lexical.splitlines()[:4])
            self._record("resolve_symbol", name, "lexical", 4, (time.monotonic() - started) * 1000)
            return f"'{name}' (lexical matches):\n{indented}"

        self._record("resolve_symbol", name, "none", 0, (time.monotonic() - started) * 1000)
        return ""

    def context_block(self, task: Any, max_candidates: int = 6, max_lines: int = 24) -> str:
        """Bounded targeted retrieval for the current step + goal.

        Extracts candidate symbols and resolves each with the cheapest
        reliable layer. Returns '' when nothing relevant is found (no
        repository dumps, ever).
        """
        ci = self._ensure_fresh()
        if ci is None:
            return ""
        candidates = _candidate_symbols(task)
        if not candidates:
            return ""

        lines = ["TARGETED CODE CONTEXT (from the code index):"]
        for name in candidates[:max_candidates]:
            block = self.resolve_symbol(name, max_lines=6)
            if not block:
                continue
            for line in block.splitlines()[:7]:
                lines.append("  " + line)
            if len(lines) >= max_lines:
                break
        return "\n".join(lines) if len(lines) > 1 else ""

    def usage_summary(self) -> str:
        """Compact observability summary for verification evidence."""
        if not self.queries:
            return "no code-intelligence queries yet"
        counts: dict[str, int] = {}
        for query in self.queries:
            counts[query.layer] = counts.get(query.layer, 0) + 1
        parts = ", ".join(f"{layer} x{n}" for layer, n in sorted(counts.items()))
        return f"{len(self.queries)} query(s) — {parts}"


def _format_hits(title: str, query: str, hits: list) -> str:
    if not hits:
        return f"No matches for '{query}'."
    lines = [f"{title} for '{query}':"]
    for hit in hits:
        lines.append(f"  - {hit.to_prompt_line()}")
    return "\n".join(lines)


def _format_edges(title: str, target: str, edges: list) -> str:
    lines = [f"{title} of {target}:"]
    for edge in edges[:20]:
        lines.append(f"  - {edge.to_prompt_line()}")
    return "\n".join(lines)


def _count_result_lines(text: str, marker: str) -> int:
    """Counts result rows under a marker header in a formatted output."""
    if marker not in text:
        return 0
    return sum(1 for line in text.splitlines() if line.startswith("  - "))
