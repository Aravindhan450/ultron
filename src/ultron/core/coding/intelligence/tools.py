"""ultron.core.coding.intelligence.tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Registered tool wrappers over the CodeIntelligence facade (Fix #4).

These are the read-only tools the ReAct agent can call. Each one creates a
:class:`CodeIntelligence` for the requested path (defaulting to the allowed
base directory), refreshes the index incrementally, runs the query, and
returns a formatted string. All of them are classified LOW by the security
boundary and never modify anything.

Tools:

- ``code_search(query, path, ...)`` — lexical + regex search (gitignore aware)
- ``find_symbol(name, path)`` — all symbols with a name
- ``find_definition(name, path)`` — where a symbol is defined
- ``find_references(name, path)`` — usage sites of a symbol
- ``get_imports(file_path, path)`` — EXACT import edges of a file
- ``get_dependents(file_path, path)`` — files importing a file (EXACT)
- ``semantic_search(query, path)`` — metadata-rich semantic/lexical hits
- ``code_index_status(path)`` — index row counts

Paths are resolved through the shared path-safety gate, so a request for a
path outside the allowed workspace returns an error string.
"""

from __future__ import annotations

from ultron.core.coding.intelligence.facade import CodeIntelligence
from ultron.core.coding.workspace import _resolve_safe_path


def _intelligence(path: str) -> CodeIntelligence | None:
    """Builds a CodeIntelligence for *path* (None when outside the workspace).

    The index DB uses the shared per-workspace default location OUTSIDE the
    scanned repository (keyed by root hash in the allowed base dir), so a
    read-only search never drops files into the repository it inspects.
    """
    resolved = _resolve_safe_path(path or ".")
    if resolved is None:
        return None
    return CodeIntelligence(root=resolved)


def code_search(
    query: str,
    path: str = ".",
    max_results: int = 30,
    regex: bool = False,
    file_pattern: str | None = None,
    case_sensitive: bool = False,
) -> str:
    """Lexical/regex search respecting .gitignore and ignored directories.

    Multi-word queries ("coding executor") that miss lexically are retried
    against their normalized identifier spellings ("CodingExecutor"), so a
    natural-language symbol phrase still finds the source.
    """
    ci = _intelligence(path)
    if ci is None:
        return "Error: access denied, that directory is outside the allowed project folder."
    with ci:
        out = ci.search(
            query,
            max_results=max_results,
            regex=regex,
            file_pattern=file_pattern,
            case_sensitive=case_sensitive,
        )
        if not regex and not case_sensitive and out.startswith("No matches"):
            from ultron.core.coding.intelligence.resolve import normalize_symbol_phrase

            for candidate in normalize_symbol_phrase(query):
                if candidate == query or not candidate.strip():
                    continue
                normalized = ci.search(
                    candidate,
                    max_results=max_results,
                    regex=False,
                    file_pattern=file_pattern,
                    case_sensitive=False,
                )
                if not normalized.startswith(("No matches", "Error:")):
                    return f"Matches for '{candidate}' (normalized from '{query}'):\n{normalized}"
        return out


def find_symbol(name: str, path: str = ".") -> str:
    """Every symbol (any kind) named *name* across the indexed workspace.

    Resolution is case-tolerant and normalization-aware ("coding executor"
    resolves to ``CodingExecutor``); the canonical identifier is reported.
    """
    if not (name or "").strip():
        return "Error: find_symbol requires a non-empty 'name'."
    ci = _intelligence(path)
    if ci is None:
        return "Error: access denied, that directory is outside the allowed project folder."
    with ci:
        ci.refresh()
        from ultron.core.coding.intelligence.resolve import (
            format_symbol_result,
            resolve_symbol,
        )

        return format_symbol_result(resolve_symbol(ci, name.strip()))


def find_definition(name: str, path: str = ".") -> str:
    """The defining location(s) of *name* (class/function/... kinds).

    Evidence-grounded: an exact / case-insensitive / normalized index hit, or
    a verified source definition line, is reported as a found definition;
    otherwise the response says explicitly that no verified definition was
    found (never a filename-convention guess).
    """
    if not (name or "").strip():
        return "Error: find_definition requires a non-empty 'name'."
    ci = _intelligence(path)
    if ci is None:
        return "Error: access denied, that directory is outside the allowed project folder."
    with ci:
        ci.refresh()
        from ultron.core.coding.intelligence.resolve import (
            format_definition_result,
            resolve_definition,
        )

        return format_definition_result(resolve_definition(ci, name.strip()))


def find_references(name: str, path: str = ".") -> str:
    """Usage sites of *name* outside its definition (bounded).

    Case-tolerant and normalization-aware: ``taskstate`` finds references to
    ``TaskState`` rather than reporting zero results.
    """
    if not (name or "").strip():
        return "Error: find_references requires a non-empty 'name'."
    ci = _intelligence(path)
    if ci is None:
        return "Error: access denied, that directory is outside the allowed project folder."
    with ci:
        ci.refresh()
        from ultron.core.coding.intelligence.resolve import (
            format_reference_result,
            resolve_references,
        )

        return format_reference_result(resolve_references(ci, name.strip()))

def get_imports(file_path: str, path: str = ".") -> str:
    """EXACT import edges of *file_path* (relative to *path*)."""
    if not (file_path or "").strip():
        return "Error: get_imports requires a non-empty 'file_path'."
    ci = _intelligence(path)
    if ci is None:
        return "Error: access denied, that directory is outside the allowed project folder."
    with ci:
        ci.refresh()
        rel = file_path.strip().lstrip("./")
        edges = ci.get_imports(rel)
        if not edges:
            return f"No imports recorded for '{rel}'."
        lines = [f"Imports of {rel}:"]
        for edge in edges[:20]:
            lines.append(f"  - {edge.to_prompt_line()}")
        return "\n".join(lines)


def get_dependents(file_path: str, path: str = ".") -> str:
    """Files that import *file_path* (EXACT reverse-import edges)."""
    if not (file_path or "").strip():
        return "Error: get_dependents requires a non-empty 'file_path'."
    ci = _intelligence(path)
    if ci is None:
        return "Error: access denied, that directory is outside the allowed project folder."
    with ci:
        ci.refresh()
        rel = file_path.strip().lstrip("./")
        dependents = ci.get_dependents(rel)
        if not dependents:
            return f"No files import '{rel}'."
        lines = [f"Files importing {rel}:"]
        for dep in dependents[:20]:
            lines.append(f"  - {dep}")
        return "\n".join(lines)


def semantic_search(query: str, path: str = ".", top_k: int = 5) -> str:
    """Metadata-rich semantic search (degrades to lexical without embeddings)."""
    if not (query or "").strip():
        return "Error: semantic_search requires a non-empty 'query'."
    ci = _intelligence(path)
    if ci is None:
        return "Error: access denied, that directory is outside the allowed project folder."
    with ci:
        ci.refresh()
        hits = ci.search_semantically(query.strip(), top_k=top_k)
        if not hits:
            return f"No matches for '{query.strip()}'."
        lines = [f"Semantic matches for '{query.strip()}':"]
        for hit in hits[:top_k]:
            lines.append(f"  - {hit.to_prompt_line()}")
        return "\n".join(lines)


def code_investigation(query: str, path: str = ".") -> str:
    """Synthesized repository investigation (\"how does X work\" /
    \"where is X implemented\").

    Resolves a verified definition when one exists (exact -> case-insensitive
    -> normalized -> lexical), then reports the primary implementation,
    supporting components (imports/dependents), and relevant tests.  For
    conceptual subjects (\"command execution\") it falls back to ranked
    semantic evidence (src-first).  Never guesses a file path.
    """
    if not (query or "").strip():
        return "Error: code_investigation requires a non-empty 'query'."
    ci = _intelligence(path)
    if ci is None:
        return "Error: access denied, that directory is outside the allowed project folder."
    with ci:
        ci.refresh()
        from ultron.core.coding.intelligence.resolve import (
            format_investigation_result,
            resolve_investigation,
        )

        return format_investigation_result(resolve_investigation(ci, query.strip()))


def code_index_status(path: str = ".") -> str:
    """Index row counts for *path* (rescans only when the index is empty)."""
    ci = _intelligence(path)
    if ci is None:
        return "Error: access denied, that directory is outside the allowed project folder."
    with ci:
        # Pure status if the index already has rows; first call indexes once
        # so a fresh workspace reports real counts instead of zeros.
        if ci.index_status().files == 0:
            ci.refresh()
        return ci.index_status_line()


def report_file(file_path: str, path: str = ".") -> str:
    """Symbols, imports and dependents of one file (targeted context)."""
    if not (file_path or "").strip():
        return "Error: report_file requires a non-empty 'file_path'."
    ci = _intelligence(path)
    if ci is None:
        return "Error: access denied, that directory is outside the allowed project folder."
    with ci:
        ci.refresh()
        rel = file_path.strip().lstrip("./")
        return ci.report_file(rel)


def report_symbol(name: str, path: str = ".") -> str:
    """Combined definition + references report for one symbol.

    Resolution-aware (case-tolerant + normalization), evidence-grounded:
    never guesses a definition from a filename convention.
    """
    if not (name or "").strip():
        return "Error: report_symbol requires a non-empty 'name'."
    ci = _intelligence(path)
    if ci is None:
        return "Error: access denied, that directory is outside the allowed project folder."
    with ci:
        ci.refresh()
        from ultron.core.coding.intelligence.resolve import (
            format_definition_result,
            format_reference_result,
            resolve_definition,
            resolve_references,
        )

        definition_block = format_definition_result(resolve_definition(ci, name.strip()))
        reference_block = format_reference_result(resolve_references(ci, name.strip()))
        return f"{definition_block}\n\n{reference_block}"
