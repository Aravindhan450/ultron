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
    """Lexical/regex search respecting .gitignore and ignored directories."""
    ci = _intelligence(path)
    if ci is None:
        return "Error: access denied, that directory is outside the allowed project folder."
    with ci:
        return ci.search(
            query,
            max_results=max_results,
            regex=regex,
            file_pattern=file_pattern,
            case_sensitive=case_sensitive,
        )


def find_symbol(name: str, path: str = ".") -> str:
    """Every symbol (any kind) named *name* across the indexed workspace."""
    if not (name or "").strip():
        return "Error: find_symbol requires a non-empty 'name'."
    ci = _intelligence(path)
    if ci is None:
        return "Error: access denied, that directory is outside the allowed project folder."
    with ci:
        ci.refresh()
        symbols = ci.find_symbol(name.strip())
        if not symbols:
            return f"No symbol named '{name.strip()}' found in the index."
        lines = [f"Symbols named '{name.strip()}':"]
        for symbol in symbols[:20]:
            lines.append(f"  - {symbol.to_prompt_line()}")
        return "\n".join(lines)


def find_definition(name: str, path: str = ".") -> str:
    """The defining location(s) of *name* (class/function/... kinds)."""
    if not (name or "").strip():
        return "Error: find_definition requires a non-empty 'name'."
    ci = _intelligence(path)
    if ci is None:
        return "Error: access denied, that directory is outside the allowed project folder."
    with ci:
        ci.refresh()
        definitions = ci.find_definition(name.strip())
        if not definitions:
            return f"No definition found for '{name.strip()}' in the index."
        lines = [f"Definitions of '{name.strip()}':"]
        for symbol in definitions[:10]:
            lines.append(f"  - {symbol.to_prompt_line()}")
        return "\n".join(lines)


def find_references(name: str, path: str = ".") -> str:
    """Usage sites of *name* outside its definition (bounded)."""
    if not (name or "").strip():
        return "Error: find_references requires a non-empty 'name'."
    ci = _intelligence(path)
    if ci is None:
        return "Error: access denied, that directory is outside the allowed project folder."
    with ci:
        ci.refresh()
        references = ci.find_references(name.strip())
        if not references:
            return f"No references found for '{name.strip()}'."
        lines = [
            f"References to '{name.strip()}' ({len(references)} found, showing up to 20):"
        ]
        for ref in references[:20]:
            lines.append(f"  - {ref.to_prompt_line()}")
        return "\n".join(lines)


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
    """Combined definition + references report for one symbol."""
    if not (name or "").strip():
        return "Error: report_symbol requires a non-empty 'name'."
    ci = _intelligence(path)
    if ci is None:
        return "Error: access denied, that directory is outside the allowed project folder."
    with ci:
        ci.refresh()
        return ci.report_symbol(name.strip())
