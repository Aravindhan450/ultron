"""ultron.core.coding.intelligence.search
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Filesystem + lexical search for the code intelligence layer (Fix #4).

This is the L1/L2 layer: reliable, bounded repository search that respects
``.gitignore`` plus the same sensible exclusions the workspace tools already
use (``.git``, ``node_modules``, ``.venv``, ``__pycache__``, ``dist``,
``build``, ...). It supports:

- file-name and file-extension matching (``file_pattern``)
- plain substring and regular-expression content search
- directory confinement through the existing path-safety gate
- deterministic, sorted, bounded output (``path:line: content``)

The ``.gitignore`` support is a pragmatic subset (comments, negation,
anchored and unanchored patterns, ``*`` / ``?`` / ``**`` / character
classes, directory-only patterns) — enough to respect the overwhelming
majority of real-world ignore files without a new dependency.

This module deliberately does NOT index anything: it is the stateless,
cheap "scan now" layer. The repository index (``index.py``) is the
incremental layer built on top of it for symbol queries.
"""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path

from ultron.core.coding.workspace import (
    _IGNORED_DIRS,
    _resolve_safe_path,
)

MAX_SEARCH_FILE_BYTES = 512 * 1024


# ---------------------------------------------------------------------------
# .gitignore parsing (pragmatic subset)
# ---------------------------------------------------------------------------


class GitIgnoreRules:
    """
    Lightweight ``.gitignore`` matcher.

    Rules are collected from ``<root>/.gitignore`` (plus optional nested
    ``.gitignore`` files) and evaluated per path. Later rules override
    earlier ones (git semantics). ``!`` negations re-include a path.
    """

    def __init__(self, root: str | Path, *, max_rules: int = 2000) -> None:
        self.root = Path(root)
        self._rules: list[tuple[re.Pattern[str], bool, bool]] = []  # (regex, negate, dir_only)
        for gitignore in _collect_gitignores(self.root):
            base = gitignore.parent.relative_to(self.root)
            self._load(gitignore, base)

    def _load(self, gitignore: Path, base: Path) -> None:
        try:
            if gitignore.stat().st_size > 128 * 1024:
                return
            lines = gitignore.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            negate = stripped.startswith("!")
            if negate:
                stripped = stripped[1:].strip()
            if not stripped:
                continue
            dir_only = stripped.endswith("/")
            if dir_only:
                stripped = stripped.rstrip("/")
            anchored = stripped.startswith("/")
            if anchored:
                stripped = stripped.lstrip("/")
            # A pattern with a slash is anchored to the .gitignore's own
            # directory; a pattern without one matches at any depth BELOW
            # that directory. Nested .gitignore files are prefixed with their
            # base so repo-relative paths match correctly.
            regex = _gitignore_pattern_to_regex(stripped, anchored or "/" in stripped, base)
            self._rules.append((regex, negate, dir_only))

    def is_ignored(self, rel_path: str, *, is_dir: bool = False) -> bool:
        """
        True when *rel_path* (relative to the root) is ignored by the rules.

        Later rules win; a negation that matches last un-ignores the path.
        """
        rel_path = rel_path.replace("\\", "/").lstrip("./")
        if not rel_path:
            return False
        ignored = False
        for regex, negate, dir_only in self._rules:
            if dir_only and not is_dir:
                continue
            if regex.search(rel_path):
                ignored = not negate
        return ignored


def _collect_gitignores(root: Path) -> list[Path]:
    """Root .gitignore plus nested ones (bounded walk, sorted)."""
    found = []
    root_gi = root / ".gitignore"
    if root_gi.is_file():
        found.append(root_gi)
    for gi in sorted(root.rglob(".gitignore")):
        if gi != root_gi and gi.is_file():
            found.append(gi)
    return found[:100]  # bound


def _gitignore_pattern_to_regex(
    pattern: str, anchored: bool, base: Path | None = None
) -> re.Pattern[str]:
    """Converts a single gitignore glob into a compiled regex."""
    # Build a glob-style regex: ** crosses directories, * doesn't.
    out: list[str] = []
    i = 0
    n = len(pattern)
    while i < n:
        ch = pattern[i]
        if ch == "*":
            if i + 1 < n and pattern[i + 1] == "*":
                # '**' — across directories; eat a following '/'.
                if i + 2 < n and pattern[i + 2] == "/":
                    out.append("(?:.*/)?")
                    i += 3
                    continue
                out.append(".*")
                i += 2
                continue
            out.append("[^/]*")
            i += 1
        elif ch == "?":
            out.append("[^/]")
            i += 1
        elif ch == "[":
            end = pattern.find("]", i + 1)
            if end == -1:
                out.append(re.escape(ch))
                i += 1
            else:
                out.append(pattern[i : end + 1])
                i = end + 1
        else:
            out.append(re.escape(ch))
            i += 1
    body = "".join(out)
    if base is not None and str(base) != ".":
        # Nested .gitignore: patterns are relative to *base*, and unanchored
        # patterns match at any depth below it (git semantics).
        prefix = re.escape(str(base).replace("\\", "/")) + "/"
        if anchored:
            regex = rf"^{prefix}{body}(?:/.*)?$"
        else:
            regex = rf"^{prefix}(?:.*/)?{body}(?:/.*)?$"
    elif anchored:
        regex = rf"^{body}(?:/.*)?$"
    else:
        regex = rf"(?:^|/){body}(?:/.*)?$"
    return re.compile(regex)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def _matches_file_pattern(filename: str, file_pattern: str | None) -> bool:
    if not file_pattern:
        return True
    return fnmatch.fnmatch(filename, file_pattern) or fnmatch.fnmatch(
        filename, f"*{file_pattern}*"
    )


def search_code(
    query: str,
    path: str = ".",
    max_results: int = 30,
    regex: bool = False,
    file_pattern: str | None = None,
    case_sensitive: bool = False,
) -> str:
    """
    Searches *path* for files whose name or content matches *query*.

    - Plain queries are case-insensitive substring matches unless
      ``case_sensitive`` is set.
    - ``regex=True`` compiles *query* as a regular expression.
    - ``file_pattern`` filters candidate files by name glob.
    - Respects ``.gitignore`` and the shared ignored-directory set.
    - Bounded: files over 512 KB are skipped; matches capped at
      ``max_results``; one match line per file (plus filename matches).

    Returns ``relative/path:line: content`` lines (filename matches are
    reported as ``path: (filename match)``); errors return an
    ``"Error: ..."`` string.
    """
    query = (query or "").strip()
    if not query:
        return "Error: search_code requires a non-empty 'query'."
    resolved = _resolve_safe_path(path)
    if resolved is None:
        return "Error: access denied, that directory is outside the allowed project folder."
    if not resolved.is_dir():
        return f"Error: {path} is not a directory"

    ignore = GitIgnoreRules(resolved)
    if regex:
        try:
            if case_sensitive:
                needle: re.Pattern[str] | None = re.compile(query)
            else:
                needle = re.compile(query, re.IGNORECASE)
        except re.error as exc:
            return f"Error: invalid regular expression: {exc}"
    else:
        needle = None
        plain = query if case_sensitive else query.lower()

    results: list[str] = []

    def walk(directory: Path) -> None:
        if len(results) >= max_results:
            return
        try:
            entries = sorted(directory.iterdir(), key=lambda e: e.name.lower())
        except (OSError, ValueError):
            return
        for entry in entries:
            if len(results) >= max_results:
                return
            name = entry.name
            if entry.is_dir():
                rel = str(entry.relative_to(resolved))
                if name in _IGNORED_DIRS or name.startswith("."):
                    continue
                if ignore.is_ignored(rel, is_dir=True):
                    continue
                walk(entry)
                continue
            if name.startswith("."):
                continue
            rel = str(entry.relative_to(resolved))
            if ignore.is_ignored(rel):
                continue
            if not _matches_file_pattern(name, file_pattern):
                continue
            if needle is None:
                if plain in name.lower():
                    results.append(f"{rel}: (filename match)")
                    continue
            elif needle.search(name):
                results.append(f"{rel}: (filename match)")
                continue
            try:
                if entry.stat().st_size > MAX_SEARCH_FILE_BYTES:
                    continue
                content = entry.read_text(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                continue
            for line_no, line in enumerate(content.splitlines(), start=1):
                if needle is None:
                    haystack = line if case_sensitive else line.lower()
                    if plain in haystack:
                        results.append(f"{rel}:{line_no}: {line.strip()[:200]}")
                        break
                else:
                    if needle.search(line):
                        results.append(f"{rel}:{line_no}: {line.strip()[:200]}")
                        break

    walk(resolved)
    if not results:
        return f"No matches for '{query}' in {path}."
    return "\n".join(results)


def list_source_files(
    path: str = ".", *, file_pattern: str | None = None, max_files: int = 500
) -> str:
    """
    Lists source files under *path* (gitignore + ignored-dir aware), one per
    line, bounded. Used by the facade's inspect/workspace summary paths.
    """
    resolved = _resolve_safe_path(path)
    if resolved is None:
        return "Error: access denied, that directory is outside the allowed project folder."
    if not resolved.is_dir():
        return f"Error: {path} is not a directory"
    ignore = GitIgnoreRules(resolved)
    out: list[str] = []

    def walk(directory: Path) -> None:
        if len(out) >= max_files:
            return
        try:
            entries = sorted(directory.iterdir(), key=lambda e: e.name.lower())
        except (OSError, ValueError):
            return
        for entry in entries:
            if len(out) >= max_files:
                return
            name = entry.name
            if entry.is_dir():
                rel = str(entry.relative_to(resolved))
                if name in _IGNORED_DIRS or name.startswith("."):
                    continue
                if ignore.is_ignored(rel, is_dir=True):
                    continue
                walk(entry)
                continue
            if name.startswith("."):
                continue
            rel = str(entry.relative_to(resolved))
            if ignore.is_ignored(rel):
                continue
            if _matches_file_pattern(name, file_pattern):
                out.append(rel)

    walk(resolved)
    return "\n".join(out) if out else "(no source files found)"
