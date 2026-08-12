"""ultron.core.coding.intelligence.index
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Repository index — an incremental SQLite-backed symbol/import/reference
index over a repository (Fix #4).

The index is the foundation the facade queries for definitions, references,
imports and dependents. Design decisions:

- **Incremental by mtime/size**: :meth:`RepositoryIndex.refresh` walks the
  tree once (stat-only), re-parses ONLY files whose (mtime, size) changed
  since the last scan, and never touches unchanged files. A single edited
  file costs one parse, not a full re-index.
- **File-authoritative**: the index is a cache. The actual source files
  remain the source of truth — every symbol/reference row records its
  file/line so the agent can read the real code.
- **Bounded**: skips ignored directories (git, node_modules, venvs, build
  artifacts), oversized files, and binary files; caps symbols per file.
- **Deterministic + dependency-free**: plain ``sqlite3``, no external
  services, no network. Parsing delegates to
  :mod:`ultron.core.coding.intelligence.parsers`.

The index is created at a per-workspace path (defaults to a data file next
to the workspace's other ``.ultron_*`` stores). Tests may repoint it at a
temp file.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
import time
from pathlib import Path
from typing import Self

from pydantic import BaseModel

from ultron.core.coding.intelligence.parsers import (
    EXTENSION_LANGUAGES,
    language_for_path,
    parse_source,
)
from ultron.core.coding.intelligence.symbols import (
    ImportEdge,
    ParseResult,
    Symbol,
    SymbolKind,
    SymbolLocation,
    SymbolReference,
)
from ultron.core.coding.workspace import _IGNORED_DIRS
from ultron.core.tools import paths as _paths

# Files larger than this are not parsed (still listed as indexed files).
MAX_FILE_BYTES = 512 * 1024
# Cap on symbols collected per file (parse bomb guard).
MAX_SYMBOLS_PER_FILE = 400
# Reference scan bound: how many definition names to search per changed file.
MAX_REFERENCE_NAMES = 2000


class IndexSummary(BaseModel):
    """One refresh() pass's outcome (also served as index status)."""

    files: int = 0
    symbols: int = 0
    imports: int = 0
    references: int = 0
    parsed: int = 0  # files actually re-parsed this pass
    unchanged: int = 0  # files skipped via mtime/size
    removed: int = 0  # files dropped because they disappeared
    duration_ms: float = 0.0

    def to_prompt_line(self) -> str:
        return (
            f"{self.files} files, {self.symbols} symbols, {self.imports} "
            f"imports, {self.references} references "
            f"(parsed {self.parsed}, unchanged {self.unchanged}, "
            f"removed {self.removed}, {self.duration_ms:.0f}ms)"
        )


_SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE NOT NULL,
    language TEXT NOT NULL DEFAULT '',
    mtime REAL NOT NULL DEFAULT 0,
    size INTEGER NOT NULL DEFAULT 0,
    indexed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS symbols (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    kind TEXT NOT NULL,
    language TEXT NOT NULL DEFAULT '',
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    line INTEGER NOT NULL DEFAULT 1,
    column INTEGER NOT NULL DEFAULT 0,
    end_line INTEGER,
    end_column INTEGER,
    scope TEXT NOT NULL DEFAULT '',
    parent TEXT,
    signature TEXT NOT NULL DEFAULT '',
    doc TEXT NOT NULL DEFAULT '',
    bases TEXT NOT NULL DEFAULT '[]',
    inferred INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name);
CREATE INDEX IF NOT EXISTS idx_symbols_file ON symbols(file_id);
CREATE TABLE IF NOT EXISTS imports (
    id INTEGER PRIMARY KEY,
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    imported TEXT NOT NULL,
    member TEXT NOT NULL DEFAULT '',
    alias TEXT,
    is_relative INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_imports_file ON imports(file_id);
CREATE INDEX IF NOT EXISTS idx_imports_imported ON imports(imported);
CREATE TABLE IF NOT EXISTS symbol_refs (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    line INTEGER NOT NULL DEFAULT 1,
    column INTEGER NOT NULL DEFAULT 0,
    context TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_symbol_refs_name ON symbol_refs(name);
CREATE INDEX IF NOT EXISTS idx_symbol_refs_file ON symbol_refs(file_id);
"""

# Kinds that count as a definition (used by find_definition).
_DEFINITION_KINDS = frozenset(
    {
        SymbolKind.CLASS,
        SymbolKind.INTERFACE,
        SymbolKind.FUNCTION,
        SymbolKind.METHOD,
        SymbolKind.VARIABLE,
        SymbolKind.CONSTANT,
        SymbolKind.STRUCT,
        SymbolKind.TRAIT,
        SymbolKind.ENUM,
        SymbolKind.TYPE_ALIAS,
    }
)


class RepositoryIndex:
    """
    Incremental, SQLite-backed symbol/import/reference index for one repo.

    Usage::

        index = RepositoryIndex(root="/path/to/repo")
        summary = index.refresh()          # first scan parses everything
        summary = index.refresh()          # no changes -> 0 parsed
        defs = index.find_definition("login")
        refs = index.find_references("UserService")
    """

    def __init__(self, root: str | Path, db_path: str | Path | None = None) -> None:
        self.root = Path(root).resolve()
        if db_path is None:
            # The index DB lives OUTSIDE the scanned repository (next to the
            # other ``.ultron_*`` stores in the allowed base dir), keyed by a
            # hash of the root — a read-only search must never drop files
            # into the repo it is scanning.
            # Read lazily (not captured at import time) so tests that
            # monkeypatch ``paths.ALLOWED_BASE_DIR`` land the DB in their
            # sandbox instead of the real project root.
            digest = hashlib.sha1(str(self.root).encode()).hexdigest()[:10]
            db_path = _paths.ALLOWED_BASE_DIR / f".ultron_code_index_{digest}.db"
        self.db_path = Path(db_path)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Closes the underlying SQLite connection."""
        try:
            self._conn.close()
        except sqlite3.Error:
            pass

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def refresh(self) -> IndexSummary:
        """
        Incremental re-index: re-parses only files whose (mtime, size)
        changed since the last scan. Removes rows for files that vanished.
        """
        started = time.monotonic()
        files: list[tuple[str, float, int]] = []
        for path in _walk_source_files(self.root):
            try:
                stat = path.stat()
            except OSError:
                continue
            files.append((str(path.relative_to(self.root)), stat.st_mtime, stat.st_size))

        known = self._known_files()
        summary = IndexSummary()

        # Files that disappeared -> drop their rows.
        for rel in known:
            if rel not in {f[0] for f in files}:
                self._delete_file(rel)
                summary.removed += 1

        changed: list[tuple[str, float, int]] = []
        for rel, mtime, size in files:
            summary.files += 1
            if rel in known and (known[rel][0], known[rel][1]) == (mtime, size):
                summary.unchanged += 1
                continue
            parse_result = self._parse_file(rel)
            self._upsert_file(rel, mtime, size, parse_result)
            changed.append((rel, mtime, size))
            summary.parsed += 1
            summary.symbols += len(parse_result.symbols)
            summary.imports += len(parse_result.imports)

        # Rebuild references ONLY for files that were re-parsed this pass
        # (their content may have changed, so their usages must be
        # recomputed). Unchanged files keep their existing reference rows.
        self._rebuild_references(changed)
        summary.references = self._reference_count()
        summary.duration_ms = (time.monotonic() - started) * 1000
        self._conn.commit()
        return summary

    def status(self) -> IndexSummary:
        """Counts currently stored rows without re-scanning the tree."""
        files = self._conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        symbols = self._conn.execute("SELECT COUNT(*) FROM symbols").fetchone()[0]
        imports = self._conn.execute("SELECT COUNT(*) FROM imports").fetchone()[0]
        references = self._conn.execute("SELECT COUNT(*) FROM symbol_refs").fetchone()[0]
        return IndexSummary(
            files=int(files),
            symbols=int(symbols),
            imports=int(imports),
            references=int(references),
        )

    # -- internals -----------------------------------------------------

    def _known_files(self) -> dict[str, tuple[float, int]]:
        rows = self._conn.execute("SELECT path, mtime, size FROM files").fetchall()
        return {r[0]: (r[1], r[2]) for r in rows}

    def _parse_file(self, rel: str) -> ParseResult:
        full = self.root / rel
        try:
            source = full.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ParseResult(file_path=rel)
        return parse_source(source, rel)

    def _upsert_file(
        self, rel: str, mtime: float, size: int, result: ParseResult
    ) -> None:
        language = language_for_path(rel) or result.language
        existing = self._conn.execute(
            "SELECT id FROM files WHERE path = ?", (rel,)
        ).fetchone()
        if existing:
            file_id = existing[0]
            self._conn.execute(
                "DELETE FROM symbols WHERE file_id = ?", (file_id,)
            )
            self._conn.execute("DELETE FROM imports WHERE file_id = ?", (file_id,))
            self._conn.execute(
                "UPDATE files SET language = ?, mtime = ?, size = ? WHERE id = ?",
                (language, mtime, size, file_id),
            )
        else:
            cursor = self._conn.execute(
                "INSERT INTO files (path, language, mtime, size) VALUES (?, ?, ?, ?)",
                (rel, language, mtime, size),
            )
            file_id = cursor.lastrowid

        for symbol in result.symbols[:MAX_SYMBOLS_PER_FILE]:
            self._conn.execute(
                "INSERT INTO symbols "
                "(name, kind, language, file_id, line, column, end_line, "
                " end_column, scope, parent, signature, doc, bases, inferred) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    symbol.name,
                    symbol.kind.value,
                    symbol.language or language,
                    file_id,
                    symbol.location.line,
                    symbol.location.column,
                    symbol.location.end_line,
                    symbol.location.end_column,
                    symbol.scope,
                    symbol.parent,
                    symbol.signature,
                    symbol.doc,
                    _serialize_bases(symbol.bases),
                    int(symbol.inferred),
                ),
            )
        for edge in result.imports:
            self._conn.execute(
                "INSERT INTO imports (file_id, imported, member, alias, is_relative) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    file_id,
                    edge.imported,
                    edge.member,
                    edge.alias,
                    int(edge.is_relative),
                ),
            )

    def _delete_file(self, rel: str) -> None:
        self._conn.execute("DELETE FROM files WHERE path = ?", (rel,))

    def _rebuild_references(self, files: list[tuple[str, float, int]]) -> None:
        """Recomputes references ONLY for files whose content changed.

        Uses the parsed definitions from the DB: for every changed file,
        scans its text for word-boundary occurrences of each definition name
        (excluding the definition line itself) and stores them as
        references. Bounded by MAX_REFERENCE_NAMES.
        """
        names = [
            row[0]
            for row in self._conn.execute(
                "SELECT DISTINCT name FROM symbols "
                "WHERE kind IN ({}) ORDER BY name".format(
                    ",".join("?" for _ in _DEFINITION_KINDS)
                ),
                tuple(k.value for k in _DEFINITION_KINDS),
            ).fetchall()
        ][:MAX_REFERENCE_NAMES]
        if not names:
            return

        # Lines on which a symbol is DEFINED, per (file_id, name) — matches
        # on those lines are the definition itself, not a reference.
        def_lines: dict[tuple[int, str], set[int]] = {}
        for row in self._conn.execute(
            "SELECT s.name, s.file_id, s.line FROM symbols s "
            "WHERE s.kind IN ({})".format(
                ",".join("?" for _ in _DEFINITION_KINDS)
            ),
            tuple(k.value for k in _DEFINITION_KINDS),
        ).fetchall():
            def_lines.setdefault((row[1], row[0]), set()).add(row[2])

        pattern = re.compile(r"\b(" + "|".join(re.escape(n) for n in names) + r")\b")

        changed_rels = {f[0] for f in files}
        for rel in changed_rels:
            file_row = self._conn.execute(
                "SELECT id FROM files WHERE path = ?", (rel,)
            ).fetchone()
            if file_row is None:
                continue
            file_id = file_row[0]
            self._conn.execute(
                "DELETE FROM symbol_refs WHERE file_id = ?", (file_id,)
            )
            full = self.root / rel
            try:
                text = full.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                col = match.start() - (text.rfind("\n", 0, match.start()) + 1)
                name = match.group(1)
                if line in def_lines.get((file_id, name), set()):
                    continue  # the definition line itself is not a reference
                context = _line_at(text, line)[:200]
                self._conn.execute(
                    "INSERT INTO symbol_refs (name, file_id, line, column, context) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (name, file_id, line, col, context),
                )

    def _reference_count(self) -> int:
        return int(
            self._conn.execute("SELECT COUNT(*) FROM symbol_refs").fetchone()[0]
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def find_definition(self, name: str) -> list[Symbol]:
        """The defining symbol(s) for *name* (class/function/... kinds)."""
        rows = self._conn.execute(
            "SELECT s.name, s.kind, s.language, s.file_id, s.line, s.column, "
            " s.end_line, s.end_column, s.scope, s.parent, s.signature, "
            " s.doc, s.bases, s.inferred, f.path "
            "FROM symbols s JOIN files f ON f.id = s.file_id "
            "WHERE s.name = ? AND s.kind IN ({}) "
            "ORDER BY CASE s.kind WHEN 'class' THEN 0 WHEN 'interface' THEN 1 "
            "WHEN 'function' THEN 2 WHEN 'method' THEN 3 ELSE 4 END, s.line".format(
                ",".join("?" for _ in _DEFINITION_KINDS)
            ),
            (name, *tuple(k.value for k in _DEFINITION_KINDS)),
        ).fetchall()
        return [_row_to_symbol(r) for r in rows]

    def find_symbol(self, name: str) -> list[Symbol]:
        """Every symbol (any kind) named *name* across the index."""
        rows = self._conn.execute(
            "SELECT s.name, s.kind, s.language, s.file_id, s.line, s.column, "
            " s.end_line, s.end_column, s.scope, s.parent, s.signature, "
            " s.doc, s.bases, s.inferred, f.path "
            "FROM symbols s JOIN files f ON f.id = s.file_id "
            "WHERE s.name = ? ORDER BY s.line",
            (name,),
        ).fetchall()
        return [_row_to_symbol(r) for r in rows]

    def find_symbols_in_file(self, rel_path: str) -> list[Symbol]:
        """Every symbol defined in one file (relative path within the repo)."""
        row = self._conn.execute(
            "SELECT id FROM files WHERE path = ?", (rel_path,)
        ).fetchone()
        if row is None:
            return []
        rows = self._conn.execute(
            "SELECT s.name, s.kind, s.language, s.file_id, s.line, s.column, "
            " s.end_line, s.end_column, s.scope, s.parent, s.signature, "
            " s.doc, s.bases, s.inferred, f.path "
            "FROM symbols s JOIN files f ON f.id = s.file_id "
            "WHERE s.file_id = ? ORDER BY s.line",
            (row[0],),
        ).fetchall()
        return [_row_to_symbol(r) for r in rows]

    def find_references(self, name: str) -> list[SymbolReference]:
        """Usage sites of *name* outside its definition (bounded, ordered)."""
        rows = self._conn.execute(
            "SELECT r.name, r.file_id, r.line, r.column, r.context, f.path "
            "FROM symbol_refs r JOIN files f ON f.id = r.file_id "
            "WHERE r.name = ? ORDER BY f.path, r.line LIMIT 300",
            (name,),
        ).fetchall()
        return [
            SymbolReference(
                name=r[0],
                location=SymbolLocation(file=r[5], line=r[2], column=r[3]),
                context=r[4],
            )
            for r in rows
        ]

    def get_imports(self, rel_path: str) -> list[ImportEdge]:
        """Import edges of one file (relative path within the repo)."""
        row = self._conn.execute(
            "SELECT id FROM files WHERE path = ?", (rel_path,)
        ).fetchone()
        if row is None:
            return []
        edges = self._conn.execute(
            "SELECT imported, member, alias, is_relative FROM imports "
            "WHERE file_id = ? ORDER BY id",
            (row[0],),
        ).fetchall()
        return [
            ImportEdge(
                source=rel_path,
                imported=e[0],
                member=e[1],
                alias=e[2],
                is_relative=bool(e[3]),
            )
            for e in edges
        ]

    def get_dependents(self, rel_path: str) -> list[str]:
        """Files that import *rel_path* (EXACT reverse-import edges).

        Resolves both ``import pkg.mod`` and ``from pkg.mod import x`` forms
        (including relative imports like ``from .models import User``): the
        target file is matched when the imported module name, its dotted
        form, or its ``.py``-suffixed stem refers to it.

        ``src/``-layout support: a module ``auth.service`` lives at
        ``src/auth/service.py`` while the import is written as
        ``auth.service`` (the source root is on the path). Candidates are
        therefore generated BOTH with and without the leading path/dotted
        segment, so absolute imports resolve to src-layout files and to
        flat layouts alike.
        """
        stem = rel_path.removesuffix(".py")
        basename = stem.rsplit("/", 1)[-1]
        module_candidates = [rel_path, stem, stem.replace("/", "."), basename]
        # Add the same forms with the leading segment stripped (src/ layout).
        parts = stem.split("/")
        if len(parts) > 1:
            tail_path = "/".join(parts[1:])
            tail_dotted = ".".join(parts[1:])
            module_candidates.extend([tail_path, f"{tail_path}.py", tail_dotted])
        placeholders = ",".join("?" for _ in module_candidates)
        rows = self._conn.execute(
            "SELECT DISTINCT f.path FROM imports i JOIN files f ON f.id = i.file_id "
            f"WHERE i.imported IN ({placeholders}) "
            f"OR i.imported || '.py' IN ({placeholders}) "
            f"OR i.imported || '.' || i.member IN ({placeholders})",
            (*module_candidates, *module_candidates, *module_candidates),
        ).fetchall()
        return sorted(r[0] for r in rows)

    def all_symbol_names(self) -> list[str]:
        """Sorted distinct definition names (used by the facade/semantic layer)."""
        rows = self._conn.execute(
            "SELECT DISTINCT name FROM symbols WHERE kind IN ({}) ORDER BY name".format(
                ",".join("?" for _ in _DEFINITION_KINDS)
            ),
            tuple(k.value for k in _DEFINITION_KINDS),
        ).fetchall()
        return [r[0] for r in rows]

    def file_language(self, rel_path: str) -> str:
        row = self._conn.execute(
            "SELECT language FROM files WHERE path = ?", (rel_path,)
        ).fetchone()
        return row[0] if row else language_for_path(rel_path)


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------


def _walk_source_files(root: Path):
    """Walks *root* yielding parseable source files, skipping ignored dirs.

    Deterministic (sorted), bounded by ignored-dir and extension rules; does
    NOT read file contents (stat-only, so refresh is cheap for unchanged
    trees).
    """
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        parts = path.relative_to(root).parts
        if any(part in _IGNORED_DIRS or part.startswith(".") for part in parts[:-1]):
            continue
        if path.name.startswith("."):
            continue
        if path.suffix.lower() not in EXTENSION_LANGUAGES:
            continue
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                continue
        except OSError:
            continue
        yield path


def _serialize_bases(bases: list[str]) -> str:
    import json

    return json.dumps(bases)


def _row_to_symbol(row) -> Symbol:
    import json

    try:
        bases = json.loads(row[12] or "[]")
    except (ValueError, TypeError):
        bases = []
    return Symbol(
        name=row[0],
        kind=SymbolKind(row[1]),
        language=row[2],
        location=SymbolLocation(
            file=row[14],
            line=row[4],
            column=row[5],
            end_line=row[6],
            end_column=row[7],
        ),
        scope=row[8],
        parent=row[9],
        signature=row[10],
        doc=row[11],
        bases=bases,
        inferred=bool(row[13]),
    )


def _line_at(text: str, line: int) -> str:
    lines = text.splitlines()
    if 1 <= line <= len(lines):
        return lines[line - 1].strip()
    return ""
