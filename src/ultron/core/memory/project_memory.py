"""ultron.core.memory.project_memory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Project-scoped memory: facts about a specific repository/workspace.

Design (FIX #6):

- **Workspace identity** — every record is keyed by ``(workspace, kind,
  name)``. Two different projects never see each other's facts, and the
  same fact in two workspaces coexists independently.
- **Persistence** — one SQLite file per workspace root
  (``<root>/.ultron_project_memory.db``), created idempotently like the
  existing memory/API-schema stores. Survives process restarts.
- **Invalidation without erasure** — storing a new value for an existing
  key marks the old row ``superseded`` (history preserved) instead of
  silently overwriting it; ``invalidate()`` marks rows ``stale`` so the
  ContextManager can drop or de-prioritize them without deleting history.
- **Sources & confidence** — every row stores ``source`` and
  ``confidence``; retrieval and the ContextManager can prefer
  DIRECT_OBSERVATION / USER_PROVIDED over LLM inference.
- **Versioning** — when git is available the current commit (short SHA) is
  captured at store time; never mandatory.
- **Secrets** — the existing secret scanner (`security.scanners.secret`)
  is the write guard: content that matches credential patterns is refused
  (returns False) so secrets never reach disk. The raw content is never
  logged.

The store is deterministic and pure-SQLite; no LLM is involved.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import time
from pathlib import Path

from ultron.core.memory.models import (
    MemoryConfidence,
    MemoryKind,
    MemoryRecord,
    MemorySource,
    MemoryValidity,
)
from ultron.security.scanners.secret import scan_secrets

_SCHEMA = """
CREATE TABLE IF NOT EXISTS project_memory (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    kind          TEXT NOT NULL,
    name          TEXT NOT NULL,
    content       TEXT NOT NULL,
    source        TEXT NOT NULL,
    confidence    TEXT NOT NULL,
    workspace     TEXT NOT NULL,
    revision      TEXT,
    validity      TEXT NOT NULL DEFAULT 'valid',
    created_at    REAL NOT NULL,
    updated_at    REAL NOT NULL,
    supersedes_id INTEGER,
    metadata      TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_pm_workspace ON project_memory(workspace, kind, name);
CREATE INDEX IF NOT EXISTS idx_pm_validity ON project_memory(validity);
"""


def project_memory_db_path(workspace_root: str | Path) -> Path:
    """The SQLite file for a workspace root."""
    return Path(workspace_root).resolve() / ".ultron_project_memory.db"


def current_revision(workspace_root: str | Path) -> str | None:
    """Best-effort short git SHA for a workspace; None when not a git repo."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(workspace_root), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if proc.returncode == 0:
            return proc.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        pass
    return None


_COERCE_FALLBACK = {
    MemorySource: MemorySource.LLM_INFERENCE,
    MemoryConfidence: MemoryConfidence.UNKNOWN,
}


def _coerce(value, enum_type):
    """Accepts either the enum member or its string value; falls back safely."""
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(str(value))
    except ValueError:
        fallback = _COERCE_FALLBACK.get(enum_type)
        return fallback if fallback is not None else next(iter(enum_type))


class ProjectMemoryStore:
    """Workspace-scoped memory store (one SQLite file per workspace root)."""

    # Revision cache: ``git rev-parse`` is a subprocess, and formation now
    # calls ``store()`` repeatedly per agent turn — compute it at most once
    # per short window instead of per fact.
    _revision_ttl: float = 5.0

    def __init__(self, workspace_root: str | Path) -> None:
        self.root = Path(workspace_root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = project_memory_db_path(self.root)
        self._revision_cache: tuple[float, str | None] | None = None
        self._init_schema()

    def _cached_revision(self) -> str | None:
        now = time.time()
        if self._revision_cache is not None and now - self._revision_cache[0] < self._revision_ttl:
            return self._revision_cache[1]
        revision = current_revision(self.root)
        self._revision_cache = (now, revision)
        return revision

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.executescript(_SCHEMA)
        except (sqlite3.Error, OSError):
            pass  # degrade to read-only memory rather than crash

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(
            id=row["id"],
            kind=MemoryKind(row["kind"]),
            name=row["name"],
            content=row["content"],
            source=MemorySource(row["source"]),
            confidence=MemoryConfidence(row["confidence"]),
            workspace=row["workspace"],
            revision=row["revision"],
            validity=MemoryValidity(row["validity"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            supersedes_id=row["supersedes_id"],
            metadata=json.loads(row["metadata"] or "{}"),
        )

    def _query(self, sql: str, params: tuple = ()) -> list[MemoryRecord]:
        try:
            with self._connect() as conn:
                return [self._row_to_record(r) for r in conn.execute(sql, params)]
        except (sqlite3.Error, OSError):
            return []

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def store(
        self,
        name: str,
        content: str,
        *,
        kind: MemoryKind | str = MemoryKind.PROJECT,
        source: MemorySource | str = MemorySource.REPOSITORY_INSPECTION,
        confidence: MemoryConfidence | str = MemoryConfidence.UNKNOWN,
        revision: str | None = None,
        metadata: dict | None = None,
        allow_secrets: bool = False,
    ) -> MemoryRecord | None:
        """
        Stores a project fact, superseding any previous value with the same
        (workspace, kind, name) key. Returns the new record, or None when
        the content was refused (credential pattern detected and
        ``allow_secrets`` is False).

        ``allow_secrets`` is a deliberate escape hatch for system-initiated
        facts that must be stored regardless; user/tool-derived content
        should always go through the default secret-guarded path.
        """
        name = (name or "").strip()
        content = (content or "").strip()
        if not name or not content:
            return None

        if not allow_secrets and scan_secrets(f"{name}: {content}"):
            return None  # never persist credential patterns

        if revision is None:
            revision = self._cached_revision()

        # The secret guard scans name, content AND metadata values — a
        # credential smuggled through metadata must not reach disk.
        scan_text = f"{name}: {content} {json.dumps(metadata or {})}"
        if not allow_secrets and scan_secrets(scan_text):
            return None

        kind_enum = _coerce(kind, MemoryKind)
        source_enum = _coerce(source, MemorySource)
        confidence_enum = _coerce(confidence, MemoryConfidence)
        now = time.time()
        workspace = str(self.root)

        try:
            with self._connect() as conn:
                cur = conn.execute(
                    "SELECT * FROM project_memory "
                    "WHERE workspace=? AND kind=? AND name=? AND validity='valid' "
                    "ORDER BY id DESC LIMIT 1",
                    (workspace, kind_enum.value, name),
                )
                current = cur.fetchone()
                # Idempotency: storing an identical fact (same content,
                # source, confidence AND revision) is a no-op (returns the
                # existing record) instead of superseding it — so
                # deterministic formation can run on every turn without
                # churning the history with byte-identical versions. An
                # explicit new revision is a meaningful update, never a
                # no-op.
                if current is not None and (
                    current["content"] == content
                    and current["source"] == source_enum.value
                    and current["confidence"] == confidence_enum.value
                    and current["revision"] == revision
                ):
                    return self._row_to_record(current)
                # Mark the previous valid value superseded (keep history).
                if current is not None:
                    conn.execute(
                        "UPDATE project_memory SET validity='superseded', "
                        "updated_at=? WHERE id=?",
                        (now, current["id"]),
                    )
                cursor = conn.execute(
                    "INSERT INTO project_memory "
                    "(kind, name, content, source, confidence, workspace, revision, "
                    " validity, created_at, updated_at, supersedes_id, metadata) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, 'valid', ?, ?, ?, ?)",
                    (
                        kind_enum.value,
                        name,
                        content,
                        source_enum.value,
                        confidence_enum.value,
                        workspace,
                        revision,
                        now,
                        now,
                        current["id"] if current is not None else None,
                        json.dumps(metadata or {}),
                    ),
                )
                new_id = cursor.lastrowid
        except (sqlite3.Error, OSError):
            return None

        return MemoryRecord(
            id=new_id,
            kind=kind_enum,
            name=name,
            content=content,
            source=source_enum,
            confidence=confidence_enum,
            workspace=workspace,
            revision=revision,
            validity=MemoryValidity.VALID,
            created_at=now,
            updated_at=now,
            supersedes_id=current["id"] if current is not None else None,
            metadata=dict(metadata or {}),
        )

    def invalidate(self, name: str | None = None) -> int:
        """
        Marks matching valid records STALE (not erased). With no name, all
        project facts for this workspace become stale.
        """
        workspace = str(self.root)
        try:
            with self._connect() as conn:
                if name:
                    cur = conn.execute(
                        "UPDATE project_memory SET validity='stale', updated_at=? "
                        "WHERE workspace=? AND name=? AND validity='valid'",
                        (time.time(), workspace, name),
                    )
                else:
                    cur = conn.execute(
                        "UPDATE project_memory SET validity='stale', updated_at=? "
                        "WHERE workspace=? AND validity='valid'",
                        (time.time(), workspace),
                    )
                return cur.rowcount
        except (sqlite3.Error, OSError):
            return 0

    def forget(self, name: str) -> bool:
        """Permanently removes one fact (all versions) for this workspace."""
        try:
            with self._connect() as conn:
                cur = conn.execute(
                    "DELETE FROM project_memory WHERE workspace=? AND name=?",
                    (str(self.root), name),
                )
                return cur.rowcount > 0
        except (sqlite3.Error, OSError):
            return False

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def recall(
        self,
        name: str | None = None,
        *,
        include_invalid: bool = False,
        limit: int = 50,
    ) -> list[MemoryRecord]:
        """Valid project facts; optionally the full history for a name."""
        workspace = str(self.root)
        sql = "SELECT * FROM project_memory WHERE workspace=?"
        params: list = [workspace]
        if include_invalid:
            sql += " AND name=?"
            params.append(name)
        else:
            sql += " AND validity='valid'"
            if name:
                sql += " AND name=?"
                params.append(name)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        return self._query(sql, tuple(params))

    def search(self, query: str, *, limit: int = 8) -> list[MemoryRecord]:
        """Keyword retrieval over valid project facts (deterministic LIKE)."""
        workspace = str(self.root)
        pattern = f"%{query.strip()}%"
        return self._query(
            "SELECT * FROM project_memory "
            "WHERE workspace=? AND validity='valid' "
            "AND (name LIKE ? OR content LIKE ?) "
            "ORDER BY created_at DESC LIMIT ?",
            (workspace, pattern, pattern, limit),
        )

    def all_valid(self) -> list[MemoryRecord]:
        return self.recall()

    def history(self, name: str) -> list[MemoryRecord]:
        """Every version of one fact, oldest first (superseded included)."""
        return list(
            reversed(
                self.recall(name=name, include_invalid=True, limit=500)
            )
        )

    def count(self) -> int:
        workspace = str(self.root)
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT COUNT(*) AS n FROM project_memory "
                    "WHERE workspace=? AND validity='valid'",
                    (workspace,),
                ).fetchone()
                return int(row["n"]) if row else 0
        except (sqlite3.Error, OSError):
            return 0

    def all_names(self) -> list[str]:
        """Stable keys currently stored, for task-scoped relevance lookup."""
        return [
            r.name for r in self.recall(limit=500) if r.name
        ]
