"""ultron.core.memory.task_store
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Task persistence (FIX #6): TaskState snapshots that survive process
restarts and interrupted sessions.

Snapshots live in ``<workspace_root>/.ultron/tasks/`` — one JSON file per
task plus a ``latest.json`` pointer, so ``load_task`` restores the goal,
plan, completed/failed/remaining steps, execution history and transcript
intact. The plan is the source of truth and is persisted WITH the task —
never only inside an LLM prompt.

Safety rules:

- **Never Ultron's own repository** — snapshots are never written into the
  Ultron repo itself (unit tests run from it; the CLI gates via the same
  check).
- **Secrets** — the existing credential scanner guards the serialized
  payload: a snapshot whose transcript carries a credential pattern is
  refused (returns None) rather than persisted.
- **Resume hygiene** — a loaded task has any pending confirmation cleared
  and is resumed (WAITING_CONFIRMATION -> RUNNING). Stale tool execution is
  never restored as success: on resume the agent re-verifies against the
  current repository state through the normal executor/security path.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from ultron.core.types import TaskState
from ultron.security.scanners.secret import scan_secrets

# The Ultron repository root (src/ultron/core/memory/task_store.py -> parents:
# [0]=memory, [1]=core, [2]=ultron, [3]=src, [4]=repo).
_ULTRON_ROOT = Path(__file__).resolve().parents[4]


def task_store_dir(workspace_root) -> Path:
    """The snapshot directory for a workspace root."""
    return Path(workspace_root).resolve() / ".ultron" / "tasks"


def _safe_root(workspace_root) -> Path | None:
    """Resolved workspace root, or None when persistence must not happen
    there (Ultron's own repository)."""
    try:
        root = Path(workspace_root).resolve()
    except (OSError, ValueError):
        return None
    if root == _ULTRON_ROOT or _ULTRON_ROOT in root.parents:
        return None
    return root


def _slug(goal: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (goal or "").lower()).strip("-")
    return (slug or "task")[:48]


def save_task(
    task: TaskState | None,
    workspace_root,
    *,
    max_snapshots: int = 10,
) -> Path | None:
    """Persists one TaskState snapshot; returns its path, or None when the
    task cannot (or must not) be persisted.

    Refusal reasons: no task, unsafe workspace root (Ultron's own repo), a
    serialization failure, or credential patterns in the serialized payload.
    Older snapshots are pruned to ``max_snapshots`` (newest kept) so long
    sessions never accumulate an unbounded snapshot pile.
    """
    root = _safe_root(workspace_root)
    if root is None or task is None:
        return None
    try:
        raw = task.model_dump_json()
    except (ValueError, TypeError, AttributeError):  # pragma: no cover - defensive
        return None
    # Never persist transcripts carrying credential patterns.
    if scan_secrets(raw):
        return None

    store_dir = task_store_dir(root)
    try:
        store_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = store_dir / f"{_slug(task.goal)}-{stamp}.json"
    try:
        path.write_text(raw, encoding="utf-8")
        (store_dir / "latest.json").write_text(
            json.dumps(
                {
                    "goal": task.goal,
                    "path": path.name,
                    "saved_at": time.time(),
                }
            ),
            encoding="utf-8",
        )
    except OSError:
        return None
    _prune_snapshots(store_dir, max_snapshots)
    return path


def _prune_snapshots(store_dir: Path, keep: int) -> None:
    """Keeps the ``keep`` most recent snapshot files (never latest.json)."""
    try:
        snapshots = [
            p for p in store_dir.glob("*.json") if p.name != "latest.json"
        ]
    except OSError:
        return
    if len(snapshots) <= keep:
        return
    snapshots.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for stale in snapshots[keep:]:
        try:
            stale.unlink()
        except OSError:  # pragma: no cover - best-effort pruning
            pass


def load_task(workspace_root) -> TaskState | None:
    """Loads the most recent saved task for a workspace, cleared for
    resumption (stale pending confirmation dropped, state resumed).

    Returns None when no snapshot exists or the payload cannot be restored.
    """
    store_dir = task_store_dir(workspace_root)
    raw = _read_latest_snapshot(store_dir)
    if raw is None:
        return None
    try:
        task = TaskState.model_validate_json(raw)
    except (ValueError, TypeError, AttributeError):
        return None
    # Resume hygiene: a persisted WAITING_CONFIRMATION must not resume as a
    # pending action (the tool call was not executed across the restart);
    # the agent re-decides from the restored transcript + current repo state.
    task.pending_action = None
    task.last_observation = None
    if task.is_waiting_confirmation:
        task.resume()
    return task


def _read_latest_snapshot(store_dir: Path) -> str | None:
    """Reads the newest snapshot (latest.json pointer, else newest file)."""
    try:
        payload = json.loads((store_dir / "latest.json").read_text(encoding="utf-8"))
        return (store_dir / payload["path"]).read_text(encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError):
        try:
            files = sorted(store_dir.glob("*.json"))
        except OSError:
            return None
        if not files:
            return None
        try:
            return files[-1].read_text(encoding="utf-8")
        except OSError:
            return None


def saved_task_info(workspace_root) -> dict | None:
    """Metadata about the latest saved task (for /resume affordances)."""
    try:
        latest = task_store_dir(workspace_root) / "latest.json"
        return json.loads(latest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
