"""ultron.core.coding.edits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Safe file modification tracking and edit operations for the coding agent.

- :class:`FileModification` — one recorded change (path, action, step,
  original/resulting state, success, error).
- :class:`ModificationTracker` — ordered history of changes, with optional
  git status/diff integration when the workspace is a git repo.
- Edit operations (``create_file``, ``replace_file``, ``replace_in_file``,
  ``append_to_file``, ``delete_file``, ``rename_file``) — path-safe,
  deterministic, and registered as tools. They return "Error: ..." strings
  on failure exactly like the existing file tools, and they NEVER bypass
  the security boundary (the agent layer gates them like write_file).

Targeted edits are preferred over whole-file replacement: ``replace_in_file``
changes only the matched region and reports how many occurrences changed.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

from ultron.core.tools.paths import is_path_safe


class EditAction(str, Enum):
    """What kind of change a FileModification represents."""

    CREATE = "create"
    REPLACE = "replace"
    TARGETED_EDIT = "targeted_edit"
    APPEND = "append"
    DELETE = "delete"
    RENAME = "rename"


class FileModification(BaseModel):
    """One recorded file change made by Ultron."""

    path: str
    action: EditAction
    step: int | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    original_state: str | None = None  # prior content (or old path for rename)
    resulting_state: str | None = None  # new content (or new path for rename)
    success: bool = True
    error: str | None = None

    def describe(self) -> str:
        """Human-readable one-line description."""
        status = "ok" if self.success else f"FAILED: {self.error}"
        return f"[{self.action.value}] {self.path} ({status})"


class ModificationTracker(BaseModel):
    """Ordered history of file modifications, with optional git integration."""

    modifications: list[FileModification] = Field(default_factory=list)

    def record(
        self,
        path: str,
        action: EditAction,
        *,
        step: int | None = None,
        original_state: str | None = None,
        resulting_state: str | None = None,
        success: bool = True,
        error: str | None = None,
    ) -> FileModification:
        """Appends one modification and returns it."""
        mod = FileModification(
            path=path,
            action=action,
            step=step,
            original_state=original_state,
            resulting_state=resulting_state,
            success=success,
            error=error,
        )
        self.modifications.append(mod)
        return mod

    def recent(self, limit: int = 10) -> list[FileModification]:
        """The most recent *limit* modifications."""
        return self.modifications[-limit:]

    def git_status(self, cwd: str | None = None) -> str:
        """Best-effort ``git status --short`` for the workspace ('' when no git)."""
        from ultron.core.coding.workspace import git_status

        return git_status(cwd)

    def git_diff(self, cwd: str | None = None, max_chars: int = 8000) -> str:
        """Best-effort ``git diff`` for the workspace ('' when no git)."""
        from ultron.core.coding.workspace import git_diff

        return git_diff(cwd, max_chars)


# ---------------------------------------------------------------------------
# Path safety helpers
# ---------------------------------------------------------------------------


def _resolve(path: str) -> Path | None:
    """Resolves *path* inside ALLOWED_BASE_DIR; None when unsafe/blank."""
    raw = str(path or "").strip()
    if not raw:
        return None
    if not os.path.isabs(raw):
        raw = os.path.join(os.getcwd(), raw)
    try:
        ok, resolved = is_path_safe(raw)
    except (OSError, ValueError):
        return None
    return resolved if ok else None


def _read_text(path: Path) -> str:
    """Reads a file's full text without raising."""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except (OSError, ValueError):
        return ""


def _write_text(path: Path, content: str) -> None:
    """Writes text (creating parents) — raises OSError on failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Edit operations (registered as tools; agent layer gates them via boundary)
# ---------------------------------------------------------------------------


def create_file(file_path: str, content: str) -> str:
    """Creates a new file; refuses to overwrite an existing one."""
    resolved = _resolve(file_path)
    if resolved is None:
        return "Error: access denied, that file is outside the allowed project folder."
    if resolved.exists():
        return f"Error: file already exists at {file_path}. Use replace_file to overwrite it."
    try:
        _write_text(resolved, content)
        return f"Created '{file_path}' ({len(content)} characters)."
    except (OSError, ValueError) as exc:
        return f"Error writing file: {exc}"


def replace_file(file_path: str, content: str) -> str:
    """Replaces a file's entire contents (creating it if missing)."""
    resolved = _resolve(file_path)
    if resolved is None:
        return "Error: access denied, that file is outside the allowed project folder."
    try:
        _write_text(resolved, content)
        return f"Replaced '{file_path}' ({len(content)} characters)."
    except (OSError, ValueError) as exc:
        return f"Error writing file: {exc}"


def replace_in_file(file_path: str, old: str, new: str) -> str:
    """
    Targeted edit: replaces every occurrence of *old* with *new*.

    Only the matched region changes — unrelated content is preserved. Errors
    when the old text is missing or blank so the agent never silently
    corrupts a file with a no-op/mistaken edit.
    """
    resolved = _resolve(file_path)
    if resolved is None:
        return "Error: access denied, that file is outside the allowed project folder."
    if not resolved.exists():
        return f"Error: file not found at {file_path}"
    if not old:
        return "Error: replace_in_file requires non-empty 'old' text."
    content = _read_text(resolved)
    count = content.count(old)
    if count == 0:
        return f"Error: text not found in {file_path}. Nothing was changed."
    updated = content.replace(old, new)
    try:
        _write_text(resolved, updated)
    except (OSError, ValueError) as exc:
        return f"Error writing file: {exc}"
    return f"Replaced {count} occurrence(s) in '{file_path}'."


def append_to_file(file_path: str, content: str) -> str:
    """Appends *content* to a file (creating it if missing)."""
    resolved = _resolve(file_path)
    if resolved is None:
        return "Error: access denied, that file is outside the allowed project folder."
    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        with open(resolved, "a", encoding="utf-8") as fh:
            fh.write(content)
        return f"Appended {len(content)} characters to '{file_path}'."
    except (OSError, ValueError) as exc:
        return f"Error writing file: {exc}"


def delete_file(file_path: str) -> str:
    """Deletes a file; refuses directories and missing paths."""
    resolved = _resolve(file_path)
    if resolved is None:
        return "Error: access denied, that file is outside the allowed project folder."
    if not resolved.exists():
        return f"Error: file not found at {file_path}"
    if resolved.is_dir():
        return f"Error: {file_path} is a directory; delete_file only removes files."
    try:
        resolved.unlink()
        return f"Deleted '{file_path}'."
    except (OSError, ValueError) as exc:
        return f"Error deleting file: {exc}"


def rename_file(file_path: str, new_path: str) -> str:
    """Renames/moves a file; both paths must stay inside the allowed folder."""
    resolved = _resolve(file_path)
    target = _resolve(new_path)
    if resolved is None or target is None:
        return "Error: access denied, one of the paths is outside the allowed project folder."
    if not resolved.exists():
        return f"Error: file not found at {file_path}"
    if target.exists():
        return f"Error: target already exists at {new_path}."
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        resolved.rename(target)
        return f"Renamed '{file_path}' to '{new_path}'."
    except (OSError, ValueError) as exc:
        return f"Error renaming file: {exc}"


# ---------------------------------------------------------------------------
# Tool-result recording (wires the tracker into agent execution)
# ---------------------------------------------------------------------------

# Maps the registered coding edit tool names to their EditAction. Used by
# record_tool_result so the agent loop can record every executed edit into
# the task's modification tracker without the edit helpers knowing about the
# tracker (they stay stateless tool functions).
EDIT_TOOL_ACTIONS: dict[str, EditAction] = {
    "create_file": EditAction.CREATE,
    "replace_file": EditAction.REPLACE,
    "replace_in_file": EditAction.TARGETED_EDIT,
    "append_to_file": EditAction.APPEND,
    "delete_file": EditAction.DELETE,
    "rename_file": EditAction.RENAME,
}


def record_tool_result(
    tracker: ModificationTracker,
    tool_name: str,
    target: str,
    result: str,
    *,
    step: int | None = None,
    success: bool | None = None,
) -> FileModification | None:
    """
    Records a coding edit tool's execution result into *tracker*.

    No-op (returns None) for tools that are not coding edits, so callers can
    invoke it unconditionally. ``result`` is the tool's return string. When
    ``success`` is not supplied it is derived from the "Error: ..." prefix,
    but callers that already know the authoritative verdict (e.g. the agent's
    ``_observation_succeeded``, which also treats guardrail blocks and user
    cancellations as failures) MUST pass it so the tracker always agrees with
    ``execution_history``.
    """
    action = EDIT_TOOL_ACTIONS.get(tool_name)
    if action is None:
        return None
    text = str(result or "")
    if success is None:
        success = not text.startswith("Error:")
    return tracker.record(
        path=str(target or ""),
        action=action,
        step=step,
        success=success,
        error=None if success else text,
    )
