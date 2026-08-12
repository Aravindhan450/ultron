"""ultron.core.nlp.workspace
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Workspace-context resolution for the natural-language → tool layer.

The routing layer must resolve natural-language location phrases
("current directory", "here", "this folder", "project root") to the *actual*
workspace root — never to a literal string or the process CWD by accident.
This module is a thin, read-only wrapper around the existing coding workspace
abstractions (``ultron.core.coding.workspace.discover_workspace`` /
``detect_project`` / git detection).  It does NOT create a second workspace
state: the coding workspace remains the single source of truth; this module
only adds location-phrase → path resolution and environment-root discovery on
top of it.

Security: path resolution never escapes the workspace — relative paths are
joined against the project root and absolute paths are returned as-is so the
existing security boundary (``ALLOWED_BASE_DIR`` / file policy) still decides
access.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

# Natural-language location phrases that mean "the workspace/project root".
_LOCATION_PHRASES: frozenset[str] = frozenset(
    {
        ".",
        "..",
        "here",
        "root",
        "workspace",
        "project",
        "cwd",
        "pwd",
        "current",
        "current directory",
        "current folder",
        "current dir",
        "working directory",
        "working dir",
        "project directory",
        "project dir",
        "project root",
        "workspace root",
        "workspace directory",
        "this directory",
        "this folder",
        "this dir",
    }
)

# Relative path token we accept after "in/of/at" — anything that does not
# look like a location phrase or a stopword is treated as a candidate path.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "the", "a", "an", "my", "your", "our", "their", "its", "his", "her",
        "this", "that", "these", "those", "some", "any", "all", "every",
        "then", "there", "now", "here", "current", "same", "other",
    }
)


@dataclass
class WorkspaceContext:
    """Resolved workspace facts used by the routing layer."""

    workspace_root: str  # the root the routing layer treats as "here"
    current_directory: str  # the process cwd
    project_root: str  # nearest project root (marker-based walk)
    git_root: str | None = None
    environment_root: str | None = None  # virtualenv root (.venv / venv)
    project_type: str = "generic"
    test_framework: str | None = None
    extra: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Location-phrase resolution
    # ------------------------------------------------------------------

    def resolve_path(self, expr: str | None) -> str:
        """Resolves a location expression to a real absolute path.

        - location phrases (``current directory`` / ``here`` / ``this
          folder`` / ``.`` / ``project root``) -> the workspace/project root
        - ``src/`` / ``./src`` / relative paths -> joined against the project
          root (never against an arbitrary CWD)
        - absolute paths -> returned unchanged (security decides access)
        """
        text = (expr or "").strip()
        if not text:
            return self.workspace_root

        key = text.lower().strip("/\\").strip()
        if key in _LOCATION_PHRASES:
            return self.project_root

        path = Path(text)
        if path.is_absolute():
            return str(path)
        # Relative paths ("src", "src/", "./src", "tests") resolve against the
        # project root — never against an arbitrary process CWD.
        resolved = (Path(self.project_root) / path).resolve()
        return str(resolved)

    def summary(self) -> str:
        bits = [f"root={self.project_root}", f"type={self.project_type}"]
        if self.git_root:
            bits.append(f"git={self.git_root}")
        if self.environment_root:
            bits.append(f"env={self.environment_root}")
        if self.test_framework:
            bits.append(f"tests={self.test_framework}")
        return "WORKSPACE " + " | ".join(bits)


def _looks_pathlike(text: str) -> bool:
    """True when *text* is path-shaped (has a slash, dot, tilde or dash)."""
    return any(ch in text for ch in "/\\.~-")


def _find_git_root(root: Path) -> str | None:
    """Best-effort git root. Never raises; returns None when not a repo."""
    if not (root / ".git").exists():
        return None
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=3, check=False,
        )
        if proc.returncode == 0:
            return proc.stdout.strip() or str(root)
    except (OSError, subprocess.SubprocessError):
        return None
    return str(root)


def _find_environment_root(root: Path) -> str | None:
    """Returns the virtualenv root when the project has one."""
    for name in (".venv", "venv"):
        candidate = root / name
        if candidate.is_dir() and (candidate / "bin" / "python").exists():
            return str(candidate)
    return None


def resolve_workspace(cwd: str | None = None) -> WorkspaceContext:
    """Discovers the workspace context for *cwd* (defaults to process CWD).

    Read-only: pure filesystem inspection + git queries.  Reuses the coding
    workspace detector so there is a single source of truth for project
    detection.
    """
    from ultron.core.coding.workspace import discover_workspace

    ws = discover_workspace(cwd)
    root = Path(ws.project_root)
    return WorkspaceContext(
        workspace_root=root.resolve().__str__(),
        current_directory=ws.cwd,
        project_root=ws.project_root,
        git_root=_find_git_root(root),
        environment_root=_find_environment_root(root),
        project_type=ws.project_type,
        test_framework=ws.test_framework,
        extra={
            "package_manager": ws.package_manager,
            "build_system": ws.build_system,
            "languages": list(ws.languages),
            "source_dirs": list(ws.source_dirs),
            "test_dirs": list(ws.test_dirs),
            "config_files": list(ws.config_files),
            "is_git_repo": ws.is_git_repo,
        },
    )


def resolve_location_path(expr: str | None, cwd: str | None = None) -> str:
    """Convenience: resolves a location expression to an absolute path."""
    return resolve_workspace(cwd).resolve_path(expr)


def git_changed_files(cwd: str | None = None, limit: int = 200) -> list[str]:
    """Returns changed file paths (working tree) from git, bounded.

    Read-only and never raises: not a git repo or a git failure returns [].
    Relative paths are returned as git reports them (relative to the repo
    root), so callers should join against ``project_root`` when needed.
    """
    workspace = resolve_workspace(cwd)
    root = workspace.git_root or workspace.project_root
    try:
        proc = subprocess.run(
            ["git", "-C", root, "status", "--porcelain"],
            capture_output=True, text=True, timeout=3, check=False,
        )
        if proc.returncode != 0:
            return []
        changed: list[str] = []
        for line in proc.stdout.splitlines():
            if len(line) < 4:
                continue
            # porcelain: "XY path" (XY may include a space for untracked).
            # Renames are "R  old -> new" — keep the destination path.
            entry = line[3:].strip()
            if " -> " in entry:
                entry = entry.split(" -> ", 1)[-1].strip()
            if entry and entry not in changed:
                changed.append(entry)
            if len(changed) >= limit:
                break
        return changed
    except (OSError, subprocess.SubprocessError):
        return []
