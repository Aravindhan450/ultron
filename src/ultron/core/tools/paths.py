import os
from pathlib import Path

# The default base directory where file operations are allowed (process CWD).
ALLOWED_BASE_DIR = Path.cwd().resolve()


def get_configured_workspace() -> Path | None:
    """
    Returns the resolved external workspace directory configured via ULTRON_WORKSPACE,
    or None if not set.
    Expands ~ and environment variables, resolves symlinks and normalizes the path.
    """
    env_val = os.environ.get("ULTRON_WORKSPACE")
    if not env_val:
        from ultron.core.config import settings

        env_val = settings.workspace
    if not env_val or not str(env_val).strip():
        return None
    try:
        resolved = Path(str(env_val).strip()).expanduser().resolve()
        return resolved
    except (OSError, ValueError):
        return None


def get_allowed_base_dirs() -> list[Path]:
    """
    Returns the list of base directories within which file operations are confined.
    Always includes ALLOWED_BASE_DIR (which can be monkeypatched by tests),
    and includes the configured ULTRON_WORKSPACE if set.
    """
    dirs = [ALLOWED_BASE_DIR]
    ws = get_configured_workspace()
    if ws is not None and ws != ALLOWED_BASE_DIR:
        dirs.append(ws)
    return dirs


def is_path_safe(file_path: str | Path) -> tuple[bool, Path]:
    """
    Checks if a given file path resolves inside any of the allowed base directories
    (ALLOWED_BASE_DIR or configured ULTRON_WORKSPACE).
    Returns a tuple of (is_safe, resolved_path).
    Prevents path traversal and escapes.
    """
    try:
        path_obj = Path(file_path).expanduser().resolve()
        for base in get_allowed_base_dirs():
            try:
                path_obj.relative_to(base)
                return True, path_obj
            except (OSError, ValueError):
                continue
        return False, path_obj
    except (OSError, ValueError):
        return False, Path(file_path)


# Active project directory context for artifact creation/execution.
_ACTIVE_PROJECT_DIR: Path | None = None


def set_active_project_dir(path: str | Path | None) -> None:
    """Sets the active project directory within the workspace."""
    global _ACTIVE_PROJECT_DIR
    if path is None:
        _ACTIVE_PROJECT_DIR = None
    else:
        _ACTIVE_PROJECT_DIR = Path(path).expanduser().resolve()


def get_active_project_dir() -> Path | None:
    """Returns the current active project directory if set."""
    return _ACTIVE_PROJECT_DIR


def resolve_project_path(path: str | Path) -> Path:
    """
    Resolves a path against the active project directory if set,
    otherwise against the current working directory.
    """
    raw = Path(path).expanduser()
    if raw.is_absolute():
        return raw.resolve()
    base = get_active_project_dir() or Path.cwd()
    return (base / raw).resolve()
