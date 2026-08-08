from pathlib import Path

# The base directory where file operations (reading and writing) are allowed.
# By default, this resolves to the current working directory of the project.
ALLOWED_BASE_DIR = Path.cwd().resolve()

def is_path_safe(file_path: str | Path) -> tuple[bool, Path]:
    """
    Checks if a given file path resolves inside the ALLOWED_BASE_DIR.
    Returns a tuple of (is_safe, resolved_path).
    """
    try:
        path_obj = Path(file_path).resolve()
        # Check if the resolved path is relative to or equal to ALLOWED_BASE_DIR
        path_obj.relative_to(ALLOWED_BASE_DIR)
        return True, path_obj
    except (OSError, ValueError):
        return False, Path(file_path)
