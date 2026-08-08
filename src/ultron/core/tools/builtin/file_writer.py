import os

from ultron.core.tools.paths import is_path_safe


def write_file(file_path: str, content: str, overwrite: bool = False) -> str:
    """
    Writes text content to a file at the specified path.

    Safety rules (checked in order):
    1. Rejects empty or whitespace-only paths immediately.
    2. Checks that the resolved path is NOT a directory.
    3. Checks that the target is inside ALLOWED_BASE_DIR.
    4. Prevents overwriting existing files unless overwrite=True.
    5. Automatically creates missing parent directories if needed.
    """
    # Guard 1: empty / blank path
    if not file_path or not str(file_path).strip():
        return "Error: Missing or invalid file path. Cannot write to a directory."

    # Resolve relative paths against CWD so bare filenames like
    # "overwrite_test.txt" land in the project root, not some unknown dir.
    file_path = str(file_path).strip()
    if not os.path.isabs(file_path):
        file_path = os.path.join(os.getcwd(), file_path)

    # Guard 2: path must not be an existing directory
    if os.path.isdir(file_path):
        return "Error: Missing or invalid file path. Cannot write to a directory."

    try:
        is_safe, resolved_path = is_path_safe(file_path)
        if not is_safe:
            return "Error: access denied, that file is outside the allowed project folder."
    except (OSError, ValueError) as e:
        return f"Error resolving file path: {e!s}"

    # Guard 3: resolved path must not be a directory (catches edge cases after symlink resolution)
    if resolved_path.is_dir():
        return "Error: Missing or invalid file path. Cannot write to a directory."

    # Guard 4: prevent accidental overwrite
    if resolved_path.exists() and not overwrite:
        return "Error: file already exists. Set overwrite=True to replace it."

    try:
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        with open(resolved_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote {len(content)} characters to '{resolved_path}'."
    except (OSError, ValueError) as e:
        return f"Error writing to file: {e!s}"
