from pathlib import Path
from ultron.core.tools.paths import is_path_safe

def write_file(file_path: str, content: str, overwrite: bool = False) -> str:
    """
    Writes text content to a file at the specified path.
    
    - Checks that the target file is inside the ALLOWED_BASE_DIR for safety.
    - Prevents overwriting existing files unless overwrite=True.
    - Automatically creates missing parent directories if needed.
    """
    try:
        # Check safety using our shared path helper
        is_safe, resolved_path = is_path_safe(file_path)
        if not is_safe:
            return "Error: access denied, that file is outside the allowed project folder."
    except Exception as e:
        return f"Error resolving file path: {str(e)}"

    # Check if the file already exists to prevent accidental data loss
    if resolved_path.exists() and not overwrite:
        return "Error: file already exists. Set overwrite=True to replace it."

    try:
        # Create parent directories if they don't exist yet
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write content to the file with UTF-8 encoding
        with open(resolved_path, "w", encoding="utf-8") as f:
            f.write(content)

        return f"Successfully wrote {len(content)} characters to '{file_path}'."
    except Exception as e:
        return f"Error writing to file: {str(e)}"
