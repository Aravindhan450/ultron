from ultron.core.tools.paths import is_path_safe

def read_file(file_path: str) -> str:
    """
    Reads the content of a file on the disk and returns it as a string.
    
    If the file does not exist, it returns an error message instead of failing.
    It reads at most 5000 characters to prevent using too much memory.
    """
    try:
        # Check safety using our shared path helper
        is_safe, resolved_path = is_path_safe(file_path)
        if not is_safe:
            return "Error: access denied, file is outside the allowed directory"
    except Exception as e:
        return f"Error resolving file path: {str(e)}"

    # Check if the file actually exists on the computer
    if not resolved_path.exists():
        return f"Error: file not found at {file_path}"
        
    # Check if the path is a file (and not a directory/folder)
    if not resolved_path.is_file():
        return f"Error: {file_path} is not a file"
        
    try:
        # Open and read the file using UTF-8 encoding
        with open(resolved_path, "r", encoding="utf-8") as f:
            content = f.read(5001)  # Read up to 5001 characters to check if it's over 5000
            
        # If the content is longer than 5000 characters, truncate it and add a message
        if len(content) > 5000:
            return content[:5000] + "...[truncated]"
            
        return content
    except Exception as e:
        # If any other error happens (like permission denied), return the error message
        return f"Error reading file: {str(e)}"
