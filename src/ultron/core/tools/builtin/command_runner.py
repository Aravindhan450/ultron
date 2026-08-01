import subprocess

def run_command(command: str) -> str:
    """
    Executes a shell command using Python's subprocess module.
    
    - Times out after 15 seconds to prevent hanging processes.
    - Captures stdout and stderr.
    - Safety and user confirmation are handled at the agent level before calling this tool.
    """
    try:
        # Execute the command with a shell environment and 15s timeout
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=15
        )
        
        output_parts = [f"Exit code: {result.returncode}"]
        if result.stdout:
            output_parts.append(f"Output:\n{result.stdout.strip()}")
        if result.stderr:
            output_parts.append(f"Error Output:\n{result.stderr.strip()}")
            
        return "\n".join(output_parts)
    except subprocess.TimeoutExpired:
        return "Error: command timed out after 15 seconds."
    except Exception as e:
        return f"Error: {str(e)}"
