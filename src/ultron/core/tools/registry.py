from typing import Any, Callable
from ultron.core.tools.builtin.file_reader import read_file
from ultron.core.tools.builtin.file_writer import write_file
from ultron.core.tools.builtin.command_runner import run_command
from ultron.core.tools.memory.sqlite import add_memory, get_all_memories, search_memories

# A dictionary that maps a tool's name (as a string) to the actual Python function.
# This makes it easy to lookup and call tools dynamically by their names.
TOOLS: dict[str, Callable[..., Any]] = {
    "read_file": read_file,
    "write_file": write_file,
    "run_command": run_command,
    "add_memory": add_memory,
    "get_all_memories": get_all_memories,
    "search_memories": search_memories,
}

def get_tool(name: str) -> Callable[..., Any] | None:
    """
    Looks up a tool by its name in the registry.
    
    Returns the tool function if found, or None if the tool doesn't exist.
    """
    # Return the function from the dictionary, or None if the name is not a key
    return TOOLS.get(name)
