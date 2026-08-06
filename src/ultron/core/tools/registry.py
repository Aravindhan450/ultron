from typing import Any, Callable
from ultron.core.tools.builtin.file_reader import read_file
from ultron.core.tools.builtin.file_writer import write_file
from ultron.core.tools.builtin.command_runner import run_command
from ultron.core.tools.builtin.http_client import make_http_request
from ultron.core.tools.builtin.web_search import fetch_page_text, search_web
from ultron.core.tools.builtin.database import run_query
from ultron.core.tools.memory.sqlite import add_memory, get_all_memories, search_memories

# A dictionary that maps a tool's name (as a string) to the actual Python function.
# This makes it easy to lookup and call tools dynamically by their names.
TOOLS: dict[str, Callable[..., Any]] = {
    "read_file": read_file,
    "write_file": write_file,
    "run_command": run_command,
    "make_http_request": make_http_request,
    "search_web": search_web,
    "fetch_page_text": fetch_page_text,
    "run_query": run_query,
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

def get_tools_schema() -> list[dict[str, Any]]:
    """
    Dynamically generates a JSON Schema format list describing all registered tools,
    their parameters, type hints, and docstrings.
    """
    import inspect

    schemas = []
    for name, func in TOOLS.items():
        sig = inspect.signature(func)
        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            param_type = "string"
            if param.annotation == int:
                param_type = "integer"
            elif param.annotation == bool:
                param_type = "boolean"
            elif param.annotation == float:
                param_type = "number"

            properties[param_name] = {
                "type": param_type,
                "description": f"Parameter '{param_name}' for {name}"
            }
            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        doc = inspect.getdoc(func) or f"Tool function {name}"
        # Keep description clean (first paragraph)
        description = doc.split("\n\n")[0].strip()

        schemas.append({
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        })
    return schemas
