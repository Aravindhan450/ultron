from collections.abc import Callable
from typing import Any

from ultron.core.tools.definitions import TOOL_DEFINITIONS

# TOOLS is DERIVED from the canonical definitions table — the single source
# of truth for tool identity (STEP 2A). Tool functions and their metadata
# are declared once in ``ultron.core.tools.definitions``; this mapping is
# just the executable view of it and must never be edited independently.
TOOLS: dict[str, Callable[..., Any]] = {
    name: definition.func for name, definition in TOOL_DEFINITIONS.items()
}


def get_tool(name: str) -> Callable[..., Any] | None:
    """
    Looks up a tool by its name in the registry.

    Returns the tool function if found, or None if the tool doesn't exist.
    """
    return TOOLS.get(name)


def get_tools_schema() -> list[dict[str, Any]]:
    """
    Dynamically generates a JSON Schema format list describing all registered tools,
    their parameters, type hints, and docstrings.

    Names and descriptions come from the canonical definitions table
    (``ultron.core.tools.definitions``); parameter schemas are derived from
    the bound function signatures.
    """
    import inspect

    schemas = []
    for name, definition in TOOL_DEFINITIONS.items():
        func = definition.func
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

        schemas.append({
            "name": name,
            "description": definition.resolved_description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        })
    return schemas
