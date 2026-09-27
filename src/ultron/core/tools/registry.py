import inspect
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


def _resolved_signature(func: Callable[..., Any]) -> inspect.Signature:
    """
    The function signature with string annotations evaluated.

    Tool modules use ``from __future__ import annotations``, so
    ``param.annotation`` is the *string* ``"int"`` rather than the ``int``
    type. Schema generation and argument coercion need the real type, so
    evaluate the annotations (falling back to the raw signature if any
    annotation cannot be resolved).
    """
    try:
        return inspect.signature(func, eval_str=True)
    except (NameError, TypeError, ValueError):
        return inspect.signature(func)


def coerce_tool_arguments(
    func: Callable[..., Any], arguments: dict[str, Any]
) -> dict[str, Any]:
    """
    Coerces string tool arguments to the function's annotated scalar types.

    Models routinely emit JSON scalars as strings (``"1000"``, ``"false"``),
    and a tool that annotated a parameter as ``int``/``float``/``bool`` then
    raises deep inside (e.g. ``'<=' not supported between int and str``),
    which the loop records as a failed call and retries. Coercing here keeps
    the invocation faithful to the signature the model was shown without
    changing any tool's semantics; uncoercible values are left untouched so
    the tool can produce its own error.
    """
    sig = _resolved_signature(func)
    coerced = dict(arguments)
    for name, param in sig.parameters.items():
        if name not in coerced:
            continue
        value = coerced[name]
        if not isinstance(value, str):
            continue
        annotation = param.annotation
        try:
            if annotation is bool:
                coerced[name] = value.strip().lower() in {"1", "true", "yes", "on"}
            elif annotation is int:
                coerced[name] = int(value)
            elif annotation is float:
                coerced[name] = float(value)
        except (TypeError, ValueError):
            pass
    return coerced


def get_tools_schema() -> list[dict[str, Any]]:
    """
    Dynamically generates a JSON Schema format list describing all registered tools,
    their parameters, type hints, and docstrings.

    Names and descriptions come from the canonical definitions table
    (``ultron.core.tools.definitions``); parameter schemas are derived from
    the bound function signatures.
    """
    schemas = []
    for name, definition in TOOL_DEFINITIONS.items():
        func = definition.func
        sig = _resolved_signature(func)
        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            annotation = param.annotation
            if annotation is bool:
                param_type = "boolean"
            elif annotation is int:
                param_type = "integer"
            elif annotation is float:
                param_type = "number"
            else:
                param_type = "string"

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
