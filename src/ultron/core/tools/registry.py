from collections.abc import Callable
from typing import Any

from ultron.core.coding.edits import (
    append_to_file,
    create_file,
    delete_file,
    rename_file,
    replace_file,
    replace_in_file,
)
from ultron.core.coding.intelligence.tools import (
    code_index_status,
    code_search,
    find_definition,
    find_references,
    find_symbol,
    get_dependents,
    get_imports,
    report_file,
    report_symbol,
    semantic_search,
)
from ultron.core.coding.workspace import (
    discover_workspace_summary,
    list_directory,
    search_files,
)
from ultron.core.intelligence.debug_context import (
    check_dependency,
    diagnose_failure,
    get_debug_context,
)
from ultron.core.intelligence.parallel_tools import (
    run_tool_batch,
    synthesize_analysis,
)
from ultron.core.intelligence.planning import (
    analyze_dependencies,
    list_plan_actions,
    preflight_plan_tool,
)
from ultron.core.intelligence.structured_output import (
    enforce_schema,
    list_schemas,
    schema_validate,
)
from ultron.core.learning.api_schema import (
    api_usage_hint,
    forget_api,
    get_api_knowledge,
    learn_api_schema,
)
from ultron.core.learning.associations import (
    discover_connections,
    explain_relation,
    memory_connections,
    related_facts,
)
from ultron.core.tools.builtin.command_runner import run_command, run_parallel
from ultron.core.tools.builtin.database import run_query
from ultron.core.tools.builtin.file_reader import read_file
from ultron.core.tools.builtin.file_writer import write_file
from ultron.core.tools.builtin.http_client import make_http_request
from ultron.core.tools.builtin.retrieval import check_connectivity, retrieve
from ultron.core.tools.builtin.web_search import fetch_page_text, search_web
from ultron.core.tools.memory.graph import (
    add_triple,
    get_all_triples,
    query_triples,
    search_triples,
    store_memory_text,
)
from ultron.core.tools.memory.sqlite import (
    get_all_memories,
    search_memories,
)
from ultron.core.tools.resource_monitor import check_resources, resource_forecast

# A dictionary that maps a tool's name (as a string) to the actual Python function.
# This makes it easy to lookup and call tools dynamically by their names.
#
# ``add_memory`` is the unified memory write: it extracts subject/predicate/
# object triples from the sentence when possible (knowledge graph) and falls
# back to the flat fact store otherwise — see ultron/core/tools/memory/graph.py.
TOOLS: dict[str, Callable[..., Any]] = {
    "read_file": read_file,
    "write_file": write_file,
    # --- Fix #3 coding workspace + execution context tools ---
    "list_directory": list_directory,
    "search_files": search_files,
    "discover_workspace_summary": discover_workspace_summary,
    # --- Fix #4 code intelligence (read-only) ---
    "code_search": code_search,
    "find_symbol": find_symbol,
    "find_definition": find_definition,
    "find_references": find_references,
    "get_imports": get_imports,
    "get_dependents": get_dependents,
    "semantic_search": semantic_search,
    "code_index_status": code_index_status,
    "report_file": report_file,
    "report_symbol": report_symbol,
    "create_file": create_file,
    "replace_file": replace_file,
    "replace_in_file": replace_in_file,
    "append_to_file": append_to_file,
    "delete_file": delete_file,
    "rename_file": rename_file,
    "run_command": run_command,
    "run_parallel": run_parallel,
    "make_http_request": make_http_request,
    "retrieve": retrieve,
    "check_connectivity": check_connectivity,
    "learn_api_schema": learn_api_schema,
    "api_usage_hint": api_usage_hint,
    "get_api_knowledge": get_api_knowledge,
    "forget_api": forget_api,
    "check_resources": check_resources,
    "resource_forecast": resource_forecast,
    "memory_connections": memory_connections,
    "related_facts": related_facts,
    "discover_connections": discover_connections,
    "explain_relation": explain_relation,
    "enforce_schema": enforce_schema,
    "schema_validate": schema_validate,
    "list_schemas": list_schemas,
    "preflight_plan": preflight_plan_tool,
    "analyze_dependencies": analyze_dependencies,
    "list_plan_actions": list_plan_actions,
    "get_debug_context": get_debug_context,
    "diagnose_failure": diagnose_failure,
    "check_dependency": check_dependency,
    "run_tool_batch": run_tool_batch,
    "synthesize_analysis": synthesize_analysis,
    "search_web": search_web,
    "fetch_page_text": fetch_page_text,
    "run_query": run_query,
    "add_memory": store_memory_text,
    "add_triple": add_triple,
    "get_all_memories": get_all_memories,
    "search_memories": search_memories,
    "query_triples": query_triples,
    "search_triples": search_triples,
    "get_all_triples": get_all_triples,
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
