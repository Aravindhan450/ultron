"""ultron.core.runtime.policy_gate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RUNTIME POLICY GATE — the deterministic boundary evaluating tool invocations
against the authoritative ExecutionPolicy BEFORE the security boundary.

Invariants:
- The Policy Gate executes BEFORE the Security Gate.
- Rejects any tool requiring capabilities not granted by the active ExecutionPolicy.
- Blocks file/state mutation when mutation_authority is FORBIDDEN.
- Blocks command/process execution when execution_authority is FORBIDDEN.
- Flags confirmation requirements when required by the policy.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel

from ultron.core.types import (
    AuthorityLevel,
    Capability,
    ExecutionPolicy,
)

# Regex to detect safe, read-only git subcommands
_GIT_READ_RE = re.compile(
    r"^git\s+(status|log|diff|branch|show|rev-parse|describe|tag\s+-l|config\s+--get)\b",
    re.IGNORECASE,
)

# Regex to detect test runner commands
_TEST_CMD_RE = re.compile(
    r"\b(pytest|unittest|npm\s+test|cargo\s+test|go\s+test|vitest|jest|ruff|mypy|flake8)\b",
    re.IGNORECASE,
)

# Regex to detect mutating SQL statements
_SQL_MUTATION_RE = re.compile(
    r"^\s*(insert|update|delete|drop|alter|create|replace|truncate)\b",
    re.IGNORECASE,
)


class PolicyVerdict(BaseModel):
    """Result of evaluating a tool invocation against the active ExecutionPolicy."""

    allowed: bool
    requires_confirmation: bool = False
    reason: str = ""
    violation_type: str | None = None
    required_capability: Capability | None = None


def classify_tool_capability(
    tool_name: str,
    arguments: dict[str, Any] | None = None,
) -> tuple[Capability, bool, bool]:
    """
    Classifies a tool invocation into (primary_capability, is_mutating, is_executing).
    """
    args = arguments or {}

    # 1. Filesystem mutations
    if tool_name in (
        "write_file",
        "append_to_file",
        "replace_file",
        "replace_in_file",
        "create_file",
        "rename_file",
    ):
        return Capability.WRITE_FILES, True, False

    if tool_name == "delete_file":
        return Capability.DELETE_FILES, True, False

    # 2. Filesystem reads
    if tool_name in ("read_file", "list_directory"):
        return Capability.READ_FILES, False, False

    # 3. Repository inspection & search
    if tool_name in (
        "search_files",
        "discover_workspace_summary",
        "code_index_status",
        "code_investigation",
        "code_search",
        "semantic_search",
        "repo_map",
        "report_file",
    ):
        return Capability.SEARCH_REPOSITORY, False, False

    # 4. Symbol inspection
    if tool_name in (
        "find_definition",
        "find_references",
        "find_symbol",
        "report_symbol",
        "get_dependents",
        "get_imports",
    ):
        return Capability.INSPECT_SYMBOLS, False, False

    # 5. Command & script execution
    if tool_name in ("run_command", "run_parallel"):
        cmd = str(args.get("command", "")).strip()
        if _GIT_READ_RE.search(cmd):
            return Capability.GIT_READ, False, False
        if _TEST_CMD_RE.search(cmd):
            return Capability.RUN_TESTS, False, True
        return Capability.RUN_COMMANDS, True, True

    # 6. Network tools
    if tool_name in ("search_web", "fetch_page_text", "retrieve", "check_connectivity"):
        return Capability.NETWORK, False, False

    if tool_name == "make_http_request":
        method = str(args.get("method", "GET")).upper()
        if method in ("POST", "PUT", "DELETE", "PATCH"):
            return Capability.NETWORK, True, False
        return Capability.NETWORK, False, False

    # 7. Database tools
    if tool_name == "run_query":
        sql = str(args.get("sql", args.get("query", ""))).strip()
        if _SQL_MUTATION_RE.search(sql):
            return Capability.EXTERNAL_ACTIONS, True, False
        return Capability.EXTERNAL_ACTIONS, False, False

    # 8. Memory mutation tools
    if tool_name in ("add_memory", "add_triple", "forget_api", "learn_api_schema"):
        return Capability.WRITE_FILES, True, False

    # 9. Read-only memory & analysis tools
    if tool_name in (
        "get_all_memories",
        "get_all_triples",
        "search_memories",
        "search_triples",
        "query_triples",
        "memory_connections",
        "discover_connections",
        "explain_relation",
        "related_facts",
        "analyze_dependencies",
        "list_plan_actions",
        "preflight_plan",
        "enforce_schema",
        "list_schemas",
        "schema_validate",
        "api_usage_hint",
        "get_api_knowledge",
        "check_dependency",
        "diagnose_failure",
        "get_debug_context",
        "check_resources",
        "resource_forecast",
        "run_tool_batch",
        "synthesize_analysis",
    ):
        return Capability.READ_FILES, False, False

    # Fallback for arbitrary/unknown tools
    if tool_name.startswith(("read_", "get_", "list_", "find_", "search_")):
        return Capability.READ_FILES, False, False
    return Capability.EXTERNAL_ACTIONS, True, True


def check_runtime_policy(
    policy: ExecutionPolicy | None,
    tool_name: str,
    arguments: dict[str, Any] | None = None,
) -> PolicyVerdict:
    """
    Evaluates a tool invocation against the active ExecutionPolicy.

    When policy is None, defaults to ExecutionPolicy.read_only().
    """
    active_policy = policy if policy is not None else ExecutionPolicy.read_only()
    capability, is_mutating, is_executing = classify_tool_capability(tool_name, arguments)

    # 1. Capability membership check
    if capability not in active_policy.allowed_capabilities:
        return PolicyVerdict(
            allowed=False,
            reason=(
                f"Capability '{capability.value}' required by tool '{tool_name}' "
                f"is not granted by the active policy."
            ),
            violation_type="FORBIDDEN_CAPABILITY",
            required_capability=capability,
        )

    # 2. Mutation authority check
    if is_mutating:
        if active_policy.mutation_authority == AuthorityLevel.FORBIDDEN:
            return PolicyVerdict(
                allowed=False,
                reason=(
                    f"Tool '{tool_name}' performs mutation, which is forbidden "
                    f"by the active execution policy."
                ),
                violation_type="MUTATION_FORBIDDEN",
                required_capability=capability,
            )
        if active_policy.mutation_authority == AuthorityLevel.REQUIRES_CONFIRMATION:
            return PolicyVerdict(
                allowed=True,
                requires_confirmation=True,
                reason=f"Tool '{tool_name}' performs mutation requiring confirmation.",
                required_capability=capability,
            )

    # 3. Execution authority check
    if is_executing:
        if active_policy.execution_authority == AuthorityLevel.FORBIDDEN:
            return PolicyVerdict(
                allowed=False,
                reason=(
                    f"Tool '{tool_name}' performs command/process execution, which is forbidden "
                    f"by the active execution policy."
                ),
                violation_type="EXECUTION_FORBIDDEN",
                required_capability=capability,
            )
        if active_policy.execution_authority == AuthorityLevel.REQUIRES_CONFIRMATION:
            return PolicyVerdict(
                allowed=True,
                requires_confirmation=True,
                reason=f"Tool '{tool_name}' performs execution requiring confirmation.",
                required_capability=capability,
            )

    return PolicyVerdict(
        allowed=True,
        required_capability=capability,
    )
