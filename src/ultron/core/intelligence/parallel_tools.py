"""ultron.core.intelligence.parallel_tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Inter-tool process optimization — parallel tool dispatch with synthesis.

When one request needs results from several *different* tools (read a file,
search the web, check a site, query the database), Ultron used to run them
one after another and return disconnected replies. This module runs the
calls **concurrently** and weaves their outputs into **one coherent
analysis**:

- :func:`run_tool_batch` — the registered tool. Takes a JSON array of
  ``{"tool": ..., "arguments": {...}}`` calls, gates **each** call through
  the security boundary (same ``check_action`` the single-call paths use),
  executes every auto-allowed call concurrently in a thread pool, and
  returns a synthesized report. A ``deny`` verdict never executes; a
  ``confirm`` verdict is reported as "needs approval" instead of running
  silently — the batch is only as safe as its most dangerous member.
- :func:`synthesize_results` — deterministic synthesis: per-tool summaries,
  cross-tool keyword connections, and a combined "what we learned" section.
- :func:`plan_tool_batch` — the LLM planner (like ``plan_task``) that turns
  an ambiguous request ("gather everything about X") into the batch.
- :func:`synthesize_analysis` — standalone tool combining a list of
  ``{tool, result}`` outputs into one analysis block.

Concurrency changes *timing*, never *permission*: the gating inside a batch
is byte-for-byte the same decision the single-call paths enforce.

See docs/parallel-tools.md.
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor

# NOTE: check_action / _generic_target_content / get_tool are imported lazily
# inside the functions that need them to avoid import cycles
# (agents -> registry -> intelligence -> agents).

MAX_BATCH_CALLS = 8
_MAX_RESULT_CHARS = 240

# Boundary *action* names <-> registry *tool* names. The security layer and
# the agent detectors speak action names ("web_search"), while the tool
# registry keys the function under its own name ("search_web"). Batch calls
# may arrive in either spelling — the ReAct agent emits registry names (from
# get_tools_schema), detectors/planner emit action names. Every name is
# canonicalized to the *action* name for gating + wave classification + labels,
# and only resolved to the registry key at execution time.
#
# Aliases are owned by the canonical tool definitions table (STEP 2A); these
# helpers derive from it lazily (parallel_tools is imported by definitions,
# so a module-level import would be circular).


def _canonical_action_name(name: str) -> str:
    """Maps any spelling (action or registry tool name) to the boundary action name.

    A tool with canonical aliases maps to its first alias (``search_web`` ->
    ``web_search``); an already-alias spelling maps to the same alias; tools
    without aliases are unchanged.
    """
    from ultron.core.tools.definitions import TOOL_DEFINITIONS

    for definition in TOOL_DEFINITIONS.values():
        if name == definition.name or name in definition.aliases:
            if definition.aliases:
                return definition.aliases[0]
            return definition.name
    return name


def _registry_tool_name(action_name: str) -> str:
    """Maps any spelling to the registry tool name (identity by default)."""
    from ultron.core.tools.definitions import canonical_action_name

    return canonical_action_name(action_name)


# POLICY (concurrency scheduling), not tool metadata: tools eligible for the
# parallel read wave. Read-only status itself lives in the canonical tool
# metadata (tools/definitions.py); this set is deliberately curated —
# code-intelligence / filesystem-list tools stay sequential to keep
# exploration ordering deterministic, and run_query is treated as a read for
# batching even though canonical flags it write-capable. A state-writing tool
# (add_memory, add_triple, forget_api, …) is auto-allowed by the boundary
# (LOW) but must NOT run in parallel with other writers — concurrent SQLite
# writers collide with "database is locked". Such members still execute, but
# sequentially after the read wave.
BATCH_READONLY_TOOLS = frozenset(
    {
        "read_file",
        "web_search",
        "fetch_page_text",
        "check_connectivity",
        "retrieve",
        "get_api_knowledge",
        "api_usage_hint",
        "check_resources",
        "resource_forecast",
        "memory_connections",
        "related_facts",
        "discover_connections",
        "explain_relation",
        "enforce_schema",
        "schema_validate",
        "list_schemas",
        "preflight_plan",
        "analyze_dependencies",
        "list_plan_actions",
        "get_debug_context",
        "diagnose_failure",
        "check_dependency",
        "get_all_memories",
        "search_memories",
        "query_triples",
        "query_chain",
        "search_triples",
        "get_all_triples",
        "run_query",
        # make_http_request is deliberately excluded: a state-changing method
        # (POST/PUT/DELETE/PATCH) is auto-allowed under permissive mode and
        # must not run concurrently — it goes through the sequential wave.
    }
)

# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def _safe_run(tool_name: str, arguments: dict) -> str:
    """Runs one tool call, capturing any exception as a message."""
    from ultron.core.tools.registry import get_tool

    func = get_tool(_registry_tool_name(tool_name))
    if func is None:
        return f"Error: unknown tool '{tool_name}'."
    try:
        return str(func(**arguments))
    except Exception as exc:  # noqa: BLE001 — arbitrary tool surface
        return f"Error executing tool '{tool_name}': {exc}"


def execute_batch(calls: list[dict]) -> tuple[list[dict], float]:
    """
    Gates each call through the security boundary, then executes the batch.

    Concurrency rule: only read-only tools (``BATCH_READONLY_TOOLS``) run in
    parallel — state-writing LOW tools (e.g. add_memory) execute sequentially
    afterwards so SQLite writers never collide. Returns
    (gated_calls, elapsed_seconds) where each gated call has a ``status``:
    ``ok`` / ``error`` (executed) or ``blocked`` / ``confirm`` (never
    executed). ``deny`` verdicts never execute; ``confirm`` verdicts are
    surfaced as needing approval rather than running silently.
    """
    from ultron.core.agents.security import (
        blocked_message,
        check_action,
        is_allow,
        is_denied,
    )
    from ultron.core.agents.simple import _generic_target_content
    from ultron.core.tools.registry import get_tool

    gated: list[dict] = []
    for call in calls[:MAX_BATCH_CALLS]:
        # Canonicalize to the boundary action name up front so gating, wave
        # classification, and labels all speak the same language regardless
        # of which spelling the caller used.
        tool_name = _canonical_action_name(str(call.get("tool", "")))
        arguments = call.get("arguments") or {}
        entry = {"tool": tool_name, "arguments": arguments}

        if get_tool(_registry_tool_name(tool_name)) is None:
            entry.update(status="error", result=f"Error: unknown tool '{tool_name}'.")
            gated.append(entry)
            continue

        target, content = _generic_target_content(tool_name, arguments)
        verdict = check_action(tool_name, target, content)
        if is_denied(verdict):
            entry.update(status="blocked", result=blocked_message(verdict))
        elif not is_allow(verdict):  # confirm
            entry.update(
                status="confirm",
                result=f"Needs approval: {verdict.reason}",
            )
        else:
            entry["status"] = "run"
        gated.append(entry)

    runnable = [c for c in gated if c["status"] == "run"]
    read_wave = [c for c in runnable if c["tool"] in BATCH_READONLY_TOOLS]
    write_wave = [c for c in runnable if c["tool"] not in BATCH_READONLY_TOOLS]

    started = time.perf_counter()
    if read_wave:
        workers = min(len(read_wave), MAX_BATCH_CALLS)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_safe_run, c["tool"], c["arguments"]): c for c in read_wave
            }
            for future, call in futures.items():
                result = future.result()
                call["result"] = result
                call["status"] = "ok" if not str(result).startswith("Error") else "error"
    # State-writing LOW tools (e.g. add_memory) run after the read wave,
    # one at a time, so SQLite writers never contend.
    for call in write_wave:
        result = _safe_run(call["tool"], call["arguments"])
        call["result"] = result
        call["status"] = "ok" if not str(result).startswith("Error") else "error"
    elapsed = time.perf_counter() - started
    return gated, elapsed


def _parse_calls(calls_json: str) -> list[dict] | None:
    if not calls_json or not calls_json.strip():
        return None
    try:
        data = json.loads(calls_json)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, list):
        return None
    calls = [c for c in data if isinstance(c, dict) and c.get("tool")]
    return calls[:MAX_BATCH_CALLS] or None


def run_tool_batch(calls_json: str) -> str:
    """
    Runs several tool calls concurrently, each gated through the security
    boundary, and returns a synthesized report.

    Arguments:
        calls_json: JSON array of {"tool": ..., "arguments": {...}}.

    Safety: a call whose boundary verdict is deny is never executed and is
    reported as blocked; a confirm verdict is reported as needing approval
    instead of running silently. Only auto-allowed calls execute, all at
    the same time.
    """
    calls = _parse_calls(calls_json)
    if calls is None:
        return "Error: run_tool_batch expects a JSON array of {tool, arguments} objects."
    # Deduplicate identical calls (same tool + same arguments) so a batch is
    # never executed twice by accident. Order of first occurrence is kept.
    seen: set[tuple] = set()
    unique: list[dict] = []
    for call in calls:
        # default=str keeps the key robust against non-JSON-serializable
        # argument values (e.g. a tuple passed directly to the tool).
        key = (
            str(call.get("tool", "")),
            json.dumps(call.get("arguments") or {}, sort_keys=True, default=str),
        )
        if key not in seen:
            seen.add(key)
            unique.append(call)
    gated, elapsed = execute_batch(unique)
    return format_batch_report(gated, elapsed=elapsed)


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset(
    [
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "for",
        "from", "has", "have", "in", "into", "is", "it", "its", "of",
        "on", "or", "that", "the", "this", "to", "was", "were", "will",
        "with", "you", "your", "they", "them", "their", "we", "our",
        "not", "no", "if", "then", "than", "so", "what", "which",
        "when", "where", "how", "all", "any", "can", "could", "should",
        "would", "about", "after", "before", "between", "during", "each",
        "few", "more", "most", "other", "some", "such", "only", "own",
        "same", "too", "very", "just", "also", "new", "out", "over",
        "up", "down",
    ]
)


def _keywords(text: str) -> set[str]:
    """Lowercased content words (>= 4 chars) minus stopwords."""
    words = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{3,}", (text or "").lower())
    return {w for w in words if w not in _STOPWORDS}


def synthesize_results(results: list[dict], include_sources: bool = True) -> str:
    """
    Deterministic synthesis of executed tool results.

    ``results`` is a list of {tool, result} dicts. Produces a "what we
    learned" block: cross-source keyword connections (terms appearing in
    more than one result). When ``include_sources`` is True the per-source
    summaries are listed too; the batch report passes False because the
    per-call sections already show the full results.
    """
    if not results:
        return "No results to synthesize."

    summaries: list[str] = []
    keyword_sets: list[set[str]] = []
    for item in results:
        tool = item.get("tool", "?")
        text = str(item.get("result", ""))
        flat = re.sub(r"\s+", " ", text).strip()
        if len(flat) > _MAX_RESULT_CHARS:
            flat = flat[:_MAX_RESULT_CHARS] + "…"
        summaries.append(f"• {tool}: {flat}" if flat else f"• {tool}: (no output)")
        keyword_sets.append(_keywords(text))

    lines = ["🧠 Combined analysis"]
    shared: list[str] = []
    if len(keyword_sets) >= 2:
        common: dict[str, int] = {}
        for kw_set in keyword_sets:
            for kw in kw_set:
                common[kw] = common.get(kw, 0) + 1
        shared = sorted((kw for kw, n in common.items() if n >= 2), key=common.get, reverse=True)
        if shared:
            top = ", ".join(shared[:8])
            lines.append(f"• Shared across sources: {top}")
    if include_sources:
        lines.append("• Sources:")
        lines.extend(summaries)
    elif not shared:
        lines.append(f"• {len(results)} source{'s' if len(results) != 1 else ''} consulted — no shared terms across them.")
    return "\n".join(lines)


def synthesize_analysis(results_json: str) -> str:
    """Combines a JSON list of {tool, result} outputs into one analysis block."""
    if not results_json or not results_json.strip():
        return "Error: results_json must be a JSON array of {tool, result} objects."
    try:
        results = json.loads(results_json)
    except (json.JSONDecodeError, TypeError):
        return "Error: results_json must be a JSON array of {tool, result} objects."
    if not isinstance(results, list):
        return "Error: results_json must be a JSON array of {tool, result} objects."
    return synthesize_results([r for r in results if isinstance(r, dict)])


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _status_badge(status: str) -> str:
    if status in {"ok", "run"}:
        return "✅"
    if status == "error":
        return "⚠️"
    if status == "blocked":
        return "⛔"
    return "🛡"  # confirm


def _call_label(call: dict) -> str:
    tool = call.get("tool", "?")
    args = call.get("arguments") or {}
    if not args:
        return tool
    parts = []
    for key in ("file_path", "filename", "url", "query", "sql", "keyword"):
        if key in args:
            parts.append(f"{key}={args[key]}")
            break
    if not parts:
        first_key = next(iter(args), "")
        parts.append(f"{first_key}={args[first_key]}")
    return f"{tool} {', '.join(parts)}"


def format_batch_report(gated: list[dict], elapsed: float | None = None) -> str:
    """Renders the gated batch (executed + blocked + needs-approval) as one report."""
    if not gated:
        return "No tool calls to run."

    executed = [c for c in gated if c["status"] in {"ok", "error"}]
    timing = f" · {elapsed:.2f}s" if elapsed is not None else ""
    lines = [
        (
            f"⚡ Parallel batch — {len(gated)} call{'s' if len(gated) != 1 else ''} · "
            f"{len(executed)} executed{timing}"
        )
    ]
    lines.append("")

    for i, call in enumerate(gated, start=1):
        status = call["status"]
        badge = _status_badge(status)
        label = _call_label(call)
        result = str(call.get("result", ""))
        if status == "ok":
            flat = re.sub(r"\s+", " ", result).strip()
            if len(flat) > _MAX_RESULT_CHARS:
                flat = flat[:_MAX_RESULT_CHARS] + "…"
            lines.append(f"[{i}] {badge} {label}")
            lines.append(f"    {flat}" if flat else "    (no output)")
        elif status == "error":
            lines.append(f"[{i}] {badge} {label} — error: {result}")
        elif status == "blocked":
            lines.append(f"[{i}] {badge} {label} — blocked (never ran): {result}")
        else:  # confirm
            lines.append(f"[{i}] {badge} {label} — needs approval (not run in batch): {result}")
        lines.append("")

    ok_results = [{"tool": c["tool"], "result": c["result"]} for c in gated if c["status"] == "ok"]
    if ok_results:
        # include_sources=False: the per-call sections above already show
        # the full results — the analysis adds only the cross-source view.
        lines.append(synthesize_results(ok_results, include_sources=False))

    needs_approval = [c for c in gated if c["status"] == "confirm"]
    if needs_approval:
        lines.append("")
        lines.append(
            "ℹ️ Calls needing approval were NOT run — say 'yes' to execute them "
            "through the normal confirmation flow."
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM planner
# ---------------------------------------------------------------------------

# Tools the planner may suggest. The batch gate protects everything anyway,
# but steering the planner to read-mostly tools keeps batches predictable.
PLANNER_TOOLS = (
    "read_file",
    "web_search",
    "fetch_page_text",
    "check_connectivity",
    "run_query",
    "search_memories",
    "get_all_memories",
    "search_triples",
    "get_all_triples",
    "get_debug_context",
    "check_resources",
    "get_api_knowledge",
    "api_usage_hint",
    "memory_connections",
    "related_facts",
    "list_schemas",
)


async def plan_tool_batch(user_input: str, engine) -> list[dict] | None:
    """
    Asks the LLM to turn a request into independent, parallelizable tool
    calls. Returns a list of {tool, arguments} dicts, or None on failure.
    """
    import httpx

    allowlist = ", ".join(PLANNER_TOOLS)
    prompt = (
        "You are a parallel tool-planning assistant.\n"
        "Break the user request into INDEPENDENT tool calls that can run at "
        "the same time (no call depends on another's output).\n"
        f"Only use these tools: {allowlist}.\n"
        "Respond with ONLY a JSON array — no other text, no markdown fences.\n"
        'Each item must be exactly: {"tool": "<name>", "arguments": {...}}\n'
        "Example: "
        '[{"tool": "read_file", "arguments": {"file_path": "config.json"}}, '
        '{"tool": "web_search", "arguments": {"query": "pandas release notes"}}]\n'
        "\n"
        f"User request: {user_input}"
    )
    try:
        raw = await engine.generate([{"role": "user", "content": prompt}])
    except (httpx.HTTPError, OSError, ValueError):
        return None

    raw = (raw or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[\w]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
        raw = raw.strip()

    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, list):
        return None

    allowed = set(PLANNER_TOOLS)
    calls: list[dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        tool = str(item.get("tool", ""))
        arguments = item.get("arguments")
        if tool in allowed and isinstance(arguments, dict):
            calls.append({"tool": tool, "arguments": arguments})
    return calls[:MAX_BATCH_CALLS] or None
