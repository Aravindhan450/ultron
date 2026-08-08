"""ultron.core.intelligence.planning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Proactive dependency identification — the plan preflight.

When a multi-step request (``read file → run command → HTTP request``) is
planned, nothing runs until the whole chain has been analyzed and shown:

- ``analyze_step`` — the boundary verdict for one step (same ``check_action``
  gate the executor uses), plus missing-required-field detection.
- ``preflight_plan`` — the full analysis: per-step verdicts, allow/confirm/
  blocked counts, blocked reasons, missing info.
- ``find_dependencies`` — deterministic same-target data-flow edges between
  steps (write F → read F, write F → command mentioning F, …).
- ``format_plan_preview`` — the human-readable upfront listing with
  permission badges and dependency lines.

The preflight is informational + consent-aggregating: `handle_multistep`
offers the whole plan for one upfront approval when any step needs it, and
never offers a plan containing a blocked step. The execution-time security
gate is unchanged — every step is still re-checked before it runs.

See docs/planning.md.
"""

from __future__ import annotations

import json
import re

# NOTE: check_action is imported lazily inside analyze_step to avoid a
# circular import (agents -> registry -> planning -> agents).

# ---------------------------------------------------------------------------
# Supported plan actions
# ---------------------------------------------------------------------------

# action -> (tool name, human label, required fields)
PLAN_ACTION_SPECS: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "read_file": ("read_file", "read file", ("filename",)),
    "write_file": ("write_file", "write file", ("filename", "content")),
    "run_command": ("run_command", "run command", ("command",)),
    "run_parallel": ("run_parallel", "run commands in parallel", ("commands",)),
    "make_http_request": ("make_http_request", "HTTP request", ("url",)),
    "run_query": ("run_query", "database query", ("sql",)),
    "add_memory": ("add_memory", "remember fact", ("fact",)),
}

PLAN_ACTIONS = tuple(PLAN_ACTION_SPECS)


def list_plan_actions() -> str:
    """Lists the action types the planner can produce, with their tools."""
    lines = ["Plan steps can use these action types:"]
    for action, (tool, label, required) in PLAN_ACTION_SPECS.items():
        req = ", ".join(required)
        lines.append(f"• {action} (tool '{tool}') — {label}; requires: {req}")
    lines.append("")
    lines.append("Ask e.g. 'read config.py, run pytest, then POST the result to http://api'.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Step analysis
# ---------------------------------------------------------------------------


def step_target_content(step: dict) -> tuple[str, str | None]:
    """Maps a plan step to the (target, content) pair the boundary expects."""
    action = step.get("action", "")
    if action == "read_file":
        return str(step.get("filename", "")), None
    if action == "write_file":
        return str(step.get("filename", "")), str(step.get("content", ""))
    if action == "run_command":
        return str(step.get("command", "")), None
    if action == "run_parallel":
        return "\n".join(str(c) for c in step.get("commands", [])), None
    if action == "make_http_request":
        return str(step.get("url", "")), str(step.get("body") or step.get("method", ""))
    if action == "run_query":
        return str(step.get("sql", "")), None
    if action == "add_memory":
        return "", str(step.get("fact", ""))
    return "", None


def _action_label(step: dict) -> str:
    action = step.get("action", "")
    spec = PLAN_ACTION_SPECS.get(action)
    if not spec:
        return f"unknown action '{action}'"
    _, label, _ = spec
    return label


def _target_text(step: dict) -> str:
    action = step.get("action", "")
    if action == "read_file":
        return str(step.get("filename", ""))
    if action == "write_file":
        return str(step.get("filename", ""))
    if action == "run_command":
        return str(step.get("command", ""))
    if action == "run_parallel":
        return "; ".join(str(c) for c in step.get("commands", []))
    if action == "make_http_request":
        method = str(step.get("method", "GET")).upper()
        return f"{method} {step.get('url', '')}"
    if action == "run_query":
        return str(step.get("sql", ""))
    if action == "add_memory":
        return str(step.get("fact", ""))
    return ""


def analyze_step(step: dict, index: int = 1) -> dict:
    """
    Analyzes one plan step: boundary verdict + missing required fields.

    Returns a dict with action, tool, label, target, risk (tier value),
    decision (allow/confirm/deny), missing (list of absent required
    fields), and reason. Uses the same check_action gate the executor
    uses, so the preview matches what execution will enforce.
    """
    action = step.get("action", "")
    spec = PLAN_ACTION_SPECS.get(action)
    target, content = step_target_content(step)

    missing: list[str] = []
    if spec:
        for field in spec[2]:
            if not step.get(field):
                missing.append(field)

    from ultron.core.agents.security import check_action  # lazy: breaks import cycle

    verdict = check_action(action, target, content)
    return {
        "index": index,
        "action": action,
        "tool": spec[0] if spec else action,
        "label": _action_label(step),
        "target": _target_text(step),
        "risk": verdict.tier.value,
        "decision": verdict.decision.value,
        "reason": verdict.reason,
        "missing": missing,
    }


def preflight_plan(steps: list[dict]) -> dict:
    """
    Full preflight of an ordered step list.

    Returns:
        steps: per-step analyze_step() results
        summary: {auto, confirm, blocked, missing} counts
        blocked: list of (index, action, reason) for denied steps
        missing_info: list of (index, action, missing fields)
    """
    analyses = [analyze_step(step, i) for i, step in enumerate(steps, start=1)]

    auto = sum(1 for a in analyses if a["decision"] == "allow")
    confirm = sum(1 for a in analyses if a["decision"] == "confirm")
    # Security denials are deny verdicts on steps that are otherwise complete.
    # A deny caused by an *empty target* ("File target is empty") means the
    # step is incomplete, not a security violation — it belongs in
    # missing_info so the plan is offered for approval with the gap shown.
    blocked = [
        (a["index"], a["action"], a["reason"])
        for a in analyses
        if a["decision"] == "deny" and not a["missing"]
    ]
    missing_info = [
        (a["index"], a["action"], a["missing"]) for a in analyses if a["missing"]
    ]

    return {
        "steps": analyses,
        "summary": {
            "auto": auto,
            "confirm": confirm,
            "blocked": len(blocked),
            "missing": len(missing_info),
        },
        "blocked": blocked,
        "missing_info": missing_info,
    }


# ---------------------------------------------------------------------------
# Dependency edges
# ---------------------------------------------------------------------------


def _step_filename(step: dict) -> str:
    action = step.get("action", "")
    if action in {"read_file", "write_file"}:
        return str(step.get("filename", ""))
    return ""


def _step_command_text(step: dict) -> str:
    action = step.get("action", "")
    if action == "run_command":
        return str(step.get("command", ""))
    if action == "run_parallel":
        return " ".join(str(c) for c in step.get("commands", []))
    return ""


def _mentions(text: str, token: str) -> bool:
    token = (token or "").strip()
    if not token:
        return False
    base = re.sub(r"\.(?:py|txt|json|md|toml|yaml|yml|cfg|ini|csv|log|db|sql|js|ts)$", "", token, flags=re.IGNORECASE)
    # Very short bases ("a.txt" → "a") would match nearly any command;
    # only report dependencies for meaningful filenames.
    if len(base) < 3:
        return False
    base = re.escape(base)
    # Bounded on both sides so "data2" never matches base "data".
    return bool(re.search(rf"(?<![\w./-]){base}(?![\w])", text, re.IGNORECASE))


def find_dependencies(steps: list[dict]) -> list[dict]:
    """
    Deterministic same-target data-flow edges between ordered steps.

    Returns a list of {from, to, kind, reason} dicts (from < to). Only
    forward edges are emitted and pairs are deduplicated.
    """
    edges: list[dict] = []
    seen: set[tuple[int, int, str]] = set()

    def add(from_i: int, to_j: int, kind: str, reason: str) -> None:
        key = (from_i, to_j, kind)
        if key in seen:
            return
        seen.add(key)
        edges.append({"from": from_i, "to": to_j, "kind": kind, "reason": reason})

    for i, step in enumerate(steps, start=1):
        action = step.get("action", "")
        fname = _step_filename(step)
        for j in range(i + 1, len(steps) + 1):
            later = steps[j - 1]
            later_action = later.get("action", "")
            if action == "write_file" and fname:
                if later_action == "read_file" and str(later.get("filename", "")) == fname:
                    add(i, j, "producer", f"step {i} writes '{fname}', read in step {j}")
                cmd = _step_command_text(later)
                if cmd and _mentions(cmd, fname):
                    add(i, j, "producer", f"step {i} creates '{fname}', used by step {j}'s command")
            elif action == "read_file" and fname:
                cmd = _step_command_text(later)
                if cmd and _mentions(cmd, fname):
                    add(i, j, "feeds", f"step {i} reads '{fname}', likely fed to step {j}'s command")
    return edges


# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------


def _decision_badge(decision: str) -> str:
    if decision == "allow":
        return "⚡ auto"
    if decision == "confirm":
        return "🛡 needs approval"
    return "⛔ blocked"


def _command_warning(command: str) -> str:
    try:
        from ultron.core.tools.resource_monitor import (
            forecast_severity,
            forecast_warning,
        )
        severity = forecast_severity(command)
        warning = forecast_warning(command)
        if severity in {"heavy", "critical"} and warning:
            return f" ⚠ {warning}"
    except (ImportError, OSError, ValueError):
        pass
    return ""


def format_plan_preview(steps: list[dict], preflight: dict | None = None) -> str:
    """
    The upfront listing: steps + permission badges + dependencies.

    ``preflight`` may be passed in when the caller already computed it, so
    steps are classified once instead of once per preview + once per
    preflight (each classification also writes an audit entry).
    """
    if not steps:
        return "No steps to run."

    preflight = preflight or preflight_plan(steps)
    analyses = preflight["steps"]

    tools = {a["tool"] for a in analyses}
    lines = [f"📋 Plan — {len(steps)} step{'s' if len(steps) != 1 else ''} · {len(tools)} tool{'s' if len(tools) != 1 else ''}"]

    for a in analyses:
        suffix = ""
        if a["missing"]:
            suffix = f" ⚠ missing: {', '.join(a['missing'])}"
        if a["action"] == "run_command":
            suffix += _command_warning(a["target"])
        lines.append(
            f"{a['index']}. {a['label']} '{a['target']}' — {_decision_badge(a['decision'])}{suffix}"
        )

    deps = find_dependencies(steps)
    if deps:
        lines.append("Dependencies:")
        for d in deps[:5]:
            lines.append(f"  • {d['reason']}")

    s = preflight["summary"]
    lines.append(
        f"Permissions: {s['auto']} auto · {s['confirm']} confirm · "
        f"{s['blocked']} blocked"
    )

    if s["missing"]:
        lines.append(
            f"⚠ {s['missing']} step{'s' if s['missing'] != 1 else ''} missing "
            "required information — check the flagged steps."
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Registered tools (read-only local analysis)
# ---------------------------------------------------------------------------


def _parse_steps(steps_json: str) -> list[dict] | None:
    if not steps_json or not steps_json.strip():
        return None
    try:
        data = json.loads(steps_json)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, list):
        return None
    return [s for s in data if isinstance(s, dict)]


def preflight_plan_tool(steps_json: str) -> str:
    """
    Analyzes a JSON plan (list of step dicts) and returns the permission +
    dependency preview — what will run, what needs approval, what is blocked.
    """
    steps = _parse_steps(steps_json)
    if steps is None:
        return "Error: steps_json must be a JSON array of step objects."
    return format_plan_preview(steps)


def analyze_dependencies(steps_json: str) -> str:
    """Returns the same-target data-flow dependencies of a JSON plan."""
    steps = _parse_steps(steps_json)
    if steps is None:
        return "Error: steps_json must be a JSON array of step objects."
    deps = find_dependencies(steps)
    if not deps:
        return "No same-target dependencies found between steps."
    lines = [f"{len(deps)} dependency edge{'s' if len(deps) != 1 else ''}:"]
    for d in deps:
        lines.append(f"  • {d['reason']}")
    return "\n".join(lines)
