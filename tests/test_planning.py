"""
Tests for proactive dependency identification
(``ultron.core.intelligence.planning``).

Covers the per-step permission matrix (auto / confirm / blocked), missing
required-field detection, same-target dependency edges, the upfront plan
preview, the three registered tools, LOW-risk classification, the extended
planner (HTTP + SQL steps), and the agent-level gate: plans with blocked
steps are never offered, plans needing approval wait for one upfront
consent, all-auto plans run immediately with the preview shown.
"""

import asyncio
import json

import pytest

from ultron.core.agents.simple import (
    SimpleAgent,
    execute_plan,
    handle_multistep,
    plan_task,
)
from ultron.core.intelligence import planning as pl
from ultron.core.tools import registry as reg
from ultron.core.tools.registry import get_tool
from ultron.core.types import Role
from ultron.security import SecurityBoundary
from ultron.security.models import Decision, RiskTier


class FakeEngine:
    """Scripted engine — no real LLM or network access needed."""

    def __init__(self, responses=()):
        self._responses = list(responses)
        self.calls = []

    async def generate(self, messages, **kwargs):
        self.calls.append(messages)
        return self._responses.pop(0) if self._responses else ""


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Per-step permission matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "step,expected_decision",
    [
        ({"action": "read_file", "filename": "config.py"}, "allow"),
        ({"action": "run_command", "command": "ls"}, "allow"),
        ({"action": "run_command", "command": "pytest"}, "allow"),
        ({"action": "run_command", "command": "mkdir x"}, "confirm"),
        # Dangerous commands are CRITICAL → confirmation in interactive mode.
        ({"action": "run_command", "command": "rm -rf /"}, "confirm"),
        # Guardrail hard-blocks: path escape, leaked credential, unsafe URL.
        ({"action": "read_file", "filename": "../../etc/passwd"}, "deny"),
        ({"action": "make_http_request", "method": "GET", "url": "http://localhost:8000"}, "allow"),
        ({"action": "make_http_request", "method": "POST", "url": "http://localhost:8000/x"}, "confirm"),
        ({"action": "make_http_request", "method": "POST", "url": "http://example.com/x"}, "deny"),
        ({"action": "run_query", "sql": "SELECT * FROM users"}, "allow"),
        ({"action": "run_query", "sql": "DROP TABLE users"}, "confirm"),
        ({"action": "write_file", "filename": "a.txt", "content": "hi"}, "confirm"),
        ({"action": "add_memory", "fact": "Paris is in France"}, "allow"),
    ],
)
def test_analyze_step_decision_matrix(step, expected_decision):
    result = pl.analyze_step(step, 1)
    assert result["decision"] == expected_decision
    assert result["index"] == 1


def test_analyze_step_risk_tiers():
    assert pl.analyze_step({"action": "run_command", "command": "ls"}, 1)["risk"] == RiskTier.LOW.value
    assert pl.analyze_step({"action": "write_file", "filename": "a.txt", "content": "x"}, 1)["risk"] == RiskTier.HIGH.value
    assert pl.analyze_step({"action": "run_command", "command": "rm -rf /"}, 1)["risk"] == RiskTier.CRITICAL.value


def test_analyze_step_missing_fields():
    result = pl.analyze_step({"action": "read_file"}, 1)
    assert "filename" in result["missing"]
    result = pl.analyze_step({"action": "make_http_request", "method": "GET"}, 2)
    assert "url" in result["missing"]
    result = pl.analyze_step({"action": "run_command", "command": "ls"}, 3)
    assert result["missing"] == []


def test_preflight_summary():
    steps = [
        {"action": "read_file", "filename": "config.py"},
        {"action": "run_command", "command": "mkdir x"},
        {"action": "read_file", "filename": "../../etc/passwd"},
    ]
    pre = pl.preflight_plan(steps)
    assert pre["summary"] == {"auto": 1, "confirm": 1, "blocked": 1, "missing": 0}
    assert pre["blocked"][0][0] == 3


# ---------------------------------------------------------------------------
# Dependency edges
# ---------------------------------------------------------------------------


def test_dependency_write_then_read():
    steps = [
        {"action": "write_file", "filename": "out.txt", "content": "data"},
        {"action": "read_file", "filename": "out.txt"},
    ]
    deps = pl.find_dependencies(steps)
    assert len(deps) == 1
    assert deps[0]["from"] == 1
    assert deps[0]["to"] == 2
    assert deps[0]["kind"] == "producer"


def test_dependency_write_then_command_mentions_file():
    steps = [
        {"action": "write_file", "filename": "data.csv", "content": "a,b"},
        {"action": "run_command", "command": "python analyze data.csv"},
    ]
    deps = pl.find_dependencies(steps)
    assert len(deps) == 1
    assert deps[0]["kind"] == "producer"
    assert "data.csv" in deps[0]["reason"]


def test_dependency_read_then_command():
    steps = [
        {"action": "read_file", "filename": "notes.txt"},
        {"action": "run_command", "command": "grep test notes.txt"},
    ]
    deps = pl.find_dependencies(steps)
    assert len(deps) == 1
    assert deps[0]["kind"] == "feeds"


def test_short_filename_no_spurious_dependency():
    # Review fix: a 1-2 char base ("a.txt" → "a") would match almost any
    # command text; only meaningful filenames produce dependency edges.
    steps = [
        {"action": "write_file", "filename": "a.txt", "content": "x"},
        {"action": "run_command", "command": "grep -r a ."},
        {"action": "run_command", "command": "ls -la"},
    ]
    assert pl.find_dependencies(steps) == []


def test_dependency_partial_token_no_match():
    # Review fix: base "data" must not match "data2" (bounded both sides).
    steps = [
        {"action": "write_file", "filename": "data.csv", "content": "a"},
        {"action": "run_command", "command": "rm data2.csv"},
    ]
    assert pl.find_dependencies(steps) == []


def test_no_false_dependencies():
    steps = [
        {"action": "read_file", "filename": "a.txt"},
        {"action": "read_file", "filename": "b.txt"},
        {"action": "add_memory", "fact": "hello"},
    ]
    assert pl.find_dependencies(steps) == []


def test_dependencies_only_forward_and_deduped():
    steps = [
        {"action": "write_file", "filename": "out.log", "content": "1"},
        {"action": "run_command", "command": "cat out.log"},
        {"action": "run_command", "command": "cat out.log"},
    ]
    deps = pl.find_dependencies(steps)
    # 1→2 and 1→3, no duplicates, no backward edges.
    assert {(d["from"], d["to"]) for d in deps} == {(1, 2), (1, 3)}


# ---------------------------------------------------------------------------
# Plan preview
# ---------------------------------------------------------------------------


def test_format_plan_preview_content():
    steps = [
        {"action": "read_file", "filename": "config.py"},
        {"action": "run_command", "command": "mkdir build"},
        {"action": "read_file", "filename": "../../etc/passwd"},
    ]
    preview = pl.format_plan_preview(steps)
    assert "📋 Plan — 3 steps" in preview
    assert "⚡ auto" in preview
    assert "🛡 needs approval" in preview
    assert "⛔ blocked" in preview
    assert "Permissions: 1 auto · 1 confirm · 1 blocked" in preview


def test_format_plan_preview_missing_info():
    steps = [{"action": "make_http_request", "method": "POST"}]
    preview = pl.format_plan_preview(steps)
    assert "missing: url" in preview


def test_preview_heavy_command_warning(tmp_path, monkeypatch):
    from ultron.core.tools import resource_monitor as rm

    monkeypatch.setattr(rm, "RESOURCES_DB_PATH", tmp_path / "r.db")
    steps = [{"action": "run_command", "command": "pip install requests"}]
    preview = pl.format_plan_preview(steps)
    assert "⚠" in preview


# ---------------------------------------------------------------------------
# Registered tools + security
# ---------------------------------------------------------------------------


def test_tools_registered():
    for name in ("preflight_plan", "analyze_dependencies", "list_plan_actions"):
        assert callable(get_tool(name))


def test_tools_classified_low_risk():
    boundary = SecurityBoundary(mode="interactive")
    for name in ("preflight_plan", "analyze_dependencies", "list_plan_actions"):
        assert boundary.classify_action(name, "") == RiskTier.LOW
        assert boundary.check(name, "", "sample").decision == Decision.ALLOW


def test_preflight_plan_tool():
    out = get_tool("preflight_plan")(
        json.dumps([{"action": "run_command", "command": "mkdir x"}])
    )
    assert "🛡 needs approval" in out


def test_analyze_dependencies_tool():
    out = get_tool("analyze_dependencies")(
        json.dumps(
            [
                {"action": "write_file", "filename": "a.txt", "content": "x"},
                {"action": "read_file", "filename": "a.txt"},
            ]
        )
    )
    assert "1 dependency edge" in out


def test_list_plan_actions_tool():
    out = get_tool("list_plan_actions")()
    assert "make_http_request" in out
    assert "run_query" in out


def test_preflight_plan_tool_rejects_bad_json():
    out = get_tool("preflight_plan")("not json")
    assert "Error" in out


# ---------------------------------------------------------------------------
# Planner + executor extensions
# ---------------------------------------------------------------------------


def test_plan_task_accepts_http_and_query_steps():
    engine = FakeEngine(
        [
            json.dumps(
                [
                    {"action": "read_file", "filename": "config.py"},
                    {"action": "make_http_request", "method": "POST", "url": "http://localhost:8000/x"},
                    {"action": "run_query", "sql": "SELECT 1"},
                ]
            )
        ]
    )
    steps = _run(plan_task("read config then POST and query", engine))
    assert steps is not None
    assert [s["action"] for s in steps] == ["read_file", "make_http_request", "run_query"]


def test_plan_task_rejects_unknown_actions():
    engine = FakeEngine(
        [json.dumps([{"action": "explode_the_world", "force": True}])]
    )
    assert _run(plan_task("do something", engine)) is None


def test_execute_plan_runs_run_query_step(monkeypatch):
    monkeypatch.setitem(reg.TOOLS, "run_query", lambda sql: f"rows: {sql}")
    results = _run(
        execute_plan([{"action": "run_query", "sql": "SELECT * FROM t"}])
    )
    assert "rows: SELECT * FROM t" in results[0]


# ---------------------------------------------------------------------------
# Agent-level gate
# ---------------------------------------------------------------------------


def test_multistep_blocked_plan_never_offered():
    engine = FakeEngine(['[{"action": "read_file", "filename": "../../etc/passwd"}]'])
    msg = _run(handle_multistep("read the passwd file", engine))
    assert msg.pending_action is None
    assert "will not run" in msg.content
    assert "⛔" in msg.content


def test_multistep_confirm_plan_waits_for_approval():
    engine = FakeEngine(
        ['[{"action": "write_file", "filename": "a.txt", "content": "hi"}]']
    )
    msg = _run(handle_multistep("create a.txt with hi", engine))
    assert msg.pending_action is not None
    assert msg.pending_action.action_type == "execute_plan"
    assert "📋 Plan" in msg.content
    steps = json.loads(msg.pending_action.target)
    assert steps[0]["action"] == "write_file"
    # Review fix: the preview rides on the pending action so the CLI
    # approval card is never empty.
    assert msg.pending_action.content == msg.content
    assert "Plan" in msg.pending_action.content


def test_multistep_missing_info_waits_for_approval():
    # An add_memory step with no fact is auto-tiered but incomplete — the
    # plan waits for approval with the missing field called out up front.
    engine = FakeEngine(['[{"action": "add_memory"}]'])
    msg = _run(handle_multistep("remember something", engine))
    assert msg.pending_action is not None
    assert msg.pending_action.action_type == "execute_plan"
    assert "missing: fact" in msg.content


def test_multistep_all_auto_runs_immediately(monkeypatch):
    monkeypatch.setitem(reg.TOOLS, "run_command", lambda command: f"ran: {command}")
    engine = FakeEngine(
        [
            json.dumps(
                [
                    {"action": "run_command", "command": "ls"},
                    {"action": "run_command", "command": "pwd"},
                ]
            )
        ]
    )
    msg = _run(handle_multistep("run ls then run pwd", engine))
    assert msg.pending_action is None
    assert "📋 Plan" in msg.content
    assert "ran: ls" in msg.content


def test_agent_run_multistep_all_auto(tmp_path, monkeypatch):
    from ultron.core.tools import paths

    monkeypatch.setattr(paths, "ALLOWED_BASE_DIR", tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(reg.TOOLS, "run_command", lambda command: "ok")
    engine = FakeEngine(['[{"action": "run_command", "command": "ls"}]'])
    agent = SimpleAgent(engine)
    msg = _run(agent.run("run ls then run pwd"))
    assert msg.role == Role.ASSISTANT
    assert "ok" in msg.content
