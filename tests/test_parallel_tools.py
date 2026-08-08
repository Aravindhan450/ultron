"""
Tests for inter-tool parallel processing
(``ultron.core.intelligence.parallel_tools``).

Covers the concurrent batch executor with per-call security gating (deny
never runs, confirm never runs silently), the deterministic synthesis
(cross-tool keyword connections), the detector (parallel phrasings -> batch,
single-target -> not a batch), the two registered tools, LOW-risk boundary
classification, the LLM planner against a fake engine, and the agent-level
flow.
"""

import json

import pytest

from ultron.core.agents.simple import (
    SimpleAgent,
    detect_tool_batch_intent,
    handle_parallel_tools,
)
from ultron.core.intelligence import parallel_tools as pt
from ultron.core.tools import registry as reg
from ultron.core.tools.registry import get_tool
from ultron.core.types import Role
from ultron.security import SecurityBoundary
from ultron.security.models import Decision, RiskTier

# ---------------------------------------------------------------------------
# run_tool_batch — execution + gating
# ---------------------------------------------------------------------------


def test_run_tool_batch_executes_reads_in_parallel():
    calls = [
        {"tool": "read_file", "arguments": {"file_path": "README.md"}},
        {"tool": "read_file", "arguments": {"file_path": "pyproject.toml"}},
    ]
    report = pt.run_tool_batch(json.dumps(calls))
    assert "Parallel batch" in report
    assert "2 executed" in report
    assert "README.md" in report
    assert "pyproject.toml" in report
    assert "✅" in report
    # The combined analysis section appears
    assert "Combined analysis" in report


def test_run_tool_batch_denied_call_never_runs():
    # read_file escaping the base dir is denied by guardrails.
    calls = [
        {"tool": "read_file", "arguments": {"file_path": "/etc/shadow"}},
        {"tool": "read_file", "arguments": {"file_path": "README.md"}},
    ]
    report = pt.run_tool_batch(json.dumps(calls))
    assert "blocked" in report
    assert "Blocked by security" in report
    # The safe member still executed.
    assert "README.md" in report


def test_run_tool_batch_confirm_call_never_runs_silently():
    # A state-changing command is HIGH -> confirm under interactive mode.
    calls = [
        {"tool": "run_command", "arguments": {"command": "echo hi > /tmp/ultron_pt_test.txt"}},
        {"tool": "read_file", "arguments": {"file_path": "README.md"}},
    ]
    report = pt.run_tool_batch(json.dumps(calls))
    assert "needs approval" in report
    assert "not run" in report.lower() or "not run in batch" in report
    # The confirm call must not have executed.
    import os

    assert not os.path.exists("/tmp/ultron_pt_test.txt")
    # The allowed member still ran.
    assert "README.md" in report


def test_run_tool_batch_unknown_tool_isolated():
    calls = [
        {"tool": "no_such_tool", "arguments": {}},
        {"tool": "read_file", "arguments": {"file_path": "README.md"}},
    ]
    report = pt.run_tool_batch(json.dumps(calls))
    assert "unknown tool" in report
    assert "README.md" in report


def test_run_tool_batch_malformed_input():
    assert "Error" in pt.run_tool_batch("not json")
    assert "Error" in pt.run_tool_batch(json.dumps({"tool": "read_file"}))
    assert "Error" in pt.run_tool_batch("")


def test_execute_batch_marks_deny_and_confirm():
    gated, _ = pt.execute_batch(
        [
            {"tool": "read_file", "arguments": {"file_path": "/etc/shadow"}},
            {"tool": "run_command", "arguments": {"command": "touch /tmp/ultron_pt2"}},
        ]
    )
    statuses = {g["status"] for g in gated}
    assert "blocked" in statuses
    assert "confirm" in statuses


def test_run_tool_batch_dedupes_identical_calls():
    # The same call repeated must execute once, not twice.
    calls = [
        {"tool": "read_file", "arguments": {"file_path": "README.md"}},
        {"tool": "read_file", "arguments": {"file_path": "README.md"}},
    ]
    report = pt.run_tool_batch(json.dumps(calls))
    assert "1 executed" in report
    assert report.count("README.md") <= 2  # once in the call section


def test_execute_batch_read_and_write_waves(tmp_path, monkeypatch):
    # Read-only tools run concurrently; a state-writing LOW tool (add_memory)
    # still executes but is separated from the read wave (no SQLite lock).
    from ultron.core.tools.memory import graph, sqlite

    monkeypatch.setattr(sqlite, "MEMORY_DB_PATH", tmp_path / "mem.db")
    monkeypatch.setattr(graph, "MEMORY_DB_PATH", tmp_path / "mem.db")
    # The schema is normally created at module import (against the real path),
    # so create the tables in the throwaway file before storing anything.
    sqlite.init_memory_db()

    gated, _ = pt.execute_batch(
        [
            {"tool": "read_file", "arguments": {"file_path": "README.md"}},
            {"tool": "add_memory", "arguments": {"text": "ultron parallel test fact 4711"}},
        ]
    )
    statuses = {g["tool"]: g["status"] for g in gated}
    assert statuses["read_file"] == "ok"
    assert statuses["add_memory"] == "ok"


def test_run_tool_batch_resolves_action_name_to_registry_tool(monkeypatch):
    # The boundary/detector speak "web_search", the registry keys "search_web".
    # A batch call in EITHER spelling must gate to allow and execute — never
    # "unknown tool", never a silent confirm-skip. The search backend is
    # stubbed so the test proves name resolution without touching the network.
    assert pt._canonical_action_name("search_web") == "web_search"
    assert pt._canonical_action_name("web_search") == "web_search"
    assert pt._registry_tool_name("web_search") == "search_web"
    assert pt._registry_tool_name("read_file") == "read_file"

    monkeypatch.setitem(
        reg.TOOLS, "search_web", lambda query="": f"stubbed search for {query}"
    )
    for spelling in ("web_search", "search_web"):
        calls = [
            {"tool": spelling, "arguments": {"query": "pandas release notes"}},
            {"tool": "read_file", "arguments": {"file_path": "README.md"}},
        ]
        report = pt.run_tool_batch(json.dumps(calls))
        assert "unknown tool" not in report, spelling
        assert "needs approval" not in report, spelling
        assert "stubbed search for pandas release notes" in report, spelling
        assert "README.md" in report, spelling


def test_batch_readonly_set_excludes_sqlite_writers():
    # SQLite writers must never join the concurrent read wave — a future edit
    # silently re-adding one would reintroduce "database is locked" races.
    for writer in ("learn_api_schema", "add_memory", "add_triple", "forget_api", "write_file", "run_command"):
        assert writer not in pt.BATCH_READONLY_TOOLS, writer
    # make_http_request is excluded too: a POST is auto-allowed under
    # permissive mode and must not run concurrently.
    assert "make_http_request" not in pt.BATCH_READONLY_TOOLS
    # The reads that form typical batches are all present.
    for reader in ("read_file", "web_search", "check_connectivity", "run_query", "get_debug_context"):
        assert reader in pt.BATCH_READONLY_TOOLS, reader


def test_run_tool_batch_dedup_robust_to_nonserializable_args():
    # A non-JSON-serializable argument value must not crash the dedup key,
    # and identical calls still collapse to one (both carry the same tuple).
    calls = [
        {"tool": "read_file", "arguments": {"file_path": "README.md"}},
        {"tool": "read_file", "arguments": {"file_path": "README.md"}},
    ]
    for call in calls:
        call["arguments"]["extra"] = ("a", "b")  # tuple: not JSON-serializable
    report = pt.run_tool_batch(json.dumps(calls, default=str))
    assert "1 call" in report  # deduped to a single unique call


def test_run_tool_batch_include_sources_false_no_repeat():
    # The combined-analysis block must NOT re-list the full per-call results
    # (include_sources=False) — the per-call sections already show them.
    calls = [
        {"tool": "read_file", "arguments": {"file_path": "README.md"}},
        {"tool": "read_file", "arguments": {"file_path": "pyproject.toml"}},
    ]
    report = pt.run_tool_batch(json.dumps(calls))
    analysis = report.split("Combined analysis", 1)[-1]
    assert "• Sources:" not in analysis


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------


def test_synthesize_results_finds_shared_keywords():
    results = [
        {"tool": "read_file", "result": "pandas 2.0 is pinned in requirements.txt"},
        {"tool": "web_search", "result": "pandas 2.0.1 released with new features"},
    ]
    out = pt.synthesize_results(results)
    assert "Combined analysis" in out
    assert "pandas" in out
    assert "Shared across sources" in out


def test_synthesize_results_empty():
    assert "No results" in pt.synthesize_results([])


def test_synthesize_analysis_tool():
    out = pt.synthesize_analysis(
        json.dumps(
            [
                {"tool": "read_file", "result": "api docs mention auth"},
                {"tool": "web_search", "result": "auth tokens explained"},
            ]
        )
    )
    assert "Combined analysis" in out
    assert "auth" in out


# ---------------------------------------------------------------------------
# Registered tools + boundary
# ---------------------------------------------------------------------------


def test_tools_registered():
    assert get_tool("run_tool_batch") is pt.run_tool_batch
    assert get_tool("synthesize_analysis") is pt.synthesize_analysis
    assert "run_tool_batch" in reg.TOOLS
    assert "synthesize_analysis" in reg.TOOLS


def test_boundary_classifies_tools_low():
    boundary = SecurityBoundary()
    for tool, target, content in [
        ("run_tool_batch", "", '[{"tool": "read_file", "arguments": {}}]'),
        ("synthesize_analysis", "", "[]"),
    ]:
        result = boundary.check(tool, target, content)
        assert result.tier == RiskTier.LOW
        assert result.decision == Decision.ALLOW


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "user_input",
    [
        "read config.json and notes.txt",
        "open config.json and world.txt",
        "search for pandas and search for numpy",
        "check https://example.com and https://example.org",
        "gather info about the API at the same time",
        "compare the files in parallel",
    ],
)
def test_detect_tool_batch_intent_fires(user_input):
    result = detect_tool_batch_intent(user_input)
    assert result is not None
    assert result.get("calls") or result.get("planner")


@pytest.mark.parametrize(
    "user_input",
    [
        "read config.json",
        "run ls and pwd in parallel",
        "hello there",
        "search for pandas",
        "what is the capital of France",
    ],
)
def test_detect_tool_batch_intent_single_target(user_input):
    assert detect_tool_batch_intent(user_input) is None


def test_detect_tool_batch_intent_extracts_reads():
    result = detect_tool_batch_intent("read config.json and notes.txt")
    assert result["calls"] == [
        {"tool": "read_file", "arguments": {"file_path": "config.json"}},
        {"tool": "read_file", "arguments": {"file_path": "notes.txt"}},
    ]


def test_detect_tool_batch_intent_extracts_urls():
    result = detect_tool_batch_intent("check https://example.com and https://example.org")
    tools = [c["tool"] for c in result["calls"]]
    assert tools == ["check_connectivity", "check_connectivity"]


def test_detect_tool_batch_intent_bare_domains():
    # "check example.com and example.org" — no scheme needed.
    result = detect_tool_batch_intent("check example.com and example.org")
    assert result is not None
    urls = [c["arguments"]["url"] for c in result["calls"]]
    assert urls == ["example.com", "example.org"]


def test_detect_tool_batch_intent_bare_domain_not_filename():
    # config.json has a non-TLD extension and must stay a file, not a domain.
    result = detect_tool_batch_intent("read config.json and notes.txt")
    assert result["calls"] == [
        {"tool": "read_file", "arguments": {"file_path": "config.json"}},
        {"tool": "read_file", "arguments": {"file_path": "notes.txt"}},
    ]


def test_detect_tool_batch_intent_no_double_url_match():
    # An explicit URL must not also be captured as a bare domain.
    result = detect_tool_batch_intent("check https://example.com and example.org")
    urls = [c["arguments"]["url"] for c in result["calls"]]
    assert urls == ["https://example.com", "example.org"]
    assert len(urls) == 2


# ---------------------------------------------------------------------------
# LLM planner (fake engine)
# ---------------------------------------------------------------------------


class FakeEngine:
    """Scripted engine stub — returns a canned response, never calls a model."""

    def __init__(self, response: str = "", error: Exception | None = None):
        self.response = response
        self.error = error
        self.calls: list[list] = []

    async def generate(self, messages, **kwargs):
        self.calls.append(messages)
        if self.error:
            raise self.error
        return self.response


def test_plan_tool_batch_parses_json():
    engine = FakeEngine(
        response=json.dumps(
            [
                {"tool": "read_file", "arguments": {"file_path": "a.txt"}},
                {"tool": "web_search", "arguments": {"query": "pandas"}},
            ]
        )
    )

    async def run():
        return await pt.plan_tool_batch("gather everything about X", engine)

    calls = asyncio_run(run())
    assert calls == [
        {"tool": "read_file", "arguments": {"file_path": "a.txt"}},
        {"tool": "web_search", "arguments": {"query": "pandas"}},
    ]


def test_plan_tool_batch_rejects_non_allowed_tools():
    # The planner may suggest run_command; it is filtered out of the batch.
    engine = FakeEngine(
        response=json.dumps(
            [
                {"tool": "run_command", "arguments": {"command": "rm -rf /"}},
                {"tool": "read_file", "arguments": {"file_path": "a.txt"}},
            ]
        )
    )

    async def run():
        return await pt.plan_tool_batch("do stuff", engine)

    calls = asyncio_run(run())
    assert calls == [{"tool": "read_file", "arguments": {"file_path": "a.txt"}}]


def test_plan_tool_batch_bad_json_returns_none():
    for response in ("not json", "[]", "[42]", "```json\n{}\n```", ""):
        engine = FakeEngine(response=response)

        async def run(engine=engine):
            return await pt.plan_tool_batch("do stuff", engine)

        assert asyncio_run(run()) is None


# ---------------------------------------------------------------------------
# Agent flow
# ---------------------------------------------------------------------------


def asyncio_run(coro):
    import asyncio

    return asyncio.new_event_loop().run_until_complete(coro)


def test_handle_parallel_tools_agent_flow_with_fake_engine():
    # The planner path: no deterministic calls, so the fake engine plans them.
    engine = FakeEngine(
        response=json.dumps(
            [
                {"tool": "read_file", "arguments": {"file_path": "README.md"}},
                {"tool": "read_file", "arguments": {"file_path": "pyproject.toml"}},
            ]
        )
    )

    async def run():
        return await handle_parallel_tools("gather info about the project at the same time", engine)

    msg = asyncio_run(run())
    assert msg.role == Role.ASSISTANT
    assert "Parallel batch" in msg.content
    assert "README.md" in msg.content


def test_handle_parallel_tools_with_deterministic_calls():
    calls = [
        {"tool": "read_file", "arguments": {"file_path": "README.md"}},
        {"tool": "read_file", "arguments": {"file_path": "pyproject.toml"}},
    ]

    async def run():
        return await handle_parallel_tools("read README.md and pyproject.toml", None, calls=calls)

    msg = asyncio_run(run())
    assert "Parallel batch" in msg.content
    assert "README.md" in msg.content


def test_simple_agent_run_dispatches_batch():
    agent = SimpleAgent(engine=FakeEngine(""))
    msg = asyncio_run(agent.run("read README.md and pyproject.toml"))
    assert "Parallel batch" in msg.content
    assert "README.md" in msg.content
    assert "pyproject.toml" in msg.content


def test_handle_parallel_tools_no_calls_message():
    async def run():
        return await handle_parallel_tools("gather info at the same time", FakeEngine(""))

    msg = asyncio_run(run())
    assert "couldn't" in msg.content.lower()
