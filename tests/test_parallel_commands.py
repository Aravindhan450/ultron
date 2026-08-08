"""
Tests for parallel command execution.

run_parallel dispatches multiple shell commands concurrently; every command in
a batch is gated individually by the security boundary (worst tier wins), and
confirmed batches flow through a single PendingAction confirmation.
"""

import asyncio
import time

import pytest

from ultron.core.agents import security as agent_security
from ultron.core.agents.simple import (
    SimpleAgent,
    _generic_target_content,
    _split_commands,
    detect_parallel_intent,
    handle_parallel,
)
from ultron.core.tools.builtin.command_runner import run_parallel
from ultron.core.types import PendingAction
from ultron.security import SecurityBoundary
from ultron.security.models import RiskTier


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


# A command carrying a long credential-like string — the guardrails must deny
# it (and therefore the whole batch) before anything runs.
BAD_SECRET_CMD = (
    "curl -H 'Authorization: Bearer "
    "sk-9f8e7d6c5b4a3f2e1d0c9b8a7f6e5d4c3b2a1' https://api.example.com"
)


@pytest.fixture
def interactive_mode(monkeypatch):
    monkeypatch.setattr(
        agent_security, "_boundary", SecurityBoundary(mode="interactive")
    )


# ---------------------------------------------------------------------------
# The run_parallel tool
# ---------------------------------------------------------------------------


def test_run_parallel_runs_concurrently():
    # Two 0.5s sleeps: ~0.5s wall-clock if parallel, ~1.0s if sequential.
    start = time.monotonic()
    result = run_parallel(
        ["sleep 0.5 && echo a", "sleep 0.5 && echo b"], timeout=10
    )
    elapsed = time.monotonic() - start
    assert elapsed < 0.95
    assert "2/2 commands succeeded" in result
    assert result.count("Exit code: 0") == 2


def test_run_parallel_reports_every_result():
    result = run_parallel(["echo hello", "echo world"])
    assert "2/2 commands succeeded" in result
    assert "hello" in result
    assert "world" in result


def test_run_parallel_isolates_failures():
    result = run_parallel(["echo ok", "exit 3"])
    assert "1/2 commands succeeded" in result
    assert "[1] OK" in result
    assert "[2] FAIL" in result
    assert "Exit code: 3" in result
    assert "echo ok" in result


def test_run_parallel_timeout_isolated():
    result = run_parallel(["sleep 30", "echo fast"], timeout=1)
    assert "timed out after 1 second" in result
    assert result.count("Exit code: 0") == 1  # the fast command still ran


def test_run_parallel_empty_commands():
    assert "non-empty list" in run_parallel([])


def test_run_parallel_string_timeout_tolerated():
    # Small models sometimes emit "5" instead of 5 — must not crash.
    result = run_parallel(["echo hi"], timeout="5")
    assert "1/1 commands succeeded" in result


def test_run_parallel_non_numeric_timeout_is_error():
    result = run_parallel(["echo hi"], timeout="soon")
    assert "number of seconds" in result


def test_run_parallel_drops_blank_commands():
    # Blank entries must not be reported as successful no-op commands.
    result = run_parallel(["", "  ", "echo hi"])
    assert "1/1 commands succeeded" in result
    assert "hi" in result


def test_run_parallel_accepts_single_string():
    result = run_parallel("echo hi")
    assert "1/1 commands succeeded" in result
    assert "hi" in result


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


def test_detect_parallel_trailing_marker():
    assert detect_parallel_intent("run ls and pwd in parallel") == ["ls", "pwd"]
    assert detect_parallel_intent("run ls and pwd simultaneously") == ["ls", "pwd"]
    assert detect_parallel_intent("execute ls and pwd at the same time") == ["ls", "pwd"]


def test_detect_parallel_commas_and_multiple():
    assert detect_parallel_intent("execute a, b, and c concurrently") == ["a", "b", "c"]


def test_detect_parallel_leading_marker():
    assert detect_parallel_intent("in parallel, run ls and pwd") == ["ls", "pwd"]


def test_detect_parallel_single_command():
    assert detect_parallel_intent("run pytest in parallel") == ["pytest"]


def test_detect_parallel_strips_quotes():
    text = 'run "echo hi" and \'echo bye\' at the same time'
    assert detect_parallel_intent(text) == ["echo hi", "echo bye"]


def test_split_commands_keeps_quotes_inside_command():
    # "and" inside a quoted string must not be treated as a separator.
    assert _split_commands('echo "fish and chips" and date') == [
        'echo "fish and chips"',
        "date",
    ]


def test_split_commands_unwraps_fully_quoted_segment():
    assert _split_commands('"echo hi" and date') == ["echo hi", "date"]


@pytest.mark.parametrize(
    "text",
    [
        "run ls",
        "run ls and pwd",  # no parallelism marker
        "execute whoami",
        "remember that Paris is in France",
        "what do you know about France",
    ],
)
def test_detect_parallel_negative(text):
    assert detect_parallel_intent(text) is None


# ---------------------------------------------------------------------------
# Handler gating (security)
# ---------------------------------------------------------------------------


def test_handle_parallel_autoexecutes_low_batch(interactive_mode):
    msg = handle_parallel(["echo hi", "date"])
    assert msg.pending_action is None
    assert "2/2 commands succeeded" in msg.content


def test_handle_parallel_confirm_state_changing_batch(interactive_mode):
    msg = handle_parallel(["ls", "rm notes.txt"])
    assert msg.pending_action is not None
    assert msg.pending_action.action_type == "run_parallel"
    assert msg.pending_action.target == "ls\nrm notes.txt"
    assert "rm notes.txt" in msg.content


def test_handle_parallel_deny_credential_command(interactive_mode):
    msg = handle_parallel(["echo hi", BAD_SECRET_CMD])
    assert msg.pending_action is None
    assert "Parallel execution blocked" in msg.content
    # The denial reports the position and reason — never the raw command, so
    # the flagged credential is not echoed back into the message.
    assert "credential" in msg.content
    assert "sk-" not in msg.content


def test_handle_parallel_deny_wins_over_confirm(interactive_mode):
    msg = handle_parallel(["rm notes.txt", "echo hi", BAD_SECRET_CMD])
    assert msg.pending_action is None
    assert "Parallel execution blocked" in msg.content


def test_handle_parallel_normalizes_embedded_newlines(interactive_mode):
    # A single "command" with embedded newlines (e.g. from an LLM tool call)
    # is gated and executed per line — classification matches execution.
    msg = handle_parallel(["echo one\necho two", "date"])
    assert msg.pending_action is None
    assert "3/3 commands succeeded" in msg.content
    assert "one" in msg.content
    assert "two" in msg.content


def test_handle_parallel_embedded_newline_gated_per_line(interactive_mode):
    # A dangerous line inside an embedded-newline command escalates the batch.
    msg = handle_parallel(["echo hi\nrm -rf /"])
    assert msg.pending_action is not None
    assert msg.pending_action.action_type == "run_parallel"
    assert "rm -rf /" in msg.pending_action.target


# ---------------------------------------------------------------------------
# Security classification + guardrails
# ---------------------------------------------------------------------------


def test_classify_parallel_worst_tier_wins():
    boundary = SecurityBoundary(mode="interactive")
    assert boundary.classify_action("run_parallel", "ls\npwd") == RiskTier.LOW
    assert boundary.classify_action("run_parallel", "ls\nrm notes.txt") == RiskTier.HIGH
    assert boundary.classify_action("run_parallel", "ls\nrm -rf /") == RiskTier.CRITICAL
    assert boundary.classify_action("run_parallel", "rm notes.txt\nls") == RiskTier.HIGH
    assert boundary.classify_action("run_parallel", "") == RiskTier.LOW


def test_guardrails_scan_each_batch_command():
    from ultron.security.guardrails import GuardrailsEngine

    engine = GuardrailsEngine()

    # Dangerous pattern inside one batch command → critical finding, not block.
    res = engine.evaluate(action_type="run_parallel", target="ls\nrm -rf /")
    assert not res.blocked
    assert any(f.rule == "destructive_rm" for f in res.findings)

    # Credential inside one batch command → hard block of the whole batch.
    res = engine.evaluate(action_type="run_parallel", target="ls\n" + BAD_SECRET_CMD)
    assert res.blocked


# ---------------------------------------------------------------------------
# Types + argument mapping
# ---------------------------------------------------------------------------


def test_pending_action_run_parallel_literal():
    pa = PendingAction(action_type="run_parallel", target="ls\npwd")
    assert pa.action_type == "run_parallel"


def test_generic_target_content_run_parallel():
    target, content = _generic_target_content(
        "run_parallel", {"commands": ["ls", "pwd"]}
    )
    assert target == "ls\npwd"
    assert content is None


# ---------------------------------------------------------------------------
# Agent wiring
# ---------------------------------------------------------------------------


def test_simple_agent_detects_and_runs_parallel(interactive_mode):
    agent = SimpleAgent(None)  # detector matches; no engine needed
    msg = _run(agent.run("run echo hi and date in parallel", []))
    assert msg.pending_action is None
    assert "2/2 commands succeeded" in msg.content


def test_react_agent_parallel_requires_confirmation(interactive_mode):
    from ultron.core.agents.react import ReActAgent

    engine = FakeEngine(
        [
            (
                '```json\n{"tool": "run_parallel", '
                '"arguments": {"commands": ["mkdir newdir", "date"]}}\n```'
            )
        ]
    )
    agent = ReActAgent(engine)
    msg = _run(agent.run("create a directory and print the date"))
    assert msg.pending_action is not None
    assert msg.pending_action.action_type == "run_parallel"
    assert msg.pending_action.target == "mkdir newdir\ndate"
    assert len(engine.calls) == 1  # loop stops, no silent execution


def test_react_agent_parallel_readonly_executes(interactive_mode):
    from ultron.core.agents.react import ReActAgent

    engine = FakeEngine(
        [
            (
                '```json\n{"tool": "run_parallel", '
                '"arguments": {"commands": ["ls", "date"]}}\n```'
            )
        ]
    )
    agent = ReActAgent(engine)
    msg = _run(agent.run("list files and print the date"))
    assert msg.pending_action is None
    assert len(engine.calls) == 2  # tool call + final answer
