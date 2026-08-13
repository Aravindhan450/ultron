"""Tests for the ReAct-loop routing correction (FIX #8 extension).

The ReAct loop lets the LLM pick tools. This layer applies the same
deterministic routing the SimpleAgent path already has:

- a `search_web` call on a repository question (\"How does the Supervisor
  delegate work?\") is redirected to `code_investigation` — repository
  questions never hit the web;
- a question-shaped argument on a generic code tool
  (\"code_search(query='Where is TaskState defined?')\") is redirected to
  the specific capability with the correctly extracted symbol;
- genuine external questions (`search_web('search the web for Python 3.12')`)
  stay as web searches;
- security action names stay symmetric: `search_web` gates LOW/allow exactly
  like `web_search`.
"""

from __future__ import annotations

import asyncio

import pytest

from ultron.core.agents.react import ReActAgent, route_llm_tool_call
from ultron.core.tools import paths as tools_paths
from ultron.security.boundary import SecurityBoundary


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """A temp workspace that is the ALLOWED_BASE_DIR (so tools work)."""
    monkeypatch.setattr(tools_paths, "ALLOWED_BASE_DIR", tmp_path)
    monkeypatch.chdir(tmp_path)
    return tmp_path


class FakeEngine:
    """Scripted engine: returns responses in order, records every call."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    async def generate(self, messages, **kwargs):
        self.calls.append(messages)
        return self._responses.pop(0) if self._responses else ""

    async def stream(self, messages, **kwargs):
        yield ""


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# route_llm_tool_call — redirect matrix
# ---------------------------------------------------------------------------


def test_repo_question_on_search_web_redirects_to_investigation():
    assert route_llm_tool_call(
        "search_web", {"query": "How does the Supervisor delegate work?"}
    ) == ("code_investigation", {"query": "Supervisor"})


def test_definition_question_on_search_web_redirects():
    assert route_llm_tool_call(
        "search_web", {"query": "Where is TaskState defined?"}
    ) == ("find_definition", {"name": "TaskState"})


def test_external_web_question_stays_as_is():
    for query in (
        "search the web for Python 3.12 release date",
        "What is the latest Python release?",
        "look up LangGraph alternatives",
    ):
        assert route_llm_tool_call("search_web", {"query": query}) is None


def test_question_shaped_arg_on_code_search_redirects():
    assert route_llm_tool_call(
        "code_search", {"query": "Where is TaskState defined?"}
    ) == ("find_definition", {"name": "TaskState"})
    assert route_llm_tool_call(
        "code_search", {"query": "find where the supervisor is defined"}
    ) == ("find_definition", {"name": "supervisor"})


def test_plain_code_search_stays_as_is():
    assert route_llm_tool_call("code_search", {"query": "pytest"}) is None


def test_bare_symbol_name_never_redirected():
    # find_definition(name='taskstate') is already handled case-insensitively
    # inside the tool — no redirect needed.
    assert route_llm_tool_call("find_definition", {"name": "taskstate"}) is None


def test_redirect_never_targets_state_modifying_tools():
    for tool in ("run_command", "write_file", "delete_file", "run_query"):
        assert route_llm_tool_call(tool, {"command": "pwd"}) is None


def test_empty_arguments_no_redirect():
    assert route_llm_tool_call("search_web", {}) is None
    assert route_llm_tool_call("search_web", {"query": "  "}) is None


# ---------------------------------------------------------------------------
# Turn-level correction — the user's request decides the tool
# ---------------------------------------------------------------------------


def test_bare_symbol_on_code_search_redirected_by_user_input():
    """The model emits code_search(query='taskstate') for a reference
    question; the turn's request classifies to find_references, so the
    runtime executes the specific capability — never a lexical dump."""
    assert route_llm_tool_call(
        "code_search",
        {"query": "taskstate"},
        user_input="Where is taskstate used?",
    ) == ("find_references", {"name": "taskstate"})


def test_bare_symbol_on_semantic_search_redirected_by_user_input():
    assert route_llm_tool_call(
        "semantic_search",
        {"query": "taskstate"},
        user_input="Find references to TaskState",
    ) == ("find_references", {"name": "TaskState"})


def test_bare_symbol_on_search_web_redirected_by_user_input():
    """The model stripped the argument to a bare symbol AND chose web search;
    the turn's reference question must still win — repository questions never
    produce a web call, even with a non-classifying argument."""
    assert route_llm_tool_call(
        "search_web",
        {"query": "taskstate"},
        user_input="Where is taskstate used?",
    ) == ("find_references", {"name": "taskstate"})
    assert route_llm_tool_call(
        "search_web",
        {"query": "supervisor"},
        user_input="Where is the Supervisor defined?",
    ) == ("find_definition", {"name": "Supervisor"})


def test_external_web_turn_never_redirected():
    # Genuine external questions do not classify to a specific symbol tool,
    # so the turn-level correction leaves a web call untouched.
    assert route_llm_tool_call(
        "search_web",
        {"query": "Python 3.12 release"},
        user_input="What is the latest Python release?",
    ) is None


def test_user_input_redirect_definition_case():
    assert route_llm_tool_call(
        "code_search",
        {"query": "supervisor"},
        user_input="Where is the Supervisor defined?",
    ) == ("find_definition", {"name": "Supervisor"})


def test_user_input_ignored_without_generic_tool():
    # find_definition is already the specific tool; the turn-level correction
    # only fires for generic code tools.
    assert route_llm_tool_call(
        "find_definition",
        {"name": "taskstate"},
        user_input="Where is taskstate used?",
    ) is None


def test_user_input_ignored_when_lexical_intent():
    # "Find files containing X" is a lexical request — the runtime must NOT
    # force a reference lookup.
    assert route_llm_tool_call(
        "code_search",
        {"query": "TaskState"},
        user_input="Find files containing TaskState",
    ) is None


def test_user_input_ignored_without_turn_question():
    # No user_input (e.g. a later tool call in the loop) -> fall back to the
    # argument-only rules; a bare symbol classifies to nothing, so no
    # redirect.
    assert route_llm_tool_call("code_search", {"query": "taskstate"}) is None


def test_user_input_redirect_is_read_only():
    # The corrected target is always a read-only code-intel tool.
    tool, _ = route_llm_tool_call(
        "code_search",
        {"query": "taskstate"},
        user_input="Where is taskstate used?",
    )
    assert tool in ("find_definition", "find_references")


# ---------------------------------------------------------------------------
# Security symmetry — search_web gates like web_search
# ---------------------------------------------------------------------------


def test_search_web_gates_low_like_web_search():
    boundary = SecurityBoundary()
    assert boundary.check("web_search", "Python 3.12").decision.value == "allow"
    assert boundary.check("search_web", "Python 3.12").decision.value == "allow"


def test_search_web_target_content_mapped():
    from ultron.core.agents.simple import _generic_target_content

    assert _generic_target_content("search_web", {"query": "Python 3.12"}) == (
        "Python 3.12",
        None,
    )
    assert _generic_target_content("web_search", {"query": "Python 3.12"}) == (
        "Python 3.12",
        None,
    )


# ---------------------------------------------------------------------------
# End-to-end: the ReAct loop applies the correction
# ---------------------------------------------------------------------------


def test_react_loop_redirects_repo_question_away_from_web():
    """The LLM emits search_web for a repo question; the loop must execute
    code_investigation instead — the observation never is a web search."""
    engine = FakeEngine(
        [
            # Turn 1: the model misroutes to web_search.
            (
                'Thought: I should search the web.\n'
                '```json\n{"tool": "search_web", '
                '"arguments": {"query": "How does the Supervisor delegate work?"}}\n```'
            ),
            # Turn 2: final answer.
            "Supervisor is implemented in the orchestration delegation module.",
        ]
    )
    agent = ReActAgent(engine, max_iterations=5)
    msg = _run(agent.run("How does the Supervisor delegate work?"))

    assert "delegation" in msg.content.lower()
    # The observation fed back must contain the investigation output, never a
    # web-search confirmation or a network request. Engine calls receive
    # OpenAI-format dicts: {"role": "tool", "name": ..., "content": ...}.
    observations = [
        str(m["content"])
        for m in engine.calls[-1]
        if m.get("role") == "tool"
        and m.get("name") in ("code_investigation", "search_web")
    ]
    assert any("Repository investigation" in o for o in observations), observations
    assert not any("Search the web" in o for o in observations), observations


def test_react_loop_keeps_external_web_search():
    """A genuine external query stays a web search (not redirected)."""
    engine = FakeEngine(
        [
            '{"tool": "search_web", "arguments": {"query": "Python 3.12 release date"}}',
            "Python 3.12 was released in October 2023.",
        ]
    )
    agent = ReActAgent(engine, max_iterations=5)
    _run(agent.run("What is the latest Python release?"))
    tools_used = [m.get("name") for m in engine.calls[-1] if m.get("role") == "tool"]
    assert "search_web" in tools_used, tools_used
    assert "code_investigation" not in tools_used, tools_used


def test_react_loop_first_tool_call_only_turn_correction():
    """The turn-level correction applies on the FIRST tool call; a later
    generic search (mid-loop exploration) is left alone."""
    engine = FakeEngine(
        [
            # Turn 1: generic code_search with a bare symbol — corrected to
            # find_references because the turn asks for references.
            '{"tool": "code_search", "arguments": {"query": "taskstate"}}',
            # Turn 2: mid-loop exploration — a bare code_search must NOT be
            # re-corrected.
            '{"tool": "code_search", "arguments": {"query": "other_symbol"}}',
            # Turn 3: final answer.
            "TaskState is referenced across the codebase.",
        ]
    )
    agent = ReActAgent(engine, max_iterations=5)
    _run(agent.run("Where is taskstate used?"))
    tool_names = [m.get("name") for m in engine.calls[-1] if m.get("role") == "tool"]
    assert tool_names == ["find_references", "code_search"], tool_names


def test_react_loop_turn_correction_reference_lookup():
    """The exact live-failure scenario: model emits code_search with a bare
    symbol; the loop must execute find_references (VERIFIED evidence), so the
    observation cites source files, not a fabricated Java answer."""
    engine = FakeEngine(
        [
            '{"tool": "code_search", "arguments": {"query": "taskstate"}}',
            "TaskState is used in the state-management layer.",
        ]
    )
    agent = ReActAgent(engine, max_iterations=5)
    _run(agent.run("Where is taskstate used?"))
    tools_used = [m.get("name") for m in engine.calls[-1] if m.get("role") == "tool"]
    assert "find_references" in tools_used, tools_used
    assert "code_search" not in tools_used, tools_used


def test_react_loop_redirects_question_shaped_code_search():
    """code_search with a definition question routes to find_definition."""
    engine = FakeEngine(
        [
            '{"tool": "code_search", "arguments": {"query": "Where is TaskState defined?"}}',
            "TaskState is in src/ultron/core/types.py.",
        ]
    )
    agent = ReActAgent(engine, max_iterations=5)
    _run(agent.run("Where is TaskState defined?"))
    tools_used = [m.get("name") for m in engine.calls[-1] if m.get("role") == "tool"]
    assert "find_definition" in tools_used, tools_used
    assert "code_search" not in tools_used, tools_used


def test_react_loop_unknown_tool_still_error_not_crash():
    """An unrecognized tool name returns a clear error observation."""
    engine = FakeEngine(
        [
            '{"tool": "does_not_exist", "arguments": {}}',
            "OK",
        ]
    )
    agent = ReActAgent(engine, max_iterations=5)
    _run(agent.run("do something"))
    observations = [str(m["content"]) for m in engine.calls[-1] if m.get("role") == "tool"]
    assert any("unknown tool" in o for o in observations), observations


# ---------------------------------------------------------------------------
# Bridge supports the investigation operation (executor deterministic path)
# ---------------------------------------------------------------------------


def test_bridge_code_investigation_operation(sandbox):
    from ultron.core.coding.intelligence_bridge import CodeIntelligenceBridge

    (sandbox / "src").mkdir(parents=True)
    (sandbox / "src" / "auth.py").write_text(
        "class AuthService:\n    def login(self):\n        return 'ok'\n",
        encoding="utf-8",
    )
    (sandbox / "tests").mkdir()
    (sandbox / "tests" / "test_auth.py").write_text(
        "from src.auth import AuthService\n", encoding="utf-8"
    )
    bridge = CodeIntelligenceBridge()
    assert bridge.enable(str(sandbox)) is True
    try:
        out = bridge.query("code_investigation", query="auth service")
        assert "Primary implementation" in out
        assert "AuthService" in out
        assert "src/auth.py" in out
    finally:
        bridge.close()
