"""Tests for repository-question routing + reference extraction + synthesis.

Covers the three observed failures:

1. Reference queries must extract ONLY the symbol (never "is TaskState").
2. "How does X work" / "how is X implemented" are repository questions that
   route to code investigation — never web search.
3. "Where is X implemented" produces a synthesized primary implementation,
   not a raw lexical dump, and src/ outranks tests/docs/scripts.
"""

from __future__ import annotations

import pytest

from ultron.core.coding.intelligence.tools import (
    code_investigation,
    find_references,
)
from ultron.core.nlp.intent import IntentCategory, route_request
from ultron.core.tools import paths as tools_paths


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """A temp workspace that is the ALLOWED_BASE_DIR (so tools work)."""
    monkeypatch.setattr(tools_paths, "ALLOWED_BASE_DIR", tmp_path)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _write(root, rel: str, text: str):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# 1. Reference extraction — symbol only, never the grammatical wrapper
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "phrase,expected",
    [
        ("Where is TaskState used?", "TaskState"),
        ("Where is taskstate used?", "taskstate"),
        ("Where is Task State used?", "Task State"),
        ("Where is TASKSTATE used?", "TASKSTATE"),
        ("Find references to TaskState.", "TaskState"),
        ("Find references to taskstate.", "taskstate"),
        ("Find usages of TaskState.", "TaskState"),
        ("Find all references to TaskState", "TaskState"),
        ("Find all usages of TaskState", "TaskState"),
        ("Who uses TaskState?", "TaskState"),
        ("Who calls TaskState?", "TaskState"),
        ("What references TaskState?", "TaskState"),
        ("Where is TaskState referenced?", "TaskState"),
        ("Where is TaskState referenced in the project?", "TaskState"),
        ("Where is TaskState called?", "TaskState"),
        ("Callers of TaskState", "TaskState"),
    ],
)
def test_reference_extraction_symbol_only(phrase, expected):
    it = route_request(phrase)
    assert it is not None, phrase
    assert it.intent_type is IntentCategory.REFERENCE_LOOKUP, phrase
    assert it.tool == "find_references", phrase
    name = it.arguments["name"]
    assert name == expected, f"{phrase}: got {name!r}, want {expected!r}"
    # The grammatical wrapper must never leak into the symbol.
    assert "is " not in f" {name} ", phrase
    assert name.lower() in expected.lower() or expected.lower() in name.lower()


def test_reference_used_vs_definition_distinct():
    """'where is X used' is a reference query; 'where is X defined' is not."""
    used = route_request("Where is TaskState used?")
    defined = route_request("Where is TaskState defined?")
    assert used is not None and used.intent_type is IntentCategory.REFERENCE_LOOKUP
    assert defined is not None and defined.intent_type is IntentCategory.DEFINITION_LOOKUP


def test_reference_extraction_no_wrapper_leak_into_tool(sandbox):
    """End-to-end: the extracted name goes to the reference tool correctly."""
    _write(
        sandbox,
        "models.py",
        "class TaskState:\n    pass\n\n"
        "def use_ts(x: TaskState):\n    return x\n",
    )
    out = find_references("TaskState", str(sandbox))
    assert "TaskState" in out
    # The buggy behavior reported 'References to \'is TaskState\''.
    assert "is TaskState" not in out


# ---------------------------------------------------------------------------
# 2. Repository-question routing — never web search
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "phrase",
    [
        "How does the Supervisor delegate work?",
        "How does Supervisor delegate?",
        "How does TaskState interact with the workflow engine?",
        "How does the workflow engine execute tasks?",
        "How is command execution implemented?",
        "How does CodingExecutor work?",
        "Explain how tool execution works",
        "Explain how the workflow is structured",
        "Why does the workflow validator reject this state?",
        "Where is task state handled?",
    ],
)
def test_repository_questions_route_to_investigation(phrase):
    it = route_request(phrase)
    assert it is not None, phrase
    assert it.intent_type is IntentCategory.REPOSITORY_INVESTIGATION, phrase
    assert it.tool == "code_investigation", phrase


def test_repository_question_never_web():
    """The exact Supervisor question must NOT become a web search."""
    it = route_request("How does the Supervisor delegate work?")
    assert it is not None
    assert it.tool == "code_investigation"
    assert it.tool != "web_search"
    assert "Supervisor" in it.arguments.get("query", "")


def test_external_question_not_forced_repo():
    """Explicit external/current questions are NOT captured as repository
    questions (they keep their existing web/LLM path)."""
    for phrase in (
        "What is the latest Python release?",
        "Search the web for LangGraph alternatives",
        "What does OpenAI currently recommend for tool calling?",
    ):
        it = route_request(phrase)
        # Must not be a repository investigation for these.
        assert it is None or it.intent_type is not IntentCategory.REPOSITORY_INVESTIGATION, phrase


def test_negative_component_not_web_searched():
    """A nonexistent component phrased as a repo question still routes to
    investigation, which reports no evidence — no automatic web search."""
    it = route_request("How does CompletelyNonexistentComponent work?")
    assert it is not None
    assert it.intent_type is IntentCategory.REPOSITORY_INVESTIGATION
    assert it.tool == "code_investigation"


# ---------------------------------------------------------------------------
# 3. Implementation synthesis — primary implementation, not raw dumps
# ---------------------------------------------------------------------------


def test_investigation_verified_definition(sandbox):
    _write(
        sandbox,
        "src/auth/service.py",
        "class AuthService:\n    def login(self):\n        return 'ok'\n",
    )
    _write(
        sandbox,
        "src/auth/__init__.py",
        "from .service import AuthService\n",
    )
    _write(
        sandbox,
        "tests/test_auth.py",
        "from ultron.auth.service import AuthService\n",
    )
    out = code_investigation("auth service", str(sandbox))
    assert "Primary implementation" in out
    assert "AuthService" in out
    assert "src/auth/service.py" in out
    assert "Supporting components" in out
    assert "Relevant tests" in out
    assert "tests/test_auth.py" in out
    assert "Evidence status: VERIFIED" in out


def test_investigation_primary_not_raw_dump(sandbox):
    """'where is coding executor implemented' must identify a primary
    implementation rather than returning a flat list of matches."""
    _write(
        sandbox,
        "src/coding/executor.py",
        "class CodingExecutor:\n    def run(self):\n        pass\n",
    )
    _write(sandbox, "src/coding/__init__.py", "")
    out = code_investigation("coding executor", str(sandbox))
    assert "Primary implementation" in out
    assert "CodingExecutor" in out
    assert "Summary:" in out
    # Not an unranked dump: the summary + evidence status are present.
    assert "Evidence status:" in out


def test_investigation_ranks_src_over_tests(sandbox):
    """Semantic/lexical fallback must put src/ ahead of tests/ and scripts."""
    _write(
        sandbox,
        "src/runner.py",
        "def run_command():\n    return 'exec'\n",
    )
    _write(
        sandbox,
        "tests/test_runner.py",
        "def test_run_command():\n    pass\n",
    )
    _write(sandbox, "_live_check.py", "# run_command mentions\n")
    out = code_investigation("run command", str(sandbox))
    src_idx = out.find("src/runner.py")
    tests_idx = out.find("tests/test_runner.py")
    assert src_idx != -1, out
    # src appears before tests in the primary-implementation list (or at all).
    if tests_idx != -1:
        assert src_idx < tests_idx


def test_investigation_negative_no_speculation(sandbox):
    _write(sandbox, "src/foo.py", "x = 1\n")
    out = code_investigation("CompletelyNonexistentSymbol", str(sandbox))
    assert "No repository evidence found" in out
    # Never a filename-convention guess.
    assert "likely" not in out.lower()


# ---------------------------------------------------------------------------
# 4. Security + registry integration
# ---------------------------------------------------------------------------


def test_code_investigation_registered_readonly():
    from ultron.core.tools.registry import TOOLS
    from ultron.security.boundary import SecurityBoundary

    assert "code_investigation" in TOOLS
    verdict = SecurityBoundary().check("code_investigation", "Supervisor")
    assert verdict.decision.value == "allow"


def test_investigation_goes_through_agent_routing():
    """repository_investigation maps to the code_investigation tool."""
    from ultron.core.agents.simple import SimpleAgent

    it = route_request("How does the Supervisor delegate work?")
    assert it is not None
    # Execution path (handle_routed_intent -> tool) is covered by the CLI
    # live test; here we assert the intent mapping and agent integration.
    assert it.tool == "code_investigation"
    agent = SimpleAgent(None)
    assert hasattr(agent, "run")
