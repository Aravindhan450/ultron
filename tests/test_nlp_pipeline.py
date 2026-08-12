"""FIX #8 — Natural-language → intent → tool selection → argument extraction
→ normalization → security → execution routing.

Deterministic tests for the NLP routing layer:

- terminal normalization matrix (wrappers never leak into arguments; inner
  quotes preserved; prose never becomes a shell command)
- intent routing (filesystem / code intelligence / project commands /
  info→command) through ``route_request``
- project command discovery (discovered from config, never invented)
- agent-level routing through ``SimpleAgent.run`` — tool selection,
  extracted arguments, and security-visible normalized command
- LLM-fallback hardening (model text is normalized or refused)
- result interpretation (never fabricates success)
- observability records
- multi-step requests never become a single shell command
"""

from __future__ import annotations

import asyncio
import json

import pytest

from ultron.core.agents import simple as simple_mod
from ultron.core.agents.security import Decision
from ultron.core.nlp.intent import IntentCategory, route_request
from ultron.core.nlp.interpret import interpret_command_result
from ultron.core.nlp.normalize import (
    detect_explicit_test_command,
    normalize_terminal_command,
)
from ultron.core.nlp.observe import (
    clear_action_records,
    recent_actions,
    record_action,
)
from ultron.core.nlp.project import discover_project_command

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Verdict:
    """Minimal stand-in for a security BoundaryResult verdict."""

    def __init__(self, decision):
        self.decision = decision
        self.tier = "low"
        self.reason = "test"


def _allow():
    return _Verdict(Decision.ALLOW)


def _confirm():
    return _Verdict(Decision.CONFIRM)


class FakeEngine:
    """Scripted engine — no real LLM or network access needed."""

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


def _tool_call(tool, **arguments):
    return f"```json\n{json.dumps({'tool': tool, 'arguments': arguments})}\n```"


def _patch(monkeypatch, **kwargs):
    """Monkeypatches simple_mod attributes and returns an execution spy.

    ``check_action=_allow`` installs a variadic stand-in so the boundary is
    not consulted during routing tests (its classification is tested
    elsewhere); ``execute_tool=True`` installs a capture spy.
    """
    captured = {}

    def _capture(name):
        def spy(*args, **kw):
            captured[name] = {"args": args, "kwargs": kw}
            return "ok"

        return spy

    for name, value in kwargs.items():
        if value is True:
            monkeypatch.setattr(simple_mod, name, _capture(name))
        elif callable(value):
            # Bind the factory via a default arg — a bare closure over the loop
            # variable would capture the last value in kwargs.
            monkeypatch.setattr(simple_mod, name, lambda *a, _v=value, **k: _v())
        else:
            monkeypatch.setattr(simple_mod, name, value)
    return captured


# ---------------------------------------------------------------------------
# 1. Terminal normalization matrix
# ---------------------------------------------------------------------------

TERMINAL_VARIANTS = [
    ("pwd", "pwd"),
    ("Run pwd", "pwd"),
    ("Run the command pwd", "pwd"),
    ("Run the command `pwd`", "pwd"),
    ("Execute pwd", "pwd"),
    ("Execute: pwd", "pwd"),
    ("Please execute `pwd`", "pwd"),
    ("Please run: `pwd`", "pwd"),
    ("Can you execute pwd?", "pwd"),
    ("Could you run pwd", "pwd"),
    ("Will you run pwd", "pwd"),
    ("Would you please execute pwd", "pwd"),
    ("Use the terminal to run pwd", "pwd"),
    ("Use the terminal tool to execute pwd", "pwd"),
    ("Go ahead and run pwd", "pwd"),
    ("Run git status", "git status"),
    ("Execute: git diff", "git diff"),
    ("Execute the command `python -m pytest`", "python -m pytest"),
    ("Please run `pytest tests/`", "pytest tests/"),
]


@pytest.mark.parametrize("request_text,expected", TERMINAL_VARIANTS)
def test_terminal_normalization_matrix(request_text, expected):
    assert normalize_terminal_command(request_text) == expected


def test_inner_content_never_stripped():
    """Wrapper stripping must not touch content INSIDE the command."""
    assert normalize_terminal_command('echo "Execute: pwd"') == 'echo "Execute: pwd"'
    assert normalize_terminal_command("python -c 'print(\"Run: pwd\")'") == "python -c 'print(\"Run: pwd\")'"
    assert normalize_terminal_command("ls -la .") == "ls -la ."
    assert normalize_terminal_command("git commit -m \"Execute: done\"") == "git commit -m \"Execute: done\""


def test_trailing_sentence_punctuation_stripped():
    assert normalize_terminal_command("Run pwd.") == "pwd"
    assert normalize_terminal_command("Execute: git status?") == "git status"
    # A trailing period preceded by whitespace is an argument, not punctuation.
    assert normalize_terminal_command("ls -la .") == "ls -la ."


PROSE_NOT_COMMANDS = [
    "run the tests",
    "please run the tests",
    "run it",
    "run this",
    "the tests",
    "what is the current directory",
    "tell me about pwd",
    "how does pwd work",
    "find where pwd is used",
]


@pytest.mark.parametrize("prose", PROSE_NOT_COMMANDS)
def test_prose_never_becomes_a_command(prose):
    assert normalize_terminal_command(prose) is None


def test_explicit_test_command_extraction():
    assert detect_explicit_test_command("Run pytest tests/test_api.py") == "pytest tests/test_api.py"
    assert detect_explicit_test_command("Run python -m pytest tests/ -q") == "python -m pytest tests/ -q"
    assert detect_explicit_test_command("Run the tests") is None
    assert detect_explicit_test_command("Please run pytest tests/auth/test_login.py") == "pytest tests/auth/test_login.py"


# ---------------------------------------------------------------------------
# 2. Intent routing
# ---------------------------------------------------------------------------


def test_terminal_intent():
    it = route_request("Execute: pwd")
    assert it is not None
    assert it.intent_type is IntentCategory.TERMINAL_EXECUTION
    assert it.tool == "run_command"
    assert it.arguments == {"command": "pwd"}


def test_directory_list_intent():
    it = route_request("List the files here")
    assert it is not None
    assert it.intent_type is IntentCategory.DIRECTORY_LIST
    assert it.tool == "list_directory"


def test_definition_intent():
    it = route_request("Where is TaskState defined?")
    assert it is not None
    assert it.intent_type is IntentCategory.DEFINITION_LOOKUP
    assert it.tool == "find_definition"
    assert it.arguments == {"name": "TaskState"}


def test_reference_intent():
    it = route_request("Find references to TaskState")
    assert it is not None
    assert it.intent_type is IntentCategory.REFERENCE_LOOKUP
    assert it.tool == "find_references"
    assert it.arguments == {"name": "TaskState"}


def test_usage_and_caller_intents():
    assert route_request("Find all usages of UserService").tool == "find_references"
    assert route_request("Where is authenticate() used?").tool == "find_references"
    assert route_request("What calls authenticate()?").tool == "find_references"


def test_where_implemented_routes_to_code_search():
    it = route_request("Where is authentication implemented?")
    assert it is not None
    assert it.intent_type is IntentCategory.CODE_SEARCH
    assert it.tool == "code_search"


def test_semantic_intent():
    it = route_request("Find the code responsible for user login")
    assert it is not None
    assert it.intent_type is IntentCategory.SEMANTIC_SEARCH
    assert it.tool == "semantic_search"


def test_symbol_intent():
    it = route_request("Find the symbol UserService")
    assert it is not None
    assert it.intent_type is IntentCategory.SYMBOL_SEARCH
    assert it.tool == "find_symbol"


def test_file_delete_intent():
    it = route_request("Delete test.txt")
    assert it is not None
    assert it.intent_type is IntentCategory.FILE_DELETE
    assert it.tool == "delete_file"
    assert it.arguments == {"file_path": "test.txt"}


def test_file_rename_intent():
    it = route_request("Rename foo.py to bar.py")
    assert it is not None
    assert it.intent_type is IntentCategory.FILE_RENAME
    assert it.tool == "rename_file"
    assert it.arguments == {"file_path": "foo.py", "new_path": "bar.py"}


def test_make_directory_intent():
    it = route_request("Make a directory called TodoList")
    assert it is not None
    assert it.intent_type is IntentCategory.DIRECTORY_CREATE
    assert it.arguments == {"command": "mkdir -p TodoList"}


def test_file_search_intent():
    it = route_request("Search for files named config")
    assert it is not None
    assert it.intent_type is IntentCategory.FILE_SEARCH
    assert it.tool == "search_files"


def test_info_to_command_intent():
    it = route_request("What is the current working directory?")
    assert it is not None
    assert it.intent_type is IntentCategory.TERMINAL_EXECUTION
    assert it.arguments == {"command": "pwd"}


@pytest.mark.parametrize(
    "request_text,expected_category",
    [
        ("run the linter", IntentCategory.LINT),
        ("Run the type checker", IntentCategory.TYPECHECK),
        ("Build the project", IntentCategory.BUILD),
        ("Start the backend", IntentCategory.APPLICATION_START),
        ("Stop the development server", IntentCategory.APPLICATION_STOP),
        ("Format the code", IntentCategory.FORMAT),
    ],
)
def test_project_command_requests(request_text, expected_category):
    it = route_request(request_text)
    assert it is not None
    assert it.intent_type is expected_category


AMBIGUOUS_OR_OTHER = [
    "hello there",
    "fix it",
    "run it",
    "check the backend",
    "what do you think about authentication?",
    "create a file called hello.py and run it",
]


@pytest.mark.parametrize("request_text", AMBIGUOUS_OR_OTHER)
def test_ambiguous_prose_not_forced_into_a_tool(request_text):
    """Ambiguous requests must NOT be hallucinated into a concrete action."""
    it = route_request(request_text)
    assert it is None or it.arguments.get("command", "") not in {
        request_text,
        request_text.strip("?"),
    }


# ---------------------------------------------------------------------------
# 3. Project command discovery
# ---------------------------------------------------------------------------


def test_python_test_discovery(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[tool.pytest.ini_options]\n")
    assert discover_project_command("test", cwd=str(tmp_path)) == "pytest -v"


def test_no_invention_without_evidence(tmp_path):
    (tmp_path / "README.md").write_text("no build system here")
    assert discover_project_command("build", cwd=str(tmp_path)) is None
    assert discover_project_command("test", cwd=str(tmp_path)) is None
    assert discover_project_command("start", cwd=str(tmp_path)) is None


def test_node_discovery_from_scripts(tmp_path):
    (tmp_path / "package.json").write_text(
        json.dumps(
            {
                "scripts": {
                    "test": "jest",
                    "build": "vite build",
                    "lint": "eslint .",
                    "start": "vite dev",
                }
            }
        )
    )
    assert discover_project_command("test", cwd=str(tmp_path)) == "npm test"
    assert discover_project_command("build", cwd=str(tmp_path)) == "npm run build"
    assert discover_project_command("lint", cwd=str(tmp_path)) == "npm run lint"
    assert discover_project_command("start", cwd=str(tmp_path)) == "npm start"


def test_unknown_project_type_no_command(tmp_path):
    assert discover_project_command("test", cwd=str(tmp_path)) is None


# ---------------------------------------------------------------------------
# 4. Result interpretation
# ---------------------------------------------------------------------------


def test_interpret_success():
    out = interpret_command_result(
        "Exit code: 0\nOutput:\n/Users/x/ultron\n[resources] elapsed 0.01s"
    )
    assert "succeeded" in out


def test_interpret_failure_surfaces_error():
    out = interpret_command_result(
        "Exit code: 1\nError Output:\nFAILED tests/test_auth.py - AssertionError: expected 5 got 4\n[resources] elapsed 0.1s"
    )
    assert "failed with exit code 1" in out
    assert "AssertionError" in out


def test_interpret_command_not_found():
    out = interpret_command_result("Exit code: 127\nError Output:\nsh: nope: not found\n[resources] elapsed 0.01s")
    assert "failed with exit code 127" in out
    assert "not found" in out


def test_interpret_timeout():
    out = interpret_command_result("Error: command timed out after 15 seconds.\n[resources] timed out after 15.0s")
    assert "timed out" in out


def test_interpret_never_fabricates_success():
    assert "succeeded" not in interpret_command_result("Exit code: 2\n[resources] elapsed 0.1s")


# ---------------------------------------------------------------------------
# 5. Observability
# ---------------------------------------------------------------------------


def test_action_records_ring_buffer():
    clear_action_records()
    try:
        record_action("terminal_execution", "run_command", {"command": "pwd"})
        record_action(
            "code_search", "code_search", {"query": "auth"},
            security_decision="allow", execution_result="success",
        )
        recs = recent_actions(2)
        assert len(recs) == 2
        assert recs[0].selected_tool == "code_search"  # newest first
        assert recs[0].extracted_arguments == {"query": "auth"}
        assert recs[0].security_decision == "allow"
        assert recs[1].user_intent == "terminal_execution"
    finally:
        clear_action_records()
    assert recent_actions() == []


# ---------------------------------------------------------------------------
# 6. Agent-level routing (SimpleAgent.run)
# ---------------------------------------------------------------------------


def test_agent_execute_colon_runs_normalized_command(monkeypatch):
    """'Execute: pwd' must reach the tool as the command 'pwd'."""
    captured = _patch(monkeypatch, check_action=_allow, execute_tool=True)
    agent = simple_mod.SimpleAgent(None)
    _run(agent.run("Execute: pwd"))
    assert captured["execute_tool"]["kwargs"] == {"command": "pwd"}


def test_agent_normalized_command_is_what_security_sees(monkeypatch):
    """The wrapper must be stripped BEFORE the security boundary evaluates."""
    captured = {}
    monkeypatch.setattr(simple_mod, "_gate_command", lambda cmd: (captured.__setitem__("command", cmd), None)[1])
    agent = simple_mod.SimpleAgent(None)
    msg = _run(agent.run("Execute: rm -rf temp/"))
    assert captured["command"] == "rm -rf temp/"
    assert msg.pending_action is not None
    assert msg.pending_action.target == "rm -rf temp/"


def test_agent_run_the_command_phrase(monkeypatch):
    captured = {}
    monkeypatch.setattr(simple_mod, "_gate_command", lambda cmd: (captured.__setitem__("command", cmd), None)[1])
    agent = simple_mod.SimpleAgent(None)
    msg = _run(agent.run("Run the command `pwd`"))
    assert captured["command"] == "pwd"
    assert msg.pending_action.target == "pwd"


def test_agent_prose_run_tests_routes_to_test_handler(monkeypatch):
    """'Please run the tests' resolves a pytest command from the project
    environment — never the shell command 'the tests'."""
    captured = {}
    monkeypatch.setattr(simple_mod, "_gate_command", lambda cmd: (captured.__setitem__("command", cmd), None)[1])
    agent = simple_mod.SimpleAgent(None)
    _run(agent.run("Please run the tests"))
    assert "pytest" in captured["command"]
    assert not captured["command"].startswith("the")


def test_agent_explicit_test_target_preserved(monkeypatch):
    """'Run pytest tests/test_api.py' keeps its target, resolved to the
    project environment (venv-wrapped) — never drops the file."""
    captured = {}
    monkeypatch.setattr(simple_mod, "_gate_command", lambda cmd: (captured.__setitem__("command", cmd), None)[1])
    agent = simple_mod.SimpleAgent(None)
    _run(agent.run("Run pytest tests/test_api.py"))
    assert "tests/test_api.py" in captured["command"]
    assert "pytest" in captured["command"]


def test_agent_code_intelligence_uses_dedicated_tool(monkeypatch):
    """'Where is TaskState defined?' must use find_definition, not the shell."""
    captured = _patch(monkeypatch, check_action=_allow, execute_tool=True)
    agent = simple_mod.SimpleAgent(None)
    msg = _run(agent.run("Where is TaskState defined?"))
    # The lookup searches the workspace root (explicit path), plus the name.
    assert captured["execute_tool"]["kwargs"]["name"] == "TaskState"
    assert captured["execute_tool"]["kwargs"].get("path")
    assert msg.pending_action is None  # read-only lookup auto-executes


def test_agent_directory_list_uses_list_directory(monkeypatch):
    captured = _patch(monkeypatch, check_action=_allow, execute_tool=True)
    agent = simple_mod.SimpleAgent(None)
    _run(agent.run("List the files here"))
    # "here" is resolved to the actual workspace root, never the literal word.
    assert captured["execute_tool"]["kwargs"]["path"] != "."
    assert captured["execute_tool"]["kwargs"]["path"].startswith("/")


def test_agent_mkdir_pending_action_target(monkeypatch):
    """'Make a directory called TodoList' → mkdir -p TodoList via confirmation."""
    _patch(monkeypatch, check_action=_confirm)
    agent = simple_mod.SimpleAgent(None)
    msg = _run(agent.run("Make a directory called TodoList"))
    assert msg.pending_action is not None
    assert msg.pending_action.target == "mkdir -p TodoList"


def test_agent_delete_file_pending_action(monkeypatch):
    _patch(monkeypatch, check_action=_confirm)
    agent = simple_mod.SimpleAgent(None)
    msg = _run(agent.run("Delete test.txt"))
    assert msg.pending_action is not None
    assert msg.pending_action.action_type == "delete_file"
    assert msg.pending_action.target == "test.txt"


def test_agent_info_question_runs_pwd(monkeypatch):
    """'What is the current working directory?' → pwd, not a chat reply."""
    captured = _patch(monkeypatch, check_action=_allow, execute_tool=True)
    agent = simple_mod.SimpleAgent(None)
    _run(agent.run("What is the current working directory?"))
    assert captured["execute_tool"]["kwargs"] == {"command": "pwd"}


def test_agent_git_phrasing_still_routes_to_git(monkeypatch):
    """Existing git routing is unchanged: 'show git status' → git status."""
    captured = _patch(monkeypatch, check_action=_allow, execute_tool=True)
    agent = simple_mod.SimpleAgent(None)
    _run(agent.run("Show git status"))
    assert captured["execute_tool"]["kwargs"] == {"command": "git status"}


def test_agent_ambiguous_request_no_hallucinated_command(monkeypatch):
    """'Fix it' must not produce a fake command or pending action."""
    engine = FakeEngine(["none", "I'll need more details about what to fix."])
    agent = simple_mod.SimpleAgent(engine)
    msg = _run(agent.run("fix it"))
    assert msg.pending_action is None
    assert "fix" in msg.content.lower() or "detail" in msg.content.lower()


def test_llm_fallback_normalizes_model_command(monkeypatch):
    """A model tool call with a wrapped command is normalized before execution."""
    captured = _patch(monkeypatch, execute_tool=True)
    engine = FakeEngine(["none", _tool_call("run_command", command="Execute: pwd")])
    agent = simple_mod.SimpleAgent(engine)
    msg = _run(agent.run("do the thing"))
    assert captured["execute_tool"]["kwargs"] == {"command": "pwd"}
    assert msg.pending_action is None


def test_llm_fallback_refuses_prose_as_command(monkeypatch):
    """A model tool call whose command is prose is refused, never executed."""
    captured = _patch(monkeypatch, execute_tool=True)
    engine = FakeEngine(["none", _tool_call("run_command", command="the tests are passing")])
    agent = simple_mod.SimpleAgent(engine)
    msg = _run(agent.run("do the thing"))
    assert "execute_tool" not in captured
    assert "won't run" in msg.content


def test_adversarial_same_action_different_phrasing(monkeypatch):
    """All phrasings of 'pwd' resolve to the identical tool call."""
    captured = _patch(monkeypatch, check_action=_allow, execute_tool=True)
    agent = simple_mod.SimpleAgent(None)
    for request_text, expected in TERMINAL_VARIANTS:
        captured.clear()
        _run(agent.run(request_text))
        assert captured["execute_tool"]["kwargs"] == {"command": expected}


def test_multistep_request_is_not_a_single_command():
    """Compound requests stay with the multi-step planner, never one shell line."""
    from ultron.core.agents.simple import detect_multistep_intent
    assert detect_multistep_intent("create a file and then run it") is True
    assert route_request("create a file and then run it") is None
