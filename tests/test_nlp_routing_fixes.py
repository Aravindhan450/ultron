"""FIX #8.5 — Capability-routing fixes: workspace context, code-intelligence
routing, project-aware test execution, and structured tool failures.

Deterministic tests for the three observed smoke-test failures plus the full
routing matrix:

- WORKSPACE: ``WorkspaceContext`` resolves location phrases (\"current
  directory\" / \"here\" / \"this folder\" / \".\" / relative paths / project
  root) to the actual workspace root — never to the literal phrase.
- FILESYSTEM: list / read / create / delete route to dedicated tools.
- CODE: \"where is X defined\" (all word orders), references, lexical and
  semantic search, architecture questions.
- TEST: explicit pytest commands, \"run the tests\", \"run the relevant
  tests\" (affected-test selection), \"run the full test suite\", virtualenv
  resolution, missing-test-executable fallback.
- ROUTING: dedicated tools preferred over terminal; safe fallbacks;
  ambiguous requests never hallucinate a command.
- ERRORS: deterministic tool failures surface as structured messages — never
  the generic \"couldn't generate a response\".
"""

from __future__ import annotations

import asyncio

from ultron.core.agents import simple as simple_mod
from ultron.core.agents.security import Decision
from ultron.core.coding.test_selection import select_affected_tests
from ultron.core.nlp import workspace as workspace_mod
from ultron.core.nlp.intent import IntentCategory, route_request
from ultron.core.nlp.project import (
    resolve_explicit_test_command,
    resolve_test_command,
)
from ultron.core.nlp.workspace import (
    git_changed_files,
    resolve_location_path,
    resolve_workspace,
)


def _run(coro):
    return asyncio.run(coro)


class _Verdict:
    def __init__(self, decision):
        self.decision = decision
        self.tier = "low"
        self.reason = "test"


def _allow():
    return _Verdict(Decision.ALLOW)


# ---------------------------------------------------------------------------
# 1. WORKSPACE CONTEXT
# ---------------------------------------------------------------------------


def test_workspace_root_resolution(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\n")
    (tmp_path / "src").mkdir()
    ws = resolve_workspace(str(tmp_path))
    assert ws.project_root == str(tmp_path)
    assert ws.project_type == "python"
    assert ws.workspace_root == str(tmp_path)


def test_workspace_detects_venv(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\n")
    (tmp_path / ".venv" / "bin").mkdir(parents=True)
    (tmp_path / ".venv" / "bin" / "python").write_text("")
    ws = resolve_workspace(str(tmp_path))
    assert ws.environment_root is not None
    assert ws.environment_root.endswith(".venv")


def test_location_phrases_resolve_to_root(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\n")
    for phrase in ("current directory", "current folder", "here", "this folder",
                   "this directory", ".", "working directory", "project root",
                   "workspace"):
        assert resolve_location_path(phrase, cwd=str(tmp_path)) == str(tmp_path), phrase
    # Empty expression also means the workspace root.
    assert resolve_location_path("", cwd=str(tmp_path)) == str(tmp_path)


def test_relative_path_resolves_against_project_root(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\n")
    (tmp_path / "src").mkdir()
    assert resolve_location_path("src", cwd=str(tmp_path)) == str(tmp_path / "src")
    assert resolve_location_path("./src", cwd=str(tmp_path)) == str(tmp_path / "src")
    assert resolve_location_path("src/", cwd=str(tmp_path)) == str(tmp_path / "src")


def test_absolute_path_passthrough(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\n")
    (tmp_path / "other").mkdir()
    assert resolve_location_path(str(tmp_path / "other"), cwd=str(tmp_path)) == str(
        tmp_path / "other"
    )


def test_directory_list_current_directory_routes_with_root_path(monkeypatch):
    """FAILURE 1: 'List the files in the current directory' must list the
    actual workspace — the extracted argument is '.' and dispatch resolves it
    to the workspace root (never path='the')."""
    it = route_request("List the files in the current directory")
    assert it is not None
    assert it.intent_type is IntentCategory.DIRECTORY_LIST
    assert it.tool == "list_directory"
    assert it.arguments == {"path": "."}

    captured = {}

    def fake_execute(tool, **kwargs):
        captured["tool"] = tool
        captured["path"] = kwargs.get("path")
        return "ok"

    monkeypatch.setattr(simple_mod, "execute_tool", fake_execute)
    monkeypatch.setattr(simple_mod, "check_action", lambda *a, **k: _allow())
    monkeypatch.setattr(simple_mod, "is_denied", lambda r: False)
    monkeypatch.setattr(simple_mod, "is_confirm", lambda r: False)
    monkeypatch.setattr(simple_mod, "is_allow", lambda r: True)
    monkeypatch.setattr(simple_mod, "security_mode", lambda: "interactive")
    monkeypatch.setattr(simple_mod, "blocked_message", lambda r: "blocked")
    _run(simple_mod.SimpleAgent(None).run("List the files in the current directory"))
    assert captured["tool"] == "list_directory"
    # Dispatch resolved the location phrase to the actual workspace root.
    assert captured["path"] == resolve_workspace().project_root


def test_directory_list_this_folder_same_semantics(monkeypatch):
    it = route_request("Show me the files in this folder")
    assert it is not None
    assert it.intent_type is IntentCategory.DIRECTORY_LIST
    assert it.arguments == {"path": "."}


def test_directory_list_explicit_path_kept():
    it = route_request("what's in src")
    assert it is not None
    assert it.intent_type is IntentCategory.DIRECTORY_LIST
    assert it.arguments == {"path": "src"}


# ---------------------------------------------------------------------------
# 2. FILESYSTEM ROUTING
# ---------------------------------------------------------------------------


def test_file_read_routes_to_dedicated_tool():
    # The deterministic file-read detector owns this before route_request.
    assert simple_mod.detect_file_read_intent("Read pyproject.toml") == "pyproject.toml"
    assert simple_mod.detect_file_read_intent("Open src/main.py") == "src/main.py"


def test_file_delete_routes_to_dedicated_tool():
    it = route_request("Delete test.txt")
    assert it is not None
    assert it.intent_type is IntentCategory.FILE_DELETE
    assert it.tool == "delete_file"
    assert it.arguments == {"file_path": "test.txt"}


def test_file_create_never_becomes_shell_command():
    # Multi-step / create phrasing must not be forced into a single command.
    it = route_request("Create a file called hello.py")
    assert it is None or it.tool != "run_command"


# ---------------------------------------------------------------------------
# 3. CODE INTELLIGENCE ROUTING
# ---------------------------------------------------------------------------


def test_definition_lookup_word_orders():
    """FAILURE 2: all word orders of 'where is X defined' route to
    find_definition — never to the LLM fallback."""
    for phrase in (
        "Where is TaskState defined?",
        "Find where TaskState is defined",
        "Where TaskState is defined",
        "Where is TaskState declared?",
        "Find the definition of TaskState",
        "Definition of Supervisor",
    ):
        it = route_request(phrase)
        assert it is not None, phrase
        assert it.intent_type is IntentCategory.DEFINITION_LOOKUP, phrase
        assert it.tool == "find_definition", phrase
        assert "TaskState" in it.arguments.get("name", "") or "Supervisor" in it.arguments.get("name", "")


def test_reference_lookup_variants():
    for phrase in (
        "Where is TaskState used?",
        "Find references to TaskState",
        "Find all usages of UserService",
        "What calls authenticate()?",
        "Who calls CodingExecutor?",
    ):
        it = route_request(phrase)
        assert it is not None, phrase
        assert it.intent_type is IntentCategory.REFERENCE_LOOKUP, phrase
        assert it.tool == "find_references", phrase


def test_where_implemented_routes_to_code_search():
    for phrase in (
        "Where is authentication implemented?",
        "Find where command execution is implemented",
        "Where is command execution handled?",
    ):
        it = route_request(phrase)
        assert it is not None, phrase
        assert it.intent_type is IntentCategory.CODE_SEARCH, phrase
        assert it.tool == "code_search", phrase


def test_lexical_search_routes_to_code_search():
    it = route_request('Search the code for "pytest"')
    assert it is not None
    assert it.intent_type is IntentCategory.CODE_SEARCH
    assert it.tool == "code_search"


def test_semantic_search_phrases():
    for phrase in (
        "Find the code responsible for user login",
        "Find where authentication is implemented",
        "Search semantically for task planning",
    ):
        it = route_request(phrase)
        assert it is not None, phrase
        assert it.intent_type in (IntentCategory.SEMANTIC_SEARCH, IntentCategory.CODE_SEARCH), phrase


def test_architecture_question_never_shell():
    """Architecture questions ('how does X work') must not become shell
    commands, and must not force a hallucinated action."""
    for phrase in (
        "How does TaskState flow through the agent?",
        "Explain how tool execution works",
        "How does the Supervisor delegate work?",
    ):
        it = route_request(phrase)
        assert it is None or it.tool != "run_command", phrase


# ---------------------------------------------------------------------------
# 4. TEST INTELLIGENCE ROUTING
# ---------------------------------------------------------------------------


def test_explicit_pytest_command_keeps_target():
    assert simple_mod.detect_explicit_test_command("Run pytest tests/test_agent.py") == (
        "pytest tests/test_agent.py"
    )
    assert simple_mod.detect_explicit_test_command("Run python -m pytest tests/ -q") == (
        "python -m pytest tests/ -q"
    )


def test_explicit_command_resolved_to_venv(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\n")
    (tmp_path / ".venv" / "bin").mkdir(parents=True)
    (tmp_path / ".venv" / "bin" / "python").write_text("")
    resolved = resolve_explicit_test_command("pytest tests/test_api.py", cwd=str(tmp_path))
    assert resolved == f"{tmp_path}/.venv/bin/python -m pytest tests/test_api.py"
    # No venv -> unchanged.
    other = tmp_path / "plain"
    other.mkdir()
    assert resolve_explicit_test_command("pytest tests/test_api.py", cwd=str(other)) == (
        "pytest tests/test_api.py"
    )


def test_run_the_tests_resolves_project_env(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\n[tool.pytest.ini_options]\n")
    (tmp_path / ".venv" / "bin").mkdir(parents=True)
    (tmp_path / ".venv" / "bin" / "python").write_text("")
    resolved = resolve_test_command(str(tmp_path))
    assert resolved is not None
    assert resolved.framework == "pytest"
    assert resolved.environment == "venv"
    assert resolved.executable == f"{tmp_path}/.venv/bin/python"
    assert resolved.arguments == ["-m", "pytest"]


def test_run_the_tests_no_venv_uses_python_module(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\n[tool.pytest.ini_options]\n")
    resolved = resolve_test_command(str(tmp_path))
    assert resolved is not None
    assert resolved.executable == "python"
    assert resolved.arguments == ["-m", "pytest"]


def test_missing_test_framework_no_command(tmp_path):
    # No pyproject/package.json/config -> resolver must not invent a command.
    resolved = resolve_test_command(str(tmp_path))
    assert resolved is None


def test_agent_relevant_tests_routes_to_affected_selection(monkeypatch, tmp_path):
    """FAILURE 3: 'Run the relevant tests' must select affected tests with the
    project environment — never bare pytest."""
    (tmp_path / "pyproject.toml").write_text("[project]\n[tool.pytest.ini_options]\n")
    (tmp_path / "src" / "auth").mkdir(parents=True)
    (tmp_path / "src" / "auth" / "service.py").write_text("x = 1\n")
    (tmp_path / "tests" / "auth").mkdir(parents=True)
    (tmp_path / "tests" / "auth" / "test_service.py").write_text("def test_x(): pass\n")

    captured = {}
    monkeypatch.setattr(
        workspace_mod, "resolve_workspace", lambda cwd=None: resolve_workspace(str(tmp_path))
    )
    monkeypatch.setattr(
        workspace_mod, "git_changed_files", lambda cwd=None, limit=200: ["src/auth/service.py"]
    )
    # route handle_relevant_tests -> handle_command(_gate_command)
    monkeypatch.setattr(
        simple_mod, "_gate_command",
        lambda cmd: (captured.__setitem__("command", cmd), None)[1],
    )
    _run(simple_mod.SimpleAgent(None).run("Run the relevant tests"))
    assert "pytest" in captured["command"]
    assert "test_service.py" in captured["command"]
    assert not captured["command"].startswith("pytest ")


def test_select_affected_tests_deterministic(tmp_path):
    (tmp_path / "tests" / "auth").mkdir(parents=True)
    (tmp_path / "tests" / "auth" / "test_service.py").write_text("def test_x(): pass\n")
    selected = select_affected_tests(["src/auth/service.py"], str(tmp_path))
    assert "tests/auth/test_service.py" in selected


def test_git_changed_files_readonly(tmp_path):
    # Not a git repo -> [] (never raises).
    assert git_changed_files(str(tmp_path)) == []


# ---------------------------------------------------------------------------
# 5. ROUTING PRIORITY + FALLBACK
# ---------------------------------------------------------------------------


def test_dedicated_tool_preferred_over_terminal():
    # Code-intelligence phrasing must never become 'grep' via the shell.
    for phrase in ("Find references to TaskState", "Where is TaskState used?"):
        it = route_request(phrase)
        assert it is not None
        assert it.tool != "run_command"


def test_terminal_still_works():
    for phrase, expected in (
        ("Run pwd", "pwd"),
        ("Execute: git status", "git status"),
        ("Run the command `pwd`", "pwd"),
        ("Use the terminal and execute pwd", "pwd"),
    ):
        it = route_request(phrase)
        assert it is not None
        assert it.intent_type is IntentCategory.TERMINAL_EXECUTION
        assert it.arguments["command"] == expected


def test_full_test_suite_routes_to_test_handler(monkeypatch):
    """'Run the full test suite' must resolve a project test command — never
    fall to the LLM (which would hang with the engine down)."""
    captured = {}
    monkeypatch.setattr(
        simple_mod, "_gate_command",
        lambda cmd: (captured.__setitem__("command", cmd), None)[1],
    )
    _run(simple_mod.SimpleAgent(None).run("Run the full test suite"))
    assert "pytest" in captured["command"]
    assert not captured["command"].startswith("the")


def test_git_phrasing_routes_to_git_handler():
    assert simple_mod.detect_git_intent("Show me the current git diff") == "git diff"
    assert simple_mod.detect_git_intent("Show git status") == "git status"


def test_ambiguous_request_no_hallucinated_command():
    for phrase in ("run it", "fix it", "check the backend", "look at the project"):
        it = route_request(phrase)
        assert it is None or it.tool != "run_command" or it.arguments.get("command") != phrase


# ---------------------------------------------------------------------------
# 6. STRUCTURED FAILURES — never the generic LLM message
# ---------------------------------------------------------------------------


def test_routed_failures_return_tool_output_not_llm_message():
    """When a request is deterministically routed, its result is the tool's
    structured output — the generic 'couldn't generate a response' fallback is
    only reachable for genuinely unroutable requests."""
    it = route_request("Find where TaskState is defined")
    assert it is not None
    assert it.tool == "find_definition"
    assert "couldn't generate" not in it.objective


def test_definition_lookup_no_result_is_structured(monkeypatch):
    captured = {}

    def fake_execute(tool, **kwargs):
        captured["tool"] = tool
        return "No definition found for 'Nope' in the index."

    monkeypatch.setattr(simple_mod, "execute_tool", fake_execute)
    monkeypatch.setattr(simple_mod, "check_action", lambda *a, **k: _allow())
    monkeypatch.setattr(simple_mod, "is_denied", lambda r: False)
    monkeypatch.setattr(simple_mod, "is_confirm", lambda r: False)
    monkeypatch.setattr(simple_mod, "is_allow", lambda r: True)
    monkeypatch.setattr(simple_mod, "security_mode", lambda: "interactive")
    monkeypatch.setattr(simple_mod, "blocked_message", lambda r: "blocked")
    msg = _run(simple_mod.SimpleAgent(None).run("Find where Nope is defined"))
    assert captured["tool"] == "find_definition"
    assert "No definition found" in msg.content
    assert "couldn't generate" not in msg.content


def test_tool_missing_is_structured_failure(monkeypatch):
    def fake_execute(tool, **kwargs):
        return f"Error: Tool '{tool}' not found in registry."

    monkeypatch.setattr(simple_mod, "execute_tool", fake_execute)
    monkeypatch.setattr(simple_mod, "check_action", lambda *a, **k: _allow())
    monkeypatch.setattr(simple_mod, "is_denied", lambda r: False)
    monkeypatch.setattr(simple_mod, "is_confirm", lambda r: False)
    monkeypatch.setattr(simple_mod, "is_allow", lambda r: True)
    monkeypatch.setattr(simple_mod, "security_mode", lambda: "interactive")
    monkeypatch.setattr(simple_mod, "blocked_message", lambda r: "blocked")
    msg = _run(simple_mod.SimpleAgent(None).run("Find where TaskState is defined"))
    assert "Error: Tool" in msg.content
    assert "couldn't generate" not in msg.content
