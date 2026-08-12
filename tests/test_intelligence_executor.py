"""
Fix #4 integration tests: Code Intelligence in the CodingExecutor.

Covers the full requirements list from the integration spec:

1.  definition lookup            9.  bug fixing using code intelligence
2.  reference lookup            10. refactoring using references
3.  symbol search               11. feature implementation
4.  semantic search             12. code review (read-only)
5.  dependency lookup           13. stale index invalidation
6.  LSP fallback                14. modified file re-indexing
7.  semantic fallback           15. targeted context retrieval
8.  repository exploration      16. large repository behavior

Plus: tool-selection guidance, observability, serialization survival across
confirmation, exploration recording, and security boundaries.

All filesystem tests use temporary repositories; the real Ultron repository
is never modified or indexed. No network, no real LLM, no real language
servers — the ReAct-loop scenarios use a scripted FakeEngine against the
REAL agent loop and REAL tools.
"""

import asyncio
import json
import sys

import pytest

from ultron.core.agents.react import ReActAgent
from ultron.core.coding.context import CodeContext
from ultron.core.coding.intelligence_bridge import CodeIntelligenceBridge
from ultron.core.coding.workspace import discover_workspace
from ultron.core.tools import paths as tools_paths
from ultron.core.types import (
    FailureStrategy,
    PlanStep,
    TaskPlan,
    TaskState,
    TaskType,
    WorkspaceKind,
)
from ultron.main import (
    continue_task_after_confirmation,
    execute_pending_action,
)

PYTHON = sys.executable


class FakeEngine:
    """Scripted engine — deterministic responses, no real LLM."""

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


def _pytest_call(project_file: str = "test_math.py") -> str:
    return _tool_call(
        "run_command",
        command=f"{PYTHON} -B -m pytest -q -p no:cacheprovider {project_file}",
    )


def _all_satisfied(step_criteria, plan_criteria) -> str:
    return json.dumps(
        {
            "step_criteria": [{"description": c, "satisfied": True} for c in step_criteria],
            "plan_criteria": [{"description": c, "satisfied": True} for c in plan_criteria],
            "step_failed": False,
            "plan_revision": None,
        }
    )


def _make_task(
    goal: str,
    task_type: TaskType,
    steps: list[PlanStep],
    plan_criteria: list[str],
    verification: list[str],
    workspace: WorkspaceKind,
    cwd: str,
) -> TaskState:
    """Hand-builds a planned TaskState with a CodeContext (no planner LLM)."""
    task = TaskState(goal=goal, task_type=task_type)
    task.attach_plan(
        TaskPlan(
            goal=goal,
            task_type=task_type,
            workspace=workspace,
            steps=steps,
            completion_criteria=plan_criteria,
            verification_requirements=verification,
        )
    )
    task.code_context = CodeContext(workspace=discover_workspace(cwd))
    task.code_context.attach_task(task)
    return task


def _step(
    step_id: int,
    description: str,
    criteria: list[str],
    deps: list[int] | None = None,
) -> PlanStep:
    return PlanStep(
        id=step_id,
        description=description,
        purpose=f"Purpose of {description}",
        expected_outcome=f"Outcome of {description}",
        completion_criteria=criteria,
        dependencies=deps or [],
        failure_strategy=FailureStrategy.STOP,
    )


def _write(root, rel: str, text: str):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """A temp workspace that IS the ALLOWED_BASE_DIR (so tools + bridge work)."""
    monkeypatch.setattr(tools_paths, "ALLOWED_BASE_DIR", tmp_path)
    monkeypatch.chdir(tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# Bridge basics: definition / reference / symbol / dependency lookups
# ---------------------------------------------------------------------------


def test_bridge_definition_lookup(sandbox):
    _write(sandbox, "src/service.py", (
        "from .models import User\n\nclass UserService:\n    def login(self, name):\n        return User(name)\n"
    ))
    _write(sandbox, "src/models.py", "class User:\n    pass\n")
    bridge = CodeIntelligenceBridge()
    assert bridge.enable(str(sandbox))
    bridge.refresh()

    out = bridge.query("find_definition", name="UserService")
    assert "Definitions of 'UserService'" in out
    assert "src/service.py" in out

    out2 = bridge.query("find_definition", name="Missing")
    assert "No definition found" in out2
    bridge.close()


def test_bridge_reference_lookup(sandbox):
    _write(sandbox, "src/service.py", "class UserService:\n    def login(self):\n        return 1\n")
    _write(sandbox, "src/api.py", "from .service import UserService\n\nsvc = UserService()\n")
    bridge = CodeIntelligenceBridge()
    assert bridge.enable(str(sandbox))
    bridge.refresh()

    out = bridge.query("find_references", name="UserService")
    assert "src/api.py" in out
    assert "src/service.py" not in out  # definition line is not a reference
    bridge.close()


def test_bridge_symbol_search(sandbox):
    _write(sandbox, "app.py", "import os\n\nVALUE = 1\n\ndef run():\n    return VALUE\n")
    bridge = CodeIntelligenceBridge()
    assert bridge.enable(str(sandbox))
    bridge.refresh()

    out = bridge.query("find_symbol", name="run")
    assert "run" in out and "function" in out
    out2 = bridge.query("find_symbol", name="os")
    assert "os" in out2  # imports are symbols too
    bridge.close()


def test_bridge_dependency_lookup(sandbox):
    _write(sandbox, "src/service.py", "from .models import User\n")
    _write(sandbox, "src/models.py", "class User:\n    pass\n")
    _write(sandbox, "src/api.py", "from .service import UserService\n")
    bridge = CodeIntelligenceBridge()
    assert bridge.enable(str(sandbox))
    bridge.refresh()

    out = bridge.query("get_imports", file_path="src/service.py")
    assert "Imports of src/service.py" in out
    assert "models" in out

    out2 = bridge.query("get_dependents", file_path="src/service.py")
    assert "src/api.py" in out2
    bridge.close()


# ---------------------------------------------------------------------------
# Semantic search + fallbacks (L5 -> L2)
# ---------------------------------------------------------------------------


class _FakeEmbedder:
    """Deterministic embedder for the semantic path (no real model)."""

    def __init__(self, available: bool = True):
        self._available = available

    def available(self) -> bool:
        return self._available

    def embed(self, texts):
        vectors = []
        for text in texts:
            vectors.append([
                float("login" in text),
                float("user" in text.lower()),
                float("class" in text),
                float("handler" in text),
            ])
        return vectors


def test_bridge_semantic_search_metadata(sandbox):
    _write(sandbox, "a.py", "class AuthService:\n    def login(self, user):\n        return user\n")
    _write(sandbox, "b.py", "class Renderer:\n    def render(self):\n        return 'x'\n")
    bridge = CodeIntelligenceBridge()
    assert bridge.enable(str(sandbox))
    bridge.refresh()

    out = bridge.query("semantic_search", query="login")
    assert "Semantic matches for 'login'" in out
    assert "a.py" in out
    assert "lexical_fallback" in out  # no embedder configured -> lexical fallback
    bridge.close()


def test_bridge_semantic_fallback_with_embedder(sandbox):
    from ultron.core.coding.intelligence.facade import CodeIntelligence

    _write(sandbox, "a.py", "class AuthService:\n    def login(self, user):\n        return user\n")
    ci = CodeIntelligence(root=str(sandbox), embedder=_FakeEmbedder(available=True))
    ci.refresh()
    hits = ci.search_semantically("login")
    assert hits and hits[0].mode == "semantic"

    # Unavailable embedder degrades to lexical_fallback.
    ci2 = CodeIntelligence(root=str(sandbox), embedder=_FakeEmbedder(available=False))
    ci2.refresh()
    hits2 = ci2.search_semantically("login")
    assert hits2 and hits2[0].mode == "lexical_fallback"
    ci.close()
    ci2.close()


def test_bridge_lsp_unavailable_degrades(sandbox):
    # The bridge/facade must keep working when no LSP server exists.
    bridge = CodeIntelligenceBridge()
    assert bridge.enable(str(sandbox))
    bridge.refresh()
    assert bridge.query("find_definition", name="x").startswith("No definition")
    # LSP layer is unavailable but the index path still answers.
    ci = bridge._build()
    assert ci is not None and ci.lsp_available() is False
    bridge.close()


# ---------------------------------------------------------------------------
# Repository exploration + observability
# ---------------------------------------------------------------------------


def test_bridge_repository_exploration(sandbox):
    _write(sandbox, "a.py", "class A:\n    pass\n")
    _write(sandbox, "b.py", "class B:\n    pass\n")
    bridge = CodeIntelligenceBridge()
    assert bridge.enable(str(sandbox))
    summary = bridge.refresh()
    assert "2 files" in summary and "2 symbols" in summary
    # Incremental: second refresh parses nothing.
    summary2 = bridge.refresh()
    assert "parsed 0" in summary2 and "unchanged 2" in summary2
    bridge.close()


def test_bridge_observability_records_queries(sandbox):
    _write(sandbox, "a.py", "class A:\n    pass\n")
    bridge = CodeIntelligenceBridge()
    bridge.enable(str(sandbox))
    bridge.refresh()
    bridge.query("find_definition", name="A")
    bridge.query("find_references", name="A")
    assert len(bridge.queries) == 3  # refresh + definition + references
    summary = bridge.usage_summary()
    assert "query(s)" in summary and "symbol" in summary
    # to_prompt_line is debuggable.
    line = bridge.queries[1].to_prompt_line()
    assert "find_definition('A')" in line
    bridge.close()


def test_bridge_exploration_tools_recorded(sandbox):
    _write(sandbox, "a.py", "class A:\n    pass\n")
    task = _make_task(
        "Find the A class",
        TaskType.SOFTWARE_ENGINEERING,
        [_step(1, "Locate the A implementation", ["located"])],
        ["done"],
        ["done"],
        WorkspaceKind.EXISTING_PROJECT,
        str(sandbox),
    )
    assert task.code_context.intelligence.enabled
    task.code_context.executor.record_observation(
        "find_definition", {"name": "A", "path": "."}, "Definitions of 'A'", succeeded=True
    )
    searches = task.code_context.executor.exploration.searches
    assert any("find_definition" in s for s in searches)
    assert task.code_context.executor.exploration.has_inspected()


# ---------------------------------------------------------------------------
# Tool selection strategy + guidance
# ---------------------------------------------------------------------------


def test_step_guidance_includes_intelligence_hint(sandbox):
    _write(sandbox, "a.py", "class A:\n    pass\n")
    task = _make_task(
        "Fix the authentication flow",
        TaskType.DEBUGGING,
        [_step(1, "Trace the authentication endpoint flow", ["traced"])],
        ["done"],
        ["done"],
        WorkspaceKind.EXISTING_PROJECT,
        str(sandbox),
    )
    guidance = task.code_context.executor.step_guidance(task)
    # Exploration advice mentions the code-intelligence tools.
    assert "find_definition" in guidance or "find_references" in guidance
    assert "code intelligence" in guidance


def test_intelligence_guidance_targeted_context(sandbox):
    _write(sandbox, "src/service.py", (
        "class UserService:\n    def login(self, user):\n        return user\n"
    ))
    task = _make_task(
        "Fix the UserService login bug",
        TaskType.DEBUGGING,
        [_step(1, "Inspect the UserService implementation", ["understood"])],
        ["done"],
        ["done"],
        WorkspaceKind.EXISTING_PROJECT,
        str(sandbox),
    )
    guidance = task.code_context.executor.intelligence_guidance(task)
    assert "CODE INTELLIGENCE" in guidance
    assert "UserService" in guidance  # targeted context includes the symbol
    assert "src/service.py" in guidance
    # Bounded: no repository dump.
    assert len(guidance.splitlines()) <= 30


def test_intelligence_guidance_empty_without_bridge(sandbox):
    # A workspace outside the allowed base dir -> bridge disabled -> no block.
    task = _make_task(
        "Do something",
        TaskType.MULTI_STEP,
        [_step(1, "Inspect something", ["ok"])],
        ["done"],
        ["done"],
        WorkspaceKind.EXISTING_PROJECT,
        str(sandbox),
    )
    task.code_context.intelligence.enabled = False
    assert task.code_context.executor.intelligence_guidance(task) == ""


# ---------------------------------------------------------------------------
# Staleness: dirty marking + incremental re-index after edits
# ---------------------------------------------------------------------------


def test_stale_index_invalidation_after_edit(sandbox):
    _write(sandbox, "a.py", "class A:\n    pass\n")
    bridge = CodeIntelligenceBridge()
    bridge.enable(str(sandbox))
    bridge.refresh()
    assert "Definitions of 'A'" in bridge.query("find_definition", name="A")

    # A new symbol appears on disk (as if the agent edited the file).
    _write(sandbox, "a.py", "class A:\n    pass\n\nclass B:\n    pass\n")
    bridge.mark_dirty()

    # The next query refreshes first and sees the new symbol — no stale data.
    out = bridge.query("find_definition", name="B")
    assert "Definitions of 'B'" in out
    assert bridge.dirty is False  # refresh cleared the flag
    bridge.close()


def test_modified_file_reindexing_incremental(sandbox):
    _write(sandbox, "a.py", "class A:\n    pass\n")
    _write(sandbox, "b.py", "class B:\n    pass\n")
    bridge = CodeIntelligenceBridge()
    bridge.enable(str(sandbox))
    bridge.refresh()

    _write(sandbox, "b.py", "class B:\n    pass\n\nclass B2:\n    pass\n")
    bridge.mark_dirty()
    summary = bridge.refresh()
    assert "parsed 1" in summary  # only the changed file re-parsed
    assert "unchanged 1" in summary
    bridge.close()


def test_dirty_flag_set_by_react_loop_on_edit(sandbox):
    # End-to-end: a confirmed edit in the real loop invalidates the index and
    # the resumed loop refreshes it — so the new symbol is visible to the
    # very next intelligence query (no stale source information).
    _write(sandbox, "app.py", "x = 1\n")
    engine = FakeEngine(
        [
            _tool_call(
                "replace_in_file",
                file_path="app.py",
                old="x = 1",
                new="x = 2\n\nclass Fresh:\n    pass\n",
            ),
        ]
    )
    task = _make_task(
        "Change x",
        TaskType.MULTI_STEP,
        [_step(1, "Change x", ["changed"])],
        ["done"],
        ["done"],
        WorkspaceKind.EXISTING_PROJECT,
        str(sandbox),
    )
    agent = ReActAgent(engine, max_iterations=4)
    msg = _run(agent.run(task.goal, [], task=task))
    result = _run(execute_pending_action(msg.pending_action))
    _run(continue_task_after_confirmation(agent, task, result, []))

    # The confirmed edit marked the index dirty; the resumed loop refreshed
    # it, so the symbol added by the edit is now findable.
    out = task.code_context.intelligence.query("find_definition", name="Fresh")
    assert "Definitions of 'Fresh'" in out


# ---------------------------------------------------------------------------
# Serialization across confirmation
# ---------------------------------------------------------------------------


def test_bridge_survives_task_serialization(sandbox):
    _write(sandbox, "a.py", "class A:\n    pass\n")
    task = _make_task(
        "Fix A",
        TaskType.SOFTWARE_ENGINEERING,
        [_step(1, "Fix A", ["fixed"])],
        ["done"],
        ["done"],
        WorkspaceKind.EXISTING_PROJECT,
        str(sandbox),
    )
    bridge = task.code_context.intelligence
    assert bridge.enabled
    bridge.refresh()
    bridge.query("find_definition", name="A")

    restored = TaskState.model_validate_json(task.model_dump_json())
    assert restored.code_context is not None
    restored_bridge = restored.code_context.intelligence
    assert restored_bridge.enabled is True
    assert restored_bridge.root == str(sandbox)
    assert len(restored_bridge.queries) == len(bridge.queries)
    # Rebuilt lazily and still answers.
    out = restored_bridge.query("find_definition", name="A")
    assert "Definitions of 'A'" in out
    restored_bridge.close()


# ---------------------------------------------------------------------------
# ReAct-loop integration scenarios
# ---------------------------------------------------------------------------

# Scenario 9 — bug fixing USING code intelligence instead of blind grep.


def test_loop_bug_fix_uses_intelligence(sandbox):
    _write(sandbox, "pyproject.toml", "[tool.pytest.ini_options]\ntestpaths=['.']\n")
    _write(sandbox, "math_util.py", "def add(a, b):\n    return a - b\n")
    _write(sandbox, "test_math.py", "from math_util import add\n\ndef test_add():\n    assert add(2, 3) == 5\n")

    engine = FakeEngine(
        [
            _tool_call("find_definition", name="add", path="."),
            _tool_call("read_file", file_path="math_util.py"),
            _pytest_call(),
            _tool_call(
                "replace_in_file",
                file_path="math_util.py",
                old="return a - b",
                new="return a + b",
            ),
            _pytest_call(),
            "The tests pass now.",
            _all_satisfied(["bug fixed", "tests pass"], ["tests pass"]),
        ]
    )
    task = _make_task(
        "Fix the failing tests",
        TaskType.DEBUGGING,
        [_step(1, "Fix the failing tests", ["bug fixed", "tests pass"])],
        ["tests pass"],
        ["tests pass"],
        WorkspaceKind.EXISTING_PROJECT,
        str(sandbox),
    )
    agent = ReActAgent(engine, max_iterations=6)
    msg = _run(agent.run(task.goal, [], task=task))
    for _ in range(6):
        if msg.pending_action is None:
            break
        result = _run(execute_pending_action(msg.pending_action))
        msg = _run(continue_task_after_confirmation(agent, task, result, []))

    assert msg.pending_action is None
    assert task.is_complete() is True
    assert "a + b" in (sandbox / "math_util.py").read_text(encoding="utf-8")
    # The intelligence lookup was actually used and recorded.
    tools = [e.tool_name for e in task.execution_history]
    assert "find_definition" in tools
    assert any("find_definition" in s for s in task.code_context.executor.exploration.searches)


# Scenario 10 — refactoring driven by find_references.


def test_loop_refactor_using_references(sandbox):
    _write(sandbox, "pyproject.toml", "[tool.pytest.ini_options]\ntestpaths=['.']\n")
    _write(sandbox, "widgets.py", (
        "class Widget:\n    pass\n\n"
        "def make():\n    return Widget()\n\n"
        "def make_two():\n    return Widget()\n"
    ))
    _write(sandbox, "test_widgets.py", (
        "from widgets import Widget, make, make_two\n\n"
        "def test_make():\n    assert isinstance(make(), Widget)\n\n"
        "def test_make_two():\n    assert isinstance(make_two(), Widget)\n"
    ))
    engine = FakeEngine(
        [
            _tool_call("find_references", name="make", path="."),
            _tool_call("find_references", name="make_two", path="."),
            _tool_call(
                "replace_in_file",
                file_path="widgets.py",
                old="def make():\n    return Widget()\n\ndef make_two():\n    return Widget()",
                new="def _new():\n    return Widget()\n\ndef make():\n    return _new()\n\ndef make_two():\n    return _new()",
            ),
            _pytest_call("test_widgets.py"),
            "Refactored — behavior preserved.",
            _all_satisfied(["duplication removed", "tests pass"], ["refactor done"]),
        ]
    )
    task = _make_task(
        "Refactor this module to remove the duplication without changing behavior",
        TaskType.SOFTWARE_ENGINEERING,
        [_step(1, "Refactor the module", ["duplication removed", "tests pass"])],
        ["refactor done"],
        ["refactor done"],
        WorkspaceKind.EXISTING_PROJECT,
        str(sandbox),
    )
    agent = ReActAgent(engine, max_iterations=6)
    msg = _run(agent.run(task.goal, [], task=task))
    for _ in range(6):
        if msg.pending_action is None:
            break
        result = _run(execute_pending_action(msg.pending_action))
        msg = _run(continue_task_after_confirmation(agent, task, result, []))

    assert msg.pending_action is None
    assert task.is_complete() is True
    text = (sandbox / "widgets.py").read_text(encoding="utf-8")
    assert "_new" in text
    tools = [e.tool_name for e in task.execution_history]
    assert tools.count("find_references") == 2


# Scenario 11 — feature implementation guided by symbol discovery.


def test_loop_feature_implementation_with_intelligence(sandbox):
    _write(sandbox, "pyproject.toml", "[tool.pytest.ini_options]\ntestpaths=['.']\n")
    _write(sandbox, "app.py", "class App:\n    def handle(self):\n        return 'hello'\n")
    _write(sandbox, "test_app.py", (
        "from app import App\n\n"
        "def test_handle():\n    assert App().handle() == 'hello'\n\n"
        "def test_health():\n    assert App().health() == 'ok'\n"
    ))
    engine = FakeEngine(
        [
            _tool_call("find_symbol", name="App", path="."),
            _tool_call("read_file", file_path="app.py"),
            _tool_call(
                "replace_in_file",
                file_path="app.py",
                old="return 'hello'",
                new="return 'hello'\n\n    def health(self):\n        return 'ok'",
            ),
            _pytest_call("test_app.py"),
            "Feature added.",
            _all_satisfied(["health endpoint added", "tests pass"], ["feature done"]),
        ]
    )
    task = _make_task(
        "Add a health method to the App class",
        TaskType.SOFTWARE_ENGINEERING,
        [_step(1, "Add the health method", ["health endpoint added", "tests pass"])],
        ["feature done"],
        ["feature done"],
        WorkspaceKind.EXISTING_PROJECT,
        str(sandbox),
    )
    agent = ReActAgent(engine, max_iterations=6)
    msg = _run(agent.run(task.goal, [], task=task))
    for _ in range(6):
        if msg.pending_action is None:
            break
        result = _run(execute_pending_action(msg.pending_action))
        msg = _run(continue_task_after_confirmation(agent, task, result, []))

    assert msg.pending_action is None
    assert task.is_complete() is True
    assert "def health" in (sandbox / "app.py").read_text(encoding="utf-8")
    tools = [e.tool_name for e in task.execution_history]
    assert "find_symbol" in tools


# Scenario 12 — code review must NOT modify anything.


def test_loop_code_review_read_only(sandbox):
    _write(sandbox, "src/auth.py", (
        "class Authenticator:\n    def check(self, token):\n        return token == 'x'\n"
    ))
    _write(sandbox, "src/api.py", "from .auth import Authenticator\n\nauth = Authenticator()\n")
    engine = FakeEngine(
        [
            _tool_call("report_file", file_path="src/auth.py", path="."),
            _tool_call("find_references", name="Authenticator", path="."),
            "The authenticator compares the token literally. No code was modified.",
            _all_satisfied(["review findings reported"], ["review done"]),
        ]
    )
    task = _make_task(
        "Review this authentication module",
        TaskType.CODE_REVIEW,
        [_step(1, "Review the authentication module", ["review findings reported"])],
        ["review done"],
        ["review done"],
        WorkspaceKind.EXISTING_PROJECT,
        str(sandbox),
    )
    agent = ReActAgent(engine, max_iterations=5)
    msg = _run(agent.run(task.goal, [], task=task))
    for _ in range(4):
        if msg.pending_action is None:
            break
        result = _run(execute_pending_action(msg.pending_action))
        msg = _run(continue_task_after_confirmation(agent, task, result, []))

    assert msg.pending_action is None
    assert task.is_complete() is True
    # Read-only review: no file modifications, no state-changing tools.
    assert len(task.code_context.tracker.modifications) == 0
    tools = [e.tool_name for e in task.execution_history]
    assert "replace_in_file" not in tools and "write_file" not in tools
    assert "report_file" in tools


# Scenario 13/14 — stale index handled inside the real loop.


def test_loop_stale_index_refreshed_after_edit(sandbox):
    # The agent edits a file then asks the index a question about it — the
    # registered tool re-indexes incrementally, so no stale answers.
    _write(sandbox, "app.py", "def greet():\n    return 'hi'\n")
    engine = FakeEngine(
        [
            _tool_call("find_definition", name="greet", path="."),
            _tool_call(
                "replace_in_file",
                file_path="app.py",
                old="return 'hi'",
                new="return 'hi'\n\ndef bye():\n    return 'bye'\n",
            ),
            _tool_call("find_definition", name="bye", path="."),
            "Done.",
            _all_satisfied(["bye exists"], ["done"]),
        ]
    )
    task = _make_task(
        "Add a bye function",
        TaskType.SOFTWARE_ENGINEERING,
        [_step(1, "Add the bye function", ["bye exists"])],
        ["done"],
        ["done"],
        WorkspaceKind.EXISTING_PROJECT,
        str(sandbox),
    )
    agent = ReActAgent(engine, max_iterations=5)
    msg = _run(agent.run(task.goal, [], task=task))
    for _ in range(4):
        if msg.pending_action is None:
            break
        result = _run(execute_pending_action(msg.pending_action))
        msg = _run(continue_task_after_confirmation(agent, task, result, []))

    assert msg.pending_action is None
    assert task.is_complete() is True
    assert "def bye" in (sandbox / "app.py").read_text(encoding="utf-8")


# Scenario 15 — targeted context retrieval stays bounded.


def test_context_block_bounded_targeted(sandbox):
    for i in range(20):
        _write(sandbox, f"mod{i}.py", f"class Class{i}:\n    def method{i}(self):\n        return {i}\n")
    task = _make_task(
        "Fix the Class3 bug",
        TaskType.DEBUGGING,
        [_step(1, "Inspect Class3 and its method3", ["understood"])],
        ["done"],
        ["done"],
        WorkspaceKind.EXISTING_PROJECT,
        str(sandbox),
    )
    block = task.code_context.executor.intelligence_guidance(task)
    assert "Class3" in block
    assert len(block.splitlines()) <= 30
    # Does NOT dump every module.
    assert "Class19" not in block


# Scenario 16 — large repository behavior stays bounded.


def test_large_repo_bounded_results(sandbox):
    for i in range(60):
        _write(sandbox, f"pkg/mod{i}.py", f"def func_{i}():\n    return {i}\n")
    bridge = CodeIntelligenceBridge()
    assert bridge.enable(str(sandbox))
    summary = bridge.refresh()
    assert "60 files" in summary
    out = bridge.query("find_definition", name="func_1")
    assert "Definitions of 'func_1'" in out
    assert len(out.splitlines()) <= 12  # bounded output
    # Bounded observability log.
    for _ in range(10):
        bridge.query("find_symbol", name="func_2")
    assert len(bridge.queries) <= bridge.max_queries
    bridge.close()


# ---------------------------------------------------------------------------
# Security + self-guard
# ---------------------------------------------------------------------------


def test_bridge_refuses_ultron_own_repo():
    # The bridge must never auto-index Ultron's own repository.
    import ultron.core.coding.intelligence_bridge as bridge_mod

    bridge = CodeIntelligenceBridge()
    assert bridge.enable(str(bridge_mod._ULTRON_ROOT)) is False
    assert bridge.enabled is False


def test_bridge_refuses_outside_allowed_base(tmp_path, monkeypatch):
    monkeypatch.setattr(tools_paths, "ALLOWED_BASE_DIR", tmp_path)
    outside = tmp_path.parent / "outside_dir"
    outside.mkdir(exist_ok=True)
    bridge = CodeIntelligenceBridge()
    assert bridge.enable(str(outside)) is False


def test_verification_evidence_includes_intelligence_usage(sandbox):
    _write(sandbox, "a.py", "class A:\n    pass\n")
    task = _make_task(
        "Fix A",
        TaskType.SOFTWARE_ENGINEERING,
        [_step(1, "Fix A", ["fixed"])],
        ["done"],
        ["done"],
        WorkspaceKind.EXISTING_PROJECT,
        str(sandbox),
    )
    bridge = task.code_context.intelligence
    bridge.refresh()
    bridge.query("find_definition", name="A")
    evidence = task.code_context.executor.verification_evidence(task)
    assert "Code intelligence:" in evidence
    assert "symbol" in evidence


def test_intelligence_never_bypasses_security(sandbox):
    # Intelligence tools are read-only and LOW tier — the boundary still
    # gates every call; a path escape stays denied.
    from ultron.security import SecurityBoundary

    boundary = SecurityBoundary(mode="interactive")
    for tool in _INTELLIGENCE_TOOL_NAMES:
        verdict = boundary.check(tool, ".")
        assert verdict.decision.value == "allow", tool
        assert verdict.tier.value == "low", tool

    from ultron.security.guardrails import GuardrailsEngine

    engine = GuardrailsEngine()
    for tool in _INTELLIGENCE_TOOL_NAMES:
        assert engine.evaluate(action_type=tool, target="/etc/passwd").blocked, tool


_INTELLIGENCE_TOOL_NAMES = (
    "code_search",
    "find_symbol",
    "find_definition",
    "find_references",
    "get_imports",
    "get_dependents",
    "semantic_search",
    "code_index_status",
    "report_file",
    "report_symbol",
)


def test_report_file_and_symbol_route_through_loop(sandbox):
    # report_file / report_symbol are REAL registered tools the model can
    # call — the code-review scenario must not rely on a silently-failing
    # unknown-tool call.
    from ultron.core.tools.registry import TOOLS

    assert "report_file" in TOOLS
    assert "report_symbol" in TOOLS
    _write(sandbox, "src/auth.py", (
        "class Authenticator:\n    def check(self, token):\n        return token == 'x'\n"
    ))
    engine = FakeEngine(
        [
            _tool_call("report_file", file_path="src/auth.py", path="."),
            _tool_call("report_symbol", name="Authenticator", path="."),
            "Reported.",
            _all_satisfied(["reported"], ["done"]),
        ]
    )
    task = _make_task(
        "Review the auth module",
        TaskType.CODE_REVIEW,
        [_step(1, "Review the auth module", ["reported"])],
        ["done"],
        ["done"],
        WorkspaceKind.EXISTING_PROJECT,
        str(sandbox),
    )
    agent = ReActAgent(engine, max_iterations=5)
    msg = _run(agent.run(task.goal, [], task=task))
    for _ in range(4):
        if msg.pending_action is None:
            break
        result = _run(execute_pending_action(msg.pending_action))
        msg = _run(continue_task_after_confirmation(agent, task, result, []))

    assert task.is_complete() is True
    # Both calls actually executed (not recorded as unknown-tool errors).
    history = task.execution_history
    assert all(e.success for e in history), [e.detail for e in history if not e.success]
    tools = [e.tool_name for e in history]
    assert "report_file" in tools and "report_symbol" in tools
