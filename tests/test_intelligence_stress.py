"""
FIX #4 stress test: can Ultron actually UNDERSTAND a codebase?

The 20 scenarios from the stress-test spec are exercised against a rich
representative repository (multiple source dirs, duplicated symbol names,
nested modules, unrelated code, tests, config, docs). Deterministic: no
real LLM (scripted FakeEngine), no real language servers, no network, all
filesystem work in temporary directories.

What is genuinely exercised vs simulated:
- lexical search / AST symbol index / dependency graph: REAL (actual index)
- semantic search: REAL pipeline, FAKE deterministic concept embedder
- LSP: abstraction with a FAKE client for the available case and the real
  NoLSPServers fallback for the unavailable case
- the ReAct-loop scenarios run the REAL loop and REAL tools

The final test reports timings (indexing, incremental, lookups, context
size) so the performance review has concrete numbers.
"""

import asyncio
import json
import sys
import time
from typing import ClassVar

import pytest

from ultron.core.agents.react import ReActAgent
from ultron.core.coding.intelligence.dependencies import EdgeConfidence
from ultron.core.coding.intelligence.facade import CodeIntelligence
from ultron.core.coding.intelligence.lsp import (
    LSPCapabilities,
    LSPFacade,
    LSPLocation,
    LSPOperation,
    NoLSPServers,
)
from ultron.core.coding.intelligence.semantic import Embedder
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


def _pytest_call(project_file: str = "tests/test_auth.py") -> str:
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


def _make_task(goal, task_type, steps, plan_criteria, verification, workspace, cwd):
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
    task.code_context = __import__(
        "ultron.core.coding.context", fromlist=["CodeContext"]
    ).CodeContext(workspace=discover_workspace(cwd))
    task.code_context.attach_task(task)
    return task


def _step(step_id, description, criteria, deps=None):
    return PlanStep(
        id=step_id,
        description=description,
        purpose=f"Purpose of {description}",
        expected_outcome=f"Outcome of {description}",
        completion_criteria=criteria,
        dependencies=deps or [],
        failure_strategy=FailureStrategy.STOP,
    )


def _write(root, rel, text):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(tools_paths, "ALLOWED_BASE_DIR", tmp_path)
    monkeypatch.chdir(tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# The representative repository
# ---------------------------------------------------------------------------

# Note: auth uses sign_in / authenticate / verify_credentials — deliberately
# NO literal "login" — so the semantic test (test 6) proves conceptual
# retrieval rather than substring luck. UserService exists in BOTH auth and
# billing (test 10). billing/legacy code and docs are unrelated surface.


def build_rich_repo(root):
    """Builds the representative repository; returns the root Path."""
    # pythonpath="src" makes `from auth.service import ...` resolvable when
    # pytest runs from the repo root (a realistic src-layout project).
    _write(
        root,
        "pyproject.toml",
        "[tool.pytest.ini_options]\ntestpaths=['.']\npythonpath = ['src']\n",
    )
    _write(root, "README.md", "A sample service with authentication and billing.\n")
    _write(root, "docs/architecture.md", "# Architecture\n\nAuth via tokens. Billing separate.\n")

    _write(root, "src/app/__init__.py", "")
    _write(root, "src/app/config.py", (
        "TOKEN_TTL = 3600\n"
        "SECRET = 'dev-secret'\n"
        "class Settings:\n"
        "    def __init__(self):\n"
        "        self.debug = True\n"
    ))
    _write(root, "src/app/main.py", (
        "from auth.service import UserService\n"
        "from auth.middleware import AuthMiddleware\n"
        "\n"
        "def route_login(token):\n"
        "    svc = UserService()\n"
        "    return svc.authenticate(token)\n"
        "\n"
        "app = AuthMiddleware(route_login)\n"
    ))

    _write(root, "src/auth/__init__.py", "")
    _write(root, "src/auth/models.py", "class User:\n    def __init__(self, name):\n        self.name = name\n")
    _write(root, "src/auth/repository.py", (
        "from .models import User\n"
        "class UserRepository:\n"
        "    def __init__(self):\n"
        "        self._users = {'alice': 'tok-1'}\n"
        "    def token_for(self, user):\n"
        "        return self._users.get(user)\n"
    ))
    _write(root, "src/auth/service.py", (
        "from .repository import UserRepository\n"
        "\n"
        "class UserService:\n"
        "    def __init__(self):\n"
        "        self.repo = UserRepository()\n"
        "    def authenticate(self, token):\n"
        "        return token == 'tok-1'\n"
        "    def sign_in(self, user):\n"
        "        return self.repo.token_for(user)\n"
    ))
    _write(root, "src/auth/middleware.py", (
        "class AuthMiddleware:\n"
        "    def __init__(self, handler):\n"
        "        self.handler = handler\n"
        "    def __call__(self, token):\n"
        "        return self.handler(token)\n"
    ))
    _write(root, "src/auth/nested/__init__.py", "")
    _write(root, "src/auth/nested/tokens.py", (
        "class TokenManager:\n"
        "    def issue(self, user):\n"
        "        return 'tok-' + user\n"
    ))

    _write(root, "src/billing/__init__.py", "")
    _write(root, "src/billing/service.py", (
        "class UserService:\n"
        "    def charge(self, amount):\n"
        "        return amount * 100\n"
    ))

    _write(root, "src/utils/__init__.py", "")
    _write(root, "src/utils/logging.py", "class Logger:\n    def log(self, msg):\n        print(msg)\n")
    _write(root, "src/utils/legacy.py", (
        "def parse_csv(text):\n"
        "    return [line.split(',') for line in text.splitlines()]\n"
    ))

    _write(root, "tests/test_auth.py", (
        "from auth.service import UserService\n"
        "from auth.models import User\n"
        "\n"
        "def test_authenticate_ok():\n"
        "    assert UserService().authenticate('tok-1') is True\n"
        "\n"
        "def test_authenticate_rejects_bad():\n"
        "    assert UserService().authenticate('wrong') is False\n"
        "\n"
        "def test_sign_in():\n"
        "    assert UserService().sign_in('alice') == 'tok-1'\n"
    ))
    _write(root, "tests/test_billing.py", (
        "from billing.service import UserService\n"
        "def test_charge():\n"
        "    assert UserService().charge(2) == 200\n"
    ))
    _write(root, "tests/test_main.py", (
        "from app.main import route_login\n"
        "def test_route():\n"
        "    assert route_login('tok-1') is True\n"
    ))
    return root


# ---------------------------------------------------------------------------
# Concept embedder for the semantic tests (deterministic, fake)
# ---------------------------------------------------------------------------


class _ConceptEmbedder(Embedder):
    """Deterministic synonym-aware embedder over a fixed concept vocabulary.

    Maps surface words to concepts (login->sign_in, auth->authenticate, ...)
    and returns one-hot concept vectors, so a query can retrieve code that
    means the same thing without sharing a single character with it. Purely
    a test double — real embeddings are a later FIX.
    """

    VOCAB: ClassVar[tuple[str, ...]] = ("auth", "user", "login", "payment", "report", "parse")

    SYNONYMS: ClassVar[dict[str, frozenset[str]]] = {
        "login": frozenset({"login", "sign_in", "signin", "sign-in", "credential"}),
        "auth": frozenset({"auth", "authenticate", "token", "middleware"}),
        "user": frozenset({"user", "userrepository", "userservice"}),
        "payment": frozenset({"billing", "charge", "payment"}),
        "report": frozenset({"report", "log", "logger"}),
        "parse": {"parse", "csv", "parser"},
    }

    def __init__(self, available: bool = True):
        self._available = available

    def available(self) -> bool:
        return self._available

    def _concepts(self, text: str) -> set[str]:
        lowered = text.lower()
        concepts = set()
        for concept, synonyms in self.SYNONYMS.items():
            if any(syn in lowered for syn in synonyms):
                concepts.add(concept)
        return concepts

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            concepts = self._concepts(text)
            vectors.append([1.0 if c in concepts else 0.0 for c in self.VOCAB])
        return vectors


# ---------------------------------------------------------------------------
# TESTS 1-5: symbol / definition / references / callers / imports
# ---------------------------------------------------------------------------


def test_1_symbol_search_is_symbol_aware(sandbox):
    build_rich_repo(sandbox)
    ci = CodeIntelligence(root=str(sandbox))
    ci.refresh()

    # find_symbol also returns IMPORT-kind symbols (the 3 importing files), so
    # filter to the actual class definitions for the definition assertions.
    symbols = ci.find_symbol("UserService")
    definitions = [s for s in symbols if s.kind.value == "class"]
    assert len(definitions) == 2  # auth + billing, both surfaced (no hidden ambiguity)
    # Symbol-aware: file + line + type, not just text occurrences.
    for symbol in definitions:
        assert symbol.location.file.endswith("service.py")
        assert symbol.location.line >= 1
        assert symbol.kind.value == "class"
    files = {s.location.file for s in definitions}
    assert files == {"src/auth/service.py", "src/billing/service.py"}
    # The imports are ALSO surfaced as symbols (main.py / tests import it).
    import_kinds = {s.kind.value for s in symbols}
    assert "import" in import_kinds
    assert "src/app/main.py" in {s.location.file for s in symbols}
    ci.close()


def test_2_definition_lookup(sandbox):
    build_rich_repo(sandbox)
    ci = CodeIntelligence(root=str(sandbox))
    ci.refresh()

    defs = ci.find_definition("authenticate")
    assert len(defs) == 1
    assert defs[0].location.file == "src/auth/service.py"
    assert defs[0].kind.value == "method"
    assert defs[0].parent == "UserService"  # scope is known
    ci.close()


def test_3_references_lookup(sandbox):
    build_rich_repo(sandbox)
    ci = CodeIntelligence(root=str(sandbox))
    ci.refresh()

    refs = ci.find_references("authenticate")
    files = {r.location.file for r in refs}
    # Used in app/main.py and the auth tests.
    assert "src/app/main.py" in files
    assert "tests/test_auth.py" in files
    # The definition site itself is NOT a reference.
    assert "src/auth/service.py" not in files
    ci.close()


def test_4_callers_inferred(sandbox):
    build_rich_repo(sandbox)
    ci = CodeIntelligence(root=str(sandbox))
    ci.refresh()

    callers = ci.dependency_callers("authenticate")
    assert callers
    assert all(edge.confidence is EdgeConfidence.INFERRED for edge in callers)
    assert all(edge.kind == "calls" for edge in callers)
    assert "src/app/main.py" in {e.source for e in callers}
    ci.close()


def test_5_import_graph_dependents(sandbox):
    build_rich_repo(sandbox)
    ci = CodeIntelligence(root=str(sandbox))
    ci.refresh()

    # src/ layout: `from auth.service import UserService` (in app/main.py and
    # tests/test_auth.py) must resolve to src/auth/service.py.
    dependents = ci.dependency_dependents("src/auth/service.py")
    assert dependents, "get_dependents must resolve src/ layout absolute imports"
    assert all(edge.confidence is EdgeConfidence.EXACT for edge in dependents)
    files = {e.source for e in dependents}
    assert "src/app/main.py" in files
    assert "tests/test_auth.py" in files

    # The nested token module is NOT a dependent (no import edge).
    assert "src/auth/nested/tokens.py" not in files

    # Same resolution through the raw index API used by the tool.
    raw = ci.get_dependents("src/auth/service.py")
    assert "src/app/main.py" in raw and "tests/test_auth.py" in raw
    ci.close()


# ---------------------------------------------------------------------------
# TEST 6: semantic search WITHOUT the literal word
# ---------------------------------------------------------------------------


def test_6_semantic_search_concept_retrieval(sandbox):
    build_rich_repo(sandbox)
    # The auth code contains sign_in / authenticate — never the word "login".
    source = (sandbox / "src/auth/service.py").read_text(encoding="utf-8")
    assert "login" not in source.lower()

    ci = CodeIntelligence(root=str(sandbox), embedder=_ConceptEmbedder(available=True))
    ci.refresh()
    hits = ci.search_semantically("login", top_k=5)
    assert hits, "semantic search returned nothing"
    assert hits[0].mode == "semantic"
    # The conceptually-relevant auth code is found despite zero substring match.
    assert any(h.chunk.file.endswith("auth/service.py") for h in hits)
    assert any("sign_in" in h.chunk.symbol or "authenticate" in h.chunk.symbol for h in hits)

    # WITHOUT the embedder the lexical fallback cannot make the conceptual
    # jump: no chunk mentions "login", so the auth code is not retrieved.
    ci2 = CodeIntelligence(root=str(sandbox))
    ci2.refresh()
    lexical = ci2.search_semantically("login", top_k=5)
    assert not any(h.chunk.file.endswith("auth/service.py") for h in lexical)
    ci.close()
    ci2.close()


# ---------------------------------------------------------------------------
# TEST 7: repository exploration — targeted context, no repo dump
# ---------------------------------------------------------------------------


def test_7_repository_exploration_targeted(sandbox):
    build_rich_repo(sandbox)
    task = _make_task(
        "Explain how authentication works in this repository",
        TaskType.CODE_REVIEW,
        [
            _step(
                1,
                "Trace the authentication flow: route_login -> UserService.authenticate -> UserRepository.token_for",
                ["flow understood"],
            )
        ],
        ["explained"],
        ["explained"],
        WorkspaceKind.EXISTING_PROJECT,
        str(sandbox),
    )
    block = task.code_context.executor.intelligence_guidance(task)
    assert "CODE INTELLIGENCE" in block
    # The targeted context covers the chain (endpoint -> service -> repo)
    # without dumping the repository.
    assert "src/app/main.py" in block or "route_login" in block
    assert "UserService" in block
    assert "authenticate" in block or "sign_in" in block
    assert "UserRepository" in block
    # Bounded: docs, billing internals, utils, nested modules are NOT dumped.
    assert "parse_csv" not in block
    assert "charge" not in block
    assert "architecture.md" not in block
    assert len(block) <= 6000
    task.code_context.intelligence.close()


# ---------------------------------------------------------------------------
# TEST 8: bug fix guided by code intelligence
# ---------------------------------------------------------------------------


def test_8_bug_fix_with_intelligence(sandbox):
    build_rich_repo(sandbox)
    # Introduce a bug: authenticate() always returns True.
    _write(sandbox, "src/auth/service.py", (
        "from .repository import UserRepository\n"
        "\n"
        "class UserService:\n"
        "    def __init__(self):\n"
        "        self.repo = UserRepository()\n"
        "    def authenticate(self, token):\n"
        "        return True\n"  # BUG
        "    def sign_in(self, user):\n"
        "        return self.repo.token_for(user)\n"
    ))
    engine = FakeEngine(
        [
            _tool_call("find_definition", name="authenticate", path="."),
            _tool_call("read_file", file_path="src/auth/service.py"),
            _pytest_call("tests/test_auth.py"),
            _tool_call(
                "replace_in_file",
                file_path="src/auth/service.py",
                old="return True",
                new="return token == 'tok-1'",
            ),
            _pytest_call("tests/test_auth.py"),
            "The authentication bug is fixed.",
            _all_satisfied(["bug fixed", "tests pass"], ["tests pass"]),
        ]
    )
    task = _make_task(
        "Fix the authentication bug",
        TaskType.DEBUGGING,
        [_step(1, "Fix the authentication bug", ["bug fixed", "tests pass"])],
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
    assert "token == 'tok-1'" in (sandbox / "src/auth/service.py").read_text(encoding="utf-8")
    tools = [e.tool_name for e in task.execution_history]
    assert "find_definition" in tools
    # The failing test was actually observed first.
    assert any(e.success is False for e in task.execution_history)
    task.code_context.intelligence.close()


# ---------------------------------------------------------------------------
# TEST 9: refactor — rename UserService (auth) to AccountService
# ---------------------------------------------------------------------------


def test_9_refactor_rename_with_references(sandbox):
    build_rich_repo(sandbox)
    engine = FakeEngine(
        [
            _tool_call("find_definition", name="UserService", path="."),
            _tool_call("find_references", name="UserService", path="."),
            _tool_call("read_file", file_path="src/auth/service.py"),
            _tool_call(
                "replace_in_file",
                file_path="src/auth/service.py",
                old="class UserService",
                new="class AccountService",
            ),
            _tool_call(
                "replace_in_file",
                file_path="src/app/main.py",
                old="from auth.service import UserService",
                new="from auth.service import AccountService",
            ),
            _tool_call(
                "replace_in_file",
                file_path="src/app/main.py",
                old="svc = UserService()",
                new="svc = AccountService()",
            ),
            _tool_call("replace_in_file", file_path="tests/test_auth.py",
                       old="from auth.service import UserService",
                       new="from auth.service import AccountService"),
            _tool_call("replace_in_file", file_path="tests/test_auth.py",
                       old="UserService()",
                       new="AccountService()"),
            _pytest_call("tests/test_auth.py"),
            "Renamed to AccountService.",
            _all_satisfied(["renamed", "tests pass"], ["refactor done"]),
        ]
    )
    task = _make_task(
        "Rename the auth UserService to AccountService",
        TaskType.SOFTWARE_ENGINEERING,
        [_step(1, "Rename UserService to AccountService", ["renamed", "tests pass"])],
        ["refactor done"],
        ["refactor done"],
        WorkspaceKind.EXISTING_PROJECT,
        str(sandbox),
    )
    agent = ReActAgent(engine, max_iterations=10)
    msg = _run(agent.run(task.goal, [], task=task))
    for _ in range(10):
        if msg.pending_action is None:
            break
        result = _run(execute_pending_action(msg.pending_action))
        msg = _run(continue_task_after_confirmation(agent, task, result, []))

    assert msg.pending_action is None
    assert task.is_complete() is True
    assert "class AccountService" in (sandbox / "src/auth/service.py").read_text(encoding="utf-8")
    main = (sandbox / "src/app/main.py").read_text(encoding="utf-8")
    assert "AccountService" in main and "UserService" not in main
    tools = [e.tool_name for e in task.execution_history]
    assert tools.count("find_references") >= 1
    task.code_context.intelligence.close()


# ---------------------------------------------------------------------------
# TEST 10: same symbol name in two modules — context-preserving selection
# ---------------------------------------------------------------------------


def test_10_duplicate_symbol_surfaced_with_context(sandbox):
    build_rich_repo(sandbox)
    ci = CodeIntelligence(root=str(sandbox))
    ci.refresh()

    # Both UserService definitions are returned WITH their file locations —
    # the ambiguity is surfaced, not hidden, so the model can disambiguate.
    symbols = ci.find_symbol("UserService")
    files = {s.location.file for s in symbols if s.kind.value == "class"}
    assert files == {"src/auth/service.py", "src/billing/service.py"}
    # The importing files are surfaced as import-kind symbols too.
    assert len(symbols) >= 5  # 2 class defs + 3 import sites

    # A loop given the auth-specific request uses the surfaced locations and
    # inspects the RIGHT file (auth), never billing.
    engine = FakeEngine(
        [
            _tool_call("find_definition", name="UserService", path="."),
            _tool_call("read_file", file_path="src/auth/service.py"),
            "The authentication UserService is in src/auth/service.py — it handles tokens.",
            _all_satisfied(["located"], ["done"]),
        ]
    )
    task = _make_task(
        "Find the authentication UserService",
        TaskType.CODE_REVIEW,
        [_step(1, "Locate the authentication UserService", ["located"])],
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
    files_read = [f for f in task.code_context.executor.exploration.files_read]
    assert any("auth/service.py" in f for f in files_read)
    assert not any("billing/service.py" in f for f in files_read)
    ci.close()
    task.code_context.intelligence.close()


# ---------------------------------------------------------------------------
# TEST 11: large repository — bounded work
# ---------------------------------------------------------------------------


def test_11_large_repository_bounded(sandbox):
    build_rich_repo(sandbox)
    # 150 irrelevant generated modules.
    for i in range(150):
        _write(sandbox, f"src/gen/mod{i:03}.py", (
            f"def generated_func_{i}():\n    return {i}\n\n"
            f"class GeneratedClass{i}:\n    def method(self):\n        return {i}\n"
        ))

    ci = CodeIntelligence(root=str(sandbox))
    t0 = time.monotonic()
    summary = ci.refresh()
    index_seconds = time.monotonic() - t0
    assert summary.files >= 150 + 15  # all source files indexed

    # A targeted symbol lookup does NOT scan the repository.
    t0 = time.monotonic()
    defs = ci.find_definition("authenticate")
    lookup_ms = (time.monotonic() - t0) * 1000
    assert defs and defs[0].location.file == "src/auth/service.py"

    # The injected context block for an auth task does not mention gen modules.
    task = _make_task(
        "Fix the authentication test",
        TaskType.DEBUGGING,
        [_step(1, "Fix the authentication test", ["fixed"])],
        ["tests pass"],  # plan criteria MUST match the verifier payload
        ["tests pass"],
        WorkspaceKind.EXISTING_PROJECT,
        str(sandbox),
    )
    block = task.code_context.executor.intelligence_guidance(task)
    assert "mod0" not in block and "generated_func" not in block
    assert len(block) <= 6000

    # The bug-fix loop inspects only auth-relevant files, not the 150 modules.
    engine = FakeEngine(
        [
            _tool_call("find_definition", name="authenticate", path="."),
            _tool_call("read_file", file_path="src/auth/service.py"),
            _pytest_call("tests/test_auth.py"),
            _tool_call(
                "replace_in_file",
                file_path="src/auth/service.py",
                old="return token == 'tok-1'",
                new="return token == 'tok-1' or token == 'tok-2'",
            ),
            _pytest_call("tests/test_auth.py"),
            "Fixed.",
            _all_satisfied(["fixed", "tests pass"], ["tests pass"]),
        ]
    )
    agent = ReActAgent(engine, max_iterations=6)
    msg = _run(agent.run(task.goal, [], task=task))
    for _ in range(6):
        if msg.pending_action is None:
            break
        result = _run(execute_pending_action(msg.pending_action))
        msg = _run(continue_task_after_confirmation(agent, task, result, []))

    assert task.is_complete() is True
    inspected = task.code_context.executor.exploration
    assert len(inspected.files_read) <= 5  # NOT the 150 generated files
    assert not any("gen" in f for f in inspected.files_read)
    print(f"\n[perf] 150-file repo: index {index_seconds:.2f}s, "
          f"symbol lookup {lookup_ms:.1f}ms, files read {len(inspected.files_read)}")
    ci.close()
    task.code_context.intelligence.close()


# ---------------------------------------------------------------------------
# TEST 12/13: LSP available (fake client) / unavailable (NoLSPServers)
# ---------------------------------------------------------------------------


class _FakeLSPClient:
    server_id = "fake"
    capabilities = LSPCapabilities(
        operations=frozenset({LSPOperation.DEFINITION, LSPOperation.REFERENCES, LSPOperation.HOVER})
    )

    def __init__(self):
        self.initialized = False
        self.shutdown_called = False

    def initialize(self, root: str) -> bool:
        self.initialized = True
        return True

    def shutdown(self) -> None:
        self.shutdown_called = True

    def definition(self, uri, line, character):
        return [LSPLocation(uri=uri, line=line, character=character)]

    def references(self, uri, line, character):
        return [LSPLocation(uri=uri, line=line + 10, character=0)]

    def hover(self, uri, line, character):
        return "def authenticate(token) -> bool"

    def document_symbols(self, uri):
        return None

    def workspace_symbols(self, query):
        return None

    def implementation(self, uri, line, character):
        return None

    def call_hierarchy(self, uri, line, character):
        return None


class _FakeManager:
    def __init__(self):
        self.client = _FakeLSPClient()

    def detect(self):
        return ["fake"]

    def start(self, server_id, root):
        return self.client

    def stop(self, client):
        client.shutdown_called = True


def test_12_lsp_available_lifecycle():
    manager = _FakeManager()
    facade = LSPFacade(manager=manager)
    assert facade.start("/tmp/repo") is True
    assert facade.available is True
    assert manager.client.initialized is True

    locs = facade.definition("file:///auth.py", 3, 0)
    assert locs is not None and locs[0].line == 3
    refs = facade.references("file:///auth.py", 3, 0)
    assert refs is not None and refs[0].line == 13
    assert facade.hover("file:///auth.py", 3, 0) == "def authenticate(token) -> bool"

    facade.stop()
    assert manager.client.shutdown_called
    assert facade.available is False


def test_13_lsp_unavailable_falls_back(sandbox):
    build_rich_repo(sandbox)
    facade = LSPFacade(manager=NoLSPServers())
    assert facade.available is False
    assert facade.definition("file:///auth.py", 0, 0) is None
    assert facade.references("file:///auth.py", 0, 0) is None
    assert facade.workspace_symbols("q") is None
    facade.stop()  # no-op, never raises

    # The index path still answers without any LSP (fallback works).
    ci = CodeIntelligence(root=str(sandbox))
    ci.refresh()
    assert ci.lsp_available() is False
    assert ci.find_definition("authenticate")
    ci.close()


# ---------------------------------------------------------------------------
# TEST 14: stale index — new content discoverable after edit
# ---------------------------------------------------------------------------


def test_14_stale_index_refreshed(sandbox):
    build_rich_repo(sandbox)
    ci = CodeIntelligence(root=str(sandbox))
    ci.refresh()
    assert ci.find_definition("refresh_tokens") == []

    # Simulate an edit: add a new symbol to an indexed file.
    _write(sandbox, "src/auth/nested/tokens.py", (
        "class TokenManager:\n"
        "    def issue(self, user):\n"
        "        return 'tok-' + user\n"
        "    def refresh_tokens(self):\n"
        "        return ['tok-1', 'tok-2']\n"
    ))
    ci.refresh()  # incremental — only the changed file is re-parsed
    defs = ci.find_definition("refresh_tokens")
    assert defs and defs[0].location.file == "src/auth/nested/tokens.py"
    ci.close()


def test_14b_incremental_only_changed_file(sandbox):
    build_rich_repo(sandbox)
    ci = CodeIntelligence(root=str(sandbox))
    first = ci.refresh()
    _write(sandbox, "src/auth/service.py", (
        "from .repository import UserRepository\n"
        "class UserService:\n"
        "    def authenticate(self, token):\n"
        "        return token == 'tok-1'\n"
        "    def sign_in(self, user):\n"
        "        return self.repo.token_for(user)\n"
        "    def logout(self, user):\n"
        "        return True\n"
    ))
    second = ci.refresh()
    assert second.parsed == 1
    assert second.unchanged == first.files - 1
    ci.close()


# ---------------------------------------------------------------------------
# TEST 15: malformed code degrades gracefully
# ---------------------------------------------------------------------------


def test_15_malformed_code_degrades(sandbox):
    build_rich_repo(sandbox)
    _write(sandbox, "src/auth/broken.py", "def broken(:\n    pass\n")
    ci = CodeIntelligence(root=str(sandbox))
    summary = ci.refresh()  # must NOT raise on the malformed file
    assert summary.files >= 1
    # The healthy symbols are still indexed; lexical search still works.
    assert ci.find_definition("authenticate")
    lexical = ci.search("broken", max_results=5)
    assert "broken.py" in lexical
    ci.close()


def test_15b_malformed_code_loop_never_crashes(sandbox):
    build_rich_repo(sandbox)
    _write(sandbox, "src/auth/broken.py", "def broken(:\n    pass\n")
    engine = FakeEngine(
        [
            _tool_call("find_definition", name="authenticate", path="."),
            "Found it despite the malformed file.",
            _all_satisfied(["located"], ["done"]),
        ]
    )
    task = _make_task(
        "Locate authenticate",
        TaskType.CODE_REVIEW,
        [_step(1, "Locate authenticate", ["located"])],
        ["done"],
        ["done"],
        WorkspaceKind.EXISTING_PROJECT,
        str(sandbox),
    )
    agent = ReActAgent(engine, max_iterations=4)
    msg = _run(agent.run(task.goal, [], task=task))
    assert msg.pending_action is None
    assert task.is_complete() is True
    task.code_context.intelligence.close()


# ---------------------------------------------------------------------------
# TEST 16: unsupported language — lexical still works
# ---------------------------------------------------------------------------


def test_16_unsupported_language_fallback(sandbox):
    build_rich_repo(sandbox)
    # C is recognized by extension but has NO parser in the registry (ruby,
    # php, swift, shell etc. DO have regex parsers — C is genuinely missing).
    _write(
        sandbox,
        "src/utils/old_script.c",
        "int legacy_run(void) {\n    return 0;\n}\n",
    )
    ci = CodeIntelligence(root=str(sandbox))
    ci.refresh()
    # No parser for C -> no symbols indexed.
    assert ci.find_symbol("legacy_run") == []
    # Filesystem/lexical search still finds the content.
    lexical = ci.search("legacy", max_results=5)
    assert "old_script.c" in lexical
    # language_for_path reports the language but the registry has no parser.
    from ultron.core.coding.intelligence.parsers import get_parser, language_for_path

    assert language_for_path("old_script.c") == "c"
    assert get_parser("c") is None  # advanced capability clearly unsupported
    ci.close()


# ---------------------------------------------------------------------------
# TEST 17: code review — read-only
# ---------------------------------------------------------------------------


def test_17_code_review_read_only(sandbox):
    build_rich_repo(sandbox)
    engine = FakeEngine(
        [
            _tool_call("report_file", file_path="src/auth/service.py", path="."),
            _tool_call("find_references", name="authenticate", path="."),
            ("authenticate() compares the token to a hardcoded value — "
             "entry points: src/app/main.py, tests/test_auth.py."),
            _all_satisfied(["findings reported"], ["review done"]),
        ]
    )
    task = _make_task(
        "Review authentication for security issues",
        TaskType.CODE_REVIEW,
        [_step(1, "Review the authentication module", ["findings reported"])],
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

    assert task.is_complete() is True
    # Read-only: nothing modified.
    assert len(task.code_context.tracker.modifications) == 0
    tools = [e.tool_name for e in task.execution_history]
    assert "report_file" in tools and "find_references" in tools
    assert not any(t in tools for t in ("replace_in_file", "write_file", "create_file"))
    task.code_context.intelligence.close()


# ---------------------------------------------------------------------------
# TEST 18: feature implementation — add caching to UserService
# ---------------------------------------------------------------------------


def test_18_feature_implementation(sandbox):
    build_rich_repo(sandbox)
    engine = FakeEngine(
        [
            _tool_call("find_symbol", name="UserService", path="."),
            _tool_call("find_references", name="UserService", path="."),
            _tool_call("read_file", file_path="src/auth/service.py"),
            _tool_call(
                "replace_in_file",
                file_path="src/auth/service.py",
                old="    def __init__(self):\n        self.repo = UserRepository()",
                new="    def __init__(self):\n        self.repo = UserRepository()\n        self._cache = {}",
            ),
            _tool_call(
                "replace_in_file",
                file_path="src/auth/service.py",
                old="    def authenticate(self, token):\n        return token == 'tok-1'",
                new="    def authenticate(self, token):\n        if token in self._cache:\n            return self._cache[token]\n        result = token == 'tok-1'\n        self._cache[token] = result\n        return result",
            ),
            _pytest_call("tests/test_auth.py"),
            "Caching added to UserService.",
            _all_satisfied(["cache added", "tests pass"], ["feature done"]),
        ]
    )
    task = _make_task(
        "Add caching to UserService",
        TaskType.SOFTWARE_ENGINEERING,
        [_step(1, "Add caching to UserService", ["cache added", "tests pass"])],
        ["feature done"],
        ["feature done"],
        WorkspaceKind.EXISTING_PROJECT,
        str(sandbox),
    )
    agent = ReActAgent(engine, max_iterations=8)
    msg = _run(agent.run(task.goal, [], task=task))
    for _ in range(8):
        if msg.pending_action is None:
            break
        result = _run(execute_pending_action(msg.pending_action))
        msg = _run(continue_task_after_confirmation(agent, task, result, []))

    assert msg.pending_action is None
    assert task.is_complete() is True
    text = (sandbox / "src/auth/service.py").read_text(encoding="utf-8")
    assert "_cache" in text
    tools = [e.tool_name for e in task.execution_history]
    assert "find_symbol" in tools and "find_references" in tools
    task.code_context.intelligence.close()


# ---------------------------------------------------------------------------
# TEST 19: false context — semantic hit is a hint, not truth
# ---------------------------------------------------------------------------


def test_19_false_context_validated_against_source(sandbox):
    build_rich_repo(sandbox)
    # The concept embedder maps "login" -> sign_in; but a WRONG hit can still
    # surface (e.g. the report/parse code scores high on a loose query). The
    # agent must verify against the actual file before acting.
    _write(sandbox, "src/reports/reportgen.py", (
        "class ReportGenerator:\n"
        "    def render(self):\n"
        "        return 'report'\n"
    ))
    engine = FakeEngine(
        [
            # Model trusts a semantic hit and tries to edit the WRONG file...
            _tool_call("semantic_search", query="authentication", path="."),
            _tool_call("read_file", file_path="src/reports/reportgen.py"),
            # ...discovers it is unrelated, then locates the real implementation.
            _tool_call("find_definition", name="authenticate", path="."),
            _tool_call("read_file", file_path="src/auth/service.py"),
            "The semantic hit was unrelated; the real authentication code is in src/auth/service.py.",
            _all_satisfied(["verified", "located"], ["done"]),
        ]
    )
    task = _make_task(
        "Find the code responsible for authentication",
        TaskType.CODE_REVIEW,
        [_step(1, "Locate the authentication implementation", ["verified", "located"])],
        ["done"],
        ["done"],
        WorkspaceKind.EXISTING_PROJECT,
        str(sandbox),
    )
    agent = ReActAgent(engine, max_iterations=6)
    msg = _run(agent.run(task.goal, [], task=task))
    for _ in range(5):
        if msg.pending_action is None:
            break
        result = _run(execute_pending_action(msg.pending_action))
        msg = _run(continue_task_after_confirmation(agent, task, result, []))

    assert task.is_complete() is True
    # The final answer came AFTER reading the authoritative source, and the
    # wrong semantic target was never edited (no modifications at all).
    assert len(task.code_context.tracker.modifications) == 0
    files_read = task.code_context.executor.exploration.files_read
    assert any("auth/service.py" in f for f in files_read)
    tools = [e.tool_name for e in task.execution_history]
    assert "semantic_search" in tools and "find_definition" in tools
    task.code_context.intelligence.close()


# ---------------------------------------------------------------------------
# TEST 20: security — workspace boundaries respected
# ---------------------------------------------------------------------------


def test_20_intelligence_respects_boundaries(sandbox):
    build_rich_repo(sandbox)
    ci = CodeIntelligence(root=str(sandbox))
    ci.refresh()

    # The registered tools refuse paths outside the allowed workspace.
    from ultron.core.coding.intelligence.tools import (
        code_search,
        find_definition,
    )

    assert find_definition("UserService", "/etc").startswith("Error: access denied")
    assert code_search("x", "/etc").startswith("Error: access denied")
    assert find_definition("UserService", "../outside").startswith("Error: access denied")

    # Guardrails hard-block path escapes for every intelligence tool.
    from ultron.security.guardrails import GuardrailsEngine

    engine = GuardrailsEngine()
    for tool in (
        "code_search", "find_symbol", "find_definition", "find_references",
        "get_imports", "get_dependents", "semantic_search", "code_index_status",
        "report_file", "report_symbol",
    ):
        assert engine.evaluate(action_type=tool, target="/etc/passwd").blocked, tool

    # The bridge refuses to enable outside the allowed base dir.
    bridge = __import__(
        "ultron.core.coding.intelligence_bridge", fromlist=["CodeIntelligenceBridge"]
    ).CodeIntelligenceBridge()
    assert bridge.enable("/etc") is False
    ci.close()


# ---------------------------------------------------------------------------
# Performance instrumentation (informational, generous bounds)
# ---------------------------------------------------------------------------


def test_perf_measurements(sandbox):
    build_rich_repo(sandbox)
    for i in range(100):
        _write(sandbox, f"src/gen/mod{i:03}.py", f"def fn{i}():\n    return {i}\n")

    ci = CodeIntelligence(root=str(sandbox))
    t0 = time.monotonic()
    ci.refresh()
    index_ms = (time.monotonic() - t0) * 1000

    t0 = time.monotonic()
    ci.refresh()  # no changes
    incremental_ms = (time.monotonic() - t0) * 1000

    t0 = time.monotonic()
    ci.search("authenticate", max_results=20)
    lexical_ms = (time.monotonic() - t0) * 1000

    t0 = time.monotonic()
    ci.find_definition("authenticate")
    symbol_ms = (time.monotonic() - t0) * 1000

    t0 = time.monotonic()
    ci.search_semantically("authentication", top_k=5)
    semantic_ms = (time.monotonic() - t0) * 1000

    block = ci.report_symbol("UserService")
    context_bytes = len(block.encode("utf-8"))

    print(
        f"\n[perf] ~115-file repo: index {index_ms:.0f}ms, incremental {incremental_ms:.1f}ms, "
        f"lexical {lexical_ms:.1f}ms, symbol {symbol_ms:.1f}ms, semantic {semantic_ms:.1f}ms, "
        f"report_symbol {context_bytes} bytes"
    )
    # Generous bounds to avoid flakiness on slow CI.
    assert index_ms < 5000
    assert incremental_ms < 200
    assert lexical_ms < 1000
    assert symbol_ms < 200
    assert context_bytes < 5000
    ci.close()
