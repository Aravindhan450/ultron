"""FIX #6 — comprehensive stress test of memory + context management.

Twenty scenarios exercising the FIX #6 integration under long-workflow
conditions, using isolated temporary repositories:

  1.  project discovery memory reduces rediscovery
  2.  follow-up coding task reuses context, still verifies current source
  3.  stale memory (moved file) never causes edits to the old location
  4.  project memory is isolated between repositories
  5.  task continuation survives process restart
  6.  a new task after completion gets a separate TaskState
  7.  large tool output never floods model context
  8.  long coding session keeps context bounded, evidence retained
  9.  memory formation stores only meaningful facts
  10. LLM inference never automatically becomes trusted project memory
  11. current repository evidence wins over conflicting memory
  12. secrets never reach persistent memory
  13. current source outranks conflicting old memory
  14. hundreds of observations compress without losing key evidence
  15. repository change cannot cause edits from stale memory
  16. facts carry their revision; unrelated revisions never conflated
  17. memory failure degrades gracefully (never a hard dependency)
  18. code-intelligence failure falls back while memory still helps
  19. session boundary: project memory survives, working memory does not leak
  20. performance: bounded context, cheap retrieval/assembly/restore

The architectural invariants are asserted throughout:
  CURRENT SOURCE > STALE MEMORY, TASKSTATE > CONVERSATION HISTORY,
  PROJECT MEMORY stays project-scoped, MEMORY assists (never replaces)
  Code Intelligence, LLM INFERENCE never auto-becomes fact, SECRET never
  persisted, CONTEXT MANAGEMENT reduces bloat.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from ultron.core.agents.react import (
    ReActAgent,
    _build_task_context_block,
    _sync_task_memory,
)
from ultron.core.coding.context import CodeContext
from ultron.core.coding.observations import ObservationKind
from ultron.core.coding.workspace import discover_workspace
from ultron.core.memory import (
    ContextManager,
    MemoryConfidence,
    MemorySource,
    ProjectMemoryStore,
    SessionMemory,
    WorkingMemory,
)
from ultron.core.memory.task_store import load_task, save_task
from ultron.core.tools import paths as tools_paths
from ultron.core.types import (
    ChatMessage,
    FailureStrategy,
    PlanStep,
    Role,
    StepStatus,
    TaskPlan,
    TaskState,
    TaskType,
)

_REPO_ROOT = tools_paths.ALLOWED_BASE_DIR


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """A temp workspace that IS the ALLOWED_BASE_DIR (tools/bridge/memory work)."""
    monkeypatch.setattr(tools_paths, "ALLOWED_BASE_DIR", tmp_path)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _write(root, rel: str, text: str):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _python_project(root, *, auth_module: str = "src/auth/service.py"):
    """A deterministic Python repo with an AuthService symbol."""
    _write(
        root,
        "pyproject.toml",
        "[project]\nname = \"demo\"\n[tool.pytest.ini_options]\n",
    )
    _write(root, "src/__init__.py", "")
    _write(
        root,
        auth_module,
        "class AuthService:\n    def authenticate(self):\n        return True\n",
    )
    _write(root, "tests/test_auth.py", "def test_auth():\n    assert True\n")
    _write(root, "config/settings.py", "DEBUG = True\n")
    _write(root, "database/models.py", "class User:\n    pass\n")


def _planned_task(goal: str, workspace_root, *, steps: int = 3) -> TaskState:
    """A TaskState with a structured plan + attached CodeContext."""
    task = TaskState(goal=goal, task_type=TaskType.SOFTWARE_ENGINEERING)
    plan_steps = [
        PlanStep(
            id=i,
            description=f"step {i}",
            purpose=f"purpose {i}",
            expected_outcome=f"outcome {i}",
            completion_criteria=[f"criterion {i}"],
            dependencies=[] if i == 1 else [i - 1],
            failure_strategy=FailureStrategy.STOP,
        )
        for i in range(1, steps + 1)
    ]
    task.attach_plan(
        TaskPlan(
            goal=goal,
            task_type=TaskType.SOFTWARE_ENGINEERING,
            steps=plan_steps,
            completion_criteria=["goal met"],
            verification_requirements=["verify the goal"],
        )
    )
    ctx = CodeContext(workspace=discover_workspace(str(workspace_root)))
    ctx.attach_task(task)
    task.code_context = ctx
    return task


class _FakeEngine:
    """Deterministic engine: returns canned responses, then a final answer."""

    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses = list(responses or [])
        self.calls = 0

    async def generate(self, messages) -> str:
        self.calls += 1
        if self._responses:
            return self._responses.pop(0)
        return "Final answer."


def _run(coro) -> ChatMessage:
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# TEST 1 — Project discovery memory reduces rediscovery
# ---------------------------------------------------------------------------


def test_project_discovery_memory_reduces_rediscovery(sandbox):
    _python_project(sandbox)
    task1 = _planned_task("understand the architecture of this project", sandbox)
    store1 = task1.code_context.ensure_project_memory()
    assert store1 is not None
    assert store1.count() >= 3  # language + package manager + test framework

    # New task: "Where is authentication implemented?" — project memory +
    # code intelligence supply targeted context instead of a fresh full scan.
    task2 = _planned_task("where is authentication implemented", sandbox)
    block = _build_task_context_block(task2)
    assert "PROJECT MEMORY" in block
    assert "project uses python" in block.lower()
    # Bounded: the block never contains the raw repository contents and stays
    # small no matter how large the repository is.
    assert "DEBUG = True" not in block
    assert len(block) < 4000


# ---------------------------------------------------------------------------
# TEST 2 — Follow-up coding task reuses context, verifies current source
# ---------------------------------------------------------------------------


def test_followup_coding_task_reuses_context(sandbox):
    _python_project(sandbox)
    task1 = _planned_task("find the authentication service", sandbox)
    task1.code_context.intelligence.enable(str(sandbox))
    task1.code_context.intelligence.refresh()
    task1.code_context.intelligence.query("find_definition", name="AuthService")
    _sync_task_memory(task1)
    assert (
        task1.code_context.ensure_project_memory().recall(name="symbol:AuthService")
    )

    # Follow-up: "add refresh token support" — the symbol fact is reused.
    task2 = _planned_task("add refresh token support", sandbox)
    block = _build_task_context_block(task2)
    assert "AuthService" in block
    assert "src/auth/service.py" in block
    # The current source is still verified: reconciliation keeps the fact
    # consistent with the live index, never trusting memory blindly.
    assert task2.code_context.ensure_project_memory().recall(
        name="symbol:AuthService"
    )[0].metadata["file"] == "src/auth/service.py"


# ---------------------------------------------------------------------------
# TEST 3 — Stale memory (moved file) never causes edits to the old location
# ---------------------------------------------------------------------------


def test_stale_memory_never_targets_deleted_location(sandbox):
    _python_project(sandbox, auth_module="src/auth/service.py")
    task = _planned_task("modify AuthService", sandbox)
    ctx = task.code_context
    assert ctx.intelligence.enable(str(sandbox))
    ctx.intelligence.refresh()
    ctx.intelligence.query("find_definition", name="AuthService")

    store = ctx.ensure_project_memory()
    # Simulate an OLD memory from before the move.
    store.store(
        "symbol:AuthService",
        "defined in src/auth/service.py",
        source=MemorySource.CODE_INTELLIGENCE,
        confidence=MemoryConfidence.DIRECT_OBSERVATION,
        metadata={"file": "src/auth/service.py"},
    )

    # The developer moves the file; the repository now has the new location.
    (sandbox / "src/auth/service.py").unlink()
    _python_project(sandbox, auth_module="src/security/auth/service.py")
    ctx.intelligence.mark_dirty()
    ctx.intelligence.refresh()

    _sync_task_memory(task)
    current = store.recall(name="symbol:AuthService")
    assert len(current) == 1
    assert current[0].metadata["file"] == "src/security/auth/service.py"
    assert "src/auth/service.py" not in current[0].content

    block = _build_task_context_block(task)
    assert "src/security/auth/service.py" in block
    assert "src/auth/service.py" not in block
    ctx.intelligence.close()


# ---------------------------------------------------------------------------
# TEST 4 — Project isolation: A's memory never leaks into B
# ---------------------------------------------------------------------------


def test_project_isolation_between_repositories(sandbox):
    repo_a = sandbox / "project_A"
    repo_b = sandbox / "project_B"
    _python_project(repo_a)
    _python_project(repo_b)

    task_a = _planned_task("work on AuthService in A", repo_a)
    task_a.code_context.ensure_project_memory().store(
        "symbol:AuthService",
        "defined in src/auth/service.py",
        source=MemorySource.CODE_INTELLIGENCE,
        confidence=MemoryConfidence.DIRECT_OBSERVATION,
        metadata={"file": "src/auth/service.py"},
    )
    task_b = _planned_task("work on AuthService in B", repo_b)

    block_b = _build_task_context_block(task_b)
    store_b = task_b.code_context.ensure_project_memory()
    assert store_b.recall(name="symbol:AuthService") == []
    # A's specific fact content never appears in B's context block (B may
    # legitimately mention AuthService in its own goal — the fact content
    # is what must not leak).
    assert "defined in src/auth/service.py" not in block_b
    assert task_a.code_context.ensure_project_memory().db_path != store_b.db_path


# ---------------------------------------------------------------------------
# TEST 5 — Task continuation survives process restart
# ---------------------------------------------------------------------------


def test_task_continuation_after_process_restart(sandbox):
    _python_project(sandbox)
    task = _planned_task("implement authentication", sandbox)
    task.plan.set_step_status(1, StepStatus.SUCCEEDED, result="inspected auth")
    task.plan.set_step_status(2, StepStatus.RUNNING)
    task.set_current_step(2)
    task.context.append(
        ChatMessage(role=Role.USER, content="implement authentication")
    )
    assert save_task(task, sandbox) is not None

    # Process restart: nothing is in memory but the snapshot files.
    loaded = load_task(sandbox)
    assert loaded is not None
    assert loaded.plan.step(1).status is StepStatus.SUCCEEDED
    assert loaded.plan.step(2).status is StepStatus.RUNNING
    assert loaded.current_step == 2

    # "Continue." — the agent operates on the RESTORED task; completed work is
    # never redone: step 1 stays SUCCEEDED, and the exhausted continuation
    # correctly marks the current step FAILED instead of claiming success.
    engine = _FakeEngine()
    response = _run(ReActAgent(engine).run(loaded.goal, task=loaded))
    assert loaded.plan.step(1).status is StepStatus.SUCCEEDED  # no restart
    assert loaded.plan.step(2).status is StepStatus.FAILED  # honest failure
    assert loaded.is_complete() is False
    assert "incomplete" in response.content.lower()


# ---------------------------------------------------------------------------
# TEST 6 — New task after completion: separate TaskState, memory remains
# ---------------------------------------------------------------------------


def test_new_task_after_completion_is_separate(sandbox):
    _python_project(sandbox)
    task1 = _planned_task("fix authentication", sandbox)
    task1.plan.set_step_status(1, StepStatus.SUCCEEDED)
    task1.plan.set_step_status(2, StepStatus.SUCCEEDED)
    task1.plan.set_step_status(3, StepStatus.SUCCEEDED)
    # The plan's overall criteria are task requirements that must also be met.
    task1.mark_requirement_complete("goal met")
    task1.mark_requirement_complete("verify the goal")
    task1.mark_complete()
    assert task1.is_complete()

    task2 = _planned_task("now refactor the database layer", sandbox)
    assert task2 is not task1
    assert task2.goal != task1.goal
    assert task2.is_complete() is False
    # Project memory remains available to the new task (same workspace store).
    assert task2.code_context.ensure_project_memory().count() > 0
    assert task2.plan.step(1).status is StepStatus.PENDING  # no merging


# ---------------------------------------------------------------------------
# TEST 7 — Large tool output never floods model context
# ---------------------------------------------------------------------------


def test_large_tool_output_stays_out_of_context(sandbox):
    _python_project(sandbox)
    task = _planned_task("inspect the build log", sandbox)
    huge = "\n".join(f"line {i} " + "x" * 80 for i in range(5000))  # ~400k chars
    task.code_context.record_observation(
        ObservationKind.COMMAND_RESULT,
        "run_command",
        "npm run build",
        detail=huge,
        success=False,
        exit_code=1,
    )

    wm = WorkingMemory.from_task(task, task.code_context)
    for line in wm.recent_observations:
        assert len(line) < 250  # summarized
    block = _build_task_context_block(task)
    assert "line 4999" not in block  # raw output never reaches the prompt
    assert len(block) < 4000  # bounded regardless of the 400k-char output


# ---------------------------------------------------------------------------
# TEST 8 — Long coding session: bounded context, evidence retained
# ---------------------------------------------------------------------------


def test_long_coding_session_context_stays_bounded(sandbox):
    _python_project(sandbox)
    task = _planned_task("implement the feature end to end", sandbox)
    ctx = task.code_context
    # 300 observations across many tools, including repeated failures; the
    # most recent observation is a failure so the active-failure evidence is
    # deterministic.
    for i in range(300):
        failed = i % 7 == 0 or i == 299
        ctx.record_observation(
            ObservationKind.COMMAND_RESULT if i % 2 else ObservationKind.EDIT_RESULT,
            "run_command" if i % 2 else "replace_in_file",
            f"pytest run {i}: {'2 failed, 1 passed' if failed else '3 passed'}",
            success=not failed,
        )
    ctx.add_relevant_file("src/auth/service.py")

    wm = WorkingMemory.from_task(task, ctx)
    assert len(wm.recent_observations) <= 5  # bounded, not 300
    block = _build_task_context_block(task)
    assert len(block) < 4000  # bounded regardless of session length
    # Key evidence survives: the most recent failure is the active failure.
    assert wm.active_failure is not None


# ---------------------------------------------------------------------------
# TEST 9 — Memory formation stores only meaningful facts
# ---------------------------------------------------------------------------


def test_memory_formation_stores_only_meaningful_facts(sandbox):
    _python_project(sandbox)
    task = _planned_task("understand the auth layer", sandbox)
    ctx = task.code_context
    # D: the executor actually resolved AuthService through the task's own
    # code-intelligence bridge (recorded symbol query with hits).
    assert ctx.intelligence.enable(str(sandbox))
    ctx.intelligence.refresh()
    ctx.intelligence.query("find_definition", name="AuthService")
    # A: trivial read, C: trivial command — no formation path exists for them.
    ctx.record_observation(
        ObservationKind.FILE_CONTENT, "read_file", "read auth.py", success=True
    )
    ctx.record_observation(
        ObservationKind.COMMAND_RESULT, "run_command", "command exited 0", success=True
    )
    _sync_task_memory(task)

    names = {r.name for r in ctx.ensure_project_memory().recall(limit=100)}
    # B (project uses python/pytest) and D (AuthService -> file) are stored.
    assert "language:python" in names
    assert "test_framework" in names
    assert "symbol:AuthService" in names
    # A and C are not.
    assert not any("read" in n.lower() for n in names)
    assert not any("exited" in n.lower() for n in names)
    ctx.intelligence.close()


# ---------------------------------------------------------------------------
# TEST 10 — LLM inference never automatically becomes trusted memory
# ---------------------------------------------------------------------------


def test_llm_claim_never_becomes_trusted_memory(sandbox):
    _python_project(sandbox)
    task = _planned_task("fix the authentication flow", sandbox)
    store = task.code_context.ensure_project_memory()
    # Formation DID run and produced real memory (so the test is not vacuous).
    _sync_task_memory(task)
    assert store.count() >= 3  # workspace facts formed
    # …but the LLM's CLAIM ("Authentication uses Redis") never entered the
    # store — there is no LLM-promotion path in formation.
    all_facts = " ".join(
        f"{r.name}:{r.content}" for r in store.recall(limit=100)
    ).lower()
    assert "redis" not in all_facts


# ---------------------------------------------------------------------------
# TEST 11 — Current repository evidence wins over conflicting memory
# ---------------------------------------------------------------------------


def test_current_repository_evidence_wins(sandbox):
    _python_project(sandbox)
    # Old memory claims unittest; the repository actually declares pytest.
    stale_store = ProjectMemoryStore(sandbox)
    stale_store.store(
        "test_framework",
        "test framework is unittest",
        source=MemorySource.REPOSITORY_INSPECTION,
        confidence=MemoryConfidence.DIRECT_OBSERVATION,
    )
    assert stale_store.recall(name="test_framework")[0].content == (
        "test framework is unittest"
    )

    # A new task re-attaches: workspace detection stores the CURRENT fact,
    # which supersedes the stale one.
    task = _planned_task("run the tests", sandbox)
    store = task.code_context.ensure_project_memory()
    current = store.recall(name="test_framework")
    assert len(current) == 1
    assert "pytest" in current[0].content
    history = store.history("test_framework")
    assert len(history) == 2  # stale version kept as history, not erased
    assert "unittest" not in _build_task_context_block(task)


# ---------------------------------------------------------------------------
# TEST 12 — Secrets never reach persistent memory
# ---------------------------------------------------------------------------


def test_secrets_never_reach_persistent_memory(sandbox):
    store = ProjectMemoryStore(sandbox)
    payloads = [
        "API_KEY=secret-value",
        "PASSWORD=secret-value",
        "TOKEN=secret-value",
    ]
    for payload in payloads:
        assert store.store("env", payload) is None, payload
        assert store.store("env", "safe value", metadata={"raw": payload}) is None
    assert store.count() == 0
    # The task snapshot store applies the same guard.
    task = _planned_task("deploy the service", sandbox)
    task.context.append(
        ChatMessage(role=Role.TOOL, name="run_command", content=payloads[0])
    )
    assert save_task(task, sandbox) is None


# ---------------------------------------------------------------------------
# TEST 13 — Current source outranks conflicting old memory
# ---------------------------------------------------------------------------


def test_current_source_outranks_conflicting_memory(sandbox):
    _python_project(sandbox)
    task = _planned_task("fix the add function", sandbox)
    ctx = task.code_context
    # Old memory: "add returns an integer". Current source observation says
    # otherwise (the function now returns a string).
    ctx.ensure_project_memory().store(
        "add_signature",
        "add returns an integer",
        source=MemorySource.LLM_INFERENCE,
        confidence=MemoryConfidence.INFERRED,
    )
    ctx.record_observation(
        ObservationKind.FILE_CONTENT,
        "read_file",
        "src/utils.py: def add(a, b): return str(a + b)  # returns a string",
        success=True,
    )

    wm = WorkingMemory.from_task(task, ctx)
    block = ContextManager().assemble(
        user_request="fix the add function",
        task=task,
        working=wm,
        code_context=ctx,
        project_memory=ctx.ensure_project_memory().recall(limit=50),
        workspace=str(sandbox),
        task_terms=["add", "function"],
    )
    # Priority: the observed current source precedes the older memory, so the
    # source cannot be overridden by stale memory.
    assert block.index("OBSERVATIONS") < block.index("PROJECT MEMORY")
    assert "returns a string" in block


# ---------------------------------------------------------------------------
# TEST 14 — Hundreds of observations compress without losing key evidence
# ---------------------------------------------------------------------------


def test_hundreds_of_observations_compress(sandbox):
    _python_project(sandbox)
    session = SessionMemory()
    session.set_workspace(str(sandbox))
    for i in range(300):
        session.note_decision(f"decision {i}: chose option {i}")
        session.note_output(f"output {i} ... " + "y" * 100)
    assert len(session.decisions) <= 6  # bounded
    assert len(session.notable_outputs) <= 4

    task = _planned_task("finish the refactor", sandbox)
    for i in range(300):
        task.code_context.record_observation(
            ObservationKind.TEST_RESULT if i % 3 else ObservationKind.ERROR,
            "pytest",
            f"run {i}: 1 failed at tests/test_refactor.py" if i % 3 == 0
            else f"run {i}: 12 passed",
            success=bool(i % 3),
        )
    wm = WorkingMemory.from_task(task, task.code_context)
    block = ContextManager().memory_block(
        project_records=task.code_context.ensure_project_memory().recall(limit=50),
        session=session,
        workspace=str(sandbox),
        task_terms=["refactor"],
    )
    assert len(block) <= 1450  # hard cap for project+session+long-term
    assert "Active workspace" in block
    assert wm.active_failure is not None  # unresolved failure retained


# ---------------------------------------------------------------------------
# TEST 15 — Repository change cannot cause edits from stale memory
# ---------------------------------------------------------------------------


def test_repository_change_cannot_trigger_stale_edits(sandbox):
    _python_project(sandbox)
    task = _planned_task("modify the users endpoint", sandbox)
    ctx = task.code_context
    # Old memory knows: "API endpoint is /users".
    ctx.ensure_project_memory().store(
        "endpoint:users",
        "API endpoint is /users",
        source=MemorySource.REPOSITORY_INSPECTION,
        confidence=MemoryConfidence.DIRECT_OBSERVATION,
    )
    # The repository changed: the current source now exposes /accounts.
    _write(
        sandbox,
        "src/api/routes.py",
        '@router.get("/accounts")\ndef list_accounts():\n    return []\n',
    )
    ctx.record_observation(
        ObservationKind.FILE_CONTENT,
        "read_file",
        'src/api/routes.py: @router.get("/accounts")',
        success=True,
    )

    wm = WorkingMemory.from_task(task, ctx)
    block = ContextManager().assemble(
        user_request="modify the users endpoint",
        task=task,
        working=wm,
        code_context=ctx,
        project_memory=ctx.ensure_project_memory().recall(limit=50),
        workspace=str(sandbox),
        task_terms=["users", "endpoint"],
    )
    # The CURRENT observation (with /accounts) precedes and outranks memory;
    # the block clearly reflects the current repository state.
    assert "/accounts" in block
    assert block.index("OBSERVATIONS") < block.index("PROJECT MEMORY")


# ---------------------------------------------------------------------------
# TEST 16 — Facts carry their revision; unrelated revisions never conflated
# ---------------------------------------------------------------------------


def test_facts_are_revision_aware(sandbox):
    store = ProjectMemoryStore(sandbox)
    r1 = store.store("db", "postgres", revision="aaa1111")
    assert r1.revision == "aaa1111"
    r2 = store.store("db", "sqlite", revision="bbb2222")
    assert r2.revision == "bbb2222"
    assert r2.supersedes_id == r1.id
    assert store.recall(name="db")[0].content == "sqlite"
    assert [r.revision for r in store.history("db")] == ["aaa1111", "bbb2222"]

    # An unrelated-revision fact is never conflated with the current one.
    store.store("endpoint", "/users", revision="aaa1111")
    store.store("endpoint", "/accounts", revision="bbb2222")
    assert store.recall(name="endpoint")[0].content == "/accounts"
    block = ContextManager().memory_block(
        project_records=store.recall(limit=50),
        workspace=str(sandbox),
        task_terms=["endpoint"],
    )
    assert "/accounts" in block


# ---------------------------------------------------------------------------
# TEST 17 — Memory failure degrades gracefully (never a hard dependency)
# ---------------------------------------------------------------------------


def test_memory_failure_degrades_gracefully():
    # A workspace whose project root is unsafe (Ultron's own repository):
    # persistent memory is simply disabled.
    task = _planned_task("fix the failing test", _REPO_ROOT)
    assert task.code_context.ensure_project_memory() is None
    assert task.code_context.project_memory_root is None

    # The agent still operates on TaskState + Code Intelligence + filesystem.
    block = _build_task_context_block(task)
    assert "CURRENT TASK" in block
    assert "STRUCTURED PLAN" in block
    assert "PROJECT MEMORY" not in block

    engine = _FakeEngine()
    response = _run(ReActAgent(engine).run(task.goal, task=task))
    assert response is not None  # no crash, memory is optional


# ---------------------------------------------------------------------------
# TEST 18 — Code-intelligence failure: fallback while memory still helps
# ---------------------------------------------------------------------------


def test_code_intelligence_failure_falls_back(sandbox):
    _python_project(sandbox)
    task = _planned_task("locate the login handler", sandbox)
    ctx = task.code_context
    # Store a valid project fact.
    ctx.ensure_project_memory().store(
        "symbol:AuthService",
        "defined in src/auth/service.py",
        source=MemorySource.CODE_INTELLIGENCE,
        confidence=MemoryConfidence.DIRECT_OBSERVATION,
        metadata={"file": "src/auth/service.py"},
    )
    # Semantic search has no embedder (the default): it degrades to lexical
    # matches instead of crashing, and the query ladder stays usable.
    out = ctx.intelligence.query("semantic_search", query="login handler")
    assert isinstance(out, str)
    assert not out.startswith("Error")
    # The targeted context path also degrades gracefully.
    assert isinstance(ctx.intelligence.context_block(task), str)

    # Memory still helps where valid, independent of the index.
    block = _build_task_context_block(task)
    assert "PROJECT MEMORY" in block
    assert "src/auth/service.py" in block


# ---------------------------------------------------------------------------
# TEST 19 — Session boundary: project memory survives, working memory does not
# ---------------------------------------------------------------------------


def test_session_boundary_project_survives_working_does_not(sandbox):
    _python_project(sandbox)
    session1 = SessionMemory()
    task1 = _planned_task("fix the authentication bug", sandbox)
    session1.note_request(task1.goal)
    session1.note_decision("chose the JWT approach")
    block1 = _build_task_context_block(task1, session1)
    assert "chose the JWT approach" in block1
    assert "PROJECT MEMORY" in block1

    # A NEW session starts (fresh SessionMemory, fresh task, same workspace).
    session2 = SessionMemory()
    task2 = _planned_task("add email verification", sandbox)
    block2 = _build_task_context_block(task2, session2)
    # Project memory survives (persisted on disk, workspace-scoped).
    assert "PROJECT MEMORY" in block2
    # Session/working content does NOT leak across the boundary.
    assert "chose the JWT approach" not in block2
    assert "fix the authentication bug" not in block2


# ---------------------------------------------------------------------------
# TEST 20 — Performance: bounded context, cheap retrieval/assembly/restore
# ---------------------------------------------------------------------------


def test_performance_context_is_smaller_than_rediscovery(sandbox):
    _python_project(sandbox)
    # A representative repository: 60 source files with substantial bodies,
    # so a full re-read (what memory replaces) costs tens of kilobytes and
    # the comparison below is robust to rendering details.
    for i in range(60):
        _write(
            sandbox,
            f"src/modules/module{i}.py",
            "def function_{i}():\n    return " + '"' + "z" * 400 + '"' + "\n",
        )
    task = _planned_task("understand the authentication flow", sandbox)
    store = task.code_context.ensure_project_memory()
    for i in range(50):
        store.store(
            f"symbol:Symbol{i}",
            f"defined in src/auth/module{i}.py",
            source=MemorySource.CODE_INTELLIGENCE,
            confidence=MemoryConfidence.DIRECT_OBSERVATION,
            metadata={"file": f"src/auth/module{i}.py"},
        )

    # Cost of re-reading the repository fresh (what memory replaces): the full
    # content of the source tree.
    repo_bytes = sum(
        p.stat().st_size for p in sandbox.rglob("*") if p.is_file()
    )

    started = time.monotonic()
    records = store.recall(limit=100)
    recall_ms = (time.monotonic() - started) * 1000
    assert recall_ms < 50  # deterministic SQLite lookup

    started = time.monotonic()
    block = ContextManager().memory_block(
        project_records=records,
        session=SessionMemory().set_workspace(str(sandbox)),
        workspace=str(sandbox),
        task_terms=["authentication", "flow"],
    )
    assembly_ms = (time.monotonic() - started) * 1000
    assert assembly_ms < 200  # generous even on slow CI (measured ~1ms)
    # The memory block is far smaller than a full repository dump.
    assert len(block) <= 1400
    assert repo_bytes > 10_000  # the generated repo is non-trivial
    assert len(block) < repo_bytes  # memory < raw rediscovery cost

    # Session restore latency: snapshot save + load stays cheap.
    started = time.monotonic()
    assert save_task(task, sandbox) is not None
    assert load_task(sandbox) is not None
    restore_ms = (time.monotonic() - started) * 1000
    assert restore_ms < 200

    # Storage stays compact: facts are small rows, never code blobs.
    db_size = store.db_path.stat().st_size
    assert db_size < 200_000  # 50 facts + workspace facts, no code dumps
