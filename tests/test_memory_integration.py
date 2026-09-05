"""FIX #6 — Memory + ContextManager integration into the agent architecture.

Covers the 15 integration scenarios from the FIX #6 integration spec:

  1.  project memory reused during a follow-up task
  2.  current code overrides stale memory
  3.  task state survives confirmation
  4.  task survives process restart
  5.  interrupted task resumes
  6.  unrelated new task gets a separate TaskState
  7.  meaningful observation becomes project memory
  8.  irrelevant observation is not persisted
  9.  long context is compressed
  10. large command output is summarized
  11. stale memory is invalidated
  12. project memory is isolated between repositories
  13. secrets are not persisted
  14. context priority works correctly
  15. code intelligence outranks stale memory

Plus the session-continuity and task-persistence plumbing (react.py session
threading, memory formation sync, /resume task store).

All tests use temporary workspaces (never the Ultron repository) and the
established sandbox pattern (ALLOWED_BASE_DIR -> tmp) so the code-intelligence
bridge, workspace tools and the project-memory store behave as in production.
"""

from __future__ import annotations

import asyncio

import pytest

from ultron.core.agents.react import (
    ReActAgent,
    _build_task_context_block,
    _sync_task_memory,
)
from ultron.core.coding.context import CodeContext
from ultron.core.coding.intelligence_bridge import CodeIntelligenceBridge
from ultron.core.coding.observations import ObservationKind
from ultron.core.coding.workspace import discover_workspace
from ultron.core.context import RepositoryContextManager
from ultron.core.memory import (
    MemoryConfidence,
    MemoryProvider,
    MemorySource,
    ProjectMemoryStore,
    SessionMemory,
    WorkingMemory,
)
from ultron.core.memory.formation import reconcile_project_memory
from ultron.core.memory.task_store import load_task, save_task, task_store_dir
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


def _pyproject(root):
    """Minimal Python project markers so workspace detection is deterministic."""
    _write(
        root,
        "pyproject.toml",
        "[project]\nname = \"demo\"\n[tool.pytest.ini_options]\n",
    )
    _write(root, "src/pkg/__init__.py", "def helper():\n    return 1\n")
    _write(root, "tests/test_pkg.py", "def test_helper():\n    assert helper() == 1\n")


def _planned_task(goal: str, workspace_root, *, steps: int = 3) -> TaskState:
    """A TaskState with a structured plan + attached CodeContext — the same
    shape ``prepare_task_for_execution`` produces (minus the LLM calls)."""
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
# 1. Project memory reused during a follow-up task
# ---------------------------------------------------------------------------


def test_project_memory_reused_in_followup_task(sandbox):
    _pyproject(sandbox)
    task1 = _planned_task("understand the auth architecture", sandbox)
    store1 = task1.code_context.ensure_project_memory()
    assert store1 is not None
    assert store1.count() > 0

    # A follow-up task in the same workspace attaches to the same store.
    task2 = _planned_task("add refresh tokens", sandbox)
    store2 = task2.code_context.ensure_project_memory()
    assert store2 is not None

    records = store2.recall(limit=50)
    names = [r.name for r in records]
    assert "language:python" in names
    assert "test_framework" in names

    # The follow-up task's context block surfaces the fact (memory-aware
    # exploration — no blind rediscovery).
    block = _build_task_context_block(task2)
    assert "PROJECT MEMORY" in block
    assert "project uses python" in block.lower()


# ---------------------------------------------------------------------------
# 2. Current code overrides stale memory (reconciliation)
# ---------------------------------------------------------------------------


def test_current_code_overrides_stale_memory(sandbox):
    _write(sandbox, "src/service.py", "class UserService:\n    pass\n")
    bridge = CodeIntelligenceBridge()
    assert bridge.enable(str(sandbox))
    bridge.refresh()
    bridge.query("find_definition", name="UserService")  # populate log + index

    store = ProjectMemoryStore(sandbox)
    store.store(
        "symbol:UserService",
        "defined in src/old/service.py",
        source=MemorySource.CODE_INTELLIGENCE,
        confidence=MemoryConfidence.DIRECT_OBSERVATION,
        metadata={"file": "src/old/service.py"},
    )

    updated = reconcile_project_memory(store, bridge)
    assert updated == 1

    current = store.recall(name="symbol:UserService")
    assert len(current) == 1
    assert "src/service.py" in current[0].content
    assert current[0].metadata["file"] == "src/service.py"
    # History preserved: the stale version is superseded, not erased.
    history = store.history("symbol:UserService")
    assert len(history) == 2
    assert history[0].validity.value == "superseded"
    bridge.close()


# ---------------------------------------------------------------------------
# 3. Task state survives confirmation
# ---------------------------------------------------------------------------


def test_task_state_survives_confirmation(sandbox):
    _pyproject(sandbox)
    task = _planned_task("add a health endpoint", sandbox)
    task.wait_for_confirmation()
    task.context.append(
        ChatMessage(role=Role.ASSISTANT, content='{"tool": "write_file"}')
    )

    raw = task.model_dump_json()
    restored = TaskState.model_validate_json(raw)

    assert restored.goal == task.goal
    assert restored.plan is not None and len(restored.plan.steps) == 3
    assert restored.is_waiting_confirmation
    assert restored.code_context is not None
    # The workspace-scoped store handle is rebuilt lazily after round-trip.
    store = restored.code_context.ensure_project_memory()
    assert store is not None
    assert store.recall(limit=50)
    assert len(restored.context) == 1


# ---------------------------------------------------------------------------
# 4. Task survives process restart (snapshot persistence)
# ---------------------------------------------------------------------------


def test_task_survives_process_restart(sandbox):
    _pyproject(sandbox)
    task = _planned_task("implement the feature", sandbox)
    task.record_tool_execution(
        tool_name="read_file",
        target="src/pkg/__init__.py",
        success=True,
        detail="read the module",
    )

    path = save_task(task, sandbox)
    assert path is not None
    assert path.parent == task_store_dir(sandbox)
    assert path.exists()

    # Fresh process: new store objects, same files.
    loaded = load_task(sandbox)
    assert loaded is not None
    assert loaded.goal == task.goal
    assert loaded.task_type is TaskType.SOFTWARE_ENGINEERING
    assert loaded.plan is not None and len(loaded.plan.steps) == 3
    assert len(loaded.execution_history) == 1
    assert loaded.is_complete() is False


# ---------------------------------------------------------------------------
# 5. Interrupted task resumes (completed work preserved)
# ---------------------------------------------------------------------------


def test_interrupted_task_resumes(sandbox):
    _pyproject(sandbox)
    task = _planned_task("fix the failing tests", sandbox)
    # Work done: step 1 succeeded, step 2 running, step 3 pending.
    task.plan.set_step_status(1, StepStatus.SUCCEEDED, result="inspected")
    task.plan.set_step_status(2, StepStatus.RUNNING)
    task.set_current_step(2)
    task.record_tool_execution(
        tool_name="read_file",
        target="src/pkg/__init__.py",
        success=True,
        detail="found the buggy function",
    )
    task.wait_for_confirmation()  # interrupted mid-confirmation
    assert save_task(task, sandbox) is not None

    loaded = load_task(sandbox)
    assert loaded is not None
    # Completed steps stay recorded; the plan is not restarted from step 1.
    assert loaded.plan.step(1).status is StepStatus.SUCCEEDED
    assert loaded.plan.step(2).status is StepStatus.RUNNING
    assert loaded.plan.step(3).status is StepStatus.PENDING
    assert loaded.current_step == 2
    assert len(loaded.execution_history) == 1
    # Stale confirmation hygiene: a pending action is not restored.
    assert loaded.pending_action is None
    assert not loaded.is_waiting_confirmation
    assert loaded.is_complete() is False


# ---------------------------------------------------------------------------
# 6. Unrelated new task gets a separate TaskState
# ---------------------------------------------------------------------------


def test_unrelated_task_gets_separate_taskstate(sandbox):
    _pyproject(sandbox)
    task_a = _planned_task("refactor the auth module", sandbox)
    task_b = _planned_task("upgrade the database driver", sandbox)

    assert task_a is not task_b
    assert task_a.goal != task_b.goal
    assert task_a.plan is not None and task_b.plan is not None
    # Separate runtime state: completing one never touches the other.
    task_a.plan.set_step_status(1, StepStatus.SUCCEEDED)
    assert task_b.plan.step(1).status is StepStatus.PENDING

    session = SessionMemory()
    session.note_request(task_a.goal)
    session.note_request(task_b.goal)
    assert len(session.recent_requests) == 2
    # Task B's own context block is about B. Session memory (bounded) is the
    # only place task A's request appears — that is session continuity, not
    # task merging.
    block = _build_task_context_block(task_b, session)
    assert "Goal: upgrade the database driver" in block
    own_block = _build_task_context_block(task_b)
    assert "auth module" not in own_block


# ---------------------------------------------------------------------------
# 7. Meaningful observation becomes project memory
# ---------------------------------------------------------------------------


def test_meaningful_observation_becomes_project_memory(sandbox):
    _pyproject(sandbox)
    task = _planned_task("understand the codebase", sandbox)
    store = task.code_context.ensure_project_memory()
    records = store.recall(limit=50)
    assert records

    names = {r.name for r in records}
    assert {"language:python", "package_manager", "test_framework"} <= names
    # Provenance is honest: repository inspection + direct observation.
    assert all(
        r.source is MemorySource.REPOSITORY_INSPECTION
        and r.confidence is MemoryConfidence.DIRECT_OBSERVATION
        for r in records
    )


# ---------------------------------------------------------------------------
# 8. Irrelevant observation is not persisted
# ---------------------------------------------------------------------------


def test_irrelevant_observation_not_persisted(sandbox):
    _pyproject(sandbox)
    task = _planned_task("list the files", sandbox)
    store = task.code_context.ensure_project_memory()
    baseline = {r.name for r in store.recall(limit=50)}

    # A turn that only read files / ran trivial commands produces no memory.
    task.code_context.record_observation(
        ObservationKind.FILE_CONTENT,
        "read_file",
        "src/pkg/__init__.py: def helper(): return 1",
        success=True,
    )
    task.code_context.record_observation(
        ObservationKind.COMMAND_RESULT,
        "run_command",
        "ls -la",
        "drwxr-xr-x  5 user  staff  160 Jan  1 00:00 .",
        success=True,
    )
    _sync_task_memory(task)

    after = {r.name for r in store.recall(limit=50)}
    assert after == baseline  # nothing irrelevant added
    assert not any("read_file" in name for name in after)
    assert not any("command" in name.lower() for name in after)


# ---------------------------------------------------------------------------
# 9. Long context is compressed
# ---------------------------------------------------------------------------


def test_long_context_compressed(sandbox):
    _pyproject(sandbox)
    store = ProjectMemoryStore(sandbox)
    for i in range(30):
        store.store(
            f"fact:{i}",
            f"some project detail number {i} that keeps going " * 4,
            source=MemorySource.REPOSITORY_INSPECTION,
            confidence=MemoryConfidence.DIRECT_OBSERVATION,
        )
    session = SessionMemory()
    for i in range(20):
        session.note_decision(f"decision {i}: choose approach A{i}")

    provider = MemoryProvider()
    proj_items = provider.provide_project_memory(
        records=store.recall(limit=100),
        workspace=str(sandbox),
        task_terms=["detail"],
    )
    sess_items = provider.provide_session_memory(session)
    assert len(proj_items) <= 6
    assert len(sess_items) <= 1
    assert not any("fact:0" in item.content for item in proj_items)


# ---------------------------------------------------------------------------
# 10. Large command output is summarized
# ---------------------------------------------------------------------------


def test_large_command_output_summarized(sandbox):
    _pyproject(sandbox)
    task = _planned_task("inspect the build output", sandbox)
    huge = "x" * 20000
    task.code_context.record_observation(
        ObservationKind.COMMAND_RESULT,
        "run_command",
        "pytest",
        detail=huge,
        success=True,
    )
    wm = WorkingMemory.from_task(task, task.code_context)
    for line in wm.recent_observations:
        assert len(line) < 250  # summarized, never the raw 20k dump
    assert all("x" * 100 not in line for line in wm.recent_observations)


# ---------------------------------------------------------------------------
# 11. Stale memory is invalidated (without erasure)
# ---------------------------------------------------------------------------


def test_stale_memory_invalidated(sandbox):
    store = ProjectMemoryStore(sandbox)
    store.store(
        "auth_location",
        "authentication lives in src/auth/",
        source=MemorySource.REPOSITORY_INSPECTION,
        confidence=MemoryConfidence.DIRECT_OBSERVATION,
    )
    assert store.count() == 1

    invalidated = store.invalidate("auth_location")
    assert invalidated == 1
    # Not erased: valid recall excludes it, history still has it.
    assert store.recall(name="auth_location") == []
    history = store.history("auth_location")
    assert len(history) == 1
    assert history[0].validity.value == "stale"

    provider = MemoryProvider()
    items = provider.provide_project_memory(
        records=store.recall(limit=50),
        workspace=str(sandbox),
        task_terms=["auth"],
    )
    assert len(items) == 0  # stale records are excluded


# ---------------------------------------------------------------------------
# 12. Project memory is isolated between repositories
# ---------------------------------------------------------------------------


def test_project_memory_isolated_between_repos(sandbox, tmp_path):
    repo_a = sandbox / "repo_a"
    repo_b = sandbox / "repo_b"
    repo_a.mkdir()
    repo_b.mkdir()

    store_a = ProjectMemoryStore(repo_a)
    store_b = ProjectMemoryStore(repo_b)
    store_a.store(
        "api_stack",
        "uses FastAPI",
        source=MemorySource.REPOSITORY_INSPECTION,
        confidence=MemoryConfidence.DIRECT_OBSERVATION,
    )

    assert store_a.count() == 1
    assert store_b.count() == 0
    assert store_a.db_path != store_b.db_path


# ---------------------------------------------------------------------------
# 13. Secrets are not persisted
# ---------------------------------------------------------------------------


def test_secrets_not_persisted(sandbox):
    store = ProjectMemoryStore(sandbox)

    assert store.store("api_key", "the key is sk-1234567890abcdefghij") is None
    assert store.store(
        "config",
        "connection settings",
        metadata={"password": "sk-9876543210abcdefghij"},
    ) is None
    assert store.count() == 0

    # The escape hatch is explicit and documented — never the default.
    assert store.store(
        "system_fact",
        "system-defined value",
        allow_secrets=True,
    ) is not None


# ---------------------------------------------------------------------------
# 14. Context priority works correctly
# ---------------------------------------------------------------------------


def test_context_priority_works(sandbox):
    _pyproject(sandbox)
    task = _planned_task("add caching to the user service", sandbox)
    store = task.code_context.ensure_project_memory()
    store.store(
        "old_fact",
        "user service was removed",
        source=MemorySource.LLM_INFERENCE,
        confidence=MemoryConfidence.INFERRED,
    )
    store.invalidate("old_fact")

    session = SessionMemory()
    session.note_decision("chose the in-memory cache approach")

    cm = RepositoryContextManager(workspace=discover_workspace(str(sandbox)))
    block = cm.build_context(
        user_request="add caching to the user service",
        task=task,
        code_context=task.code_context,
        project_memory=store.recall(limit=50),
        session=session,
        task_terms=["caching", "user", "service"],
    )

    # Strict priority order: task facts before memory; stale excluded.
    assert block.index("Active Task State") < block.index("Session Memory")
    assert "was removed" not in block  # stale record excluded entirely


# ---------------------------------------------------------------------------
# 15. Code intelligence outranks stale memory (end-to-end)
# ---------------------------------------------------------------------------


def test_code_intelligence_outranks_stale_memory(sandbox):
    _write(sandbox, "src/service.py", "class UserService:\n    pass\n")
    task = _planned_task("rename UserService", sandbox)
    ctx = task.code_context
    assert ctx.intelligence.enable(str(sandbox))
    ctx.intelligence.refresh()
    ctx.intelligence.query("find_definition", name="UserService")

    store = ctx.ensure_project_memory()
    store.store(
        "symbol:UserService",
        "defined in src/legacy/service.py",
        source=MemorySource.CODE_INTELLIGENCE,
        confidence=MemoryConfidence.DIRECT_OBSERVATION,
        metadata={"file": "src/legacy/service.py"},
    )

    # Agent turn end: formation sync reconciles against the current index.
    _sync_task_memory(task)
    current = store.recall(name="symbol:UserService")
    assert len(current) == 1
    assert "src/service.py" in current[0].content
    assert current[0].metadata["file"] == "src/service.py"

    # The context block the model sees reflects the CURRENT repository.
    block = _build_task_context_block(task)
    assert "defined in src/service.py" in block
    assert "src/legacy/service.py" not in block


# ---------------------------------------------------------------------------
# 16. Session continuity across turns (react.py wiring)
# ---------------------------------------------------------------------------


def test_session_continuity_across_turns(sandbox):
    _pyproject(sandbox)
    session = SessionMemory()
    # A plan-less task so the agent's final answer is accepted immediately
    # (a structured plan would force the verification path).
    task1 = TaskState(goal="fix the authentication bug")
    ctx1 = CodeContext(workspace=discover_workspace(str(sandbox)))
    ctx1.attach_task(task1)
    task1.code_context = ctx1

    engine = _FakeEngine()
    agent = ReActAgent(engine)
    response = _run(agent.run(task1.goal, task=task1, session=session))

    assert response.content == "Final answer."
    assert session.recent_requests == [task1.goal]
    assert str(sandbox) in session.active_workspace
    assert task1.goal[:60] in session.task_refs

    # Follow-up turn: new task, same session — context block carries session
    # memory so "now" resolves to the active project.
    task2 = _planned_task("add refresh tokens", sandbox)
    block = _build_task_context_block(task2, session)
    assert "SESSION MEMORY" in block
    assert "Active workspace" in block
    assert "fix the authentication bug" in block


# ---------------------------------------------------------------------------
# 17. Task snapshot store gates (Ultron repo + secrets)
# ---------------------------------------------------------------------------


def test_task_store_never_writes_into_ultron_repo(sandbox):
    task = _planned_task("some goal", sandbox)
    # Persistence into Ultron's own repository is refused outright.
    assert save_task(task, _REPO_ROOT) is None
    assert not (task_store_dir(_REPO_ROOT) / "latest.json").exists()


def test_task_store_refuses_secret_payloads(sandbox):
    task = _planned_task("some goal", sandbox)
    task.context.append(
        ChatMessage(
            role=Role.TOOL,
            name="run_command",
            content="echo sk-1234567890abcdefghij",
        )
    )
    assert save_task(task, sandbox) is None
    assert load_task(sandbox) is None


def test_task_store_prunes_old_snapshots(sandbox):
    _pyproject(sandbox)
    for i in range(12):
        task = _planned_task(f"task number {i}", sandbox)
        save_task(task, sandbox)
    snapshots = [
        p
        for p in task_store_dir(sandbox).glob("*.json")
        if p.name != "latest.json"
    ]
    assert len(snapshots) <= 10
    assert load_task(sandbox) is not None  # latest pointer still valid


# ---------------------------------------------------------------------------
# 18. Memory formation sync promotes symbol facts via react wiring
# ---------------------------------------------------------------------------


def test_sync_task_memory_promotes_symbol_facts(sandbox):
    _write(sandbox, "src/service.py", "class UserService:\n    pass\n")
    task = _planned_task("find UserService", sandbox)
    ctx = task.code_context
    assert ctx.intelligence.enable(str(sandbox))
    ctx.intelligence.refresh()
    ctx.intelligence.query("find_references", name="UserService")
    ctx.intelligence.query("find_definition", name="UserService")

    _sync_task_memory(task)
    store = ctx.ensure_project_memory()
    current = store.recall(name="symbol:UserService")
    assert len(current) == 1
    assert current[0].source is MemorySource.CODE_INTELLIGENCE
    assert "src/service.py" in current[0].content
    ctx.intelligence.close()
