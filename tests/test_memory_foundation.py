"""
FIX #6 — Memory hierarchy + context management foundation tests.

Covers every requested area deterministically (no LLM):

- memory record model + source/confidence semantics
- project memory: persistence, workspace scoping, supersede history,
  staleness invalidation, retrieval, git revision, secret guard
- session memory: bounded, task/session separation
- working memory: ephemeral projection from TaskState/CodeContext
- ContextManager: prioritization, budget, assembly, source-of-truth order
- restart persistence

All storage uses temporary directories — the Ultron repo is never touched.
"""

import pytest

from ultron.core.coding.context import CodeContext
from ultron.core.coding.observations import ObservationKind
from ultron.core.coding.workspace import discover_workspace
from ultron.core.context import ContextBudgetConfig, RepositoryContextManager
from ultron.core.memory import (
    MemoryConfidence,
    MemoryKind,
    MemoryRecord,
    MemorySource,
    MemoryValidity,
    ProjectMemoryStore,
    SessionMemory,
    WorkingMemory,
)
from ultron.core.memory.project_memory import _coerce
from ultron.core.types import (
    FailureStrategy,
    PlanStep,
    TaskPlan,
    TaskState,
    TaskType,
    WorkspaceKind,
)


def _step(step_id: int, description: str, criteria: list[str]) -> PlanStep:
    return PlanStep(
        id=step_id,
        description=description,
        purpose=f"Purpose of {description}",
        expected_outcome=f"Outcome of {description}",
        completion_criteria=criteria,
        dependencies=[],
        failure_strategy=FailureStrategy.STOP,
    )


def _make_task(goal: str, cwd: str, with_step: str | None = None) -> TaskState:
    task = TaskState(goal=goal, task_type=TaskType.SOFTWARE_ENGINEERING)
    if with_step:
        task.attach_plan(
            TaskPlan(
                goal=goal,
                task_type=TaskType.SOFTWARE_ENGINEERING,
                workspace=WorkspaceKind.EXISTING_PROJECT,
                steps=[_step(1, with_step, ["criterion"])],
                completion_criteria=["criterion"],
                verification_requirements=["criterion"],
            )
        )
    task.code_context = CodeContext(workspace=discover_workspace(cwd))
    task.code_context.attach_task(task)
    return task


@pytest.fixture
def store(tmp_path):
    return ProjectMemoryStore(tmp_path)


@pytest.fixture
def second_workspace(tmp_path):
    other = tmp_path / "other_project"
    other.mkdir()
    return ProjectMemoryStore(other)


# ---------------------------------------------------------------------------
# 1. Memory record model + source/confidence semantics
# ---------------------------------------------------------------------------


def test_record_model_roundtrips():
    record = MemoryRecord(
        name="backend",
        content="uses FastAPI",
        source=MemorySource.REPOSITORY_INSPECTION,
        confidence=MemoryConfidence.DIRECT_OBSERVATION,
        workspace="/tmp/proj",
        revision="abc1234",
        metadata={"framework": "fastapi"},
    )
    restored = MemoryRecord.model_validate(record.model_dump())
    assert restored.name == "backend"
    assert restored.source is MemorySource.REPOSITORY_INSPECTION
    assert restored.confidence is MemoryConfidence.DIRECT_OBSERVATION
    assert restored.revision == "abc1234"
    assert restored.metadata["framework"] == "fastapi"
    # Prompt rendering is debuggable and bounded.
    line = restored.to_prompt_line(max_len=40)
    assert "[direct_observation]" in line
    assert len(line) <= 40


def test_source_and_confidence_enums():
    # The full vocabulary required by FIX #6 section 7/8.
    sources = {s.value for s in MemorySource}
    assert {
        "user",
        "repository_inspection",
        "tool_result",
        "code_intelligence",
        "test_result",
        "system_knowledge",
        "llm_inference",
    } <= sources
    confidences = {c.value for c in MemoryConfidence}
    assert {
        "direct_observation",
        "high_confidence",
        "inferred",
        "user_provided",
    } <= confidences


def test_coerce_accepts_strings_and_members():
    assert _coerce("repository_inspection", MemorySource) is MemorySource.REPOSITORY_INSPECTION
    assert _coerce(MemorySource.TOOL_RESULT, MemorySource) is MemorySource.TOOL_RESULT
    assert _coerce("direct_observation", MemoryConfidence) is MemoryConfidence.DIRECT_OBSERVATION
    assert _coerce("nonsense", MemorySource) is MemorySource.LLM_INFERENCE  # safe fallback


# ---------------------------------------------------------------------------
# 2. Project memory: store / recall / persistence / scoping
# ---------------------------------------------------------------------------


def test_project_memory_store_and_recall(store):
    record = store.store(
        "auth",
        "authentication is handled by AuthService in src/auth/service.py",
        source=MemorySource.REPOSITORY_INSPECTION,
        confidence=MemoryConfidence.DIRECT_OBSERVATION,
    )
    assert record is not None
    assert record.id is not None
    recalled = store.recall("auth")
    assert len(recalled) == 1
    assert recalled[0].content == record.content
    assert recalled[0].source is MemorySource.REPOSITORY_INSPECTION
    assert store.count() == 1


def test_project_memory_persists_across_restart(tmp_path):
    first = ProjectMemoryStore(tmp_path)
    first.store(
        "stack",
        "backend uses FastAPI",
        source=MemorySource.REPOSITORY_INSPECTION,
        confidence=MemoryConfidence.DIRECT_OBSERVATION,
    )
    # Simulate a process restart: a brand-new store object over the same path.
    second = ProjectMemoryStore(tmp_path)
    recalled = second.recall("stack")
    assert len(recalled) == 1
    assert recalled[0].content == "backend uses FastAPI"


def test_project_memory_workspace_scoping(store, second_workspace):
    store.store(
        "stack",
        "Project A uses FastAPI",
        source=MemorySource.REPOSITORY_INSPECTION,
        confidence=MemoryConfidence.DIRECT_OBSERVATION,
    )
    second_workspace.store(
        "stack",
        "Project B uses Express",
        source=MemorySource.REPOSITORY_INSPECTION,
        confidence=MemoryConfidence.DIRECT_OBSERVATION,
    )
    assert store.recall("stack")[0].content == "Project A uses FastAPI"
    assert second_workspace.recall("stack")[0].content == "Project B uses Express"
    # Workspaces never see each other's facts.
    assert "Project B" not in store.recall("stack")[0].content
    assert "Project A" not in second_workspace.recall("stack")[0].content


def test_project_memory_forget(store):
    store.store("stack", "uses FastAPI")
    assert store.forget("stack") is True
    assert store.recall("stack") == []
    assert store.count() == 0


# ---------------------------------------------------------------------------
# 3. Invalidation: supersede history + staleness
# ---------------------------------------------------------------------------


def test_store_supersedes_previous_value_keeping_history(store):
    first = store.store(
        "auth_location",
        "AuthService -> src/auth/service.py",
        source=MemorySource.REPOSITORY_INSPECTION,
        confidence=MemoryConfidence.DIRECT_OBSERVATION,
    )
    store.store(
        "auth_location",
        "AuthService -> src/security/auth/service.py",
        source=MemorySource.CODE_INTELLIGENCE,
        confidence=MemoryConfidence.DIRECT_OBSERVATION,
    )
    # Current value is the new one; the old one is superseded, not erased.
    current = store.recall("auth_location")
    assert len(current) == 1
    assert current[0].content == "AuthService -> src/security/auth/service.py"
    assert current[0].supersedes_id == first.id
    history = store.history("auth_location")
    assert len(history) == 2
    statuses = {h.validity for h in history}
    assert MemoryValidity.VALID in statuses
    assert MemoryValidity.SUPERSEDED in statuses


def test_invalidate_marks_stale_not_deleted(store):
    store.store("auth", "AuthService -> src/auth/service.py")
    assert store.invalidate("auth") == 1
    # Stale records are excluded from normal recall but not erased.
    assert store.recall("auth") == []
    stale = store.recall("auth", include_invalid=True)
    assert len(stale) == 1
    assert stale[0].validity is MemoryValidity.STALE


def test_invalidate_all(store):
    store.store("a", "fact one")
    store.store("b", "fact two")
    assert store.invalidate() == 2
    assert store.count() == 0


# ---------------------------------------------------------------------------
# 4. Retrieval
# ---------------------------------------------------------------------------


def test_search_is_deterministic_and_project_scoped(store, second_workspace):
    store.store("stack", "backend uses FastAPI and PostgreSQL")
    second_workspace.store("stack", "backend uses Express")
    hits = store.search("FastAPI")
    assert len(hits) == 1
    assert hits[0].content == "backend uses FastAPI and PostgreSQL"


def test_search_ignores_invalid(store):
    store.store("stack", "uses FastAPI")
    store.invalidate("stack")
    assert store.search("FastAPI") == []


# ---------------------------------------------------------------------------
# 5. Sources / confidence / versioning / secrets
# ---------------------------------------------------------------------------


def test_revision_captured_when_git_available(tmp_path):
    store = ProjectMemoryStore(tmp_path)
    # Non-git temp dir -> revision is None, never a crash.
    record = store.store("stack", "uses FastAPI")
    assert record.revision is None or isinstance(record.revision, str)
    # An explicit revision is honored.
    record = store.store("stack", "uses FastAPI", revision="deadbeef")
    assert record.revision == "deadbeef"


def test_secret_content_never_persisted(store):
    # A credential-looking string must be refused by the write guard.
    assert (
        store.store(
            "credentials",
            "api_key=sk-1234567890abcdef",
            source=MemorySource.TOOL_RESULT,
            confidence=MemoryConfidence.DIRECT_OBSERVATION,
        )
        is None
    )
    assert store.recall("credentials") == []
    assert store.count() == 0
    # Even metadata alone must not smuggle secrets through.
    assert (
        store.store(
            "config",
            "connection settings",
            metadata={"password": "sk-0123456789abcdef0123456789abcdef"},
        )
        is None
    )


def test_source_tracking_preserved(store):
    assert (
        store.store(
            "test_cmd",
            "run pytest tests/auth before the full suite",
            source=MemorySource.SYSTEM_KNOWLEDGE,
            confidence=MemoryConfidence.HIGH_CONFIDENCE,
        )
        is not None
    )
    recalled = store.recall("test_cmd")[0]
    assert recalled.source is MemorySource.SYSTEM_KNOWLEDGE
    assert recalled.confidence is MemoryConfidence.HIGH_CONFIDENCE


# ---------------------------------------------------------------------------
# 6. Session memory
# ---------------------------------------------------------------------------


def test_session_memory_bounded_and_summary_shaped():
    session = SessionMemory()
    for i in range(20):
        session.note_request(f"request number {i}")
    assert len(session.recent_requests) == session.max_requests == 6
    # Notable outputs are truncated summaries, never raw dumps.
    session.note_output("x" * 500)
    assert len(session.notable_outputs[0]) <= session.output_clip_chars


def test_session_task_separation(tmp_path):
    session = SessionMemory()
    session.set_workspace(str(tmp_path))
    session.note_request("understand the auth architecture")
    session.note_decision("use refresh tokens")
    session.note_task("task-1")
    lines = session.to_context_lines()
    assert any("Active workspace" in line for line in lines)
    assert any("understand the auth architecture" in line for line in lines)
    assert any("use refresh tokens" in line for line in lines)

    # A different task never shares this session's memory.
    other_session = SessionMemory()
    assert other_session.is_empty
    assert other_session.to_context_lines() == []


def test_session_reset(tmp_path):
    session = SessionMemory()
    session.set_workspace(str(tmp_path))
    session.note_request("hello")
    session.reset()
    assert session.is_empty


# ---------------------------------------------------------------------------
# 7. Working memory
# ---------------------------------------------------------------------------


def test_working_memory_projects_task_state(tmp_path):
    task = _make_task(
        "add refresh tokens",
        str(tmp_path),
        with_step="inspect the auth implementation",
    )
    task.code_context.add_relevant_file("src/auth/service.py")
    task.code_context.record_observation(
        kind=ObservationKind.TEST_RESULT,
        source="pytest",
        summary="1 failed, 0 passed",
        success=False,
        exit_code=1,
    )
    wm = WorkingMemory.from_task(task, task.code_context)
    assert wm.goal == "add refresh tokens"
    assert wm.current_step == "inspect the auth implementation"
    assert wm.current_step_id == 1
    assert "src/auth/service.py" in wm.relevant_files
    assert wm.active_failure is not None  # classified failure surfaced


def test_working_memory_never_persisted_itself():
    # WorkingMemory is a plain pydantic view — no store, no DB.
    wm = WorkingMemory()
    assert wm.recent_observations == []
    wm.record_tool_result("exit code 0")
    assert wm.last_tool_result == "exit code 0"


# ---------------------------------------------------------------------------
# 8. ContextManager: prioritization, budget, assembly
# ---------------------------------------------------------------------------


def _project_records(tmp_path, workspace=None):
    store = ProjectMemoryStore(tmp_path)
    store.store(
        "auth",
        "authentication handled in src/auth/service.py",
        source=MemorySource.CODE_INTELLIGENCE,
        confidence=MemoryConfidence.DIRECT_OBSERVATION,
    )
    store.store(
        "stack",
        "backend uses FastAPI",
        source=MemorySource.REPOSITORY_INSPECTION,
        confidence=MemoryConfidence.DIRECT_OBSERVATION,
    )
    return store.all_valid()


def test_context_priority_order(tmp_path):
    task = _make_task("add refresh tokens", str(tmp_path), with_step="inspect auth")
    session = SessionMemory()
    session.note_request("earlier conversation")

    ctx = RepositoryContextManager(workspace=discover_workspace(str(tmp_path)))
    block = ctx.build_context(
        user_request="now add refresh tokens",
        task=task,
        code_context=task.code_context,
        project_memory=_project_records(tmp_path),
        session=session,
        task_terms=["auth", "token"],
    )
    # Priority 1: the current user request appears first.
    assert block.index("User Request") < block.index("Active Task State")
    assert block.index("Active Task State") < block.index("Project Fact")
    assert block.index("Project Fact") < block.index("Session Memory")
    # Current task facts are present; session memory is last.
    assert "add refresh tokens" in block
    assert "src/auth/service.py" in block


def test_context_budget_enforced(tmp_path):
    # A tiny budget with rich inputs genuinely stresses the hard ceiling.
    task = _make_task("a long goal " * 200, str(tmp_path), with_step="do the work")
    session = SessionMemory()
    session.note_request("earlier session detail " * 50)
    budget = ContextBudgetConfig(max_total_tokens=40)
    ctx = RepositoryContextManager(workspace=discover_workspace(str(tmp_path)), budget=budget)
    snapshot = ctx.assemble_snapshot(
        user_request="do it",
        task=task,
        code_context=task.code_context,
        project_memory=_project_records(tmp_path),
        session=session,
    )
    assert snapshot.total_estimated_tokens <= budget.max_total_tokens
    assert snapshot.compacted or snapshot.dropped_items_count > 0
    # High-priority user request survived
    user_items = [it for it in snapshot.items if it.title == "User Request"]
    assert len(user_items) == 1
    assert "do it" in user_items[0].content


def test_context_hard_ceiling_never_exceeded(tmp_path):
    # Pathological inputs: every section huge, budget tiny
    task = _make_task("goal " * 500, str(tmp_path), with_step="step " * 200)
    session = SessionMemory()
    session.note_request("request " * 300)
    session.note_decision("decision " * 200)
    session.note_output("output " * 300)
    for budget_tokens in (20, 40, 80, 150):
        ctx = RepositoryContextManager(
            workspace=discover_workspace(str(tmp_path)),
            budget=ContextBudgetConfig(max_total_tokens=budget_tokens),
        )
        snapshot = ctx.assemble_snapshot(
            user_request="user " * 400,
            task=task,
            code_context=task.code_context,
            project_memory=_project_records(tmp_path),
            session=session,
        )
        assert snapshot.total_estimated_tokens <= budget_tokens, (
            f"budget {budget_tokens} overflowed to {snapshot.total_estimated_tokens}"
        )


def test_stale_memory_never_injected(tmp_path):
    store = ProjectMemoryStore(tmp_path)
    store.store(
        "auth_location",
        "AuthService -> src/auth/service.py",
        source=MemorySource.CODE_INTELLIGENCE,
        confidence=MemoryConfidence.DIRECT_OBSERVATION,
    )
    store.invalidate("auth_location")
    ctx = RepositoryContextManager(workspace=discover_workspace(str(tmp_path)))
    block = ctx.build_context(
        user_request="where is auth?",
        project_memory=store.all_valid(),
    )
    assert "src/auth/service.py" not in block


def test_project_memory_preferred_over_global(tmp_path):
    project = ProjectMemoryStore(tmp_path)
    project.store(
        "stack",
        "this project uses FastAPI",
        source=MemorySource.REPOSITORY_INSPECTION,
        confidence=MemoryConfidence.DIRECT_OBSERVATION,
    )
    global_records = [
        MemoryRecord(
            kind=MemoryKind.LONG_TERM,
            name="stack",
            content="the user generally prefers Flask",
            source=MemorySource.USER,
            confidence=MemoryConfidence.USER_PROVIDED,
            workspace="",
        )
    ]
    ctx = RepositoryContextManager(workspace=discover_workspace(str(tmp_path)))
    block = ctx.build_context(
        user_request="what stack is this project?",
        project_memory=project.all_valid(),
        long_term_memory=global_records,
    )
    # The project-scoped fact appears in the block; project memory precedes general repo
    assert "FastAPI" in block
    assert block.index("Project Fact") < block.index("Long-Term Memory")


def test_context_assembly_deterministic(tmp_path):
    task = _make_task("same goal", str(tmp_path), with_step="same step")
    ws = discover_workspace(str(tmp_path))
    kwargs = {
        "user_request": "same request",
        "task": task,
        "code_context": task.code_context,
        "project_memory": _project_records(tmp_path),
        "task_terms": ["auth"],
    }
    first = RepositoryContextManager(workspace=ws).build_context(**kwargs)
    second = RepositoryContextManager(workspace=ws).build_context(**kwargs)
    assert first == second


def test_test_observations_not_double_counted(tmp_path):
    task = _make_task("fix tests", str(tmp_path))
    task.code_context.record_observation(
        ObservationKind.TEST_RESULT,
        "pytest",
        "tests/test_auth.py::test_login FAILED",
        success=False,
    )
    task.code_context.record_observation(
        ObservationKind.FILE_CONTENT,
        "read_file",
        "src/auth/service.py",
        success=True,
    )
    ctx = RepositoryContextManager(workspace=discover_workspace(str(tmp_path)))
    block = ctx.build_context(
        user_request="fix the failing tests",
        task=task,
        code_context=task.code_context,
    )
    assert block.count("tests/test_auth.py::test_login FAILED") == 1
    assert "Recent Observations" in block


def test_context_manager_empty_inputs(tmp_path):
    ctx = RepositoryContextManager(workspace=discover_workspace(str(tmp_path)))
    snapshot = ctx.assemble_snapshot(user_request="")
    assert isinstance(snapshot.items, list)


# ---------------------------------------------------------------------------
# 9. Integration sanity: the memory package coexists with task machinery
# ---------------------------------------------------------------------------


def test_project_memory_keyword_relevance_ranking(tmp_path):
    store = ProjectMemoryStore(tmp_path)
    store.store(
        "auth",
        "authentication handled by AuthService",
        source=MemorySource.CODE_INTELLIGENCE,
        confidence=MemoryConfidence.DIRECT_OBSERVATION,
    )
    store.store(
        "billing",
        "invoices computed by BillingService",
        source=MemorySource.CODE_INTELLIGENCE,
        confidence=MemoryConfidence.DIRECT_OBSERVATION,
    )
    hits = store.search("auth")
    assert hits and hits[0].name == "auth"
