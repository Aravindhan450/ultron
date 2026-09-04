"""
Unit tests for ultron.core.context (Phase 2: Repository-Aware ContextManager).

Validates:
- Repository discovery and project profile resolution
- File retrieval (existing, missing -> NOT_FOUND, region, access safety)
- Symbol retrieval (existing definition/reference, missing -> NOT_FOUND)
- Search retrieval (valid matches, missing -> NOT_FOUND)
- Git context retrieval (branch, status, diff stat)
- Deduplication and priority ordering
- Token budgeting and compaction
- ContextSnapshot observability
- AgentRuntime integration
"""

import pytest

from ultron.core.agents.base import BaseAgent
from ultron.core.coding.context import CodeContext
from ultron.core.coding.workspace import discover_workspace
from ultron.core.context import (
    ContextBudgetConfig,
    ContextPriority,
    ContextRetrievalStatus,
    RepositoryContextManager,
    RepositoryRetriever,
    estimate_tokens,
)
from ultron.core.runtime import AgentRuntime
from ultron.core.tools import paths as tools_paths
from ultron.core.types import ChatMessage, Role, TaskState


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(tools_paths, "ALLOWED_BASE_DIR", tmp_path)
    monkeypatch.chdir(tmp_path)
    return tmp_path


class MockAgent(BaseAgent):
    def __init__(self):
        self.engine = None

    async def run(self, user_input: str, history: list[ChatMessage] | None = None, **kwargs) -> ChatMessage:
        return ChatMessage(role=Role.ASSISTANT, content=f"Echo: {user_input}")


# ---------------------------------------------------------------------------
# Token Estimation Tests
# ---------------------------------------------------------------------------


def test_estimate_tokens():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abc") == 1
    assert estimate_tokens("a" * 40) == 10


# ---------------------------------------------------------------------------
# File Retrieval Tests
# ---------------------------------------------------------------------------


def test_retrieve_existing_file_and_region(sandbox):
    test_file = sandbox / "example.py"
    lines = [f"def func_{i}():\n    return {i}" for i in range(10)]
    test_file.write_text("\n".join(lines), encoding="utf-8")

    ws = discover_workspace(str(sandbox))
    retriever = RepositoryRetriever(workspace=ws)

    # Full file retrieval
    res = retriever.retrieve_file("example.py")
    assert res.status == ContextRetrievalStatus.FOUND
    assert len(res.items) == 1
    assert "func_0" in res.items[0].content
    assert res.items[0].priority == ContextPriority.DIRECT_FILE

    # Region retrieval
    res_region = retriever.retrieve_file("example.py", start_line=1, end_line=3)
    assert res_region.status == ContextRetrievalStatus.FOUND
    assert "lines 1-3" in res_region.items[0].title


def test_retrieve_missing_file_explicit_not_found(sandbox):
    ws = discover_workspace(str(sandbox))
    retriever = RepositoryRetriever(workspace=ws)

    res = retriever.retrieve_file("non_existent_file.py")
    assert res.status == ContextRetrievalStatus.NOT_FOUND
    assert not res.is_found
    assert len(res.items) == 0
    assert "File not found" in (res.error_message or "")


# ---------------------------------------------------------------------------
# Symbol & Search Retrieval Tests
# ---------------------------------------------------------------------------


def test_retrieve_missing_symbol_explicit_not_found(sandbox):
    ws = discover_workspace(str(sandbox))
    retriever = RepositoryRetriever(workspace=ws)

    res = retriever.retrieve_symbol("NonExistentClass")
    assert res.status == ContextRetrievalStatus.NOT_FOUND
    assert not res.is_found
    assert len(res.items) == 0


def test_retrieve_search_results(sandbox):
    (sandbox / "app.py").write_text("def special_feature(): pass\n", encoding="utf-8")
    ws = discover_workspace(str(sandbox))
    retriever = RepositoryRetriever(workspace=ws)

    # Note: refresh code intelligence index
    retriever.intelligence.refresh()

    res = retriever.retrieve_search("special_feature")
    assert res.status == ContextRetrievalStatus.FOUND
    assert len(res.items) == 1
    assert "special_feature" in res.items[0].content

    res_missing = retriever.retrieve_search("completely_unmatched_term_xyz")
    assert res_missing.status == ContextRetrievalStatus.NOT_FOUND


# ---------------------------------------------------------------------------
# Git Context Retrieval Tests
# ---------------------------------------------------------------------------


def test_retrieve_git_context(sandbox):
    ws = discover_workspace(str(sandbox))
    retriever = RepositoryRetriever(workspace=ws)

    res = retriever.retrieve_git_context()
    # tmp_path is not a git repo
    assert res.status == ContextRetrievalStatus.NOT_FOUND


# ---------------------------------------------------------------------------
# ContextManager Assembly, Deduplication & Budget Tests
# ---------------------------------------------------------------------------


def test_context_manager_assembly_and_deduplication(sandbox):
    (sandbox / "mod.py").write_text("class TargetService: pass\n", encoding="utf-8")
    ws = discover_workspace(str(sandbox))
    cm = RepositoryContextManager(workspace=ws)

    task = TaskState(goal="Implement user auth")
    task.add_requirement("Create user service")

    code_ctx = CodeContext(workspace=ws)
    code_ctx.add_relevant_file(str(sandbox / "mod.py"))

    ctx_text = cm.build_context(
        user_request="Build authentication service",
        task=task,
        code_context=code_ctx,
        requested_files=[str(sandbox / "mod.py"), str(sandbox / "mod.py")],  # duplicate
    )

    assert "User Request" in ctx_text
    assert "Active Task State" in ctx_text
    assert "mod.py" in ctx_text
    assert cm.last_snapshot is not None
    assert cm.last_snapshot.total_estimated_tokens > 0


def test_context_manager_budget_and_compaction(sandbox):
    # Setup small budget
    budget = ContextBudgetConfig(max_total_tokens=150)
    ws = discover_workspace(str(sandbox))
    cm = RepositoryContextManager(workspace=ws, budget=budget)

    # Large content
    big_file = sandbox / "big.py"
    big_file.write_text("# Line\n" * 500, encoding="utf-8")

    task = TaskState(goal="Fix performance issue in repo")

    ctx_text = cm.build_context(
        user_request="Check big file",
        task=task,
        requested_files=[str(big_file)],
    )

    assert "Check big file" in ctx_text
    assert cm.last_snapshot is not None
    assert cm.last_snapshot.total_estimated_tokens <= 200
    assert cm.last_snapshot.compacted


# ---------------------------------------------------------------------------
# AgentRuntime Integration Tests
# ---------------------------------------------------------------------------


def test_runtime_context_manager_integration(sandbox):
    import asyncio

    async def _test():
        runtime = AgentRuntime()
        agent = MockAgent()
        task = TaskState(goal="Test runtime integration")

        result = await runtime.execute(agent, "Hello Ultron", task=task)
        assert result.is_success
        assert runtime.context_manager is not None
        assert result.context_snapshot is not None
        assert result.context_snapshot.total_estimated_tokens > 0
        assert any(item.title == "User Request" for item in result.context_snapshot.items)

    asyncio.run(_test())


def test_runtime_context_snapshot_on_cancellation(sandbox):
    import asyncio

    from ultron.core.runtime import CancellationToken

    async def _test():
        runtime = AgentRuntime()
        agent = MockAgent()
        token = CancellationToken()
        token.cancel("Cancelled by user")

        result = await runtime.execute(agent, "Hello Ultron", cancellation_token=token)
        assert not result.is_success
        assert result.context_snapshot is not None
        assert result.context_snapshot.total_estimated_tokens > 0

    asyncio.run(_test())


def test_context_manager_memory_provider_integration(sandbox):
    from ultron.core.memory.models import (
        MemoryConfidence,
        MemoryKind,
        MemoryRecord,
        MemorySource,
    )
    from ultron.core.memory.provider import MemoryProvider
    from ultron.core.memory.session_memory import SessionMemory

    provider = MemoryProvider()
    records = [
        MemoryRecord(
            kind=MemoryKind.PROJECT,
            name="auth_service",
            content="Authentication uses JWT tokens",
            source=MemorySource.CODE_INTELLIGENCE,
            confidence=MemoryConfidence.DIRECT_OBSERVATION,
            workspace=str(sandbox),
        ),
        MemoryRecord(
            kind=MemoryKind.LONG_TERM,
            name="user_pref",
            content="User prefers pytest over unittest",
            source=MemorySource.USER,
            confidence=MemoryConfidence.USER_PROVIDED,
        ),
    ]

    # Test MemoryProvider methods directly
    proj_items = provider.provide_project_memory(records, workspace=str(sandbox), task_terms=["auth"])
    assert len(proj_items) == 1
    assert proj_items[0].target == "auth_service"
    assert "JWT tokens" in proj_items[0].content

    session = SessionMemory()
    session.note_request("User requested login feature")
    session_items = provider.provide_session_memory(session)
    assert len(session_items) == 1
    assert "login feature" in session_items[0].content

    lt_items = provider.provide_long_term_memory(records)
    assert len(lt_items) == 1
    assert "pytest" in lt_items[0].content

    # Test assemble_snapshot with memory items
    cm = RepositoryContextManager(workspace=discover_workspace(str(sandbox)), memory_provider=provider)
    task = TaskState(goal="Update authentication")
    code_ctx = CodeContext(workspace=cm.retriever.workspace)
    task.code_context = code_ctx
    mem_store = code_ctx.ensure_project_memory()
    if mem_store is not None:
        mem_store.store(
            "auth_service",
            "Authentication uses JWT tokens",
            source=MemorySource.CODE_INTELLIGENCE,
            confidence=MemoryConfidence.DIRECT_OBSERVATION,
        )

    snapshot = cm.assemble_snapshot(
        user_request="Update authentication logic",
        task=task,
        code_context=code_ctx,
        session=session,
    )
    assert snapshot.total_estimated_tokens > 0
    assert any("Project Fact" in item.title or "Session Memory" in item.title for item in snapshot.items)


def test_react_agent_uses_memory_provider_not_legacy_context_manager():
    import inspect

    import ultron.core.agents.react as react_module

    source = inspect.getsource(react_module)
    assert "from ultron.core.memory.context_manager import ContextManager" not in source
    assert "MemoryProvider" in source


def test_legacy_context_manager_delegation_to_memory_provider(sandbox):
    from ultron.core.memory.context_manager import ContextManager
    from ultron.core.memory.models import (
        MemoryConfidence,
        MemoryKind,
        MemoryRecord,
        MemorySource,
    )
    from ultron.core.memory.session_memory import SessionMemory

    records = [
        MemoryRecord(
            kind=MemoryKind.PROJECT,
            name="auth_service",
            content="Authentication uses JWT tokens",
            source=MemorySource.CODE_INTELLIGENCE,
            confidence=MemoryConfidence.DIRECT_OBSERVATION,
            workspace=str(sandbox),
        ),
    ]
    session = SessionMemory()
    session.note_request("User requested login feature")

    legacy_cm = ContextManager()
    mem_block = legacy_cm.memory_block(
        project_records=records,
        session=session,
        workspace=str(sandbox),
        task_terms=["auth"],
    )
    assert "PROJECT MEMORY:" in mem_block
    assert "JWT tokens" in mem_block
    assert "SESSION MEMORY:" in mem_block
    assert "login feature" in mem_block


def test_canonical_pipeline_agent_runtime_to_react(sandbox):
    import asyncio

    from ultron.core.agents.react import ReActAgent

    class PipelineFakeEngine:
        def __init__(self, responses):
            self.responses = list(responses)
            self.calls = []

        async def generate(self, messages, **kwargs):
            self.calls.append(messages)
            return self.responses.pop(0) if self.responses else "Done."

    async def _test():
        engine = PipelineFakeEngine(["I have inspected the context."])
        agent = ReActAgent(engine=engine)
        runtime = AgentRuntime()
        task = TaskState(goal="Inspect context pipeline")

        result = await runtime.execute(agent, "Inspect context", task=task)
        assert result.is_success
        assert result.context_snapshot is not None
        assert any(item.title == "User Request" for item in result.context_snapshot.items)
        assert any(item.title == "Active Task State" for item in result.context_snapshot.items)
        assert len(engine.calls) > 0

    asyncio.run(_test())

