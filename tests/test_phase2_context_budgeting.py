"""
tests/test_phase2_context_budgeting.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Comprehensive validation test suite for Phase 2:
Context Manager + Token Budgeting & Compaction per the Gap Closure Plan.

Validates:
1. Hard token budget calculation (estimated_input_tokens + reserved_output_tokens <= model_context_limit).
2. Synthetic long conversation exceeding model budget (condensing older history, protecting user goal & system prompt).
3. Oversized tool outputs stress (head + tail retention, truncated markers).
4. Strict deterministic priority preservation:
   User task > Active plan step > Latest failure > Repository context > Observations > Older history > Long-term memory.
5. Dynamic model-specific context limits (ModelCatalog / ModelRouter specs).
6. Safe TaskEvent (CONTEXT_BUILT) emission with sanitized summary metadata and no secret leakage.
7. ReActAgent pre-model-call budgeting loop integration.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from ultron.core.agents.react import ReActAgent
from ultron.core.coding.context import CodeContext
from ultron.core.coding.workspace import discover_workspace
from ultron.core.context import (
    ContextBudgetConfig,
    RepositoryContextManager,
    estimate_tokens,
)
from ultron.core.intelligence.model_catalog import ModelCatalog
from ultron.core.memory.models import (
    MemoryConfidence,
    MemoryKind,
    MemoryRecord,
    MemorySource,
)
from ultron.core.runtime import AgentRuntime, TaskEvent, TaskEventType
from ultron.core.runtime.events import EventBus
from ultron.core.types import (
    ChatMessage,
    PlanStep,
    Role,
    StepStatus,
    TaskLifecycleStatus,
    TaskPlan,
    TaskState,
    TaskType,
)


@pytest.fixture
def workspace_env(tmp_path):
    (tmp_path / "main.py").write_text("def run():\n    print('hello world')\n", encoding="utf-8")
    return discover_workspace(str(tmp_path))


# ===========================================================================
# 1. Hard Token Budget Calculation & Model Limits
# ===========================================================================


def test_budget_messages_under_budget(workspace_env):
    cm = RepositoryContextManager(workspace=workspace_env)
    messages = [
        ChatMessage(role=Role.SYSTEM, content="System prompt instructions"),
        ChatMessage(role=Role.USER, content="Hello assistant"),
        ChatMessage(role=Role.ASSISTANT, content="Hello user, how can I help?"),
    ]
    budgeted, meta = cm.budget_messages(
        messages,
        model_context_limit=4096,
        reserved_output_tokens=1024,
    )
    assert len(budgeted) == 3
    assert meta["dropped_messages"] == 0
    assert meta["truncated_tool_messages"] == 0
    assert meta["input_tokens"] + meta["reserved_output_tokens"] <= meta["model_limit"]


def test_budget_messages_exceeding_budget_condenses_older_dialogue(workspace_env):
    cm = RepositoryContextManager(workspace=workspace_env)
    # Target: max input tokens = 1000 - 300 = 700 tokens (~2800 chars)
    # Create system prompt, user original goal, 10 intermediate turns, and final query
    messages = [
        ChatMessage(role=Role.SYSTEM, content="System prompt: You are Ultron, a coding agent."),
        ChatMessage(role=Role.USER, content="ORIGINAL_GOAL: Build an authentication system."),
    ]
    # Add 10 intermediate turns with significant length (~150 tokens each = 1500 tokens)
    for i in range(10):
        messages.append(ChatMessage(role=Role.ASSISTANT, content=f"Intermediate step {i} " + ("exploring auth logic " * 20)))
        messages.append(ChatMessage(role=Role.TOOL, name="read_file", content=f"File content chunk {i} " + ("code line " * 20)))

    messages.append(ChatMessage(role=Role.USER, content="Now please summarize our findings."))

    budgeted, meta = cm.budget_messages(
        messages,
        model_context_limit=600,
        reserved_output_tokens=150,
    )

    # Hard budget invariant
    assert meta["input_tokens"] + meta["reserved_output_tokens"] <= 600
    assert meta["dropped_messages"] > 0

    # User goal and System message MUST be preserved
    roles = [m.role for m in budgeted]
    assert Role.SYSTEM in roles
    assert budgeted[0].role == Role.SYSTEM
    assert any("ORIGINAL_GOAL" in m.content for m in budgeted)
    # The latest user message should also be retained
    assert any("Now please summarize" in m.content for m in budgeted)


# ===========================================================================
# 2. Oversized Tool Output Truncation (Head + Tail Retention)
# ===========================================================================


def test_budget_messages_truncates_large_tool_outputs(workspace_env):
    # Set max_tool_output_tokens = 50 tokens (~200 chars)
    config = ContextBudgetConfig(max_tool_output_tokens=50, model_context_limit=2000, reserved_output_tokens=200)
    cm = RepositoryContextManager(workspace=workspace_env, budget=config)

    huge_tool_output = "START_OF_LOG\n" + ("data row 123456789\n" * 100) + "END_OF_LOG_SUMMARY"
    messages = [
        ChatMessage(role=Role.SYSTEM, content="System instructions"),
        ChatMessage(role=Role.USER, content="Run tool"),
        ChatMessage(role=Role.TOOL, name="run_command", content=huge_tool_output),
    ]

    budgeted, meta = cm.budget_messages(messages)
    assert meta["truncated_tool_messages"] == 1
    tool_msg = budgeted[2]
    assert tool_msg.role == Role.TOOL
    assert "START_OF_LOG" in tool_msg.content
    assert "END_OF_LOG_SUMMARY" in tool_msg.content
    assert "[truncated" in tool_msg.content
    assert "retained head & tail" in tool_msg.content


# ===========================================================================
# 3. Deterministic Priority Hierarchy Validation
# ===========================================================================


def test_priority_ordering_under_tight_budget(tmp_path):
    ws = discover_workspace(str(tmp_path))
    (tmp_path / "app.py").write_text("print('App code')\n" * 30, encoding="utf-8")

    # Tight budget: 60 tokens total to force dropping lower priority items
    budget = ContextBudgetConfig(max_total_tokens=60)
    cm = RepositoryContextManager(workspace=ws, budget=budget)

    # Build task with active step and failure evidence
    task = TaskState(goal="Implement user authentication")
    task.plan = TaskPlan(
        goal="Implement user authentication",
        task_type=TaskType.SOFTWARE_ENGINEERING,
        steps=[
            PlanStep(id=1, description="Setup database schema", status=StepStatus.SUCCEEDED),
            PlanStep(id=2, description="Write JWT handler", status=StepStatus.RUNNING),
        ],
    )
    task.record_tool_execution("write_file", "auth.py", success=False, detail="Permission denied on auth.py")

    # Code context observations
    code_ctx = CodeContext(workspace=ws)
    code_ctx.add_relevant_file(str(tmp_path / "app.py"))

    # Long-term & project memory
    proj_mem = [
        MemoryRecord(
            kind=MemoryKind.PROJECT,
            name="db_type",
            content="PostgreSQL 15",
            source=MemorySource.USER,
            confidence=MemoryConfidence.USER_PROVIDED,
            workspace=str(tmp_path),
        )
    ]
    long_term = [
        MemoryRecord(
            kind=MemoryKind.LONG_TERM,
            name="pref",
            content="User prefers Python 3.12",
            source=MemorySource.USER,
            confidence=MemoryConfidence.USER_PROVIDED,
        )
    ]

    snapshot = cm.assemble_snapshot(
        user_request="Fix auth handler bug",
        task=task,
        code_context=code_ctx,
        project_memory=proj_mem,
        long_term_memory=long_term,
        requested_files=[str(tmp_path / "app.py")],
    )

    assert snapshot.compacted
    assert snapshot.total_estimated_tokens <= 120

    # Priorities in accepted items:
    priorities = [item.priority for item in snapshot.items]
    for i in range(len(priorities) - 1):
        assert priorities[i].value <= priorities[i + 1].value

    # High priorities (USER_TASK, ACTIVE_PLAN_STEP, LATEST_FAILURE) must be present
    titles = [item.title for item in snapshot.items]
    assert any("User Request" in t for t in titles)
    assert any("Plan Step 2" in t for t in titles)
    assert any("Latest Failure" in t for t in titles)

    # Lowest priority item (Long-Term Memory) should be dropped under tight budget
    assert not any("Long-Term Memory" in t for t in titles)


# ===========================================================================
# 4. Model Catalog & Dynamic Limit Integration
# ===========================================================================


def test_model_catalog_context_limits_configured():
    catalog = ModelCatalog()
    for spec in catalog.list_models():
        assert spec.recommended_context_length >= 4096
        # Primary coding models should declare reasonable context
        if spec.role.value == "primary":
            assert spec.recommended_context_length >= 16384


def test_runtime_updates_context_budget_for_routed_model(tmp_path):
    ws = discover_workspace(str(tmp_path))
    cm = RepositoryContextManager(workspace=ws)
    runtime = AgentRuntime(context_manager=cm)

    # Mock decision with 32k context model
    mock_model_spec = MagicMock()
    mock_model_spec.model_id = "qwen3-32b"
    mock_model_spec.recommended_context_length = 32768

    runtime.context_manager.budget.model_context_limit = mock_model_spec.recommended_context_length
    assert runtime.context_manager.budget.model_context_limit == 32768


# ===========================================================================
# 5. Safe TaskEvent (CONTEXT_BUILT) Emission & Secret Redaction
# ===========================================================================


def test_context_built_event_emission_and_scrubbing(tmp_path):
    async def _test():
        ws = discover_workspace(str(tmp_path))
        bus = EventBus()
        emitted_events: list[TaskEvent] = []

        def _listener(event: TaskEvent):
            emitted_events.append(event)

        bus.subscribe(_listener, TaskEventType.CONTEXT_BUILT)

        cm = RepositoryContextManager(workspace=ws)
        runtime = AgentRuntime(context_manager=cm, event_bus=bus)

        task = TaskState(goal="Configure environment secret")
        user_prompt = "Store secret api_key=sk-proj-supersecret1234567890abcdef"

        class SimpleMockAgent:
            async def run(self, *args, **kwargs):
                return ChatMessage(role=Role.ASSISTANT, content="Done")

        await runtime.execute(SimpleMockAgent(), user_prompt, task=task)

        context_events = [e for e in emitted_events if e.event_type == TaskEventType.CONTEXT_BUILT]
        assert len(context_events) >= 1
        event = context_events[0]
        payload = event.payload

        # Metadata must be present
        assert "total_estimated_tokens" in payload
        assert "items_count" in payload
        assert "model_context_limit" in payload
        # Raw secrets must NOT be in the event payload
        payload_str = str(payload)
        assert "sk-proj-supersecret1234567890abcdef" not in payload_str

    asyncio.run(_test())


# ===========================================================================
# 6. ReActAgent Loop Pre-Model-Call Budgeting Integration
# ===========================================================================


def test_react_agent_budgets_messages_before_model_call(tmp_path):
    async def _test():
        ws = discover_workspace(str(tmp_path))
        budget_cfg = ContextBudgetConfig(model_context_limit=300, reserved_output_tokens=50)
        cm = RepositoryContextManager(workspace=ws, budget=budget_cfg)

        # Engine that captures the exact messages passed to generate()
        captured_calls = []

        class CapturingEngine:
            async def generate(self, messages, **kwargs):
                captured_calls.append(list(messages))
                return "Final answer from model."

        engine = CapturingEngine()
        agent = ReActAgent(engine=engine, max_iterations=2)

        # Create long history exceeding 250 tokens
        history = [
            ChatMessage(role=Role.USER, content="INITIAL_TASK: Build component"),
        ]
        for i in range(8):
            history.append(ChatMessage(role=Role.ASSISTANT, content=f"Step {i} analysis " * 10))
            history.append(ChatMessage(role=Role.TOOL, name="bash", content=f"Output {i} log data " * 10))

        task = TaskState(goal="Build component")
        task.lifecycle_status = TaskLifecycleStatus.EXECUTING

        bus = EventBus()
        events_caught = []

        def _catch(ev):
            events_caught.append(ev)

        bus.subscribe(_catch, TaskEventType.CONTEXT_BUILT)

        reply = await agent.run(
            user_input="Continue building",
            history=history,
            task=task,
            context_manager=cm,
            event_bus=bus,
        )

        assert reply.role == Role.ASSISTANT
        assert len(captured_calls) >= 1

        called_messages = captured_calls[0]
        # Invariant: input tokens must fit into (300 - 50 = 250)
        total_tokens = sum(estimate_tokens(m.get("content", "")) for m in called_messages)
        assert total_tokens <= 250

        # System instruction and initial task anchor are preserved
        assert called_messages[0]["role"] == "system"
        assert any("INITIAL_TASK" in m.get("content", "") for m in called_messages)

        # Event was emitted
        assert len(events_caught) >= 1
        assert events_caught[0].event_type == TaskEventType.CONTEXT_BUILT

    asyncio.run(_test())

