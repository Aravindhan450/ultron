"""
tests/test_model_call_boundary.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Regression tests for the authoritative model-call boundary (Phase 2
remediation). Verifies that ONE mechanism — ``BudgetedEngine`` installed at the
production agent factory — enforces the hard invariant::

    estimated_input_tokens + reserved_output_tokens <= model_context_limit

across ReAct, verification, planning/classification, and SimpleAgent model
calls; that the routed model's limit propagates; that compaction preserves the
latest user request; and that observability never leaks secrets.
"""

from __future__ import annotations

import asyncio
import pathlib
import types
from typing import Any

from ultron.core.agents import get_agent
from ultron.core.agents.react import ReActAgent
from ultron.core.agents.simple import classify_intent, plan_task
from ultron.core.context.budget import (
    ContextBudgetConfig,
    budget_messages,
    budget_openai_messages,
)
from ultron.core.context.invocation import (
    BudgetedEngine,
    ensure_budgeted_engine,
    model_call_scope,
)
from ultron.core.intelligence.model_catalog import get_default_catalog
from ultron.core.intelligence.task_planning import generate_task_plan
from ultron.core.runtime.event_store import InMemoryEventStore
from ultron.core.runtime.events import EventBus, TaskEventType
from ultron.core.types import (
    ChatMessage,
    Role,
    TaskLifecycleStatus,
    TaskState,
    TaskType,
)


class CapturingEngine:
    """Inner engine capturing exactly what the boundary forwarded to it."""

    def __init__(self, response: str = "done") -> None:
        self.calls: list[list[dict[str, Any]]] = []
        self.response = response

    async def generate(self, messages, **kwargs):
        self.calls.append(list(messages))
        return self.response


def _tokens(messages: list[dict[str, Any]]) -> int:
    return sum(max(1, (len(str(m.get("content", ""))) + 3) // 4) for m in messages)


def _assert_invariant(meta: dict[str, Any]) -> None:
    assert meta["input_tokens"] + meta["reserved_output_tokens"] <= meta["model_limit"]


# --------------------------------------------------------------------------
# A. Main ReAct model call is budgeted by the boundary
# --------------------------------------------------------------------------


def test_a_react_model_call_budgeted(tmp_path):
    async def _test():
        engine = CapturingEngine("Final answer.")
        boundary = BudgetedEngine(engine, ContextBudgetConfig(model_context_limit=300, reserved_output_tokens=50))
        agent = ReActAgent(engine=boundary, max_iterations=2)
        history = [ChatMessage(role=Role.USER, content="INITIAL_TASK: build component")]
        for i in range(8):
            history.append(ChatMessage(role=Role.ASSISTANT, content=f"step {i} " * 40))
            history.append(ChatMessage(role=Role.TOOL, name="bash", content=f"output {i} " * 40))
        task = TaskState(goal="build component")
        task.lifecycle_status = TaskLifecycleStatus.EXECUTING
        await agent.run(user_input="continue", history=history, task=task)
        assert engine.calls
        assert _tokens(engine.calls[0]) <= 250
        _assert_invariant(boundary.last_budget)

    asyncio.run(_test())


# --------------------------------------------------------------------------
# B. Verification model call is budgeted
# --------------------------------------------------------------------------


def test_b_verification_model_call_budgeted():
    async def _test():
        engine = CapturingEngine('[{"description": "x", "satisfied": false}]')
        boundary = BudgetedEngine(engine, ContextBudgetConfig(model_context_limit=400, reserved_output_tokens=50))
        agent = ReActAgent(engine=boundary)
        task = TaskState(goal="Do a large thing")
        for i in range(30):
            task.record_tool_execution("read_file", f"file_{i}.py", True, detail="evidence " * 60)
        accepted, _ = await agent._verify_task(task, "goal", "I am done", [])
        assert not accepted
        assert engine.calls
        assert _tokens(engine.calls[0]) <= 350
        _assert_invariant(boundary.last_budget)

    asyncio.run(_test())


# --------------------------------------------------------------------------
# C. Plan-generation model call is budgeted
# --------------------------------------------------------------------------


def test_c_plan_generation_model_call_budgeted(tmp_path):
    async def _test():
        engine = CapturingEngine("not-json")
        boundary = BudgetedEngine(engine, ContextBudgetConfig(model_context_limit=400, reserved_output_tokens=50))
        plan = await generate_task_plan(
            "Build a FastAPI service with many endpoints",
            TaskType.SOFTWARE_ENGINEERING,
            boundary,
            working_context="repository context " * 400,
            cwd=str(tmp_path),
        )
        assert plan is None or plan.steps
        assert engine.calls
        for call in engine.calls:
            assert _tokens(call) <= 350
        _assert_invariant(boundary.last_budget)

    asyncio.run(_test())


# --------------------------------------------------------------------------
# D. SimpleAgent production model calls are budgeted
# --------------------------------------------------------------------------


def test_d_simple_agent_model_calls_budgeted():
    async def _test():
        engine = CapturingEngine("none")
        boundary = BudgetedEngine(engine, ContextBudgetConfig(model_context_limit=300, reserved_output_tokens=50))
        await plan_task("do a complicated multi step thing " * 100, boundary)
        await classify_intent("please route this request " * 100, boundary)
        assert engine.calls
        for call in engine.calls:
            assert _tokens(call) <= 250
        _assert_invariant(boundary.last_budget)

    asyncio.run(_test())


# --------------------------------------------------------------------------
# E. Hard invariant holds for arbitrary inputs
# --------------------------------------------------------------------------


def test_e_hard_invariant_holds_various_limits():
    config = ContextBudgetConfig()
    messages = [
        ChatMessage(role=Role.SYSTEM, content="system " * 500),
        ChatMessage(role=Role.USER, content="goal " * 500),
        ChatMessage(role=Role.TOOL, name="read_file", content="observation " * 2000),
    ]
    for limit, reserved in [(16384, 2048), (1000, 900), (200, 100), (64, 64), (10, 8)]:
        _, meta = budget_messages(messages, config, model_context_limit=limit, reserved_output_tokens=reserved)
        _assert_invariant(meta)
        assert meta["input_tokens"] <= meta["model_limit"]


# --------------------------------------------------------------------------
# F. Small remaining budget cannot violate the invariant
# --------------------------------------------------------------------------


def test_f_small_remaining_budget_clamps_reservation():
    config = ContextBudgetConfig()
    messages = [ChatMessage(role=Role.USER, content="hello " * 100)]
    _, meta = budget_messages(messages, config, model_context_limit=100, reserved_output_tokens=99)
    # The reservation is reduced so the invariant holds and some input survives.
    _assert_invariant(meta)
    assert meta["input_tokens"] + meta["reserved_output_tokens"] <= 100
    assert meta["reserved_output_tokens"] < 99


# --------------------------------------------------------------------------
# G. Long conversation compaction
# --------------------------------------------------------------------------


def test_g_long_conversation_compacts():
    config = ContextBudgetConfig()
    messages = [ChatMessage(role=Role.SYSTEM, content="sys")]
    messages.append(ChatMessage(role=Role.USER, content="ORIGINAL goal"))
    for i in range(30):
        messages.append(ChatMessage(role=Role.ASSISTANT, content=f"assistant turn {i} " * 20))
        messages.append(ChatMessage(role=Role.TOOL, name="bash", content=f"tool output {i} " * 20))
    messages.append(ChatMessage(role=Role.USER, content="LATEST request now"))
    _, meta = budget_messages(messages, config, model_context_limit=800, reserved_output_tokens=100)
    assert meta["dropped_messages"] > 0
    _assert_invariant(meta)


# --------------------------------------------------------------------------
# H. Latest user request is retained
# --------------------------------------------------------------------------


def test_h_latest_user_request_retained():
    config = ContextBudgetConfig()
    messages = [ChatMessage(role=Role.SYSTEM, content="system instructions")]
    messages.append(ChatMessage(role=Role.USER, content="ORIGINAL goal"))
    for i in range(20):
        messages.append(ChatMessage(role=Role.ASSISTANT, content=f"analysis {i} " * 30))
        messages.append(ChatMessage(role=Role.TOOL, name="bash", content=f"observation {i} " * 30))
    messages.append(ChatMessage(role=Role.USER, content="LATEST request now"))
    budgeted, _ = budget_messages(messages, config, model_context_limit=700, reserved_output_tokens=100)
    contents = " ".join(m.content for m in budgeted)
    assert "LATEST request now" in contents
    assert "system instructions" in contents


# --------------------------------------------------------------------------
# I. Oversized tool output head/tail truncation (dict + ChatMessage paths)
# --------------------------------------------------------------------------


def test_i_oversized_tool_output_truncated_head_tail():
    config = ContextBudgetConfig(max_tool_output_tokens=50, model_context_limit=4000, reserved_output_tokens=200)
    huge = "START_OF_LOG\n" + ("data row 123456789\n" * 200) + "END_OF_LOG_SUMMARY"
    openai_msgs = [
        {"role": "system", "content": "sys"},
        {"role": "tool", "content": huge},
    ]
    budgeted, meta = budget_openai_messages(openai_msgs, config)
    assert meta["truncated_tool_messages"] == 1
    assert "START_OF_LOG" in budgeted[1]["content"]
    assert "END_OF_LOG_SUMMARY" in budgeted[1]["content"]
    assert "retained head & tail" in budgeted[1]["content"]


# --------------------------------------------------------------------------
# J. Routed model's context limit propagates to the boundary
# --------------------------------------------------------------------------


def test_j_routed_model_limit_reaches_boundary():
    async def _test():
        from ultron.core.runtime import AgentRuntime

        engine = CapturingEngine("done")
        boundary = BudgetedEngine(engine, ContextBudgetConfig(model_context_limit=16384, reserved_output_tokens=2048))
        agent = ReActAgent(engine=boundary)

        spec = get_default_catalog().list_models()[0].model_copy(
            update={"recommended_context_length": 8192}
        )
        router = types.SimpleNamespace(
            route=lambda request: types.SimpleNamespace(selected_model=spec, reason="test")
        )
        lifecycle = types.SimpleNamespace(
            ensure_loaded=lambda model: types.SimpleNamespace(
                endpoint_url="http://127.0.0.1:8080", model_spec=model
            )
        )
        runtime = AgentRuntime(
            router=router,
            lifecycle_manager=lifecycle,
            event_bus=EventBus(store=InMemoryEventStore()),
        )
        result = await runtime.execute(agent, "hello", task=TaskState(goal="hello"))
        assert result.is_success
        assert boundary.budget.model_context_limit == 8192

    asyncio.run(_test())


# --------------------------------------------------------------------------
# K. CONTEXT_BUILT observability corresponds to the actual invocation
# --------------------------------------------------------------------------


def test_k_context_built_event_matches_invocation():
    async def _test():
        engine = CapturingEngine("done")
        boundary = BudgetedEngine(engine, ContextBudgetConfig(model_context_limit=500, reserved_output_tokens=50))
        bus = EventBus(store=InMemoryEventStore())
        captured = []
        bus.subscribe(captured.append, TaskEventType.CONTEXT_BUILT)
        messages = [{"role": "user", "content": "hello world " * 200}]
        with model_call_scope(bus, "task_boundary_test", "run_boundary_test"):
            await boundary.generate(messages)
        assert captured, "boundary must emit a CONTEXT_BUILT event"
        payload = captured[0].payload
        assert payload["kind"] == "model_call_budget"
        assert payload["input_tokens"] == _tokens(engine.calls[0])
        assert payload["input_tokens"] + payload["reserved_output_tokens"] <= payload["model_limit"]

    asyncio.run(_test())


# --------------------------------------------------------------------------
# L. Secret sanitization
# --------------------------------------------------------------------------


def test_l_no_secret_leak_in_budget_event():
    async def _test():
        engine = CapturingEngine("done")
        boundary = BudgetedEngine(engine, ContextBudgetConfig())
        bus = EventBus(store=InMemoryEventStore())
        captured = []
        bus.subscribe(captured.append, TaskEventType.CONTEXT_BUILT)
        secret = "sk-test-ULTRON-BOUNDARY-SECRET"
        messages = [{"role": "user", "content": f"api_key={secret}"}]
        with model_call_scope(bus, "task_secret_test", "run_secret_test"):
            await boundary.generate(messages)
        assert captured
        assert secret not in str(captured[0].payload)

    asyncio.run(_test())


# --------------------------------------------------------------------------
# Factory guard: production agents always receive the boundary
# --------------------------------------------------------------------------


def test_production_agent_factory_installs_boundary():
    for agent_type in ("simple", "react"):
        agent = get_agent(agent_type)
        assert isinstance(agent.engine, BudgetedEngine), agent_type


def test_boundary_wrapping_is_idempotent():
    inner = CapturingEngine()
    once = ensure_budgeted_engine(inner)
    twice = ensure_budgeted_engine(once)
    assert once is twice
    assert once.inner_engine is inner


def test_no_direct_engine_construction_in_agents_factory():
    """get_agent must not hand an unwrapped engine to any agent."""
    source = pathlib.Path("src/ultron/core/agents/__init__.py").read_text(encoding="utf-8")
    assert "ensure_budgeted_engine" in source
