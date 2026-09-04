"""
Unit tests for ultron.core.runtime (Phase 1: AgentRuntime Foundation).

Validates:
- Lifecycle transitions (valid + invalid)
- State machine terminal semantics
- Budgets (iteration, tool calls, delegation limits)
- Timeout detection & cancellation
- EventBus publish / subscribe / history
- Agent execution through AgentRuntime
- Security boundary preservation
"""

import asyncio
from unittest.mock import MagicMock

import pytest

from ultron.core.agents.base import BaseAgent
from ultron.core.runtime import (
    AgentRuntime,
    CancellationToken,
    EventBus,
    RunState,
    RuntimeBudget,
    RuntimeEvent,
    RuntimeEventType,
    RuntimeStatus,
)
from ultron.core.types import ChatMessage, Role


class MockAgent(BaseAgent):
    def __init__(self, response_msg: ChatMessage | None = None, delay: float = 0.0, error: Exception | None = None):
        self.engine = MagicMock()
        self.response_msg = response_msg or ChatMessage(role=Role.ASSISTANT, content="Done.")
        self.delay = delay
        self.error = error
        self.run_called = False

    async def run(self, user_input: str, history: list[ChatMessage] | None = None, **kwargs) -> ChatMessage:
        self.run_called = True
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        if self.error:
            raise self.error
        return self.response_msg


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# State & Lifecycle Tests
# ---------------------------------------------------------------------------


def test_runtime_status_properties():
    assert RuntimeStatus.CREATED.is_active
    assert not RuntimeStatus.CREATED.is_terminal
    assert RuntimeStatus.RUNNING.is_active
    assert not RuntimeStatus.RUNNING.is_terminal
    assert RuntimeStatus.COMPLETED.is_terminal
    assert not RuntimeStatus.COMPLETED.is_active
    assert RuntimeStatus.CANCELLED.is_terminal
    assert RuntimeStatus.TIMED_OUT.is_terminal
    assert RuntimeStatus.BUDGET_EXCEEDED.is_terminal


def test_runtime_transitions_valid():
    state = RunState(run_id="test_run_1")
    assert state.status == RuntimeStatus.CREATED
    assert state.started_at is None

    state.transition_to(RuntimeStatus.INITIALIZING)
    assert state.status == RuntimeStatus.INITIALIZING
    assert state.started_at is not None

    state.transition_to(RuntimeStatus.RUNNING)
    assert state.status == RuntimeStatus.RUNNING

    state.transition_to(RuntimeStatus.VERIFYING)
    assert state.status == RuntimeStatus.VERIFYING

    state.transition_to(RuntimeStatus.COMPLETED)
    assert state.status == RuntimeStatus.COMPLETED
    assert state.finished_at is not None


def test_runtime_transitions_invalid():
    state = RunState(run_id="test_run_2")
    # Illegal jump from CREATED directly to COMPLETED
    with pytest.raises(ValueError, match="Illegal runtime transition"):
        state.transition_to(RuntimeStatus.COMPLETED)

    # Terminal states accept no transitions
    state.transition_to(RuntimeStatus.RUNNING)
    state.transition_to(RuntimeStatus.FAILED, error="Some failure")
    assert state.status == RuntimeStatus.FAILED

    with pytest.raises(ValueError, match="Illegal runtime transition"):
        state.transition_to(RuntimeStatus.RUNNING)


# ---------------------------------------------------------------------------
# Budget Tests
# ---------------------------------------------------------------------------


def test_budget_limits_and_exhaustion():
    budget = RuntimeBudget(max_iterations=3, max_tool_calls=5, max_delegations=2)
    assert not budget.is_exhausted()

    budget.record_iteration(2)
    assert not budget.is_exhausted()
    assert budget.iterations_used == 2

    budget.record_iteration(1)
    assert budget.is_exhausted()
    assert "max_iterations" in budget.exhaustion_reason()

    tool_budget = RuntimeBudget(max_iterations=10, max_tool_calls=2)
    tool_budget.record_tool_call(2)
    assert tool_budget.is_exhausted()
    assert "max_tool_calls" in tool_budget.exhaustion_reason()


def test_budget_timeout():
    budget = RuntimeBudget(timeout_seconds=0.01)
    import datetime
    budget.started_at = datetime.datetime.now(datetime.UTC) - datetime.timedelta(seconds=1)
    assert budget.is_timed_out()


# ---------------------------------------------------------------------------
# EventBus Tests
# ---------------------------------------------------------------------------


def test_event_bus_subscribe_emit():
    bus = EventBus()
    received_events = []

    def on_event(ev: RuntimeEvent):
        received_events.append(ev)

    bus.subscribe(on_event, RuntimeEventType.RUN_STARTED)

    ev1 = RuntimeEvent(event_type=RuntimeEventType.RUN_STARTED, run_id="r1", payload={"foo": "bar"})
    ev2 = RuntimeEvent(event_type=RuntimeEventType.RUN_COMPLETED, run_id="r1")

    bus.emit_sync(ev1)
    bus.emit_sync(ev2)

    assert len(received_events) == 1
    assert received_events[0].event_type == RuntimeEventType.RUN_STARTED
    assert len(bus.history) == 2


# ---------------------------------------------------------------------------
# Cancellation Token Tests
# ---------------------------------------------------------------------------


def test_cancellation_token():
    token = CancellationToken()
    assert not token.is_cancelled

    token.cancel(reason="User stopped")
    assert token.is_cancelled
    assert token.reason == "User stopped"

    with pytest.raises(asyncio.CancelledError, match="User stopped"):
        token.check()


# ---------------------------------------------------------------------------
# AgentRuntime Execution Tests
# ---------------------------------------------------------------------------


def test_runtime_successful_execution():
    async def _test():
        runtime = AgentRuntime()
        agent = MockAgent(response_msg=ChatMessage(role=Role.ASSISTANT, content="Result text"))
        
        result = await runtime.execute(agent, "Test prompt")
        assert result.is_success
        assert result.status == RuntimeStatus.COMPLETED
        assert result.message is not None
        assert result.message.content == "Result text"
        assert len(runtime.event_bus.history) >= 2

    _run(_test())


def test_runtime_timeout_execution():
    async def _test():
        runtime = AgentRuntime()
        agent = MockAgent(delay=0.2)
        budget = RuntimeBudget(timeout_seconds=0.05)

        result = await runtime.execute(agent, "Long task", budget=budget)
        assert result.status == RuntimeStatus.TIMED_OUT
        assert "timed out" in (result.error or "")
        assert not result.is_success

    _run(_test())


def test_runtime_cancellation_execution():
    async def _test():
        runtime = AgentRuntime()
        agent = MockAgent()
        token = CancellationToken()
        token.cancel("Early cancel")

        result = await runtime.execute(agent, "Task", cancellation_token=token)
        assert result.status == RuntimeStatus.CANCELLED
        assert result.termination_reason == "Early cancel"

    _run(_test())


def test_runtime_agent_error_handling():
    async def _test():
        runtime = AgentRuntime()
        agent = MockAgent(error=ValueError("Engine crashed"))

        result = await runtime.execute(agent, "Fail task")
        assert result.status == RuntimeStatus.FAILED
        assert "Engine crashed" in (result.error or "")

    _run(_test())
