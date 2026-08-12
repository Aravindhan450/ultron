"""
FIX #7 section 7.1 — Agent contract + lifecycle: deterministic tests.

Covers every required area without an LLM:

- lifecycle transitions (valid path + every valid edge)
- invalid transitions rejected (terminal states, illegal jumps)
- AgentResult (statuses, structured fields, terminal semantics)
- ExecutionContext (deny-by-default tools, permissions, cancellation)
- ExecutionBudget (limits, exhaustion, timeout)
- cancellation / failure / blocked states
- AgentState lifecycle methods + audit metadata
- agent completion does NOT complete the TaskState (separate concepts)
- the Agent contract (a minimal implementation runs + returns a result)

No real tools are executed and no repo files are touched.
"""

import asyncio

import pytest

from ultron.core.orchestration import (
    ACTIVE_STATUSES,
    TERMINAL_STATUSES,
    TRANSITIONS,
    Agent,
    AgentIdentity,
    AgentResult,
    AgentResultStatus,
    AgentState,
    AgentStatus,
    AgentType,
    ExecutionBudget,
    ExecutionContext,
    assert_transition,
    can_transition,
    transitions_from,
)
from ultron.core.types import TaskState, TaskStatus


def _run(coro):
    """Runs one async test body via asyncio.run (no pytest-asyncio dep)."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Lifecycle: valid transitions
# ---------------------------------------------------------------------------


def test_lifecycle_full_success_path():
    state = AgentState(
        task_id="task-1",
        identity=AgentIdentity(agent_id="a1", agent_type=AgentType.CODING),
        objective="implement health endpoint",
        context=ExecutionContext(task_id="task-1", agent_id="a1"),
    )
    assert state.status is AgentStatus.PENDING
    state.assign(state.context)
    assert state.status is AgentStatus.ASSIGNED
    state.start()
    assert state.status is AgentStatus.RUNNING
    state.complete(
        AgentResult(status=AgentResultStatus.SUCCESS, summary="endpoint added")
    )
    assert state.status is AgentStatus.COMPLETED
    assert state.is_terminal
    assert not state.is_active
    assert state.result is not None and state.result.is_success


def test_lifecycle_wait_resume_cycle():
    state = AgentState(
        task_id="t",
        identity=AgentIdentity(agent_id="a", agent_type=AgentType.CODING),
        objective="o",
        context=ExecutionContext(task_id="t", agent_id="a"),
    )
    state.assign(state.context)
    state.start()
    state.wait(result=AgentResult(status=AgentResultStatus.NEEDS_INPUT, summary="confirm"))
    assert state.status is AgentStatus.WAITING
    assert state.result.status is AgentResultStatus.NEEDS_INPUT
    # WAITING -> COMPLETED is invalid; must resume first.
    with pytest.raises(ValueError):
        state.complete(AgentResult(status=AgentResultStatus.SUCCESS, summary="done"))
    state.resume()
    assert state.status is AgentStatus.RUNNING
    state.complete(AgentResult(status=AgentResultStatus.SUCCESS, summary="done"))
    assert state.status is AgentStatus.COMPLETED


def test_lifecycle_failure_states_from_any_active_state():
    # Failure states are reachable from ANY active state, including PENDING
    # (e.g. an agent cancelled or blocked before it ever starts).
    for failure in (AgentStatus.FAILED, AgentStatus.BLOCKED, AgentStatus.CANCELLED):
        for active in ACTIVE_STATUSES:
            state = AgentState(
                task_id="t",
                identity=AgentIdentity(agent_id="a", agent_type=AgentType.RESEARCH),
                objective="o",
                context=ExecutionContext(task_id="t", agent_id="a"),
            )
            if active is AgentStatus.ASSIGNED:
                state.assign(state.context)
            elif active is AgentStatus.RUNNING:
                state.assign(state.context)
                state.start()
            elif active is AgentStatus.WAITING:
                state.assign(state.context)
                state.start()
                state.wait()
            assert can_transition(state.status, failure), (
                f"{state.status.value} -> {failure.value} should be legal"
            )
            if failure is AgentStatus.FAILED:
                state.fail(reason=f"failed from {active.value}")
            elif failure is AgentStatus.BLOCKED:
                state.block(reason=f"blocked from {active.value}")
            else:
                state.cancel(reason=f"cancelled from {active.value}")
            assert state.status is failure
            assert state.result is not None and state.result.status.value == failure.value


# ---------------------------------------------------------------------------
# Lifecycle: invalid transitions
# ---------------------------------------------------------------------------


def test_invalid_transitions_rejected():
    state = AgentState(
        task_id="t",
        identity=AgentIdentity(agent_id="a", agent_type=AgentType.REVIEWER),
        objective="o",
        context=ExecutionContext(task_id="t", agent_id="a"),
    )
    # Cannot start before assignment.
    with pytest.raises(ValueError):
        state.start()
    # Cannot skip ASSIGNED and jump straight to completion from PENDING.
    with pytest.raises(ValueError):
        state.complete(AgentResult(status=AgentResultStatus.SUCCESS, summary="x"))
    state.assign(state.context)
    # Cannot wait before running.
    with pytest.raises(ValueError):
        state.wait()
    state.start()
    # Cannot assign twice.
    with pytest.raises(ValueError):
        state.assign(state.context)
    # complete() with a non-SUCCESS result is a state corruption, rejected.
    with pytest.raises(ValueError):
        state.complete(AgentResult(status=AgentResultStatus.FAILED, summary="x"))
    state.complete(AgentResult(status=AgentResultStatus.SUCCESS, summary="ok"))


def test_terminal_states_accept_no_transitions():
    for terminal in TERMINAL_STATUSES:
        assert TRANSITIONS[terminal] == frozenset()
        for target in AgentStatus:
            assert not can_transition(terminal, target), (
                f"{terminal.value} -> {target.value} must be invalid"
            )
            with pytest.raises(ValueError):
                assert_transition(terminal, target)


def test_transition_table_sanity():
    # Every status has a table entry; every source is an AgentStatus member.
    assert set(TRANSITIONS) == set(AgentStatus)
    # PENDING: ASSIGNED, or terminate via a failure state (incl. CANCELLED
    # before ever starting). COMPLETED is only reachable from RUNNING.
    assert transitions_from(AgentStatus.PENDING) == frozenset(
        {
            AgentStatus.ASSIGNED,
            AgentStatus.FAILED,
            AgentStatus.BLOCKED,
            AgentStatus.CANCELLED,
        }
    )
    # RUNNING is the only state that may complete.
    for status in AgentStatus:
        if status is AgentStatus.RUNNING:
            assert can_transition(status, AgentStatus.COMPLETED)
        else:
            assert not can_transition(status, AgentStatus.COMPLETED)
    # WAITING may only resume or fail/block/cancel.
    assert AgentStatus.RUNNING in transitions_from(AgentStatus.WAITING)
    assert AgentStatus.WAITING not in transitions_from(AgentStatus.WAITING)


# ---------------------------------------------------------------------------
# AgentResult
# ---------------------------------------------------------------------------


def test_agent_result_structured_and_terminal():
    result = AgentResult(
        status=AgentResultStatus.SUCCESS,
        summary="implemented feature",
        artifacts=["src/feature.py"],
        evidence=["pytest: 12 passed"],
        changed_files=["src/feature.py"],
        tests=["tests/test_feature.py"],
        blockers=[],
        recommendations=["run full suite"],
        metadata={"duration_s": 2.5},
    )
    assert result.is_success
    assert result.is_terminal
    line = result.to_prompt_line()
    assert "[success]" in line
    assert "implemented feature" in line
    assert "1 file(s) changed" in line

    needs_input = AgentResult(status=AgentResultStatus.NEEDS_INPUT, summary="which port?")
    assert not needs_input.is_terminal


def test_agent_result_roundtrip_json():
    result = AgentResult(
        status=AgentResultStatus.BLOCKED,
        summary="security block",
        blockers=["denied by boundary"],
        changed_files=[],
        metadata={"tier": "critical"},
    )
    restored = AgentResult.model_validate_json(result.model_dump_json())
    assert restored == result
    assert restored.status is AgentResultStatus.BLOCKED


# ---------------------------------------------------------------------------
# ExecutionContext
# ---------------------------------------------------------------------------


def test_execution_context_deny_by_default_tools():
    ctx = ExecutionContext(task_id="t", agent_id="a")
    assert not ctx.allows_tool("run_command")
    assert not ctx.allows_tool("read_file")
    ctx.allowed_tools = ["read_file", "find_definition"]
    assert ctx.allows_tool("read_file")
    assert not ctx.allows_tool("run_command")  # still denied


def test_execution_context_cancellation():
    ctx = ExecutionContext(task_id="t", agent_id="a")
    assert not ctx.is_cancelled
    ctx.request_cancel()
    assert ctx.is_cancelled


def test_execution_context_scoped_view_has_no_taskstate():
    # The context references the task by id only — no TaskState inside.
    ctx = ExecutionContext(task_id="task-9", agent_id="a")
    assert ctx.task_id == "task-9"
    assert not hasattr(ctx, "goal")
    assert not hasattr(ctx, "requirements")


def test_execution_context_summary_and_permissions():
    ctx = ExecutionContext(
        task_id="t",
        agent_id="a",
        workspace="/tmp/proj",
        allowed_tools=["read_file"],
        permissions={"security_mode": "interactive"},
        current_plan_step=2,
    )
    line = ctx.to_prompt_line()
    assert "task=t" in line
    assert "/tmp/proj" in line
    assert "step 2" in line
    assert ctx.permissions["security_mode"] == "interactive"


# ---------------------------------------------------------------------------
# ExecutionBudget
# ---------------------------------------------------------------------------


def test_budget_records_and_exhausts():
    budget = ExecutionBudget(max_steps=2, max_tool_calls=3)
    assert not budget.is_exhausted()
    budget.record_step()
    budget.record_step()
    assert budget.is_exhausted()  # steps exhausted
    assert budget.steps_remaining() == 0

    fresh = ExecutionBudget(max_steps=100, max_tool_calls=2)
    fresh.record_tool_call()
    assert not fresh.is_exhausted()
    fresh.record_tool_call()
    assert fresh.is_exhausted()
    assert fresh.tool_calls_remaining() == 0


def test_budget_timeout():
    from datetime import UTC, datetime, timedelta

    budget = ExecutionBudget(timeout_seconds=10, started_at=datetime.now(UTC))
    assert not budget.timed_out()
    later = datetime.now(UTC) + timedelta(seconds=30)
    assert budget.timed_out(now=later)
    assert not ExecutionBudget(timeout_seconds=10).timed_out()  # not started


def test_budget_summary_bounded():
    budget = ExecutionBudget(max_steps=5, max_tool_calls=10, timeout_seconds=60)
    budget.record_step()
    assert "1/5" in budget.summary()
    assert "10" in budget.summary()


# ---------------------------------------------------------------------------
# AgentState: metadata / audit / reporting
# ---------------------------------------------------------------------------


def test_metadata_audit_trail_records_transitions():
    state = AgentState(
        task_id="t",
        identity=AgentIdentity(agent_id="a", agent_type=AgentType.TEST_QA),
        objective="run tests",
        context=ExecutionContext(task_id="t", agent_id="a"),
    )
    state.assign(state.context, reason="handed test plan")
    state.start()
    state.wait(reason="needs confirmation")
    state.resume()
    state.complete(
        AgentResult(status=AgentResultStatus.SUCCESS, summary="tests pass"),
        reason="all green",
    )
    history = [c.to_status for c in state.metadata.status_history]
    assert history == [
        AgentStatus.ASSIGNED,
        AgentStatus.RUNNING,
        AgentStatus.WAITING,
        AgentStatus.RUNNING,
        AgentStatus.COMPLETED,
    ]
    assert state.metadata.attempts == 2  # started, then resumed after the wait
    assert state.metadata.started_at is not None
    assert state.metadata.completed_at is not None
    assert state.metadata.elapsed_seconds is not None
    assert state.metadata.status_history[0].reason == "handed test plan"


def test_resume_counts_as_new_attempt():
    state = AgentState(
        task_id="t",
        identity=AgentIdentity(agent_id="a", agent_type=AgentType.CODING),
        objective="o",
        context=ExecutionContext(task_id="t", agent_id="a"),
    )
    state.assign(state.context)
    state.start()
    state.wait()
    state.resume()
    assert state.metadata.attempts == 2  # started twice
    state.complete(AgentResult(status=AgentResultStatus.SUCCESS, summary="ok"))
    assert state.summary().startswith("AgentState(coding:a")
    assert "status=completed" in state.summary()
    assert "result=success" in state.summary()


def test_agent_state_roundtrip_json():
    state = AgentState(
        task_id="t",
        identity=AgentIdentity(agent_id="a", agent_type=AgentType.SECURITY),
        objective="audit secrets",
        context=ExecutionContext(task_id="t", agent_id="a", allowed_tools=["read_file"]),
    )
    state.assign(state.context)
    state.start()
    state.fail(reason="no files to audit")
    restored = AgentState.model_validate_json(state.model_dump_json())
    assert restored.status is AgentStatus.FAILED
    assert restored.result is not None
    assert restored.result.status is AgentResultStatus.FAILED
    assert restored.identity.agent_type is AgentType.SECURITY
    assert len(restored.metadata.status_history) == 3  # assigned/started/failed


# ---------------------------------------------------------------------------
# Agent completion vs task completion (the critical separation)
# ---------------------------------------------------------------------------


def test_agent_completion_never_completes_taskstate():
    task = TaskState(goal="implement authentication")
    task.add_requirement("auth works")
    assert task.status is TaskStatus.TASK_STARTED

    state = AgentState(
        task_id="task-1",
        identity=AgentIdentity(agent_id="coder", agent_type=AgentType.CODING),
        objective="implement auth",
        context=ExecutionContext(task_id="task-1", agent_id="coder"),
    )
    state.assign(state.context)
    state.start()
    state.complete(AgentResult(status=AgentResultStatus.SUCCESS, summary="auth added"))

    # The agent run is complete, but the TASK is untouched: it has not been
    # marked complete, no requirement was satisfied, no tool executed.
    assert state.status is AgentStatus.COMPLETED
    assert task.status is TaskStatus.TASK_STARTED
    assert task.remaining_requirements()
    assert not task.is_complete()
    assert task.execution_history == []


def test_failed_agent_does_not_fail_task():
    task = TaskState(goal="fix login")
    state = AgentState(
        task_id="task-1",
        identity=AgentIdentity(agent_id="qa", agent_type=AgentType.TEST_QA),
        objective="verify login",
        context=ExecutionContext(task_id="task-1", agent_id="qa"),
    )
    state.assign(state.context)
    state.start()
    state.fail(reason="tests still red")
    assert state.status is AgentStatus.FAILED
    assert task.status is TaskStatus.TASK_STARTED  # task is not auto-failed
    assert task.errors == []


# ---------------------------------------------------------------------------
# The Agent contract
# ---------------------------------------------------------------------------


class _ProbeAgent(Agent):
    """Minimal contract implementation for tests — no real work."""

    def __init__(self, identity: AgentIdentity, mode: str = "ok") -> None:
        super().__init__(identity)
        self.mode = mode

    async def execute(self, objective: str, context: ExecutionContext) -> AgentResult:
        if context.is_cancelled:
            return AgentResult(status=AgentResultStatus.CANCELLED, summary="cancelled")
        if self.mode == "fail":
            return AgentResult(
                status=AgentResultStatus.FAILED,
                summary=f"could not {objective}",
                blockers=["probe failure"],
            )
        if self.mode == "block":
            return AgentResult(status=AgentResultStatus.BLOCKED, summary="probe blocked")
        if self.mode == "input":
            return AgentResult(
                status=AgentResultStatus.NEEDS_INPUT, summary="need more info"
            )
        return AgentResult(
            status=AgentResultStatus.SUCCESS,
            summary=f"did {objective}",
            changed_files=["src/probe.py"],
            evidence=["probe ran"],
        )


class _CrashAgent(Agent):
    """Contract implementation that raises — the wrapper must contain it."""

    def __init__(self) -> None:
        super().__init__(AgentIdentity(agent_id="crash", agent_type=AgentType.CODING))

    async def execute(self, objective: str, context: ExecutionContext) -> AgentResult:
        raise RuntimeError("boom")


def test_contract_implementation_runs_to_success():
    async def body():
        identity = AgentIdentity(agent_id="probe", agent_type=AgentType.CODING)
        agent = _ProbeAgent(identity)
        state = AgentState(
            task_id="t",
            identity=identity,
            objective="implement probe",
            context=ExecutionContext(task_id="t", agent_id="probe"),
        )
        result = await agent.run_with_state(state)
        assert result.is_success
        assert state.status is AgentStatus.COMPLETED
        assert state.result is result
        assert "src/probe.py" in result.changed_files
        assert result.evidence == ["probe ran"]

    _run(body())


def test_contract_implementation_failure_and_blocked():
    async def body():
        identity = AgentIdentity(agent_id="probe", agent_type=AgentType.RESEARCH)

        state = AgentState(
            task_id="t",
            identity=identity,
            objective="research",
            context=ExecutionContext(task_id="t", agent_id="probe"),
        )
        result = await _ProbeAgent(identity, mode="fail").run_with_state(state)
        assert result.status is AgentResultStatus.FAILED
        assert state.status is AgentStatus.FAILED
        assert state.result.blockers == ["probe failure"]

        blocked = AgentState(
            task_id="t",
            identity=identity,
            objective="research",
            context=ExecutionContext(task_id="t", agent_id="probe"),
        )
        result = await _ProbeAgent(identity, mode="block").run_with_state(blocked)
        assert result.status is AgentResultStatus.BLOCKED
        assert blocked.status is AgentStatus.BLOCKED

    _run(body())


def test_contract_needs_input_pauses_state():
    async def body():
        identity = AgentIdentity(agent_id="probe", agent_type=AgentType.REVIEWER)
        state = AgentState(
            task_id="t",
            identity=identity,
            objective="review",
            context=ExecutionContext(task_id="t", agent_id="probe"),
        )
        result = await _ProbeAgent(identity, mode="input").run_with_state(state)
        assert result.status is AgentResultStatus.NEEDS_INPUT
        assert state.status is AgentStatus.WAITING  # paused, not finished
        assert not state.is_terminal

    _run(body())


def test_contract_honors_cancellation():
    async def body():
        identity = AgentIdentity(agent_id="probe", agent_type=AgentType.CODING)
        # Cancellation requested mid-run: the agent checks the flag and
        # returns CANCELLED instead of doing more work.
        state = AgentState(
            task_id="t",
            identity=identity,
            objective="long work",
            context=ExecutionContext(task_id="t", agent_id="probe"),
        )
        state.assign(state.context)
        state.start()
        state.context.request_cancel()
        result = await _ProbeAgent(identity).execute("long work", state.context)
        assert result.status is AgentResultStatus.CANCELLED
        # A FRESH state run through run_with_state with a pre-cancelled
        # context routes the whole lifecycle to CANCELLED.
        fresh = AgentState(
            task_id="t",
            identity=identity,
            objective="long work",
            context=ExecutionContext(task_id="t", agent_id="probe"),
        )
        fresh.context.request_cancel()
        result = await _ProbeAgent(identity).run_with_state(fresh)
        assert result.status is AgentResultStatus.CANCELLED
        assert fresh.status is AgentStatus.CANCELLED

    _run(body())


def test_contract_converts_exception_to_failed_result():
    async def body():
        state = AgentState(
            task_id="t",
            identity=AgentIdentity(agent_id="crash", agent_type=AgentType.CODING),
            objective="work",
            context=ExecutionContext(task_id="t", agent_id="crash"),
        )
        result = await _CrashAgent().run_with_state(state)
        assert result.status is AgentResultStatus.FAILED
        assert "boom" in result.summary
        assert state.status is AgentStatus.FAILED

    _run(body())


def test_run_with_state_resumes_waiting_state():
    async def body():
        identity = AgentIdentity(agent_id="probe", agent_type=AgentType.CODING)
        state = AgentState(
            task_id="t",
            identity=identity,
            objective="work",
            context=ExecutionContext(task_id="t", agent_id="probe"),
        )
        # First run pauses on NEEDS_INPUT.
        result = await _ProbeAgent(identity, mode="input").run_with_state(state)
        assert result.status is AgentResultStatus.NEEDS_INPUT
        assert state.status is AgentStatus.WAITING
        # The SAME state resumes through run_with_state — it must not crash
        # at assign() (the one-shot-assign pitfall): WAITING -> RUNNING.
        result = await _ProbeAgent(identity).run_with_state(state)
        assert result.status is AgentResultStatus.SUCCESS
        assert state.status is AgentStatus.COMPLETED
        # A terminal state is rejected outright.
        with pytest.raises(ValueError):
            await _ProbeAgent(identity).run_with_state(state)

    _run(body())


def test_terminal_methods_validate_explicit_result_status():
    state = AgentState(
        task_id="t",
        identity=AgentIdentity(agent_id="a", agent_type=AgentType.CODING),
        objective="o",
        context=ExecutionContext(task_id="t", agent_id="a"),
    )
    state.assign(state.context)
    state.start()
    # fail() with a SUCCESS result contradicts the lifecycle — rejected.
    with pytest.raises(ValueError):
        state.fail(AgentResult(status=AgentResultStatus.SUCCESS, summary="x"))
    state.fail(AgentResult(status=AgentResultStatus.FAILED, summary="x"))
    assert state.status is AgentStatus.FAILED

    blocked = AgentState(
        task_id="t",
        identity=AgentIdentity(agent_id="a", agent_type=AgentType.CODING),
        objective="o",
        context=ExecutionContext(task_id="t", agent_id="a"),
    )
    blocked.assign(blocked.context)
    blocked.start()
    with pytest.raises(ValueError):
        blocked.block(AgentResult(status=AgentResultStatus.CANCELLED, summary="x"))
    blocked.block(AgentResult(status=AgentResultStatus.BLOCKED, summary="x"))
    assert blocked.status is AgentStatus.BLOCKED


def test_cancel_never_records_success_result():
    # A cancelled run can never store a SUCCESS result, even when the agent
    # ignored the cancellation flag and returned SUCCESS.
    state = AgentState(
        task_id="t",
        identity=AgentIdentity(agent_id="a", agent_type=AgentType.CODING),
        objective="o",
        context=ExecutionContext(task_id="t", agent_id="a"),
    )
    state.assign(state.context)
    state.start()
    state.context.request_cancel()
    state.cancel(AgentResult(status=AgentResultStatus.SUCCESS, summary="agent claims done"))
    assert state.status is AgentStatus.CANCELLED
    assert state.result is not None
    assert state.result.status is AgentResultStatus.CANCELLED  # coerced, not SUCCESS
    assert "agent claims done" in state.result.summary  # evidence preserved


def test_run_with_state_cancellation_overrides_success():
    class _IgnorantAgent(Agent):
        """Agent that never checks the cancellation flag."""

        def __init__(self) -> None:
            super().__init__(
                AgentIdentity(agent_id="ignorant", agent_type=AgentType.CODING)
            )

        async def execute(self, objective: str, context: ExecutionContext) -> AgentResult:
            return AgentResult(status=AgentResultStatus.SUCCESS, summary="done anyway")

    async def body():
        state = AgentState(
            task_id="t",
            identity=AgentIdentity(agent_id="ignorant", agent_type=AgentType.CODING),
            objective="work",
            context=ExecutionContext(task_id="t", agent_id="ignorant"),
        )
        state.context.request_cancel()
        result = await _IgnorantAgent().run_with_state(state)
        assert state.status is AgentStatus.CANCELLED
        assert result.status is AgentResultStatus.CANCELLED
        assert state.result.status is AgentResultStatus.CANCELLED

    _run(body())


def test_wait_rejects_non_needs_input_result():
    state = AgentState(
        task_id="t",
        identity=AgentIdentity(agent_id="a", agent_type=AgentType.CODING),
        objective="o",
        context=ExecutionContext(task_id="t", agent_id="a"),
    )
    state.assign(state.context)
    state.start()
    # A paused run may never carry a terminal result status.
    with pytest.raises(ValueError):
        state.wait(AgentResult(status=AgentResultStatus.SUCCESS, summary="x"))
    state.wait(AgentResult(status=AgentResultStatus.NEEDS_INPUT, summary="ask"))
    assert state.status is AgentStatus.WAITING


def test_assign_rejects_context_for_other_task():
    state = AgentState(
        task_id="task-1",
        identity=AgentIdentity(agent_id="a", agent_type=AgentType.CODING),
        objective="o",
    )
    with pytest.raises(ValueError):
        state.assign(ExecutionContext(task_id="task-2", agent_id="a"))
    assert state.status is AgentStatus.PENDING  # unchanged


def test_agent_identity_labels():
    identity = AgentIdentity(agent_id="r1", agent_type=AgentType.RESEARCH)
    assert identity.label == "research:r1"
    assert "research" in identity.to_prompt_line()
    assert "r1" in identity.to_prompt_line()


def test_agent_types_are_extensible_vocabulary():
    # The six standard roles exist; the tuple exposes them for discovery.
    assert AgentType.SUPERVISOR.value == "supervisor"
    assert AgentType.CODING.value == "coding"
    assert AgentType.TEST_QA.value == "test_qa"
    assert AgentType.REVIEWER.value == "reviewer"
    assert AgentType.SECURITY.value == "security"
    assert AgentType.RESEARCH.value == "research"
    assert len(set(AgentType)) == len(AgentType)  # unique values
