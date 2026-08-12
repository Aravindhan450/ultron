"""
FIX #7 — Section 7.4: Supervisor -> specialist delegation.

Deterministic tests (fake specialists; no live LLM, no real tools):

- DelegationRequest creation (fields, budget copy, timeout folding, frozen
  permission profile, deterministic task association)
- delegation lifecycle transitions (valid + invalid, result-status lock)
- dispatch: completion / failure / needs-input-resume / timeout / cancel
- invalid agent type -> KeyError; missing agent factory -> ValueError
- permission propagation (researcher read-only, coder confirm)
- context isolation: no transcript, no execution history, no other agents'
  internal trajectory — only the task brief + artifact summaries
- artifact return + ArtifactStore persistence
- TaskState update (observation / error; never a completion claim)
- Supervisor.decide mapping
"""

from __future__ import annotations

import asyncio

import pytest

from ultron.core.memory.models import MemoryConfidence
from ultron.core.orchestration import (
    DEFAULT_REGISTRY,
    Agent,
    AgentResult,
    AgentResultStatus,
    AgentStatus,
    AgentType,
    ArtifactStore,
    DelegationRequest,
    ExecutionContext,
    PermissionCategory,
    ResearchFinding,
    Supervisor,
    SupervisorDecision,
    classify_tool,
    task_brief,
    task_key,
)
from ultron.core.types import TaskState, TaskStatus
from ultron.security import Decision


class StubBoundary:
    """Deterministic stand-in for the security boundary: CONFIRM everything."""

    def check(self, action_type, target="", content=None):
        from types import SimpleNamespace

        return SimpleNamespace(decision=Decision.CONFIRM, reason="stub boundary")


# ---------------------------------------------------------------------------
# Fake specialists
# ---------------------------------------------------------------------------


def _research_artifact(context: ExecutionContext, extra: str = "") -> ResearchFinding:
    return ResearchFinding(
        task_id=context.task_id,
        agent_id=context.agent_id,
        summary=f"authentication lives in src/auth/service.py{extra}",
        evidence=["symbols: AuthService", "files: src/auth/service.py"],
        source="fake_researcher",
        confidence=MemoryConfidence.DIRECT_OBSERVATION,
        related_files=["src/auth/service.py"],
        related_symbols=["AuthService"],
    )


class FakeAgent(Agent):
    """Deterministic specialist: records internal reasoning that must never
    leak past its structured artifact/result."""

    def __init__(self, identity, behavior: str = "success") -> None:
        super().__init__(identity)
        self.behavior = behavior
        self.trajectory: list[str] = []

    async def execute(self, objective: str, context: ExecutionContext) -> AgentResult:
        self.trajectory.append(f"internal reasoning step: {objective} / {context.task_id}")
        self.trajectory.append(f"internal reasoning step: write_verdict={context.check_action('write_file')}")
        if self.behavior == "success":
            return AgentResult(status=AgentResultStatus.SUCCESS, summary=f"{objective} done")
        if self.behavior == "artifact":
            return _research_artifact(context).to_agent_result()
        if self.behavior == "needs_input":
            return AgentResult(
                status=AgentResultStatus.NEEDS_INPUT,
                summary="need confirmation to proceed",
            )
        if self.behavior == "failed":
            return AgentResult(
                status=AgentResultStatus.FAILED,
                summary="could not locate the authentication module",
            )
        if self.behavior == "blocked":
            return AgentResult(
                status=AgentResultStatus.BLOCKED,
                summary="security policy blocks this investigation",
            )
        if self.behavior == "slow":
            await asyncio.sleep(5.0)
            return AgentResult(status=AgentResultStatus.SUCCESS, summary="eventually done")
        if self.behavior == "cancellable":
            while not context.is_cancelled:
                await asyncio.sleep(0.005)
            # The agent ignores the cancellation for its own result; the
            # contract/supervisor must still record CANCELLED.
            return AgentResult(status=AgentResultStatus.SUCCESS, summary="ignored cancel")
        raise AssertionError(f"unknown fake behavior {self.behavior}")


def _make_supervisor(
    behavior: str = "success",
    store: ArtifactStore | None = None,
) -> Supervisor:
    def factory(spec, state) -> FakeAgent:
        return FakeAgent(state.identity, behavior=behavior)

    return Supervisor(registry=DEFAULT_REGISTRY, agent_factory=factory, store=store)


def _task(goal: str = "Understand authentication.") -> TaskState:
    task = TaskState(goal=goal)
    task.add_requirement("Authentication flow is documented")
    return task


# ---------------------------------------------------------------------------
# Delegation creation
# ---------------------------------------------------------------------------


def test_delegation_creation_fields():
    task = _task()
    supervisor = _make_supervisor()
    req = supervisor.create_delegation(
        task,
        "researcher",
        "Locate the authentication implementation",
        constraints=["read-only"],
        expected_output="ResearchFinding with affected files",
        timeout_seconds=120,
    )
    assert isinstance(req, DelegationRequest)
    assert req.status is AgentStatus.PENDING
    assert req.delegation_id
    assert req.task_id == task_key(task)
    assert req.parent_task_id is None
    assert req.agent_type is AgentType.RESEARCH
    assert req.objective == "Locate the authentication implementation"
    assert req.constraints == ["read-only"]
    assert req.expected_output.startswith("ResearchFinding")
    # Budget is a per-run copy of the researcher's spec budget, with the
    # timeout folded in.
    spec_budget = DEFAULT_REGISTRY.get("researcher").max_budget
    assert req.budget.max_steps == spec_budget.max_steps
    assert req.timeout_seconds == 120
    assert req.budget.timeout_seconds == 120
    # The frozen permission profile is attached (runtime-controlled).
    assert req.permissions is DEFAULT_REGISTRY.get("researcher").permissions
    assert supervisor.get_delegation(req.delegation_id) is req


def test_delegation_creation_invalid_agent_rejected():
    supervisor = _make_supervisor()
    with pytest.raises(KeyError):
        supervisor.create_delegation(_task(), "llm-agent", "do anything")


def test_delegation_default_id_and_ownership():
    task = _task()
    supervisor = _make_supervisor()
    a = supervisor.create_delegation(task, "researcher", "x")
    b = supervisor.create_delegation(task, "coder", "y")
    assert a.delegation_id != b.delegation_id
    assert a.task_id == b.task_id == task_key(task)


# ---------------------------------------------------------------------------
# Delegation lifecycle
# ---------------------------------------------------------------------------


def test_delegation_lifecycle_valid_and_invalid_transitions():
    req = DelegationRequest(task_id="t", agent_type=AgentType.RESEARCH, objective="o")
    assert req.status is AgentStatus.PENDING

    with pytest.raises(ValueError):  # PENDING -> RUNNING is invalid
        req.start()
    with pytest.raises(ValueError):  # PENDING -> COMPLETED is invalid
        req.complete(AgentResult(status=AgentResultStatus.SUCCESS, summary=""))

    req.assign()
    assert req.status is AgentStatus.ASSIGNED
    req.start()
    assert req.status is AgentStatus.RUNNING
    ok = AgentResult(status=AgentResultStatus.SUCCESS, summary="done")
    req.complete(ok)
    assert req.status is AgentStatus.COMPLETED
    assert req.result is ok
    assert len(req.status_history) == 3  # assigned, running, completed

    # Terminal states accept nothing.
    with pytest.raises(ValueError):
        req.cancel()
    with pytest.raises(ValueError):
        req.fail()


def test_delegation_result_status_locked_to_lifecycle():
    req = DelegationRequest(task_id="t", agent_type=AgentType.RESEARCH, objective="o")
    req.assign()
    req.start()
    # A run may only complete with a SUCCESS result…
    with pytest.raises(ValueError):
        req.complete(AgentResult(status=AgentResultStatus.FAILED, summary="no"))
    # …fail only with a FAILED result…
    with pytest.raises(ValueError):
        req.fail(AgentResult(status=AgentResultStatus.SUCCESS, summary="no"))
    # …pause only with a NEEDS_INPUT result…
    with pytest.raises(ValueError):
        req.wait(AgentResult(status=AgentResultStatus.SUCCESS, summary="no"))
    # …and cancellation always forces CANCELLED, never a success.
    req.cancel(AgentResult(status=AgentResultStatus.SUCCESS, summary="ignored"))
    assert req.status is AgentStatus.CANCELLED
    assert req.result.status is AgentResultStatus.CANCELLED


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def test_dispatch_success_completes(tmp_path):
    task = _task()
    store = ArtifactStore(tmp_path)
    supervisor = _make_supervisor(behavior="artifact", store=store)
    req = supervisor.create_delegation(
        task, "researcher", "Locate the authentication implementation"
    )

    async def body():
        return await supervisor.dispatch(req, workspace=str(tmp_path), task_state=task)

    asyncio.run(body())

    # Delegation + agent lifecycle both COMPLETED.
    assert req.status is AgentStatus.COMPLETED
    assert req.result is not None
    assert req.result.status is AgentResultStatus.SUCCESS
    assert isinstance(req.result.artifact, ResearchFinding)
    # The artifact was persisted and is owned by this task.
    stored = store.get(req.result.artifact.artifact_id)
    assert stored is not None and stored.owned_by(req.task_id, req.run_agent_id)
    # TaskState observed the outcome (but is NOT complete — invariant).
    assert task.last_observation.startswith("[delegate:research]")
    assert task.status is TaskStatus.TASK_STARTED
    assert not task.is_complete()
    # The specialist run itself reached COMPLETED.
    run = supervisor._runs[req.delegation_id]
    assert run.status is AgentStatus.COMPLETED
    # Continue decision.
    assert supervisor.decide(req) is SupervisorDecision.CONTINUE


def test_dispatch_failure_records_task_error(tmp_path):
    task = _task()
    supervisor = _make_supervisor(behavior="failed")
    req = supervisor.create_delegation(task, "researcher", "find auth")

    async def body():
        return await supervisor.dispatch(req, workspace=str(tmp_path), task_state=task)

    asyncio.run(body())

    assert req.status is AgentStatus.FAILED
    assert req.result.status is AgentResultStatus.FAILED
    assert task.errors, "a failed delegation must record a TaskError"
    assert "[delegate:research]" in task.errors[-1].message
    assert supervisor.decide(req) is SupervisorDecision.FAILED


def test_dispatch_needs_input_then_resume(tmp_path):
    task = _task()
    supervisor = _make_supervisor(behavior="needs_input")

    req = supervisor.create_delegation(task, "researcher", "understand auth")

    async def run_once():
        await supervisor.dispatch(req, workspace=str(tmp_path), task_state=task)

    asyncio.run(run_once())
    assert req.status is AgentStatus.WAITING
    assert req.result.status is AgentResultStatus.NEEDS_INPUT
    assert supervisor.decide(req) is SupervisorDecision.NEEDS_INPUT
    assert task.last_observation.startswith("[delegate:research] awaiting input")

    # Approval arrives: the supervisor re-dispatches and resumes the SAME run.
    supervisor.agent_factory = lambda spec, state: FakeAgent(state.identity, behavior="success")
    asyncio.run(run_once())
    assert req.status is AgentStatus.COMPLETED
    assert req.result.status is AgentResultStatus.SUCCESS
    assert len(req.status_history) >= 5  # …running, waiting, running, completed


def test_dispatch_timeout_marks_failed(tmp_path):
    task = _task()
    supervisor = _make_supervisor(behavior="slow")
    req = supervisor.create_delegation(task, "researcher", "slow research", timeout_seconds=1)

    async def body():
        return await supervisor.dispatch(req, workspace=str(tmp_path), task_state=task)

    asyncio.run(body())

    assert req.status is AgentStatus.FAILED
    assert req.result.metadata.get("timeout") is True
    assert "timed out" in req.result.summary
    assert "timeout" in (req.error or "")
    run = supervisor._runs[req.delegation_id]
    assert run.status is AgentStatus.CANCELLED  # the run itself was aborted
    assert run.context.is_cancelled


def test_dispatch_cancel_pending_and_running(tmp_path):
    task = _task()
    supervisor = _make_supervisor(behavior="cancellable")

    # Cancel before dispatch: immediate, no run spawned.
    req = supervisor.create_delegation(task, "researcher", "will be cancelled")
    supervisor.cancel_delegation(req, reason="pre-run cancel")
    assert req.status is AgentStatus.CANCELLED
    with pytest.raises(ValueError):
        asyncio.run(supervisor.dispatch(req, workspace=str(tmp_path), task_state=task))

    # Cancel mid-run: the specialist stops at its next checkpoint.
    req2 = supervisor.create_delegation(task, "researcher", "will be cancelled mid-run")

    async def body():
        run_task = asyncio.create_task(
            supervisor.dispatch(req2, workspace=str(tmp_path), task_state=task)
        )
        await asyncio.sleep(0.05)  # let the run start
        supervisor.cancel_delegation(req2, reason="live cancel")
        await run_task

    asyncio.run(body())
    assert req2.status is AgentStatus.CANCELLED
    assert req2.result.status is AgentResultStatus.CANCELLED
    assert supervisor.decide(req2) is SupervisorDecision.FAILED


def test_dispatch_rejects_terminal_or_unknown_state(tmp_path):
    supervisor = _make_supervisor()
    req = supervisor.create_delegation(_task(), "researcher", "x")
    req.cancel()
    with pytest.raises(ValueError):
        asyncio.run(supervisor.dispatch(req, workspace=str(tmp_path)))


def test_dispatch_without_factory_raises():
    task = _task()
    supervisor = Supervisor(registry=DEFAULT_REGISTRY)  # no agent_factory
    req = supervisor.create_delegation(task, "researcher", "x")
    with pytest.raises(ValueError, match="agent_factory"):
        asyncio.run(supervisor.dispatch(req))


# ---------------------------------------------------------------------------
# Permission propagation
# ---------------------------------------------------------------------------


def test_permission_propagation_researcher_read_only():
    # The scoped context the registry builds for a researcher run denies
    # writes and allows reads — the agent can never widen this.
    state = DEFAULT_REGISTRY.instantiate(
        AgentType.RESEARCH, "researcher-1", "t1", "understand auth"
    )
    ctx = state.context
    assert ctx.check_action("read_file") is Decision.ALLOW
    assert ctx.check_action("search_files") is Decision.ALLOW
    assert ctx.check_action("write_file") is Decision.DENY
    assert ctx.check_action("run_command") is Decision.DENY
    assert ctx.check_action("write_file", "x.py", "api_key=sk-123") is Decision.DENY


def test_permission_propagation_coder_confirm(tmp_path):
    state = DEFAULT_REGISTRY.instantiate(
        AgentType.CODING, "coder-1", "t1", "add endpoint"
    )
    ctx = state.context
    assert ctx.check_action("read_file") is Decision.ALLOW
    # State-change is CONFIRM-level, and the VERDICT is the boundary's — the
    # agent can never decide for itself (stub boundary says confirm).
    assert (
        ctx.check_action("write_file", "src/app.py", boundary=StubBoundary())
        is Decision.CONFIRM
    )
    # Test commands classify as TEST (not blind shell) and the coder's TEST
    # level is CONFIRM — subject to security, never automatic.
    assert classify_tool("run_command", "pytest") is PermissionCategory.TEST
    assert (
        DEFAULT_REGISTRY.get("coder").permissions.level_for(PermissionCategory.TEST)
        is Decision.CONFIRM
    )


def test_specialist_sees_scoped_permissions_during_run(tmp_path):
    task = _task()
    supervisor = _make_supervisor(behavior="success")
    req = supervisor.create_delegation(task, "researcher", "understand auth")

    async def body():
        await supervisor.dispatch(req, workspace=str(tmp_path), task_state=task)

    asyncio.run(body())
    run = supervisor._runs[req.delegation_id]
    assert run.context.allows_tool("read_file")
    assert not run.context.allows_tool("write_file")


# ---------------------------------------------------------------------------
# Context isolation
# ---------------------------------------------------------------------------


def test_task_brief_excludes_transcript_and_history():
    from ultron.core.types import ToolExecution

    task = _task()
    task.current_step = 1
    task.total_steps = 3
    task.execution_history.append(ToolExecution(tool_name="run_command", target="pytest"))
    task.execution_history.append(ToolExecution(tool_name="write_file", target="x.py"))
    brief = task_brief(task)
    text = str(brief)
    assert brief["goal"] == "Understand authentication."
    assert brief["requirements"] == ["Authentication flow is documented"]
    # The specialist gets the string-only brief, not the raw runtime surface.
    assert "run_command" not in text
    assert "write_file" not in text
    assert "execution_history" not in text
    assert "last_observation" not in text


def test_context_isolation_no_trajectory_no_transcript(tmp_path):
    task = _task()
    task.execution_history.append(_tool_exec("run_command", "pytest -q"))
    supervisor = _make_supervisor(behavior="artifact")

    # First delegation: the researcher's context contains only the brief.
    req1 = supervisor.create_delegation(task, "researcher", "understand auth")
    asyncio.run(supervisor.dispatch(req1, workspace=str(tmp_path), task_state=task))
    run1 = supervisor._runs[req1.delegation_id]
    ctx1 = run1.context.relevant_context
    # The specialist DOES receive the relevant TaskState subset (goal,
    # requirements) — isolation excludes the transcript/history, not the brief.
    assert ctx1["task"]["goal"] == "Understand authentication."
    assert ctx1["task"]["requirements"] == ["Authentication flow is documented"]
    assert ctx1["artifacts"] == []
    assert "pytest -q" not in str(ctx1)

    # The researcher recorded a large internal trajectory on ITSELF. The
    # next delegation receives ONLY the structured artifact — never the
    # trajectory, never the raw artifact dump, never the transcript.
    researcher = supervisor.agent_factory(
        DEFAULT_REGISTRY.get("researcher"), run1
    )
    trajectory = "\n".join(
        f"internal reasoning step {i}: searched {i} files" for i in range(200)
    )
    researcher.trajectory = trajectory.split("\n")

    artifact = req1.result.artifact
    req2 = supervisor.create_delegation(
        task,
        "coder",
        "implement refresh tokens",
        input_artifacts=[artifact],
        constraints=["follow the research"],
    )
    asyncio.run(supervisor.dispatch(req2, workspace=str(tmp_path), task_state=task))
    run2 = supervisor._runs[req2.delegation_id]
    ctx2 = run2.context.relevant_context
    ctx2_text = str(ctx2)

    # The structured summary + task brief travel; trajectory/transcript do not.
    assert "authentication lives in src/auth/service.py" in ctx2_text
    assert ctx2["task"]["goal"] == "Understand authentication."
    assert len(ctx2["artifacts"]) == 1
    assert ctx2["artifacts"][0]["related_files"] == ["src/auth/service.py"]
    assert ctx2["constraints"] == ["follow the research"]
    assert "internal reasoning step" not in ctx2_text
    assert "searched 199 files" not in ctx2_text
    assert "pytest -q" not in ctx2_text
    # And the second specialist never saw the first one's raw trajectory.
    assert "trajectory" not in ctx2_text


def _tool_exec(name: str, target: str):
    from ultron.core.types import ToolExecution

    return ToolExecution(tool_name=name, target=target)


# ---------------------------------------------------------------------------
# Artifact return + TaskState update
# ---------------------------------------------------------------------------


def test_artifact_return_owned_by_task(tmp_path):
    task = _task()
    store = ArtifactStore(tmp_path)
    supervisor = _make_supervisor(behavior="artifact", store=store)
    req = supervisor.create_delegation(task, "researcher", "understand auth")
    asyncio.run(supervisor.dispatch(req, workspace=str(tmp_path), task_state=task))

    artifact = req.result.artifact
    assert artifact.task_id == req.task_id
    assert artifact.agent_id == req.run_agent_id
    # Persisted artifact round-trips through the store.
    stored = store.get(artifact.artifact_id)
    assert stored is not None and stored.summary == artifact.summary
    assert store.load_for_task(req.task_id) == [stored]


def test_task_state_updated_but_never_completed(tmp_path):
    task = _task()
    supervisor = _make_supervisor(behavior="artifact")
    before = task.updated_at
    req = supervisor.create_delegation(task, "researcher", "understand auth")
    asyncio.run(supervisor.dispatch(req, workspace=str(tmp_path), task_state=task))

    assert task.last_observation is not None
    assert task.last_observation.startswith("[delegate:research]")
    assert task.updated_at >= before
    assert task.status is TaskStatus.TASK_STARTED  # never auto-completed
    assert task.errors == []  # success adds no errors


def test_decide_mapping(tmp_path):
    task = _task()
    supervisor = _make_supervisor()
    req = supervisor.create_delegation(task, "researcher", "x")
    # No result yet -> continue.
    assert supervisor.decide(req) is SupervisorDecision.CONTINUE

    for behavior, status, decision in (
        ("failed", AgentStatus.FAILED, SupervisorDecision.FAILED),
        ("blocked", AgentStatus.BLOCKED, SupervisorDecision.FAILED),
        ("needs_input", AgentStatus.WAITING, SupervisorDecision.NEEDS_INPUT),
        ("success", AgentStatus.COMPLETED, SupervisorDecision.CONTINUE),
    ):
        supervisor.agent_factory = lambda spec, state, b=behavior: FakeAgent(
            state.identity, behavior=b
        )
        r = supervisor.create_delegation(task, "researcher", "x")
        asyncio.run(supervisor.dispatch(r, workspace=str(tmp_path), task_state=task))
        assert r.status is status
        assert supervisor.decide(r) is decision
