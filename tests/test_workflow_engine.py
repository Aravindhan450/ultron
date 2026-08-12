"""
FIX #7 — Section 7.6: Workflow engine + sequential execution.

Deterministic tests (fake specialists; no live LLM, no real tools, no
network). Covers:

- workflow creation + structural validation (missing task id, duplicate ids,
  invalid/unregistered agent type, missing/self/circular/cross-workflow
  dependencies, empty workflows)
- dependency enforcement (a step waits until ALL dependencies complete)
- lifecycle transitions (workflow + step state machines; invalid transitions)
- sequential execution through the Supervisor (correct agent type,
  objective, constraints, expected output, task binding, artifacts)
- OrchestrationValidator gating (valid result -> step succeeds; invalid /
  false-completion / missing-evidence results -> step fails, workflow stops)
- failure handling (agent failed / test artifact with failures)
- blocked + needs-input (workflow WAITING, no further execution, resume)
- pause / resume / serialization reload (completed steps never repeat)
- cancellation (future steps stopped, artifacts preserved)
- completion + TaskState integration (workflow completes only when every
  step is done; TaskState.mark_complete() invoked through its own API and
  refused when requirements remain)
- context isolation (only dependency artifacts cross the agent boundary)
- observability / traceability (workflow -> task -> step -> delegation ->
  agent -> artifact)
"""

from __future__ import annotations

import asyncio

import pytest

from ultron.core.coding.executor import FailureAnalysis, FailureCategory
from ultron.core.memory.models import MemoryConfidence
from ultron.core.orchestration import (
    DEFAULT_REGISTRY,
    Agent,
    AgentResult,
    AgentResultStatus,
    AgentStatus,
    AgentType,
    ApprovalStatus,
    ArtifactStore,
    ExecutionContext,
    ImplementationResult,
    ResearchFinding,
    ReviewResult,
    Supervisor,
    TestResult,
    Workflow,
    WorkflowEngine,
    WorkflowStatus,
    WorkflowStep,
    WorkflowStepStatus,
    task_key,
)
from ultron.core.orchestration.workflow import WorkflowEvent
from ultron.core.types import TaskState, TaskStatus

# pytest would otherwise collect the imported artifact model as a test class.
TestResult.__test__ = False  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------
# Fake specialists
# ---------------------------------------------------------------------------


def _artifact_for(context: ExecutionContext, kind: str):
    """The structured artifact each specialist produces by default."""
    task_id, agent_id = context.task_id, context.agent_id
    base = {
        "task_id": task_id,
        "agent_id": agent_id,
        "source": f"fake_{kind}",
        "confidence": MemoryConfidence.DIRECT_OBSERVATION,
    }
    if kind == "research":
        return ResearchFinding(
            **base,
            summary="authentication lives in src/auth/service.py",
            evidence=["symbols: AuthService", "files: src/auth/service.py"],
            related_files=["src/auth/service.py"],
            related_symbols=["AuthService"],
        )
    if kind == "coding":
        return ImplementationResult(
            **base,
            summary="implemented the requested feature",
            changed_files=["src/app/main.py"],
            changed_symbols=["health"],
            tests_added=["tests/test_health.py"],
            tests_run=["pytest tests/test_health.py -q"],
        )
    if kind == "test_qa":
        return TestResult(
            **base,
            command="pytest -q",
            passed=6,
            failed=0,
        )
    if kind == "reviewer":
        return ReviewResult(
            **base,
            summary="review passed, no blockers",
            approval=ApprovalStatus.APPROVED,
        )
    raise AssertionError(f"no default artifact for {kind!r}")


class WorkflowFakeAgent(Agent):
    """Deterministic specialist whose behavior is keyed by agent type.

    ``plan`` maps agent_type value -> behavior; ``attempts`` is the state's
    execution attempt count (1 on first run, 2 after a WAITING resume), which
    lets a ``needs_input_once`` specialist pause the FIRST attempt only.
    """

    def __init__(
        self, identity, plan: dict[str, str], attempts: int = 1, resumed: bool = False
    ) -> None:
        super().__init__(identity)
        self.plan = plan
        self.attempts = attempts
        self.resumed = resumed  # True when dispatched to resume a WAITING run
        self.trajectory: list[str] = []

    async def execute(self, objective: str, context: ExecutionContext) -> AgentResult:
        self.trajectory.append(f"internal reasoning: {objective} / {context.task_id}")
        kind = self.identity.agent_type.value
        mode = self.plan.get(kind, "ok")

        if mode == "needs_input_once":
            if not self.resumed:
                return AgentResult(
                    status=AgentResultStatus.NEEDS_INPUT,
                    summary="need confirmation before proceeding",
                )
            mode = "ok"  # resumed attempt behaves normally
        if mode == "needs_input":
            return AgentResult(
                status=AgentResultStatus.NEEDS_INPUT,
                summary="need confirmation before proceeding",
            )
        if mode == "failed":
            return AgentResult(
                status=AgentResultStatus.FAILED,
                summary=f"could not {objective}",
                blockers=["fake failure"],
            )
        if mode == "blocked":
            return AgentResult(
                status=AgentResultStatus.BLOCKED,
                summary="security policy blocks this step",
            )
        if mode == "empty":
            return AgentResult(status=AgentResultStatus.SUCCESS, summary="ok")
        if mode == "claim":
            return AgentResult(
                status=AgentResultStatus.SUCCESS, summary="task is complete"
            )
        if mode == "testing_fail":
            return TestResult(
                task_id=context.task_id,
                agent_id=context.agent_id,
                command="pytest -q",
                passed=3,
                failed=2,
                failures=[
                    FailureAnalysis(
                        category=FailureCategory.TEST_ASSERTION,
                        command="pytest -q",
                        summary="expected 5 but got 4",
                    )
                ],
                source="fake_tester",
            ).to_agent_result()
        return _artifact_for(context, kind).to_agent_result()


def _make_supervisor(plan: dict[str, str], store: ArtifactStore | None = None) -> Supervisor:
    def factory(spec, state) -> WorkflowFakeAgent:
        # ``resumed`` is True exactly when the factory is asked to continue a
        # WAITING run (its lifecycle is WAITING at factory time; attempts is
        # still the pre-resume value).
        return WorkflowFakeAgent(
            state.identity,
            plan,
            attempts=state.metadata.attempts,
            resumed=state.status is AgentStatus.WAITING,
        )

    return Supervisor(registry=DEFAULT_REGISTRY, agent_factory=factory, store=store)


def _make_engine(
    plan: dict[str, str] | None = None,
    store: ArtifactStore | None = None,
) -> WorkflowEngine:
    return WorkflowEngine(_make_supervisor(plan or {}, store))


# ---------------------------------------------------------------------------
# Workflow builders
# ---------------------------------------------------------------------------


def _chain_steps() -> list[WorkflowStep]:
    """research -> implementation -> testing -> review (strict chain)."""
    return [
        WorkflowStep(
            step_id="research",
            name="Research",
            agent_type=AgentType.RESEARCH,
            objective="Understand the authentication flow",
            expected_output="ResearchFinding",
        ),
        WorkflowStep(
            step_id="implementation",
            name="Implementation",
            agent_type=AgentType.CODING,
            objective="Implement the change",
            dependencies=["research"],
            expected_output="ImplementationResult",
            constraints=["stay in src/"],
        ),
        WorkflowStep(
            step_id="testing",
            name="Testing",
            agent_type=AgentType.TEST_QA,
            objective="Run and validate tests",
            dependencies=["implementation"],
            expected_output="TestResult",
        ),
        WorkflowStep(
            step_id="review",
            name="Review",
            agent_type=AgentType.REVIEWER,
            objective="Review the change",
            dependencies=["testing"],
            expected_output="ReviewResult",
        ),
    ]


def _task(goal: str = "Fix the authentication bug.") -> TaskState:
    task = TaskState(goal=goal)
    task.add_requirement("Authentication works and tests pass")
    return task


def _happy_plan() -> dict[str, str]:
    return {
        "research": "ok",
        "coding": "ok",
        "test_qa": "ok",
        "reviewer": "ok",
    }


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Workflow creation + structural validation
# ---------------------------------------------------------------------------


def test_create_valid_workflow():
    task = _task()
    engine = _make_engine(_happy_plan())
    wf = engine.create_workflow(
        steps=_chain_steps(), task_state=task, name="auth fix"
    )
    assert wf.workflow_id.startswith("wf:")
    assert wf.task_id == task_key(task)
    assert wf.name == "auth fix"
    assert wf.status is WorkflowStatus.PENDING
    assert len(wf.steps) == 4
    assert all(s.workflow_id == wf.workflow_id for s in wf.steps)
    assert all(s.status is WorkflowStepStatus.PENDING for s in wf.steps)
    assert engine.validate_workflow(wf) == []
    assert engine.get_workflow(wf.workflow_id) is wf


def test_create_workflow_missing_task_id():
    engine = _make_engine()
    with pytest.raises(ValueError, match="task_id"):
        engine.create_workflow(steps=_chain_steps())


def test_duplicate_workflow_id_rejected():
    task = _task()
    engine = _make_engine()
    wf = engine.create_workflow(steps=_chain_steps(), task_state=task)
    with pytest.raises(ValueError, match="duplicate workflow_id"):
        engine.create_workflow(
            steps=_chain_steps(), task_state=task, workflow_id=wf.workflow_id
        )


def test_duplicate_step_id_rejected():
    steps = [
        WorkflowStep(step_id="a", name="A", agent_type=AgentType.RESEARCH, objective="o"),
        WorkflowStep(step_id="a", name="B", agent_type=AgentType.CODING, objective="o"),
    ]
    engine = _make_engine()
    with pytest.raises(ValueError, match="duplicate step id"):
        engine.create_workflow(steps=steps, task_id="t1")


def test_unregistered_agent_type_rejected():
    steps = [
        WorkflowStep(step_id="a", name="A", agent_type="dragon", objective="o"),
    ]
    engine = _make_engine()
    with pytest.raises(ValueError, match="unregistered agent type"):
        engine.create_workflow(steps=steps, task_id="t1")


def test_missing_dependency_rejected():
    steps = [
        WorkflowStep(
            step_id="a", name="A", agent_type=AgentType.RESEARCH,
            objective="o", dependencies=["ghost"],
        ),
    ]
    engine = _make_engine()
    with pytest.raises(ValueError, match="unknown step 'ghost'"):
        engine.create_workflow(steps=steps, task_id="t1")


def test_self_dependency_rejected():
    steps = [
        WorkflowStep(
            step_id="a", name="A", agent_type=AgentType.RESEARCH,
            objective="o", dependencies=["a"],
        ),
    ]
    engine = _make_engine()
    with pytest.raises(ValueError, match="depends on itself"):
        engine.create_workflow(steps=steps, task_id="t1")


def test_circular_dependency_rejected():
    steps = [
        WorkflowStep(step_id="a", name="A", agent_type=AgentType.RESEARCH, objective="o", dependencies=["b"]),
        WorkflowStep(step_id="b", name="B", agent_type=AgentType.CODING, objective="o", dependencies=["a"]),
    ]
    engine = _make_engine()
    with pytest.raises(ValueError, match="circular dependency"):
        engine.create_workflow(steps=steps, task_id="t1")


def test_cross_workflow_dependency_rejected():
    engine = _make_engine()
    engine.create_workflow(
        steps=[WorkflowStep(step_id="x", name="X", agent_type=AgentType.RESEARCH, objective="o")],
        task_id="t1",
    )
    steps = [
        WorkflowStep(
            step_id="a", name="A", agent_type=AgentType.CODING,
            objective="o", dependencies=["x"],  # 'x' belongs to another workflow
        ),
    ]
    with pytest.raises(ValueError, match="unknown step 'x'"):
        engine.create_workflow(steps=steps, task_id="t2")


def test_empty_workflow_rejected():
    engine = _make_engine()
    with pytest.raises(ValueError, match="no steps"):
        engine.create_workflow(steps=[], task_id="t1")


def test_agent_type_string_alias_accepted():
    steps = [
        WorkflowStep(step_id="a", name="A", agent_type="research", objective="o"),
    ]
    engine = _make_engine({"research": "ok"})
    wf = engine.create_workflow(steps=steps, task_id="t1")
    assert wf.steps[0].agent_type_value == "research"


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_start_transitions_pending_to_running():
    engine = _make_engine(_happy_plan())
    wf = engine.create_workflow(steps=_chain_steps(), task_state=_task())
    _run(engine.start(wf))
    assert wf.status is WorkflowStatus.RUNNING
    assert any(e.event == "WORKFLOW_STARTED" for e in wf.events)


def test_start_twice_raises():
    engine = _make_engine(_happy_plan())
    wf = engine.create_workflow(steps=_chain_steps(), task_state=_task())
    _run(engine.start(wf))
    with pytest.raises(ValueError, match="cannot start"):
        _run(engine.start(wf))


def test_pause_resume_between_steps():
    engine = _make_engine(_happy_plan())
    wf = engine.create_workflow(steps=_chain_steps(), task_state=_task())
    _run(engine.start(wf))
    _run(engine.execute_next_step(wf))  # research completes
    engine.pause(wf)
    assert wf.status is WorkflowStatus.PAUSED
    assert any(e.event == "WORKFLOW_PAUSED" for e in wf.events)
    _run(engine.resume(wf))
    assert wf.status is WorkflowStatus.RUNNING
    assert any(e.event == "WORKFLOW_RESUMED" for e in wf.events)


def test_pause_before_first_step():
    engine = _make_engine(_happy_plan())
    wf = engine.create_workflow(steps=_chain_steps(), task_state=_task())
    engine.pause(wf)  # PENDING -> PAUSED
    assert wf.status is WorkflowStatus.PAUSED
    _run(engine.resume(wf))
    step = _run(engine.execute_next_step(wf))
    assert step is not None and step.step_id == "research"


def test_pause_not_running_raises():
    engine = _make_engine(_happy_plan())
    wf = engine.create_workflow(steps=_chain_steps(), task_state=_task())
    _run(engine.start(wf))
    engine.pause(wf)
    # Pausing an already-paused workflow is an invalid transition.
    with pytest.raises(ValueError, match="cannot pause"):
        engine.pause(wf)


def test_cancel_before_execution():
    engine = _make_engine(_happy_plan())
    wf = engine.create_workflow(steps=_chain_steps(), task_state=_task())
    engine.cancel(wf)
    assert wf.status is WorkflowStatus.CANCELLED
    assert all(s.status is WorkflowStepStatus.CANCELLED for s in wf.steps)
    assert all(s.delegation_id is None for s in wf.steps)


def test_cancel_terminal_raises():
    engine = _make_engine(_happy_plan())
    wf = engine.create_workflow(steps=_chain_steps(), task_state=_task())
    _run(engine.execute_until_blocked(wf))
    assert wf.status is WorkflowStatus.COMPLETED
    with pytest.raises(ValueError, match="cannot cancel"):
        engine.cancel(wf)


def test_invalid_step_transitions_rejected():
    step = WorkflowStep(step_id="a", name="A", agent_type=AgentType.RESEARCH, objective="o")
    with pytest.raises(ValueError, match="Invalid"):
        step.complete()  # PENDING -> COMPLETED is illegal
    with pytest.raises(ValueError, match="Invalid"):
        step.start()  # PENDING -> RUNNING (must be READY first)
    step.mark_ready()
    step.start()
    with pytest.raises(ValueError, match="Invalid"):
        step.mark_ready()  # RUNNING -> READY is illegal


def test_workflow_cannot_complete_with_unfinished_steps():
    engine = _make_engine(_happy_plan())
    wf = engine.create_workflow(steps=_chain_steps(), task_state=_task())
    _run(engine.start(wf))
    _run(engine.execute_next_step(wf))  # only research done
    with pytest.raises(ValueError, match="unfinished steps"):
        wf.complete()


def test_multiple_sequential_dependencies():
    steps = [
        WorkflowStep(step_id="a", name="A", agent_type=AgentType.RESEARCH, objective="o"),
        WorkflowStep(step_id="b", name="B", agent_type=AgentType.RESEARCH, objective="o2"),
        WorkflowStep(
            step_id="c", name="C", agent_type=AgentType.CODING,
            objective="combine", dependencies=["a", "b"],
        ),
    ]
    engine = _make_engine(_happy_plan())
    wf = engine.create_workflow(steps=steps, task_state=_task())
    _run(engine.start(wf))
    _run(engine.execute_next_step(wf))  # a done
    assert wf.get_step("c").status is WorkflowStepStatus.PENDING  # b missing
    _run(engine.execute_next_step(wf))  # b done
    assert wf.get_step("c").status is WorkflowStepStatus.READY
    step = _run(engine.execute_next_step(wf))
    assert step is not None and step.step_id == "c"
    assert wf.get_step("c").status is WorkflowStepStatus.COMPLETED


# ---------------------------------------------------------------------------
# Sequential execution + delegation wiring
# ---------------------------------------------------------------------------


def test_first_step_executes():
    engine = _make_engine(_happy_plan())
    wf = engine.create_workflow(steps=_chain_steps(), task_state=_task())
    _run(engine.start(wf))
    step = _run(engine.execute_next_step(wf))
    assert step is not None and step.step_id == "research"
    research = wf.get_step("research")
    assert research.status is WorkflowStepStatus.COMPLETED
    assert research.delegation_id is not None
    assert research.result_artifact is not None
    assert research.attempts == 1
    # Workflow is NOT complete after one step.
    assert wf.status is WorkflowStatus.RUNNING


def test_execute_next_step_before_start_returns_none():
    engine = _make_engine(_happy_plan())
    wf = engine.create_workflow(steps=_chain_steps(), task_state=_task())
    assert _run(engine.execute_next_step(wf)) is None


def test_execute_next_step_when_paused_returns_none():
    engine = _make_engine(_happy_plan())
    wf = engine.create_workflow(steps=_chain_steps(), task_state=_task())
    engine.pause(wf)
    assert _run(engine.execute_next_step(wf)) is None


def test_step_waits_for_dependency():
    # Implementation listed FIRST depends on research listed second.
    steps = [
        WorkflowStep(step_id="implementation", name="Impl", agent_type=AgentType.CODING, objective="code", dependencies=["research"]),
        WorkflowStep(step_id="research", name="Res", agent_type=AgentType.RESEARCH, objective="research"),
    ]
    engine = _make_engine(_happy_plan())
    wf = engine.create_workflow(steps=steps, task_state=_task())
    _run(engine.start(wf))
    # Implementation is NOT ready, so research runs first (dependency
    # ordering beats step-list order).
    step = _run(engine.execute_next_step(wf))
    assert step is not None and step.step_id == "research"
    # Research completing unlocks implementation (dependency satisfied).
    assert wf.get_step("implementation").status is WorkflowStepStatus.READY
    step = _run(engine.execute_next_step(wf))
    assert step is not None and step.step_id == "implementation"


def test_steps_execute_in_order_and_complete():
    engine = _make_engine(_happy_plan())
    wf = engine.create_workflow(steps=_chain_steps(), task_state=_task())
    order = _run(engine.execute_until_blocked(wf))
    assert order == 4
    assert wf.status is WorkflowStatus.COMPLETED
    assert all(s.status is WorkflowStepStatus.COMPLETED for s in wf.steps)
    assert wf.current_step == "review"
    assert [s.step_id for s in wf.steps] == [
        "research", "implementation", "testing", "review",
    ]
    # Completed workflow: no step is repeated.
    assert _run(engine.execute_until_blocked(wf)) == 0


def test_engine_routes_through_supervisor():
    task = _task()
    engine = _make_engine(_happy_plan())
    wf = engine.create_workflow(steps=_chain_steps(), task_state=task)
    _run(engine.start(wf))
    _run(engine.execute_next_step(wf))
    research = wf.get_step("research")
    request = engine.supervisor.get_delegation(research.delegation_id)
    assert request is not None
    assert request.agent_type is AgentType.RESEARCH
    assert request.objective == "Understand the authentication flow"
    assert request.task_id == task_key(task)
    assert request.expected_output == "ResearchFinding"
    # The specialist run exists and completed.
    run = engine.supervisor.get_run(request.delegation_id)
    assert run is not None and run.status is AgentStatus.COMPLETED


def test_delegation_carries_constraints():
    engine = _make_engine(_happy_plan())
    wf = engine.create_workflow(steps=_chain_steps(), task_state=_task())
    _run(engine.execute_until_blocked(wf))
    impl = wf.get_step("implementation")
    request = engine.supervisor.get_delegation(impl.delegation_id)
    assert request.constraints == ["stay in src/"]
    # No per-step timeout was set, so the coder type's spec budget applies.
    spec = engine.supervisor.registry.get(AgentType.CODING)
    assert request.timeout_seconds == spec.max_budget.timeout_seconds


def test_artifact_flows_to_dependent_step(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    engine = _make_engine(_happy_plan(), store=store)
    wf = engine.create_workflow(steps=_chain_steps(), task_state=_task())
    _run(engine.execute_until_blocked(wf))
    impl = wf.get_step("implementation")
    request = engine.supervisor.get_delegation(impl.delegation_id)
    # Exactly the research artifact — nothing else.
    assert len(request.input_artifacts) == 1
    finding = request.input_artifacts[0]
    assert finding.artifact_type.value == "research_finding"
    assert finding.related_symbols == ["AuthService"]
    # The artifact persisted to the store under the task.
    assert store.get(finding.artifact_id) is not None


def test_input_artifact_type_filtering():
    # Review depends on both implementation (impl artifact) and research
    # (finding); filtering keeps only the research finding.
    steps = [
        WorkflowStep(step_id="research", name="Res", agent_type=AgentType.RESEARCH, objective="research"),
        WorkflowStep(
            step_id="implementation", name="Impl", agent_type=AgentType.CODING,
            objective="code", dependencies=["research"],
        ),
        WorkflowStep(
            step_id="review", name="Review", agent_type=AgentType.REVIEWER,
            objective="review", dependencies=["research", "implementation"],
            input_artifact_types=["research_finding"],
        ),
    ]
    engine = _make_engine(_happy_plan())
    wf = engine.create_workflow(steps=steps, task_state=_task())
    _run(engine.execute_until_blocked(wf))
    review = wf.get_step("review")
    request = engine.supervisor.get_delegation(review.delegation_id)
    assert {a.artifact_type.value for a in request.input_artifacts} == {
        "research_finding"
    }


def test_context_isolation_no_trajectory(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    engine = _make_engine(_happy_plan(), store=store)
    wf = engine.create_workflow(steps=_chain_steps(), task_state=_task())
    _run(engine.execute_until_blocked(wf))
    impl = wf.get_step("implementation")
    request = engine.supervisor.get_delegation(impl.delegation_id)
    ctx = engine.supervisor.get_run(request.delegation_id).context
    # The specialist receives the task brief + artifact summaries only.
    assert "task" in ctx.relevant_context
    assert "artifacts" in ctx.relevant_context
    blob = str(ctx.relevant_context)
    assert "internal reasoning" not in blob
    assert "execution_history" not in blob
    # Input artifacts are the ONLY cross-agent channel.
    assert len(request.input_artifacts) == 1


# ---------------------------------------------------------------------------
# Validation gating + failures
# ---------------------------------------------------------------------------


def test_valid_result_step_succeeds_with_evidence():
    steps = [
        WorkflowStep(
            step_id="implementation", name="Impl", agent_type=AgentType.CODING,
            objective="code", required_evidence=["artifact", "changed_files"],
        ),
    ]
    engine = _make_engine({"coding": "ok"})
    wf = engine.create_workflow(steps=steps, task_state=_task())
    _run(engine.execute_until_blocked(wf))
    assert wf.status is WorkflowStatus.COMPLETED
    assert wf.get_step("implementation").validation["violations"] == []


def test_missing_required_evidence_fails_step():
    steps = [
        WorkflowStep(
            step_id="implementation", name="Impl", agent_type=AgentType.CODING,
            objective="code", required_evidence=["artifact"],
        ),
    ]
    engine = _make_engine({"coding": "empty"})  # SUCCESS but no artifact
    wf = engine.create_workflow(steps=steps, task_state=_task())
    _run(engine.execute_until_blocked(wf))
    step = wf.get_step("implementation")
    assert step.status is WorkflowStepStatus.FAILED
    assert "missing evidence: artifact" in (step.error or "")
    assert wf.status is WorkflowStatus.FAILED


def test_false_completion_claim_rejected():
    steps = [
        WorkflowStep(
            step_id="review", name="Review", agent_type=AgentType.REVIEWER,
            objective="review", claims_completion=True,
            required_evidence=["artifact"],
        ),
    ]
    engine = _make_engine({"reviewer": "claim"})  # "task is complete", no artifact
    wf = engine.create_workflow(steps=steps, task_state=_task())
    _run(engine.execute_until_blocked(wf))
    step = wf.get_step("review")
    assert step.status is WorkflowStepStatus.FAILED
    assert "false_completion_claim" in step.validation["violations"]
    assert wf.status is WorkflowStatus.FAILED


def test_validation_violation_workflow_does_not_advance():
    steps = [
        WorkflowStep(step_id="implementation", name="Impl", agent_type=AgentType.CODING, objective="code", required_evidence=["artifact"]),
        WorkflowStep(step_id="testing", name="Testing", agent_type=AgentType.TEST_QA, objective="test", dependencies=["implementation"]),
    ]
    engine = _make_engine({"coding": "empty", "test_qa": "ok"})
    wf = engine.create_workflow(steps=steps, task_state=_task())
    _run(engine.execute_until_blocked(wf))
    assert wf.get_step("implementation").status is WorkflowStepStatus.FAILED
    assert wf.get_step("testing").status is WorkflowStepStatus.PENDING
    assert wf.status is WorkflowStatus.FAILED


def test_agent_failure_stops_workflow():
    plan = {"research": "failed"}
    engine = _make_engine(plan)
    wf = engine.create_workflow(steps=_chain_steps(), task_state=_task())
    _run(engine.execute_until_blocked(wf))
    research = wf.get_step("research")
    assert research.status is WorkflowStepStatus.FAILED
    assert wf.status is WorkflowStatus.FAILED
    # Implementation never dispatched.
    assert wf.get_step("implementation").delegation_id is None
    assert wf.get_step("implementation").status is WorkflowStepStatus.PENDING


def test_test_failure_stops_workflow():
    plan = _happy_plan() | {"test_qa": "testing_fail"}
    engine = _make_engine(plan)
    wf = engine.create_workflow(steps=_chain_steps(), task_state=_task())
    _run(engine.execute_until_blocked(wf))
    testing = wf.get_step("testing")
    assert testing.status is WorkflowStepStatus.FAILED
    assert "test artifact reports failures" in (testing.error or "")
    assert wf.get_step("review").delegation_id is None
    assert wf.status is WorkflowStatus.FAILED


def test_review_failure_stops_workflow():
    plan = _happy_plan() | {"reviewer": "failed"}
    engine = _make_engine(plan)
    wf = engine.create_workflow(steps=_chain_steps(), task_state=_task())
    _run(engine.execute_until_blocked(wf))
    assert wf.get_step("review").status is WorkflowStepStatus.FAILED
    assert wf.status is WorkflowStatus.FAILED


def test_failure_recorded_on_task_state():
    task = _task()
    engine = _make_engine({"research": "failed"})
    wf = engine.create_workflow(steps=_chain_steps(), task_state=task)
    _run(engine.execute_until_blocked(wf))
    assert task.errors, "workflow failure must be recorded on the TaskState"
    assert "[workflow:research]" in task.errors[-1].message
    assert task.status is not TaskStatus.TASK_COMPLETED


# ---------------------------------------------------------------------------
# Blocked + needs-input
# ---------------------------------------------------------------------------


def test_agent_blocked_workflow_blocked():
    engine = _make_engine({"research": "blocked"})
    wf = engine.create_workflow(steps=_chain_steps(), task_state=_task())
    _run(engine.execute_until_blocked(wf))
    assert wf.get_step("research").status is WorkflowStepStatus.BLOCKED
    assert wf.status is WorkflowStatus.BLOCKED  # blocked, NOT failed


def test_needs_input_workflow_waits():
    engine = _make_engine({"research": "needs_input"})
    wf = engine.create_workflow(steps=_chain_steps(), task_state=_task())
    _run(engine.execute_until_blocked(wf))
    research = wf.get_step("research")
    assert research.status is WorkflowStepStatus.WAITING
    assert wf.status is WorkflowStatus.WAITING
    # No subsequent step executes while waiting.
    assert _run(engine.execute_next_step(wf)) is None
    assert wf.get_step("implementation").delegation_id is None


def test_needs_input_resume_continues():
    engine = _make_engine({"research": "needs_input_once"})
    wf = engine.create_workflow(steps=_chain_steps(), task_state=_task())
    _run(engine.execute_until_blocked(wf))
    assert wf.status is WorkflowStatus.WAITING
    research = wf.get_step("research")
    assert research.attempts == 1
    # Input provided -> resume -> the SAME delegation is resumed.
    delegation_id = research.delegation_id
    _run(engine.resume(wf))
    step = _run(engine.execute_next_step(wf))
    assert step is not None and step.step_id == "research"
    assert research.status is WorkflowStepStatus.COMPLETED
    assert research.attempts == 2
    assert research.delegation_id == delegation_id  # resumed, not duplicated
    # The chain continues from implementation.
    impl = _run(engine.execute_next_step(wf))
    assert impl is not None and impl.step_id == "implementation"


# ---------------------------------------------------------------------------
# Pause / resume / reload
# ---------------------------------------------------------------------------


def test_pause_between_steps_resume_from_correct_step():
    engine = _make_engine(_happy_plan())
    wf = engine.create_workflow(steps=_chain_steps(), task_state=_task())
    _run(engine.start(wf))
    _run(engine.execute_next_step(wf))  # research done
    research = wf.get_step("research")
    engine.pause(wf)
    _run(engine.resume(wf))
    step = _run(engine.execute_next_step(wf))
    assert step is not None and step.step_id == "implementation"
    # Research was not repeated.
    assert research.status is WorkflowStepStatus.COMPLETED
    assert research.attempts == 1


def test_workflow_survives_serialization_reload(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    engine = _make_engine(_happy_plan(), store=store)
    wf = engine.create_workflow(steps=_chain_steps(), task_state=_task())
    _run(engine.start(wf))
    _run(engine.execute_next_step(wf))  # research completes
    engine.pause(wf)
    payload = wf.model_dump_json()

    # A fresh engine + supervisor (same store) restores the workflow.
    engine2 = _make_engine(_happy_plan(), store=store)
    wf2 = Workflow.model_validate_json(payload)
    _run(engine2.resume(wf2, task_state=_task()))
    step = _run(engine2.execute_next_step(wf2))
    assert step is not None and step.step_id == "implementation"
    # The completed step was NOT repeated.
    research = wf2.get_step("research")
    assert research.status is WorkflowStepStatus.COMPLETED
    assert research.attempts == 1
    # Implementation receives the research artifact from the store.
    impl = wf2.get_step("implementation")
    request = engine2.supervisor.get_delegation(impl.delegation_id)
    assert len(request.input_artifacts) == 1


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


def test_cancel_during_execution_preserves_artifacts(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    engine = _make_engine(_happy_plan(), store=store)
    wf = engine.create_workflow(steps=_chain_steps(), task_state=_task())
    _run(engine.start(wf))
    _run(engine.execute_next_step(wf))  # research completes + artifact saved
    research_artifact_id = wf.get_step("research").result_artifact
    engine.cancel(wf)
    assert wf.status is WorkflowStatus.CANCELLED
    assert wf.get_step("implementation").status is WorkflowStepStatus.CANCELLED
    assert wf.get_step("review").status is WorkflowStepStatus.CANCELLED
    # The already-produced artifact remains in the store.
    assert store.get(research_artifact_id) is not None
    assert wf.get_step("research").result_artifact == research_artifact_id
    # No false completion anywhere.
    assert wf.metadata.get("cancelled")
    assert wf.metadata.get("task_completion") is None


# ---------------------------------------------------------------------------
# Completion + TaskState integration
# ---------------------------------------------------------------------------


def test_completion_invokes_task_completion_api():
    task = _task()
    task.mark_requirement_complete("Authentication works and tests pass")
    engine = _make_engine(_happy_plan())
    wf = engine.create_workflow(steps=_chain_steps(), task_state=task)
    _run(engine.execute_until_blocked(wf))
    assert wf.status is WorkflowStatus.COMPLETED
    assert task.is_complete()
    assert task.status is TaskStatus.TASK_COMPLETED
    assert wf.metadata["task_completion"] == "completed"
    assert any(e.event == "WORKFLOW_COMPLETED" for e in wf.events)


def test_incomplete_task_cannot_become_complete():
    task = _task()  # requirement intentionally left unresolved
    engine = _make_engine(_happy_plan())
    wf = engine.create_workflow(steps=_chain_steps(), task_state=task)
    _run(engine.execute_until_blocked(wf))
    # Orchestration finished, but the task must NOT be complete.
    assert wf.status is WorkflowStatus.COMPLETED
    assert not task.is_complete()
    assert task.status is not TaskStatus.TASK_COMPLETED
    assert wf.metadata["task_completion"].startswith("not_completed:")


def test_cancelled_task_never_completes():
    task = _task()
    engine = _make_engine(_happy_plan())
    wf = engine.create_workflow(steps=_chain_steps(), task_state=task)
    _run(engine.start(wf))
    engine.cancel(wf)
    assert not task.is_complete()


# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------


def test_events_recorded():
    engine = _make_engine(_happy_plan())
    wf = engine.create_workflow(steps=_chain_steps(), task_state=_task())
    _run(engine.execute_until_blocked(wf))
    names = [e.event for e in wf.events]
    for expected in (
        "WORKFLOW_CREATED",
        "WORKFLOW_STARTED",
        "STEP_READY",
        "STEP_STARTED",
        "STEP_COMPLETED",
        "WORKFLOW_COMPLETED",
    ):
        assert expected in names, f"missing event {expected}"
    assert all(isinstance(e, WorkflowEvent) for e in wf.events)
    completed = [e for e in wf.events if e.event == "STEP_COMPLETED"]
    assert len(completed) == 4
    assert all(e.step_id for e in completed)


def test_traceability():
    engine = _make_engine(_happy_plan())
    wf = engine.create_workflow(steps=_chain_steps(), task_state=_task())
    _run(engine.execute_until_blocked(wf))
    rows = engine.trace_rows(wf)
    assert len(rows) == 4
    for row in rows:
        assert row["workflow_id"] == wf.workflow_id
        assert row["task_id"] == wf.task_id
        assert row["step_id"]
        assert row["status"] == "completed"
        assert row["delegation_id"]
        assert row["agent_id"]
        assert row["result_artifact"]
    # The ids are all cross-linked: delegation -> run -> task.
    first = rows[0]
    request = engine.supervisor.get_delegation(first["delegation_id"])
    run = engine.supervisor.get_run(first["delegation_id"])
    assert run.task_id == wf.task_id
    assert request.run_agent_id == first["agent_id"]


def test_engine_is_deterministic_and_sequential():
    """The engine never runs two steps concurrently and never invents work."""
    plan = _happy_plan()
    results = []
    for _ in range(2):
        engine = _make_engine(plan)
        wf = engine.create_workflow(steps=_chain_steps(), task_state=_task())
        _run(engine.execute_until_blocked(wf))
        results.append(
            (wf.status, [s.status.value for s in wf.steps], wf.metadata.get("task_completion"))
        )
    assert results[0] == results[1]
