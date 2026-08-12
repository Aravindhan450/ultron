"""
FIX #7 — Orchestration validation layer: deterministic tests.

Covers every check family the spec requires:

- lifecycle: legal history, illegal history, non-terminal-with-result,
  resume (WAITING -> RUNNING) legality
- permissions: authorized tool, unauthorized tool, unauthorized write,
  unauthorized shell (whitelist + category-DENY)
- budget: within, exceeded, exact boundary (warning)
- timeout: within, exceeded, flagged by execution layer, missing timing
- result: valid/invalid success/failure/blocked/needs-input
- artifacts: valid, malformed, wrong task, wrong agent, missing provenance
- task state: consistent, blocked-task success, failed-task success,
  completed task with unresolved steps
- completion: valid, false, missing test evidence, missing verification,
  incomplete plan, test-claim contradiction
- workspace scope: allowed, disallowed, mixed, outside workspace
- idempotency + no side effects
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from ultron.core.memory.models import MemoryConfidence
from ultron.core.orchestration import (
    DEFAULT_REGISTRY,
    AgentResult,
    AgentResultStatus,
    AgentStatus,
    AgentStatusChange,
    AgentType,
    DelegationRequest,
    ExecutionBudget,
    OrchestrationValidator,
    ResearchFinding,
    TestResult,
    ValidationContext,
    ValidationStatus,
    ViolationCode,
    task_key,
)
from ultron.core.types import (
    PlanStep,
    TaskPlan,
    TaskState,
    TaskStatus,
    TaskType,
    ToolExecution,
)

VALIDATOR = OrchestrationValidator()


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _task(goal: str = "Understand authentication.", *requirements: str) -> TaskState:
    task = TaskState(goal=goal)
    for req in requirements or ("Authentication flow is documented",):
        task.add_requirement(req)
    return task


def _research_artifact(task_id: str, agent_id: str = "researcher-1", **kw) -> ResearchFinding:
    defaults = {
        "task_id": task_id,
        "agent_id": agent_id,
        "summary": "Authentication lives in src/auth/service.py.",
        "source": "code_intelligence",
        "confidence": MemoryConfidence.DIRECT_OBSERVATION,
    }
    defaults.update(kw)
    return ResearchFinding(**defaults)


def _happy_ctx(tmp_path, task: TaskState | None = None, **overrides) -> ValidationContext:
    """A clean researcher delegation record: everything consistent."""
    task = task or _task()
    tid = task_key(task)
    delegation = DelegationRequest(
        task_id=tid,
        agent_type=AgentType.RESEARCH,
        objective="understand auth",
        permissions=DEFAULT_REGISTRY.get("research").permissions,
        budget=DEFAULT_REGISTRY.get("research").max_budget.model_copy(deep=True),
    )
    state = DEFAULT_REGISTRY.instantiate(
        AgentType.RESEARCH, "researcher-1", tid, "understand auth"
    )
    artifact = _research_artifact(tid)
    result = artifact.to_agent_result()
    # Drive the lifecycle deterministically (fully legal histories).
    state.assign(state.context)
    state.start()
    state.complete(result)
    delegation.assign()
    delegation.start()
    delegation.complete(result)
    ctx = ValidationContext(
        agent_state=state,
        result=result,
        delegation=delegation,
        task_state=task,
        artifacts=[artifact],
        tool_uses=[ToolExecution(tool_name="search_files", target="authenticate")],
        workspace=str(tmp_path),
        **overrides,
    )
    return ctx


def _result(status: AgentResultStatus, **kw) -> AgentResult:
    return AgentResult(status=status, **kw)


def _code(result: ValidationContext) -> set[ViolationCode]:
    return {v.code for v in VALIDATOR.validate(result).violations}


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_lifecycle_legal_history_passes(tmp_path):
    result = VALIDATOR.validate(_happy_ctx(tmp_path))
    assert result.status is ValidationStatus.PASS
    assert result.valid
    by_name = {c.name: c for c in result.checks}
    assert by_name["lifecycle_history_transitions"].passed
    assert by_name["lifecycle_result_status_match"].passed
    assert by_name["delegation_result_status_match"].passed


def test_lifecycle_illegal_transition_detected(tmp_path):
    ctx = _happy_ctx(tmp_path)
    # Corrupt the history: a transition out of a terminal state.
    ctx.agent_state.metadata.status_history.append(
        AgentStatusChange(
            from_status=AgentStatus.COMPLETED,
            to_status=AgentStatus.RUNNING,
            reason="corrupt",
        )
    )
    result = VALIDATOR.validate(ctx)
    assert ViolationCode.INVALID_LIFECYCLE_TRANSITION in _code(ctx)
    assert not result.valid
    assert result.status is ValidationStatus.FAIL


def test_lifecycle_non_terminal_state_with_result(tmp_path):
    ctx = _happy_ctx(tmp_path)
    # Simulate a run that never reached a terminal state.
    ctx.agent_state = DEFAULT_REGISTRY.instantiate(
        AgentType.RESEARCH, "researcher-1", ctx.delegation.task_id, "x"
    )
    ctx.agent_state.assign(ctx.agent_state.context)
    ctx.agent_state.start()  # RUNNING
    assert ViolationCode.INVALID_LIFECYCLE_TRANSITION in _code(ctx)


def test_lifecycle_result_status_mismatch(tmp_path):
    ctx = _happy_ctx(tmp_path)
    # Corrupt: delegation COMPLETED but the reported result is FAILED.
    ctx.result = _result(AgentResultStatus.FAILED, summary="oops")
    assert ViolationCode.INVALID_LIFECYCLE_TRANSITION in _code(ctx)


def test_lifecycle_resume_is_legal(tmp_path):
    task = _task()
    tid = task_key(task)
    delegation = DelegationRequest(
        task_id=tid,
        agent_type=AgentType.RESEARCH,
        objective="x",
        permissions=DEFAULT_REGISTRY.get("research").permissions,
    )
    state = DEFAULT_REGISTRY.instantiate(AgentType.RESEARCH, "r1", tid, "x")
    state.assign(state.context)
    state.start()
    state.wait(_result(AgentResultStatus.NEEDS_INPUT, summary="need approval"))
    state.resume()  # WAITING -> RUNNING (retry/resume is supported)
    state.complete(_result(AgentResultStatus.SUCCESS, summary="done"))
    delegation.assign()
    delegation.start()
    delegation.wait(_result(AgentResultStatus.NEEDS_INPUT, summary="need approval"))
    delegation.start()
    delegation.complete(_result(AgentResultStatus.SUCCESS, summary="done"))
    ctx = ValidationContext(
        agent_state=state,
        result=state.result,
        delegation=delegation,
        task_state=task,
    )
    result = VALIDATOR.validate(ctx)
    assert result.valid
    by_name = {c.name: c for c in result.checks}
    assert by_name["lifecycle_history_transitions"].passed


# ---------------------------------------------------------------------------
# Permissions / tool authorization
# ---------------------------------------------------------------------------


def test_permission_authorized_tool_passes(tmp_path):
    ctx = _happy_ctx(tmp_path)
    ctx.tool_uses = [ToolExecution(tool_name="read_file", target="src/auth/service.py")]
    result = VALIDATOR.validate(ctx)
    assert result.valid
    assert not _code(ctx)


def test_permission_unauthorized_tool(tmp_path):
    ctx = _happy_ctx(tmp_path)
    ctx.tool_uses = [ToolExecution(tool_name="write_file", target="x.py")]
    assert ViolationCode.UNAUTHORIZED_TOOL in _code(ctx)


def test_permission_unauthorized_write(tmp_path):
    # A researcher attempting any write is a violation.
    ctx = _happy_ctx(tmp_path)
    ctx.tool_uses = [ToolExecution(tool_name="create_file", target="src/new.py")]
    assert ViolationCode.UNAUTHORIZED_TOOL in _code(ctx)


def test_permission_unauthorized_shell_via_category_deny(tmp_path):
    # The tester profile whitelists run_command but DENY-classifies shell
    # commands — rm -rf is SHELL, so it is unauthorized.
    task = _task()
    tid = task_key(task)
    delegation = DelegationRequest(
        task_id=tid,
        agent_type=AgentType.TEST_QA,
        objective="run tests",
        permissions=DEFAULT_REGISTRY.get("tester").permissions,
    )
    state = DEFAULT_REGISTRY.instantiate(AgentType.TEST_QA, "tester-1", tid, "run tests")
    ctx = ValidationContext(
        agent_state=state,
        delegation=delegation,
        task_state=task,
        tool_uses=[ToolExecution(tool_name="run_command", target="rm -rf src/")],
        workspace=str(tmp_path),
    )
    assert ViolationCode.UNAUTHORIZED_TOOL in _code(ctx)


def test_permission_researcher_shell_not_whitelisted(tmp_path):
    ctx = _happy_ctx(tmp_path)
    ctx.tool_uses = [ToolExecution(tool_name="run_command", target="ls")]
    assert ViolationCode.UNAUTHORIZED_TOOL in _code(ctx)


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------


def _budget_ctx(max_steps: int, steps_used: int) -> ValidationContext:
    task = _task()
    delegation = DelegationRequest(
        task_id=task_key(task),
        agent_type=AgentType.RESEARCH,
        objective="x",
        budget=ExecutionBudget(max_steps=max_steps, max_tool_calls=10, steps_used=steps_used),
    )
    return ValidationContext(
        delegation=delegation,
        task_state=task,
        result=_result(AgentResultStatus.SUCCESS, summary="ok"),
    )


def test_budget_within_limits():
    result = VALIDATOR.validate(_budget_ctx(max_steps=5, steps_used=3))
    assert result.valid
    assert result.status is ValidationStatus.PASS


def test_budget_exceeded():
    ctx = _budget_ctx(max_steps=5, steps_used=6)
    result = VALIDATOR.validate(ctx)
    assert ViolationCode.BUDGET_EXCEEDED in _code(ctx)
    assert not result.valid


def test_budget_exact_boundary_is_warning():
    ctx = _budget_ctx(max_steps=5, steps_used=5)
    result = VALIDATOR.validate(ctx)
    assert result.valid  # at the limit is not a violation
    assert result.status is ValidationStatus.WARNING
    assert any(w.name == "budget_at_limit" for w in result.warnings)


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------


def _timeout_ctx(started_delta: timedelta | None) -> ValidationContext:
    task = _task()
    tid = task_key(task)
    state = DEFAULT_REGISTRY.instantiate(AgentType.RESEARCH, "r1", tid, "x")
    state.assign(state.context)
    state.start()
    # Control the timing data for a deterministic elapsed duration.
    state.metadata.started_at = (
        datetime.now(UTC) - started_delta if started_delta is not None else None
    )
    state.complete(_result(AgentResultStatus.SUCCESS, summary="ok"))
    delegation = DelegationRequest(
        task_id=tid,
        agent_type=AgentType.RESEARCH,
        objective="x",
        budget=ExecutionBudget(timeout_seconds=5),
    )
    return ValidationContext(
        agent_state=state,
        delegation=delegation,
        task_state=task,
        result=state.result,
    )


def test_timeout_within_limit():
    result = VALIDATOR.validate(_timeout_ctx(timedelta(seconds=1)))
    assert result.valid
    assert result.status is ValidationStatus.PASS


def test_timeout_exceeded():
    ctx = _timeout_ctx(timedelta(seconds=30))
    result = VALIDATOR.validate(ctx)
    assert ViolationCode.TIMEOUT_EXCEEDED in _code(ctx)
    assert not result.valid


def test_timeout_flagged_by_execution_layer():
    # The execution layer already flagged the timeout — validation confirms.
    ctx = _timeout_ctx(timedelta(seconds=1))
    ctx.result = _result(
        AgentResultStatus.FAILED, summary="timed out", metadata={"timeout": True}
    )
    assert ViolationCode.TIMEOUT_EXCEEDED in _code(ctx)


def test_timeout_missing_timing_data_is_warning():
    ctx = _timeout_ctx(None)
    result = VALIDATOR.validate(ctx)
    assert result.valid
    assert result.status is ValidationStatus.WARNING
    assert any(w.name == "timing_data_present" for w in result.warnings)


# ---------------------------------------------------------------------------
# Agent result
# ---------------------------------------------------------------------------


def test_result_valid_success(tmp_path):
    ctx = _happy_ctx(tmp_path)
    ctx.result = _result(
        AgentResultStatus.SUCCESS, summary="done", evidence=["verified"]
    )
    assert VALIDATOR.validate(ctx).valid


def test_result_invalid_empty_success(tmp_path):
    ctx = _happy_ctx(tmp_path)
    ctx.result = _result(AgentResultStatus.SUCCESS)
    assert ViolationCode.INVALID_AGENT_RESULT in _code(ctx)


def test_result_valid_failure():
    ctx = ValidationContext(
        result=_result(AgentResultStatus.FAILED, summary="module not found")
    )
    assert VALIDATOR.validate(ctx).valid


def test_result_invalid_failure_no_info():
    ctx = ValidationContext(result=_result(AgentResultStatus.FAILED))
    assert ViolationCode.INVALID_AGENT_RESULT in _code(ctx)


def test_result_blocked_without_blocker():
    ctx = ValidationContext(result=_result(AgentResultStatus.BLOCKED))
    assert ViolationCode.INVALID_AGENT_RESULT in _code(ctx)


def test_result_blocked_with_blocker_passes():
    ctx = ValidationContext(
        result=_result(
            AgentResultStatus.BLOCKED, blockers=["security policy denies the write"]
        )
    )
    assert VALIDATOR.validate(ctx).valid


def test_result_needs_input_without_requirement():
    ctx = ValidationContext(result=_result(AgentResultStatus.NEEDS_INPUT))
    assert ViolationCode.INVALID_AGENT_RESULT in _code(ctx)


def test_result_needs_input_with_requirement_passes():
    ctx = ValidationContext(
        result=_result(AgentResultStatus.NEEDS_INPUT, summary="approve the write")
    )
    assert VALIDATOR.validate(ctx).valid


def test_result_missing_is_critical():
    ctx = ValidationContext()
    result = VALIDATOR.validate(ctx)
    assert ViolationCode.INVALID_AGENT_RESULT in _code(ctx)
    assert result.status is ValidationStatus.BLOCKED


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------


def test_artifact_valid(tmp_path):
    ctx = _happy_ctx(tmp_path)
    result = VALIDATOR.validate(ctx)
    assert result.valid
    assert result.status is ValidationStatus.PASS


def test_artifact_malformed_missing_ids(tmp_path):
    ctx = _happy_ctx(tmp_path)
    ctx.artifacts = [_research_artifact("", "")]  # empty task_id + agent_id
    assert ViolationCode.INVALID_ARTIFACT in _code(ctx)


def test_artifact_wrong_task(tmp_path):
    ctx = _happy_ctx(tmp_path)
    ctx.artifacts = [_research_artifact(task_id="some-other-task")]
    assert ViolationCode.ARTIFACT_TASK_MISMATCH in _code(ctx)


def test_artifact_wrong_agent(tmp_path):
    ctx = _happy_ctx(tmp_path)
    ctx.artifacts = [_research_artifact(ctx.delegation.task_id, agent_id="coder-7")]
    assert ViolationCode.ARTIFACT_OWNERSHIP_VIOLATION in _code(ctx)


def test_artifact_missing_provenance_is_warning(tmp_path):
    ctx = _happy_ctx(tmp_path)
    ctx.artifacts = [
        _research_artifact(ctx.delegation.task_id, source="", confidence=None)
    ]
    result = VALIDATOR.validate(ctx)
    assert result.valid
    assert result.status is ValidationStatus.WARNING
    assert any(w.name.startswith("artifact_provenance") for w in result.warnings)


# ---------------------------------------------------------------------------
# TaskState consistency
# ---------------------------------------------------------------------------


def test_task_state_consistent(tmp_path):
    assert VALIDATOR.validate(_happy_ctx(tmp_path)).valid


def test_task_state_blocked_task_claiming_success(tmp_path):
    ctx = _happy_ctx(tmp_path)
    ctx.task_state.block("blocked by policy")
    assert ViolationCode.TASK_STATE_CONFLICT in _code(ctx)


def test_task_state_failed_task_claiming_success(tmp_path):
    ctx = _happy_ctx(tmp_path)
    ctx.task_state.status = TaskStatus.TASK_FAILED
    assert ViolationCode.TASK_STATE_CONFLICT in _code(ctx)


def test_task_state_completed_with_unresolved_steps(tmp_path):
    ctx = _happy_ctx(tmp_path)
    # Corrupt: task marked complete while a requirement remains.
    ctx.task_state.status = TaskStatus.TASK_COMPLETED
    assert ViolationCode.TASK_STATE_CONFLICT in _code(ctx)


def test_task_state_completed_with_failed_result(tmp_path):
    ctx = _happy_ctx(tmp_path)
    ctx.task_state.requirements[0].completed = True
    ctx.task_state.status = TaskStatus.TASK_COMPLETED
    ctx.result = _result(AgentResultStatus.FAILED, summary="regression")
    assert ViolationCode.TASK_STATE_CONFLICT in _code(ctx)


# ---------------------------------------------------------------------------
# Completion claims
# ---------------------------------------------------------------------------


def _completion_ctx(tmp_path, *, claims: bool, **overrides) -> ValidationContext:
    task = _task("Add a health endpoint", "Health endpoint implemented")
    task.requirements[0].completed = True
    result = _result(
        AgentResultStatus.SUCCESS,
        summary="Health endpoint added",
        changed_files=["src/main.py"],
        tests=["pytest"],
        metadata={"verified": True},
    )
    delegation = DelegationRequest(
        task_id=task_key(task),
        agent_type=AgentType.CODING,
        objective="add health endpoint",
        permissions=DEFAULT_REGISTRY.get("coder").permissions,
    )
    ctx = ValidationContext(
        task_state=task,
        result=result,
        delegation=delegation,
        workspace=str(tmp_path),
        required_evidence=["changed_files", "tests", "verification"],
        claims_completion=claims,
        **overrides,
    )
    return ctx


def test_completion_valid_with_all_evidence(tmp_path):
    ctx = _completion_ctx(tmp_path, claims=True)
    result = VALIDATOR.validate(ctx)
    assert result.valid, result.summary
    assert result.status is ValidationStatus.PASS


def test_completion_false_when_requirements_remain(tmp_path):
    ctx = _completion_ctx(tmp_path, claims=True)
    ctx.task_state.requirements[0].completed = False  # requirement not met
    result = VALIDATOR.validate(ctx)
    assert ViolationCode.FALSE_COMPLETION_CLAIM in _code(ctx)
    assert not result.valid


def test_completion_missing_test_evidence(tmp_path):
    ctx = _completion_ctx(tmp_path, claims=True)
    ctx.result = _result(
        AgentResultStatus.SUCCESS,
        summary="done",
        changed_files=["src/main.py"],
        metadata={"verified": True},
    )  # no tests
    assert ViolationCode.FALSE_COMPLETION_CLAIM in _code(ctx)


def test_completion_missing_verification(tmp_path):
    ctx = _completion_ctx(tmp_path, claims=True)
    ctx.result = _result(
        AgentResultStatus.SUCCESS,
        summary="done",
        changed_files=["src/main.py"],
        tests=["pytest"],
    )  # no verified metadata
    assert ViolationCode.FALSE_COMPLETION_CLAIM in _code(ctx)


def test_completion_incomplete_plan(tmp_path):
    ctx = _completion_ctx(tmp_path, claims=True)
    ctx.task_state.plan = TaskPlan(
        goal=ctx.task_state.goal,
        task_type=TaskType.SOFTWARE_ENGINEERING,
        steps=[PlanStep(id=1, description="implement")],
    )
    assert ViolationCode.FALSE_COMPLETION_CLAIM in _code(ctx)


def test_completion_no_claim_no_false_positive(tmp_path):
    # A mid-workflow SUCCESS that does not claim completion must not be
    # flagged — only completion claims are checked against the criteria.
    ctx = _happy_ctx(tmp_path)
    ctx.claims_completion = False
    result = VALIDATOR.validate(ctx)
    assert result.valid


def test_completion_test_claim_contradiction(tmp_path):
    ctx = _completion_ctx(tmp_path, claims=False)
    ctx.test_results = [
        TestResult(
            task_id=ctx.delegation.task_id,
            agent_id="tester-1",
            command="pytest",
            passed=1,
            failed=2,
            failures=[],
        )
    ]
    assert ViolationCode.TEST_CLAIM_CONTRADICTION in _code(ctx)


def test_completion_summary_phrasing_detected(tmp_path):
    # The heuristic catches "everything works"-style claims too.
    ctx = _completion_ctx(tmp_path, claims=False)
    ctx.result = _result(
        AgentResultStatus.SUCCESS,
        summary="Done, everything works.",
        metadata={"verified": True},
    )
    ctx.task_state.requirements[0].completed = False
    assert ViolationCode.FALSE_COMPLETION_CLAIM in _code(ctx)


def test_completion_summary_no_substring_false_positives(tmp_path):
    # Word-boundary regression: "complete"/"done" must not match inside
    # "incomplete", "completely" or "undone", and "not all done" must
    # not count as a completion claim (Fix #7 validation review catch).
    for summary in (
        "The task is incomplete — needs work.",
        "task incomplete",
        "task completely broken",
        "undone everything",
        "everything is undone",
        "not all done yet",
        "no all done",
        "never all done",
        "task is incomplete, fixing now",
    ):
        ctx = _completion_ctx(tmp_path, claims=False)
        ctx.result = _result(
            AgentResultStatus.SUCCESS,
            summary=summary,
            changed_files=["src/main.py"],
            metadata={"verified": True},
        )
        ctx.task_state.requirements[0].completed = False
        # No completion claim is detected, so the incomplete requirement is
        # not held against the agent as a false-completion claim.
        assert ViolationCode.FALSE_COMPLETION_CLAIM not in _code(ctx), summary


def test_completion_summary_legitimate_claims_still_detected(tmp_path):
    # Word-boundary regression: genuine completion phrasing must still fire.
    for summary in (
        "task is complete",
        "task completed",
        "the task is done",
        "all done",
        "all done, verified",
        "everything is done",
        "everything works",
        "done, everything works",
    ):
        ctx = _completion_ctx(tmp_path, claims=False)
        ctx.result = _result(
            AgentResultStatus.SUCCESS,
            summary=summary,
            changed_files=["src/main.py"],
            metadata={"verified": True},
        )
        ctx.task_state.requirements[0].completed = False
        assert ViolationCode.FALSE_COMPLETION_CLAIM in _code(ctx), summary


# ---------------------------------------------------------------------------
# Workspace scope
# ---------------------------------------------------------------------------


def _scope_ctx(tmp_path, changed: list[str], scope: list[str] | None = None) -> ValidationContext:
    ctx = _happy_ctx(tmp_path)
    ctx.result = _result(
        AgentResultStatus.SUCCESS, summary="implemented", changed_files=changed
    )
    ctx.workspace = str(tmp_path)
    if scope is not None:
        ctx.allowed_scope = scope
    return ctx


def test_workspace_allowed_file(tmp_path):
    result = VALIDATOR.validate(_scope_ctx(tmp_path, ["src/app.py"], scope=["src"]))
    assert result.valid


def test_workspace_disallowed_file(tmp_path):
    ctx = _scope_ctx(tmp_path, ["src/payments/payment.py"], scope=["src/auth"])
    result = VALIDATOR.validate(ctx)
    assert ViolationCode.WORKSPACE_SCOPE_VIOLATION in _code(ctx)
    assert not result.valid


def test_workspace_mixed_allowed_and_disallowed(tmp_path):
    ctx = _scope_ctx(
        tmp_path,
        ["src/auth/service.py", "src/payments/payment.py"],
        scope=["src/auth"],
    )
    result = VALIDATOR.validate(ctx)
    assert ViolationCode.WORKSPACE_SCOPE_VIOLATION in _code(ctx)
    assert not result.valid
    assert any(c.name == "scope_clean" and not c.passed for c in result.checks)


def test_workspace_outside_workspace_is_critical(tmp_path):
    ctx = _scope_ctx(tmp_path, ["/etc/passwd"], scope=["src"])
    result = VALIDATOR.validate(ctx)
    assert ViolationCode.UNAUTHORIZED_FILE_ACCESS in _code(ctx)
    assert result.status is ValidationStatus.BLOCKED


# ---------------------------------------------------------------------------
# Idempotency + no side effects
# ---------------------------------------------------------------------------


def test_validation_is_idempotent(tmp_path):
    ctx = _happy_ctx(tmp_path)
    first = VALIDATOR.validate(ctx)
    second = VALIDATOR.validate(ctx)
    # The two runs must be identical except for the diagnostic timestamp.
    first.metadata.pop("validated_at")
    second.metadata.pop("validated_at")
    assert first.model_dump() == second.model_dump()


def test_validation_has_no_side_effects(tmp_path):
    task = _task("Add a health endpoint", "Health endpoint implemented")
    tid = task_key(task)
    artifact = _research_artifact(tid)
    delegation = DelegationRequest(
        task_id=tid,
        agent_type=AgentType.RESEARCH,
        objective="x",
        permissions=DEFAULT_REGISTRY.get("research").permissions,
    )
    state = DEFAULT_REGISTRY.instantiate(AgentType.RESEARCH, "r1", tid, "x")
    ctx = ValidationContext(
        agent_state=state,
        delegation=delegation,
        task_state=task,
        artifacts=[artifact],
        workspace=str(tmp_path),
    )

    task_before = task.model_dump()
    artifact_before = artifact.model_dump()
    permissions_before = delegation.permissions.model_dump()
    ctx_before = ctx.model_dump()

    VALIDATOR.validate(ctx)

    assert task.model_dump() == task_before
    assert artifact.model_dump() == artifact_before
    assert delegation.permissions.model_dump() == permissions_before
    assert ctx.model_dump() == ctx_before
