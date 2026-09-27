"""tests.test_intent_policy_planner
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Comprehensive validation suite for the Intent Policy & Planner Architecture.
Verifies Table 14 test matrix from Ultron_Intent_Policy_Planner_Architecture.pdf:
- Intent & Policy Resolution
- Policy-Aware Planning
- Plan Validation Policy Enforcement
- Runtime Policy Gate Enforcement
- Decoupled Verification
"""

import json

import pytest

from ultron.core.agents.react import ReActAgent
from ultron.core.intelligence.intent_understanding import understand_user_intent
from ultron.core.intelligence.plan_validation import validate_plan
from ultron.core.intelligence.task_planning import fallback_plan
from ultron.core.runtime.policy_gate import check_runtime_policy
from ultron.core.types import (
    AuthorityLevel,
    Capability,
    ChatMessage,
    ExecutionIntent,
    ExecutionPolicy,
    PlanStep,
    Role,
    StepStatus,
    TaskLifecycleStatus,
    TaskPlan,
    TaskState,
    TaskType,
    WorkspaceKind,
)


class FakeEngine:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.call_count = 0

    async def generate(self, messages: list[dict], **kwargs) -> str:
        self.call_count += 1
        if not self._responses:
            return ""
        return self._responses.pop(0)


# ---------------------------------------------------------------------------
# 1. Intent & Policy Resolution Tests
# ---------------------------------------------------------------------------


def test_intent_resolver_read_only_debugging():
    prompt = (
        "Investigate why users lose authenticated state in the backend. "
        "Do not modify any files, do not delete anything, and do not run tests."
    )
    intent = understand_user_intent(prompt, task_type=TaskType.DEBUGGING)
    assert intent.task_type == TaskType.DEBUGGING
    assert intent.execution_intent == ExecutionIntent.DIAGNOSE
    assert intent.policy.is_read_only
    assert intent.policy.mutation_authority == AuthorityLevel.FORBIDDEN
    assert intent.policy.execution_authority == AuthorityLevel.FORBIDDEN
    assert Capability.WRITE_FILES not in intent.policy.allowed_capabilities
    assert Capability.DELETE_FILES not in intent.policy.allowed_capabilities
    assert Capability.RUN_COMMANDS not in intent.policy.allowed_capabilities


def test_intent_resolver_repair_debugging():
    prompt = "Fix the authentication bug where users lose session state in backend."
    intent = understand_user_intent(prompt, task_type=TaskType.DEBUGGING)
    assert intent.execution_intent == ExecutionIntent.REPAIR
    assert intent.policy.can_mutate
    assert intent.policy.can_execute


def test_intent_resolver_plan_debugging():
    prompt = "Investigate the memory leak and propose a fix plan without changing any code."
    intent = understand_user_intent(prompt, task_type=TaskType.DEBUGGING)
    assert intent.execution_intent == ExecutionIntent.PLAN
    assert intent.policy.is_read_only
    assert intent.policy.mutation_authority == AuthorityLevel.FORBIDDEN


def test_intent_resolver_explain_architecture():
    prompt = "Explain how session validation and authentication tokens connect across the backend."
    intent = understand_user_intent(prompt, task_type=TaskType.RESEARCH)
    assert intent.execution_intent == ExecutionIntent.EXPLAIN
    assert intent.policy.is_read_only
    assert intent.policy.mutation_authority == AuthorityLevel.FORBIDDEN


# ---------------------------------------------------------------------------
# 2. Policy-Aware Planning Tests
# ---------------------------------------------------------------------------


def test_debugging_plan_read_only():
    prompt = "Investigate the login failure. Do not modify files or run tests."
    plan = fallback_plan(prompt, TaskType.DEBUGGING, WorkspaceKind.EXISTING_PROJECT)
    assert plan.task_type == TaskType.DEBUGGING
    assert plan.execution_intent == ExecutionIntent.DIAGNOSE
    assert plan.execution_policy.is_read_only
    assert len(plan.steps) == 4
    # Ensure no step requires mutation or test execution
    for step in plan.steps:
        desc = step.description.lower()
        assert "apply targeted bug fix" not in desc
        assert "run regression tests" not in desc


def test_debugging_plan_repair():
    prompt = "Fix the login failure and make the tests pass."
    plan = fallback_plan(prompt, TaskType.DEBUGGING, WorkspaceKind.EXISTING_PROJECT)
    assert plan.task_type == TaskType.DEBUGGING
    assert plan.execution_intent == ExecutionIntent.REPAIR
    assert len(plan.steps) == 5
    assert any("apply" in s.description.lower() for s in plan.steps)
    assert any("test" in s.description.lower() for s in plan.steps)


# ---------------------------------------------------------------------------
# 3. Plan Validation Policy Tests
# ---------------------------------------------------------------------------


def test_plan_validator_blocks_mutating_step_under_read_only():
    step = PlanStep(
        id=1,
        description="Apply targeted bug fix to affected files",
        purpose="Modify code to resolve the defect",
        expected_outcome="Bug fix is implemented in source files",
        completion_criteria=["Code modification applied"],
    )
    plan = TaskPlan(
        goal="Investigate issue",
        task_type=TaskType.DEBUGGING,
        steps=[step],
        completion_criteria=["Investigate issue"],
        verification_requirements=["Verified"],
        execution_policy=ExecutionPolicy.read_only(),
    )
    report = validate_plan(plan)
    assert not report.valid
    assert any(i.code == "policy_violation_mutation" for i in report.issues)


def test_plan_validator_blocks_executing_step_under_read_only():
    step = PlanStep(
        id=1,
        description="Run regression tests and validate fix",
        purpose="Execute test suite to verify resolution",
        expected_outcome="Tests pass",
        completion_criteria=["Regression tests pass"],
    )
    plan = TaskPlan(
        goal="Investigate issue",
        task_type=TaskType.DEBUGGING,
        steps=[step],
        completion_criteria=["Investigate issue"],
        verification_requirements=["Verified"],
        execution_policy=ExecutionPolicy.read_only(),
    )
    report = validate_plan(plan)
    assert not report.valid
    assert any(i.code == "policy_violation_execution" for i in report.issues)


# ---------------------------------------------------------------------------
# 4. Runtime Policy Gate Enforcement Tests
# ---------------------------------------------------------------------------


def test_runtime_policy_gate_blocks_mutation_under_read_only():
    policy = ExecutionPolicy.read_only()
    verdict = check_runtime_policy(
        policy, "write_file", {"file_path": "test.py", "content": "x = 1"}
    )
    assert not verdict.allowed
    assert verdict.violation_type in ("FORBIDDEN_CAPABILITY", "MUTATION_FORBIDDEN")


def test_runtime_policy_gate_blocks_command_under_read_only():
    policy = ExecutionPolicy.read_only()
    verdict = check_runtime_policy(
        policy, "run_command", {"command": "pytest tests/"}
    )
    assert not verdict.allowed


def test_runtime_policy_gate_allows_git_read_under_read_only():
    policy = ExecutionPolicy.read_only()
    verdict = check_runtime_policy(
        policy, "run_command", {"command": "git status"}
    )
    assert verdict.allowed
    assert verdict.required_capability == Capability.GIT_READ


def test_runtime_policy_gate_allows_read_file_under_read_only():
    policy = ExecutionPolicy.read_only()
    verdict = check_runtime_policy(
        policy, "read_file", {"file_path": "main.py"}
    )
    assert verdict.allowed
    assert verdict.required_capability == Capability.READ_FILES


# ---------------------------------------------------------------------------
# 5. Decoupled Verification Tests
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_verifier_allows_diagnostic_completion_without_execution():
    goal = "Investigate login issue. Do not modify files or run tests."
    plan = fallback_plan(goal, TaskType.DEBUGGING, WorkspaceKind.EXISTING_PROJECT)
    task = TaskState(goal=goal, task_type=TaskType.DEBUGGING)
    task.attach_plan(plan)

    # Simulate read-only investigative tool executions
    task.record_tool_execution("repo_map", target=".", success=True, detail="repo overview")
    task.record_tool_execution("read_file", target="auth.py", success=True, detail="code snippet")

    # Mark plan steps completed
    for step in task.plan.steps:
        step.status = StepStatus.SUCCEEDED

    # Proposed answer providing complete diagnosis
    proposed_answer = (
        "Root cause analysis: The session tokens expire because the cookie expiry "
        "is set to 0 in auth.py. Connected components: auth.py -> router.py."
    )

    import json
    plan_criteria = [{"description": r.description, "satisfied": True} for r in task.requirements]
    verification_response = json.dumps({
        "step_criteria": [{"description": goal, "satisfied": True}],
        "plan_criteria": plan_criteria,
        "step_failed": False,
        "plan_revision": None,
    })

    engine = FakeEngine([verification_response])
    agent = ReActAgent(engine)
    messages: list[ChatMessage] = [ChatMessage(role=Role.USER, content=goal)]

    success, reply = await agent._verify_plan_task(
        task, goal, proposed_answer, messages
    )
    assert success
    assert reply is not None
    assert task.lifecycle_status == TaskLifecycleStatus.COMPLETED


@pytest.mark.anyio
async def test_verifier_requires_execution_for_repair():
    goal = "Fix the broken login and make tests pass"
    plan = fallback_plan(goal, TaskType.DEBUGGING, WorkspaceKind.EXISTING_PROJECT)
    task = TaskState(goal=goal, task_type=TaskType.DEBUGGING)
    task.attach_plan(plan)

    # Simulate only code edit, but NO command execution (tests were not run)
    task.record_tool_execution("write_file", target="auth.py", success=True, detail="applied fix")

    for step in task.plan.steps:
        step.status = StepStatus.SUCCEEDED

    proposed_answer = "I fixed the bug in auth.py."
    repair_criteria = [{"description": r.description, "satisfied": True} for r in task.requirements]
    verification_response = json.dumps({
        "step_criteria": [{"description": goal, "satisfied": True}],
        "plan_criteria": repair_criteria,
        "step_failed": False,
        "plan_revision": None,
    })

    engine = FakeEngine([verification_response])
    agent = ReActAgent(engine)
    messages: list[ChatMessage] = [ChatMessage(role=Role.USER, content=goal)]

    # Because intent is REPAIR and policy permits mutation/execution, has_exec is enforced
    success, reply = await agent._verify_plan_task(
        task, goal, proposed_answer, messages
    )
    assert not success
    assert reply is None
    assert task.lifecycle_status != TaskLifecycleStatus.COMPLETED


# ---------------------------------------------------------------------------
# 6. Read-only evidence-gated plan completion
# ---------------------------------------------------------------------------


def _readonly_plan_task(goal: str):
    """A read-only diagnostic task with one recorded investigation tool."""
    plan = fallback_plan(goal, TaskType.DEBUGGING, WorkspaceKind.EXISTING_PROJECT)
    assert plan.execution_policy.is_read_only
    task = TaskState(goal=goal, task_type=TaskType.DEBUGGING)
    task.attach_plan(plan)
    task.record_tool_execution(
        "read_file", target="auth.py", success=True, detail="read auth.py"
    )
    return task


def _all_false_verdict(task) -> str:
    """A verifier verdict that marks every criterion unsatisfied."""
    step = task.current_plan_step()
    return json.dumps(
        {
            "step_criteria": [
                {"description": c, "satisfied": False}
                for c in (step.completion_criteria if step else [])
            ],
            "plan_criteria": [
                {"description": r.description, "satisfied": False}
                for r in task.requirements
            ],
            "step_failed": False,
            "plan_revision": None,
        }
    )


@pytest.mark.anyio
async def test_readonly_step_advances_on_investigation_evidence():
    goal = "Investigate the login failure. Do not modify files or run tests."
    task = _readonly_plan_task(goal)
    step1 = task.current_plan_step()
    assert step1 is not None and step1.id == 1

    engine = FakeEngine([_all_false_verdict(task)])
    agent = ReActAgent(engine)
    messages: list[ChatMessage] = [ChatMessage(role=Role.USER, content=goal)]

    success, reply = await agent._verify_plan_task(
        task, goal, "Investigated auth.py.", messages
    )
    assert not success
    assert reply is None
    # The recorded read-only investigation advanced the step despite the
    # verifier marking its criteria unsatisfied.
    assert task.plan.steps[0].status == StepStatus.SUCCEEDED


@pytest.mark.anyio
async def test_readonly_diagnosis_completes_from_evidence_not_model_verdict():
    goal = "Investigate the login failure. Do not modify files."
    task = _readonly_plan_task(goal)
    for step in task.plan.steps:
        step.status = StepStatus.SUCCEEDED

    engine = FakeEngine([_all_false_verdict(task)])
    agent = ReActAgent(engine)
    messages: list[ChatMessage] = [ChatMessage(role=Role.USER, content=goal)]

    success, reply = await agent._verify_plan_task(
        task, goal, "Diagnosis with concrete evidence.", messages
    )
    assert success
    assert reply is not None
    assert task.lifecycle_status == TaskLifecycleStatus.COMPLETED


@pytest.mark.anyio
async def test_readonly_bare_failure_verdict_does_not_terminate_plan():
    # A verifier that claims "step_failed: true" with no reason while every
    # recorded action succeeded must not terminate a read-only diagnosis
    # (STOP strategy) on an unsubstantiated verdict.
    goal = "Investigate the login failure. Do not modify files."
    task = _readonly_plan_task(goal)
    engine = FakeEngine(
        [
            json.dumps(
                {
                    "step_criteria": [],
                    "plan_criteria": [],
                    "step_failed": True,
                    "plan_revision": None,
                }
            )
        ]
    )
    agent = ReActAgent(engine)
    messages: list[ChatMessage] = [ChatMessage(role=Role.USER, content=goal)]

    success, reply = await agent._verify_plan_task(
        task, goal, "Investigated auth.py.", messages
    )
    assert not success
    assert reply is None
    assert task.plan.steps[0].status != StepStatus.FAILED
    assert task.lifecycle_status != TaskLifecycleStatus.FAILED


@pytest.mark.anyio
async def test_readonly_substantiated_failure_terminates_plan():
    # A failure verdict WITH a reason goes through the step's failure
    # strategy as designed (step 1 retries; exhausted retries terminate).
    goal = "Investigate the login failure. Do not modify files."
    task = _readonly_plan_task(goal)
    task.plan.steps[0].attempts = task.plan.steps[0].retry_policy  # retries spent
    engine = FakeEngine(
        [
            json.dumps(
                {
                    "step_criteria": [],
                    "plan_criteria": [],
                    "step_failed": True,
                    "step_failed_reason": "repository is empty, nothing to inspect",
                    "plan_revision": None,
                }
            )
        ]
    )
    agent = ReActAgent(engine)
    messages: list[ChatMessage] = [ChatMessage(role=Role.USER, content=goal)]

    success, reply = await agent._verify_plan_task(
        task, goal, "Cannot investigate.", messages
    )
    assert success
    assert reply is not None
    assert "could not complete plan step" in reply.content.lower()


@pytest.mark.anyio
async def test_repair_step_does_not_advance_from_investigation_evidence():
    goal = "Fix the login failure and make the tests pass."
    plan = fallback_plan(goal, TaskType.DEBUGGING, WorkspaceKind.EXISTING_PROJECT)
    assert not plan.execution_policy.is_read_only
    task = TaskState(goal=goal, task_type=TaskType.DEBUGGING)
    task.attach_plan(plan)
    task.record_tool_execution(
        "read_file", target="auth.py", success=True, detail="read auth.py"
    )
    step1 = task.current_plan_step()
    assert step1 is not None and step1.id == 1

    engine = FakeEngine([_all_false_verdict(task)])
    agent = ReActAgent(engine)
    messages: list[ChatMessage] = [ChatMessage(role=Role.USER, content=goal)]

    success, reply = await agent._verify_plan_task(
        task, goal, "Still investigating.", messages
    )
    assert not success
    assert reply is None
    # A writable task must keep requiring the model's criteria to be met.
    assert task.plan.steps[0].status != StepStatus.SUCCEEDED
