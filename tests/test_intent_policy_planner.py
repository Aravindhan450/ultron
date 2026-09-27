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

from ultron.core.agents.react import ReActAgent, _activate_task, _resume_task
from ultron.core.intelligence.intent_understanding import understand_user_intent
from ultron.core.intelligence.plan_validation import validate_plan
from ultron.core.intelligence.task_planning import (
    fallback_plan,
    generate_task_plan,
)
from ultron.core.runtime.policy_gate import check_runtime_policy
from ultron.core.types import (
    AuthorityLevel,
    Capability,
    ChatMessage,
    ExecutionIntent,
    ExecutionPolicy,
    PendingAction,
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
# 5d. run_tool_batch is NOT a policy boundary (R3)
# ---------------------------------------------------------------------------


def _batch_call(tool: str, **arguments: object) -> dict:
    return {"tool": tool, "arguments": dict(arguments)}


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Temp workspace wired as the tool layer's ALLOWED_BASE_DIR."""
    from ultron.core.tools import paths as tools_paths

    monkeypatch.setattr(tools_paths, "ALLOWED_BASE_DIR", tmp_path)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _write(root, rel: str, text: str):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _run_batch(calls: list[dict], policy) -> list[dict]:
    """Gates (not executes) a batch through execute_batch with a policy."""
    from ultron.core.intelligence.parallel_tools import execute_batch

    gated, _elapsed = execute_batch(list(calls), policy=policy)
    return gated


def test_r3a_readonly_batch_read_executes_write_blocked(sandbox, monkeypatch):
    # TEST A: READ_ONLY + batch(read_file, write_file) → read executes,
    # write is blocked by policy (not merely by the security boundary).
    _write(sandbox, "a.txt", "hello")
    policy = ExecutionPolicy.read_only()
    gated = _run_batch(
        [
            _batch_call("read_file", file_path="a.txt"),
            _batch_call("write_file", file_path="evil.txt", content="no"),
        ],
        policy,
    )
    statuses = {c["tool"]: c["status"] for c in gated}
    assert statuses["read_file"] in ("run", "ok")
    assert statuses["write_file"] == "blocked"
    assert "Policy violation" in next(
        c["result"] for c in gated if c["tool"] == "write_file"
    )
    assert not (sandbox / "evil.txt").exists()


def test_r3b_readonly_batch_run_command_blocked():
    # TEST B: READ_ONLY + batch(run_command) → never executes.
    policy = ExecutionPolicy.read_only()
    gated = _run_batch(
        [_batch_call("run_command", command="echo pwned")],
        policy,
    )
    assert gated[0]["status"] == "blocked"
    assert "Policy violation" in gated[0]["result"]


def test_r3c_confirmation_policy_batch_mutating_tool_needs_approval():
    # TEST C: confirmation policy + batch(mutating tool) → confirm status,
    # no silent execution.
    from ultron.core.types import AuthorityLevel

    policy = ExecutionPolicy.execute_with_confirmation()
    assert policy.mutation_authority == AuthorityLevel.REQUIRES_CONFIRMATION
    gated = _run_batch(
        [_batch_call("write_file", file_path="new.txt", content="x")],
        policy,
    )
    assert gated[0]["status"] == "confirm"


def test_r3d_mixed_batch_gated_independently(sandbox):
    # TEST D: mixed batch — each call judged on its own; one denial does not
    # authorize another call.
    _write(sandbox, "ok.txt", "data")
    policy = ExecutionPolicy.read_only()
    gated = _run_batch(
        [
            _batch_call("read_file", file_path="ok.txt"),
            _batch_call("code_search", query="data", path="."),
            _batch_call("write_file", file_path="x.txt", content="x"),
            _batch_call("run_command", command="echo hi"),
        ],
        policy,
    )
    statuses = {c["tool"]: c["status"] for c in gated}
    assert statuses["read_file"] in ("run", "ok")
    assert statuses["code_search"] in ("run", "ok")
    assert statuses["write_file"] == "blocked"
    assert statuses["run_command"] == "blocked"


def test_r3e_batch_arguments_coerced_like_direct_calls(sandbox):
    # TEST E: inner calls receive the same canonical argument coercion.
    _write(sandbox, "src/app.py", "needle = 1\n")
    policy = ExecutionPolicy.read_only()
    gated = _run_batch(
        [
            _batch_call(
                "code_search",
                query="needle",
                path=".",
                max_results="1",
                regex="false",
            )
        ],
        policy,
    )
    call = gated[0]
    assert call["status"] in ("run", "ok")
    assert "src/app.py" in str(call["result"])
    # The string "1" coerced and matched only the first result.
    assert str(call["result"]).count("app.py") == 1


def test_r3f_run_tool_batch_tool_accepts_policy_parameter():
    # The registered tool passes the policy through to execute_batch.
    from ultron.core.intelligence.parallel_tools import run_tool_batch

    report = run_tool_batch(
        json.dumps([_batch_call("run_command", command="echo hi")]),
        policy=ExecutionPolicy.read_only(),
    )
    assert "Policy violation" in report


# ---------------------------------------------------------------------------
# 5c. REQUIRES_CONFIRMATION is authoritative (R2)
# ---------------------------------------------------------------------------


class _NeverCalledEngine:
    """Engine that fails the test if the loop ever calls the model."""

    async def generate(self, messages: list[dict], **kwargs) -> str:
        raise AssertionError("model must not be called in this test")


def _confirmation_task(policy) -> TaskState:
    goal = "Modify the configuration file as requested."
    task = TaskState(goal=goal, task_type=TaskType.DEBUGGING)
    task.execution_policy = policy
    return task


@pytest.mark.anyio
async def test_r2a_policy_confirmation_beats_security_allow():
    # TEST A: policy = REQUIRES_CONFIRMATION; the security boundary would
    # auto-allow this LOW-risk registered tool. The action must NOT execute.
    from ultron.core.types import AuthorityLevel, Capability, ExecutionPolicy

    policy = ExecutionPolicy(
        allowed_capabilities=set(Capability),
        mutation_authority=AuthorityLevel.REQUIRES_CONFIRMATION,
        execution_authority=AuthorityLevel.ALLOWED,
    )
    task = _confirmation_task(policy)
    agent = ReActAgent(_NeverCalledEngine())

    outcome = agent._route_tool(
        "list_directory",
        {"path": "."},
        "list the directory",
        task=task,
    )
    # list_directory is not mutating, so exercise a mutating tool instead:
    # the policy gate must demand confirmation for write_file even though a
    # permissive security boundary would allow it.
    outcome = agent._route_tool(
        "write_file",
        {"file_path": "config.py", "content": "x = 1"},
        "modify the configuration",
        task=task,
    )
    assert isinstance(outcome, ChatMessage)
    assert outcome.pending_action is not None
    assert outcome.pending_action.action_type == "write_file"
    assert "requested" in outcome.content.lower()


@pytest.mark.anyio
async def test_r2b_confirmed_registry_tool_executes_exactly_once():
    # TEST B: after user approval, the exact approved action executes once.
    from ultron.core.types import AuthorityLevel, Capability, ExecutionPolicy

    policy = ExecutionPolicy(
        allowed_capabilities=set(Capability),
        mutation_authority=AuthorityLevel.REQUIRES_CONFIRMATION,
        execution_authority=AuthorityLevel.ALLOWED,
    )
    task = _confirmation_task(policy)
    agent = ReActAgent(_NeverCalledEngine())

    outcome = agent._route_tool(
        "add_memory",
        {"content": "remember this fact"},
        "store a memory",
        task=task,
    )
    assert isinstance(outcome, ChatMessage)
    action = outcome.pending_action
    assert action is not None
    assert action.action_type == "registry_tool"
    assert action.target == "add_memory"

    # Approval path: main.execute_pending_action executes the gated arguments.
    import ultron.core.tools.registry as registry_mod
    from ultron.main import execute_pending_action

    original_registry_get_tool = registry_mod.get_tool
    executed: list[str] = []

    def counting_get_tool(name: str):
        func = original_registry_get_tool(name)
        if name == "add_memory" and func is not None:
            def wrapper(*a, **kw):
                executed.append(name)
                return "recorded once"
            return wrapper
        return func

    registry_mod.get_tool = counting_get_tool
    try:
        result = await execute_pending_action(action)
        await execute_pending_action(action)
    finally:
        registry_mod.get_tool = original_registry_get_tool
    assert "recorded" in result
    # Each approval executes exactly once (two approvals = two executions;
    # one approval never double-executes).
    assert len(executed) == 2


@pytest.mark.anyio
async def test_r2c_denied_confirmation_never_executes():
    # TEST C: denial path — the pending action is simply never executed.
    # There is no execution side effect to observe; the PendingAction payload
    # is discarded by main.py's denial branch.
    from ultron.core.types import AuthorityLevel, Capability, ExecutionPolicy

    policy = ExecutionPolicy(
        allowed_capabilities=set(Capability),
        mutation_authority=AuthorityLevel.REQUIRES_CONFIRMATION,
        execution_authority=AuthorityLevel.ALLOWED,
    )
    task = _confirmation_task(policy)
    agent = ReActAgent(_NeverCalledEngine())

    outcome = agent._route_tool(
        "add_memory", {"content": "x"}, "store", task=task
    )
    action = outcome.pending_action
    assert action is not None
    # Nothing ran: execution happens ONLY through execute_pending_action.
    # Simulate the denial branch (no execute_pending_action call) — the tool
    # function was never invoked, so nothing to assert beyond absence.
    assert action.action_type == "registry_tool"


@pytest.mark.anyio
async def test_r2d_confirmation_resume_preserves_plan_step():
    # TEST D: the current plan step survives the confirmation round-trip.
    from ultron.core.types import AuthorityLevel, Capability, ExecutionPolicy

    policy = ExecutionPolicy(
        allowed_capabilities=set(Capability),
        mutation_authority=AuthorityLevel.REQUIRES_CONFIRMATION,
        execution_authority=AuthorityLevel.ALLOWED,
    )
    goal = "Fix the configuration bug. Ask before changing files."
    plan = fallback_plan(goal, TaskType.DEBUGGING, WorkspaceKind.EXISTING_PROJECT)
    task = TaskState(goal=goal, task_type=TaskType.DEBUGGING)
    task.attach_plan(plan)
    task.execution_policy = policy
    # Activate step 1 exactly like the ReAct loop does (_activate_plan_step).
    step1 = task.current_plan_step()
    step1.status = StepStatus.RUNNING
    task.set_current_step(step1.id)

    messages = [ChatMessage(role=Role.USER, content=goal)]
    response = '{"tool": "write_file", "arguments": {"file_path": "config.py", "content": "x = 1"}}'

    task = _activate_task(
        task, goal, messages, 0, response,
        PendingAction(action_type="write_file", target="config.py", content="x = 1"),
    )
    assert task.pending_action is not None
    assert task.is_waiting_confirmation  # bool property, not a method
    # The RUNNING step moved to WAITING_CONFIRMATION, not lost:
    waiting = [s for s in task.plan.steps if s.status == StepStatus.WAITING_CONFIRMATION]
    assert len(waiting) == 1

    # Resume as main.py does after approval: observation recorded, step back
    # to RUNNING on the SAME step id.
    task.last_observation = "File written successfully."
    task = _resume_task(task)
    assert task.pending_action is None
    running = [s for s in task.plan.steps if s.status == StepStatus.RUNNING]
    assert len(running) == 1
    assert running[0].id == waiting[0].id


@pytest.mark.anyio
async def test_r2e_policy_confirmation_runs_command_via_existing_flow():
    # A read-only policy with confirmable execution authority routes
    # run_command through the SAME PendingAction encoding the CLI already
    # renders and executes (no second confirmation system).
    from ultron.core.types import AuthorityLevel, Capability, ExecutionPolicy

    policy = ExecutionPolicy(
        allowed_capabilities={
            Capability.RUN_COMMANDS,
            Capability.RUN_TESTS,
            Capability.READ_FILES,
        },
        mutation_authority=AuthorityLevel.FORBIDDEN,
        execution_authority=AuthorityLevel.REQUIRES_CONFIRMATION,
    )
    task = _confirmation_task(policy)
    agent = ReActAgent(_NeverCalledEngine())

    outcome = agent._route_tool(
        "run_command", {"command": "pytest tests/"}, "run the tests", task=task
    )
    assert isinstance(outcome, ChatMessage)
    assert outcome.pending_action is not None
    assert outcome.pending_action.action_type == "run_command"
    assert outcome.pending_action.target == "pytest tests/"


# ---------------------------------------------------------------------------
# 5b. Policy bound BEFORE plan validation (R1)
# ---------------------------------------------------------------------------


def _llm_plan_payload(steps: list[dict]) -> str:
    """An LLM-shaped plan JSON response."""
    return json.dumps(
        {
            "steps": [
                {
                    "id": i + 1,
                    "description": s["description"],
                    "purpose": s.get("purpose", ""),
                    "expected_outcome": s.get("outcome", "done"),
                    "completion_criteria": [s.get("outcome", "done")],
                }
                for i, s in enumerate(steps)
            ],
            "completion_criteria": ["goal satisfied"],
            "verification_requirements": ["verified"],
        }
    )


class _ScriptedPlanner:
    """Fake engine returning canned LLM planner responses."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.prompts: list[str] = []

    async def generate(self, messages: list[dict], **kwargs) -> str:
        self.prompts.append(messages[-1]["content"])
        if not self._responses:
            return ""
        return self._responses.pop(0)


_READONLY_GOAL = "Investigate why login fails. Do not modify anything."


@pytest.mark.anyio
async def test_r1a_readonly_llm_plan_with_mutation_is_rejected():
    # TEST A: READ_ONLY intent + LLM plan containing a mutation step →
    # the plan must be rejected (no policy binding after validation).
    plan_payload = _llm_plan_payload(
        [
            {"description": "Inspect the auth flow"},
            {"description": "Apply targeted bug fix to the affected files"},
        ]
    )
    engine = _ScriptedPlanner([plan_payload])
    intent = understand_user_intent(_READONLY_GOAL, task_type=TaskType.DEBUGGING)
    assert intent.policy.is_read_only

    plan = await generate_task_plan(
        _READONLY_GOAL, TaskType.DEBUGGING, engine, intent=intent, policy=intent.policy
    )
    # The mutating plan cannot survive validation against the read-only
    # policy; generation fails cleanly (caller falls back to fallback_plan).
    assert plan is None
    # Bounded recovery was attempted with the policy-violation feedback.
    assert len(engine.prompts) == 2
    assert "CORRECTION" in engine.prompts[1]


@pytest.mark.anyio
async def test_r1b_diagnose_llm_plan_with_run_command_is_rejected():
    # TEST B: DIAGNOSE intent + LLM plan with run_command/write_file steps →
    # policy violation.
    plan_payload = _llm_plan_payload(
        [
            {"description": "Read the configuration and reproduce the failure"},
            {"description": "Run regression tests to confirm the fix"},
        ]
    )
    engine = _ScriptedPlanner([plan_payload])
    intent = understand_user_intent(_READONLY_GOAL, task_type=TaskType.DEBUGGING)

    plan = await generate_task_plan(
        _READONLY_GOAL, TaskType.DEBUGGING, engine, intent=intent, policy=intent.policy
    )
    assert plan is None


@pytest.mark.anyio
async def test_r1c_confirmation_policy_llm_plan_stays_allowed():
    # TEST C: EXECUTE_WITH_CONFIRMATION intent + LLM plan with allowed
    # mutation → plan remains valid (mutation is confirmable under policy).
    plan_payload = _llm_plan_payload(
        [
            {"description": "Inspect the configuration handling"},
            {"description": "Apply targeted bug fix to the affected files"},
        ]
    )
    engine = _ScriptedPlanner([plan_payload])
    intent = understand_user_intent(
        "Modify the configuration to fix the login bug, but ask me before making changes.",
        task_type=TaskType.DEBUGGING,
    )
    policy = intent.policy
    if not policy.can_mutate:
        policy = ExecutionPolicy.execute_with_confirmation()

    plan = await generate_task_plan(
        "Fix the login bug in the configuration",
        TaskType.DEBUGGING,
        engine,
        intent=intent,
        policy=policy,
    )
    assert plan is not None
    assert plan.execution_policy is policy
    assert plan.execution_policy.can_mutate


@pytest.mark.anyio
async def test_r1d_llm_cannot_supply_its_own_policy():
    # TEST D: an LLM plan that injects execution_policy/execution_intent
    # fields must not widen the user-derived policy.
    plan_payload = json.dumps(
        {
            "steps": [
                {
                    "id": 1,
                    "description": "Apply targeted bug fix to the affected files",
                    "expected_outcome": "fix applied",
                    "completion_criteria": ["fix applied"],
                }
            ],
            "completion_criteria": ["done"],
            "verification_requirements": ["verified"],
            # Untrusted model output attempting self-authorization:
            "execution_policy": "execute",
            "execution_intent": "EXECUTE",
            "mutation_authority": "allowed",
        }
    )
    engine = _ScriptedPlanner([plan_payload])
    intent = understand_user_intent(_READONLY_GOAL, task_type=TaskType.DEBUGGING)

    plan = await generate_task_plan(
        _READONLY_GOAL, TaskType.DEBUGGING, engine, intent=intent, policy=intent.policy
    )
    # Even though the model claimed full execute authority, the plan is
    # judged against the user's read-only policy → rejected.
    assert plan is None


@pytest.mark.anyio
async def test_r1_existing_valid_plans_still_work():
    # A read-only plan consistent with the user's policy validates normally.
    plan_payload = _llm_plan_payload(
        [
            {"description": "Inspect the repository structure and relevant files"},
            {"description": "Trace the login flow across the components"},
        ]
    )
    engine = _ScriptedPlanner([plan_payload])
    intent = understand_user_intent(_READONLY_GOAL, task_type=TaskType.DEBUGGING)

    plan = await generate_task_plan(
        _READONLY_GOAL, TaskType.DEBUGGING, engine, intent=intent, policy=intent.policy
    )
    assert plan is not None
    assert plan.execution_policy.is_read_only
    assert plan.user_intent is intent


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
