"""
Integration tests for plan-aware execution (Fix #3).

The general task planning layer (Fix #2) is now connected to the ReAct
execution pipeline: a complex request is classified and planned up front
(prepare_task_for_execution), the resulting TaskState carries a validated
structured plan, and ReActAgent executes it step-by-step:

    GOAL -> TASK -> PLAN -> CURRENT STEP -> TOOL ACTION -> OBSERVATION
    -> TASKSTATE UPDATE -> NEXT STEP -> VERIFICATION -> COMPLETION

Scenarios covered (A-Q per the spec):

  A  feature implementation task          J  failed intermediate step
  B  bug fixing task                      K  retry / recovery
  C  refactoring task                     L  adaptive plan modification
  D  repository analysis                  M  final verification
  E  code review                          N  model falsely claiming completion
  F  dependency update                    O  max iteration protection
  G  configuration task                   P  existing simple commands
  H  multi-step filesystem task           Q  informational requests
  I  confirmation interruption

Central invariants under test:
- the plan is the source of truth; a step advances only when ALL its
  completion criteria are verified (no LLM skipping A -> F)
- one tool call never completes a step or the task
- confirmations, failures, retries and plan revisions never lose completed work
- the LLM saying "done" is never sufficient; TaskState + plan decide
- informational / simple requests stay on the fast path (no plan)
"""

from __future__ import annotations

import asyncio
import json

import pytest

from ultron.core.agents.react import ReActAgent
from ultron.core.intelligence.task_planning import prepare_task_for_execution
from ultron.core.tools import paths
from ultron.core.tools import registry as reg
from ultron.core.types import (
    FailureStrategy,
    PlanStep,
    StepStatus,
    TaskPlan,
    TaskState,
    TaskStatus,
    TaskType,
    WorkspaceKind,
)
from ultron.main import (
    continue_task_after_confirmation,
    execute_pending_action,
)


class FakeEngine:
    """Scripted engine — no real LLM or network access needed."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    async def generate(self, messages, **kwargs):
        self.calls.append(messages)
        return self._responses.pop(0) if self._responses else ""

    async def stream(self, messages, **kwargs):
        yield ""


def _run(coro):
    return asyncio.run(coro)


def _tool(tool, **arguments):
    return f"```json\n{json.dumps({'tool': tool, 'arguments': arguments})}\n```"


def plan_verify(
    step_criteria,
    plan_criteria,
    step_failed=False,
    revision=None,
    revision_reason=None,
) -> str:
    """Builds a plan-mode verification JSON payload."""
    payload = {
        "step_criteria": [
            {"description": d, "satisfied": s} for d, s in step_criteria
        ],
        "plan_criteria": [
            {"description": d, "satisfied": s} for d, s in plan_criteria
        ],
        "step_failed": step_failed,
        "plan_revision": revision,
    }
    if revision_reason is not None:
        payload["step_failed_reason"] = revision_reason
    return json.dumps(payload)


def feature_plan(**overrides) -> TaskPlan:
    """Generic 3-step software-engineering plan (not TodoList-specific)."""
    plan = TaskPlan(
        goal="Create a FastAPI backend",
        task_type=TaskType.SOFTWARE_ENGINEERING,
        workspace=WorkspaceKind.NEW_WORKSPACE,
        steps=[
            PlanStep(
                id=1,
                description="Establish the project workspace",
                purpose="Create the working directory",
                expected_outcome="Working directory exists",
                completion_criteria=["Project directory exists"],
            ),
            PlanStep(
                id=2,
                description="Implement the backend application",
                purpose="Build the API",
                expected_outcome="Application files exist",
                completion_criteria=["Backend application files exist"],
                dependencies=[1],
            ),
            PlanStep(
                id=3,
                description="Verify the application",
                purpose="Confirm the app works",
                expected_outcome="Application verified",
                completion_criteria=["Application verified"],
                dependencies=[2],
            ),
        ],
        completion_criteria=["FastAPI backend implemented"],
        verification_requirements=["Backend starts without errors"],
    )
    return plan.model_copy(update=overrides)


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Points cwd + the file-policy allowlist at a temp dir so real tools are safe."""
    monkeypatch.setattr(paths, "ALLOWED_BASE_DIR", tmp_path)
    monkeypatch.chdir(tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# A — Feature implementation task (full plan-driven flow, real tools)
# ---------------------------------------------------------------------------


def test_a_feature_implementation_flow(sandbox):
    task = TaskState(goal="Create a FastAPI backend")
    task.attach_plan(feature_plan())

    engine = FakeEngine(
        [
            _tool("run_command", command="mkdir backend"),
            _tool("write_file", filename="backend/main.py", content="from fastapi import FastAPI"),
            "The backend app is implemented.",
            plan_verify(
                [("Project directory exists", True)],
                [("FastAPI backend implemented", False), ("Backend starts without errors", False)],
            ),
            _tool("write_file", filename="backend/app.py", content="app = FastAPI()"),
            "The application code is complete.",
            plan_verify(
                [("Backend application files exist", True)],
                [("FastAPI backend implemented", True), ("Backend starts without errors", False)],
            ),
            _tool("run_command", command="python -c 'import backend.main'"),
            "The backend starts without errors.",
            plan_verify(
                [("Application verified", True)],
                [("FastAPI backend implemented", True), ("Backend starts without errors", True)],
            ),
        ]
    )
    agent = ReActAgent(engine)

    msg = _run(agent.run("Create a FastAPI backend", task=task))
    task = msg.task_state

    # Confirmations + execution, one per state-changing step.
    while msg.pending_action:
        result = _run(execute_pending_action(msg.pending_action))
        msg = _run(continue_task_after_confirmation(agent, task, result, []))

    assert msg.pending_action is None
    assert task.is_complete() is True
    assert task.status is TaskStatus.TASK_COMPLETED
    assert task.remaining_requirements() == []
    assert [s.status for s in task.plan.steps] == [
        StepStatus.SUCCEEDED,
        StepStatus.SUCCEEDED,
        StepStatus.SUCCEEDED,
    ]
    assert (sandbox / "backend" / "main.py").exists()
    assert (sandbox / "backend" / "app.py").exists()


# ---------------------------------------------------------------------------
# B — Bug fixing task
# ---------------------------------------------------------------------------


def test_b_bug_fixing_task(sandbox, monkeypatch):
    monkeypatch.setitem(
        reg.TOOLS,
        "read_file",
        lambda file_path: "def login(user, pwd):\n    return user == 'admin'",
    )
    plan = feature_plan(
        goal="Make the login tests pass",
        task_type=TaskType.DEBUGGING,
        steps=[
            PlanStep(id=1, description="Reproduce the failure", purpose="Run the failing test",
                     expected_outcome="Failure observed", completion_criteria=["Failure observed"]),
            PlanStep(id=2, description="Locate the root cause", purpose="Trace the implementation",
                     expected_outcome="Root cause identified", completion_criteria=["Root cause identified"], dependencies=[1]),
            PlanStep(id=3, description="Fix and verify", purpose="Apply the fix and rerun tests",
                     expected_outcome="Tests pass", completion_criteria=["Login tests pass"], dependencies=[2]),
        ],
        completion_criteria=["Authentication works"],
        verification_requirements=["Test suite passes"],
    )
    task = TaskState(goal="Make the login tests pass", task_type=TaskType.DEBUGGING)
    task.attach_plan(plan)

    engine = FakeEngine(
        [
            _tool("run_command", command="pytest tests/test_login.py"),  # step 1
            "Failure reproduced.",
            plan_verify(
                [("Failure observed", True)],
                [("Authentication works", False), ("Test suite passes", False)],
            ),
            _tool("read_file", file_path="auth.py"),  # step 2
            "Root cause identified: auth.py.",
            plan_verify(
                [("Root cause identified", True)],
                [("Authentication works", False), ("Test suite passes", False)],
            ),
            _tool("run_command", command="pytest tests/test_login.py"),  # step 3
            "All tests pass now.",
            plan_verify(
                [("Login tests pass", True)],
                [("Authentication works", True), ("Test suite passes", True)],
            ),
        ]
    )
    agent = ReActAgent(engine)

    msg = _run(agent.run("Fix the failing login tests", task=task))
    task = msg.task_state
    while msg.pending_action:
        result = _run(execute_pending_action(msg.pending_action))
        msg = _run(continue_task_after_confirmation(agent, task, result, []))

    assert task.is_complete() is True
    assert task.task_type is TaskType.DEBUGGING
    assert [e.tool_name for e in task.execution_history] == [
        "run_command",
        "read_file",
        "run_command",
    ]
    # Each step advanced only after ITS OWN criteria were verified.
    assert [s.status for s in task.plan.steps] == [
        StepStatus.SUCCEEDED,
        StepStatus.SUCCEEDED,
        StepStatus.SUCCEEDED,
    ]


# ---------------------------------------------------------------------------
# C — Refactoring task: dependencies are enforced, no skipping A -> F
# ---------------------------------------------------------------------------


def test_c_refactoring_dependencies_are_enforced():
    plan = feature_plan(
        goal="Refactor the auth service",
        task_type=TaskType.SOFTWARE_ENGINEERING,
        steps=[
            PlanStep(id=1, description="Inspect the codebase", purpose="Understand current state",
                     expected_outcome="Codebase understood", completion_criteria=["Codebase inspected"]),
            PlanStep(id=2, description="Define the target structure", purpose="Design the refactor",
                     expected_outcome="Target structure defined", completion_criteria=["Target structure defined"], dependencies=[1]),
            PlanStep(id=3, description="Apply the refactor", purpose="Rewrite the module",
                     expected_outcome="Refactor applied", completion_criteria=["Refactor applied"], dependencies=[2]),
        ],
        completion_criteria=["Refactor complete"],
        verification_requirements=["Behavior unchanged"],
    )
    task = TaskState(goal="Refactor the auth service")
    task.attach_plan(plan)

    # The model claims step 2 done while step 1 is still unmet → verification
    # must NOT advance; it can only advance the CURRENT step.
    engine = FakeEngine(
        [
            "Refactor done.",  # premature final answer
            plan_verify(
                [("Codebase inspected", False)],
                [("Refactor complete", True), ("Behavior unchanged", True)],
            ),
            _tool("read_file", file_path="auth.py"),
            "Refactor done.",
            plan_verify(
                [("Codebase inspected", True)],
                [("Refactor complete", False), ("Behavior unchanged", False)],
            ),
            "Refactor done.",
            plan_verify(
                [("Target structure defined", True)],
                [("Refactor complete", False), ("Behavior unchanged", False)],
            ),
            "Refactor done.",
            plan_verify(
                [("Refactor applied", True)],
                [("Refactor complete", True), ("Behavior unchanged", True)],
            ),
        ]
    )
    agent = ReActAgent(engine)

    msg = _run(agent.run("Refactor the auth service", task=task))
    assert msg.pending_action is None
    assert task.is_complete() is True
    # Steps completed strictly in dependency order.
    assert [s.status for s in task.plan.steps] == [
        StepStatus.SUCCEEDED,
        StepStatus.SUCCEEDED,
        StepStatus.SUCCEEDED,
    ]


# ---------------------------------------------------------------------------
# D — Repository analysis (research / read-only)
# ---------------------------------------------------------------------------


def test_d_repository_analysis(monkeypatch):
    monkeypatch.setitem(reg.TOOLS, "read_file", lambda file_path: "auth implementation")
    plan = feature_plan(
        goal="Explain how authentication works",
        task_type=TaskType.RESEARCH,
        workspace=WorkspaceKind.EXISTING_PROJECT,
        steps=[
            PlanStep(id=1, description="Inspect the repository structure", purpose="Map the codebase",
                     expected_outcome="Structure mapped", completion_criteria=["Repository inspected"]),
            PlanStep(id=2, description="Trace the authentication path", purpose="Follow the code",
                     expected_outcome="Path traced", completion_criteria=["Auth path traced"], dependencies=[1]),
            PlanStep(id=3, description="Explain with evidence", purpose="Produce the explanation",
                     expected_outcome="Explanation written", completion_criteria=["Explanation produced"], dependencies=[2]),
        ],
        completion_criteria=["Authentication explained"],
        verification_requirements=["Explanation cites source files"],
    )
    task = TaskState(goal="Explain how authentication works", task_type=TaskType.RESEARCH)
    task.attach_plan(plan)

    engine = FakeEngine(
        [
            _tool("read_file", file_path="README.md"),  # step 1
            "Repository structure mapped.",
            plan_verify(
                [("Repository inspected", True)],
                [("Authentication explained", False), ("Explanation cites source files", False)],
            ),
            _tool("read_file", file_path="auth.py"),  # step 2
            "Auth path traced through auth.py.",
            plan_verify(
                [("Auth path traced", True)],
                [("Authentication explained", False), ("Explanation cites source files", False)],
            ),
            "Here is the full explanation with citations.",  # step 3
            plan_verify(
                [("Explanation produced", True)],
                [("Authentication explained", True), ("Explanation cites source files", True)],
            ),
        ]
    )
    agent = ReActAgent(engine)

    msg = _run(agent.run("Analyze this repository and explain how authentication works", task=task))
    assert msg.pending_action is None
    assert task.is_complete() is True
    assert [e.tool_name for e in task.execution_history] == ["read_file", "read_file"]


# ---------------------------------------------------------------------------
# E — Code review task
# ---------------------------------------------------------------------------


def test_e_code_review_task(monkeypatch):
    monkeypatch.setitem(reg.TOOLS, "read_file", lambda file_path: "code under review")
    plan = feature_plan(
        goal="Review the repository for security problems",
        task_type=TaskType.CODE_REVIEW,
        workspace=WorkspaceKind.EXISTING_PROJECT,
        steps=[
            PlanStep(id=1, description="Inspect the code", purpose="Read the relevant files",
                     expected_outcome="Code read", completion_criteria=["Code inspected"]),
            PlanStep(id=2, description="Identify findings", purpose="Spot issues",
                     expected_outcome="Findings identified", completion_criteria=["Findings identified"], dependencies=[1]),
            PlanStep(id=3, description="Produce the review report", purpose="Write findings up",
                     expected_outcome="Report written", completion_criteria=["Report produced"], dependencies=[2]),
        ],
        completion_criteria=["Security review complete"],
        verification_requirements=["Findings validated"],
    )
    task = TaskState(goal="Review the repository for security problems", task_type=TaskType.CODE_REVIEW)
    task.attach_plan(plan)

    engine = FakeEngine(
        [
            _tool("read_file", file_path="app.py"),  # step 1
            "Code inspected.",
            plan_verify(
                [("Code inspected", True)],
                [("Security review complete", False), ("Findings validated", False)],
            ),
            "Findings identified.",  # step 2
            plan_verify(
                [("Findings identified", True)],
                [("Security review complete", False), ("Findings validated", False)],
            ),
            "Final report with validated findings.",  # step 3
            plan_verify(
                [("Report produced", True)],
                [("Security review complete", True), ("Findings validated", True)],
            ),
        ]
    )
    agent = ReActAgent(engine)
    _run(agent.run("Review this repository for security problems", task=task))
    assert task.is_complete() is True
    assert task.task_type is TaskType.CODE_REVIEW


# ---------------------------------------------------------------------------
# F — Dependency update task
# ---------------------------------------------------------------------------


def test_f_dependency_update(sandbox):
    plan = feature_plan(
        goal="Upgrade the project from React 18 to React 19",
        task_type=TaskType.SOFTWARE_ENGINEERING,
        workspace=WorkspaceKind.EXISTING_PROJECT,
        steps=[
            PlanStep(id=1, description="Inspect the dependency configuration", purpose="Find package.json",
                     expected_outcome="Config found", completion_criteria=["Dependencies inspected"]),
            PlanStep(id=2, description="Update the dependency", purpose="Bump React",
                     expected_outcome="Version bumped", completion_criteria=["React upgraded"], dependencies=[1]),
            PlanStep(id=3, description="Verify the upgrade", purpose="Run tests/build",
                     expected_outcome="Build passes", completion_criteria=["Upgrade verified"], dependencies=[2]),
        ],
        completion_criteria=["React 19 installed"],
        verification_requirements=["Build succeeds"],
    )
    task = TaskState(goal="Upgrade the project from React 18 to React 19")
    task.attach_plan(plan)

    engine = FakeEngine(
        [
            _tool("read_file", file_path="package.json"),  # step 1
            "Dependencies inspected.",
            plan_verify(
                [("Dependencies inspected", True)],
                [("React 19 installed", False), ("Build succeeds", False)],
            ),
            _tool("write_file", filename="package.json", content='{"react": "^19.0.0"}'),  # step 2
            "React upgraded.",
            plan_verify(
                [("React upgraded", True)],
                [("React 19 installed", False), ("Build succeeds", False)],
            ),
            _tool("run_command", command="npm test"),  # step 3
            "Build succeeds.",
            plan_verify(
                [("Upgrade verified", True)],
                [("React 19 installed", True), ("Build succeeds", True)],
            ),
        ]
    )
    agent = ReActAgent(engine)
    task = TaskState(goal="Upgrade the project from React 18 to React 19")
    task.attach_plan(plan)
    msg = _run(agent.run("Upgrade this project from React 18 to React 19", task=task))
    task = msg.task_state
    while msg.pending_action:
        result = _run(execute_pending_action(msg.pending_action))
        msg = _run(continue_task_after_confirmation(agent, task, result, []))
    assert task.is_complete() is True


# ---------------------------------------------------------------------------
# G — Configuration task
# ---------------------------------------------------------------------------


def test_g_configuration_task(sandbox, monkeypatch):
    monkeypatch.setitem(reg.TOOLS, "read_file", lambda file_path: "key=old")
    plan = feature_plan(
        goal="Configure Redis for this project",
        task_type=TaskType.CONFIGURATION,
        workspace=WorkspaceKind.EXISTING_PROJECT,
        steps=[
            PlanStep(id=1, description="Inspect the current configuration", purpose="Find the config",
                     expected_outcome="Config inspected", completion_criteria=["Config inspected"]),
            PlanStep(id=2, description="Apply the Redis configuration", purpose="Set the values",
                     expected_outcome="Redis configured", completion_criteria=["Redis configured"], dependencies=[1]),
            PlanStep(id=3, description="Verify the configuration", purpose="Check it works",
                     expected_outcome="Config verified", completion_criteria=["Config verified"], dependencies=[2]),
        ],
        completion_criteria=["Redis is configured"],
        verification_requirements=["Configuration validated"],
    )
    task = TaskState(goal="Configure Redis for this project", task_type=TaskType.CONFIGURATION)
    task.attach_plan(plan)

    engine = FakeEngine(
        [
            _tool("read_file", file_path=".env"),  # step 1
            "Config inspected.",
            plan_verify(
                [("Config inspected", True)],
                [("Redis is configured", False), ("Configuration validated", False)],
            ),
            _tool("write_file", filename=".env", content="REDIS_URL=redis://localhost:6379"),  # step 2
            "Redis configured.",
            plan_verify(
                [("Redis configured", True)],
                [("Redis is configured", False), ("Configuration validated", False)],
            ),
            "Config verified.",  # step 3
            plan_verify(
                [("Config verified", True)],
                [("Redis is configured", True), ("Configuration validated", True)],
            ),
        ]
    )
    agent = ReActAgent(engine)
    msg = _run(agent.run("Configure Redis for this project", task=task))
    task = msg.task_state
    while msg.pending_action:
        result = _run(execute_pending_action(msg.pending_action))
        msg = _run(continue_task_after_confirmation(agent, task, result, []))
    assert task.is_complete() is True


# ---------------------------------------------------------------------------
# H — Multi-step filesystem task
# ---------------------------------------------------------------------------


def test_h_multistep_filesystem_task(sandbox):
    plan = feature_plan(
        goal="Create a project scaffold in the workspace",
        task_type=TaskType.MULTI_STEP,
        workspace=WorkspaceKind.NEW_WORKSPACE,
        steps=[
            PlanStep(id=1, description="Create the directory", purpose="Make the folder",
                     expected_outcome="Directory exists", completion_criteria=["Directory exists"]),
            PlanStep(id=2, description="Write the entry file", purpose="Add the main file",
                     expected_outcome="File exists", completion_criteria=["Entry file exists"], dependencies=[1]),
            PlanStep(id=3, description="Verify the scaffold", purpose="Check the result",
                     expected_outcome="Scaffold verified", completion_criteria=["Scaffold verified"], dependencies=[2]),
        ],
        completion_criteria=["Scaffold created"],
        verification_requirements=["Files verified on disk"],
    )
    task = TaskState(goal="Create a project scaffold in the workspace", task_type=TaskType.MULTI_STEP)
    task.attach_plan(plan)

    engine = FakeEngine(
        [
            _tool("run_command", command="mkdir scaffold"),  # step 1
            "Directory created.",
            plan_verify(
                [("Directory exists", True)],
                [("Scaffold created", False), ("Files verified on disk", False)],
            ),
            _tool("write_file", filename="scaffold/main.txt", content="hello"),  # step 2
            "Entry file written.",
            plan_verify(
                [("Entry file exists", True)],
                [("Scaffold created", False), ("Files verified on disk", False)],
            ),
            "Scaffold verified.",  # step 3
            plan_verify(
                [("Scaffold verified", True)],
                [("Scaffold created", True), ("Files verified on disk", True)],
            ),
        ]
    )
    agent = ReActAgent(engine)
    msg = _run(agent.run("Create a project scaffold", task=task))
    task = msg.task_state
    while msg.pending_action:
        result = _run(execute_pending_action(msg.pending_action))
        msg = _run(continue_task_after_confirmation(agent, task, result, []))
    assert task.is_complete() is True
    assert (sandbox / "scaffold" / "main.txt").read_text() == "hello"


# ---------------------------------------------------------------------------
# I — Confirmation interruption: plan + work survive the boundary
# ---------------------------------------------------------------------------


def test_i_confirmation_interruption_preserves_plan(sandbox):
    task = TaskState(goal="Create a FastAPI backend")
    task.attach_plan(feature_plan())

    engine = FakeEngine(
        [
            _tool("run_command", command="mkdir backend"),  # step 1 (confirm-gated)
            "Workspace ready.",
            plan_verify(
                [("Project directory exists", True)],
                [("FastAPI backend implemented", False), ("Backend starts without errors", False)],
            ),
            _tool("write_file", filename="backend/main.py", content="app = FastAPI()"),  # step 2
            "Backend implemented.",
            plan_verify(
                [("Backend application files exist", True)],
                [("FastAPI backend implemented", False), ("Backend starts without errors", False)],
            ),
            _tool("run_command", command="python -m pytest"),  # step 3
            "Verified.",
            plan_verify(
                [("Application verified", True)],
                [("FastAPI backend implemented", True), ("Backend starts without errors", True)],
            ),
        ]
    )
    agent = ReActAgent(engine)

    msg = _run(agent.run("Create a FastAPI backend", task=task))
    task = msg.task_state
    goal_before = task.goal
    plan_before = task.plan

    # First confirmation round-trip: the gated action parks the plan step in
    # WAITING_CONFIRMATION.
    assert msg.pending_action is not None
    step_before = task.current_plan_step()
    assert step_before is not None
    assert step_before.status is StepStatus.WAITING_CONFIRMATION
    assert task.status is TaskStatus.WAITING_CONFIRMATION
    result = _run(execute_pending_action(msg.pending_action))
    msg = _run(continue_task_after_confirmation(agent, task, result, []))

    # Everything survived: goal, plan object, and the confirmed step completed
    # while the task advanced to the NEXT step — never a restart from step 1.
    assert task.goal == goal_before
    assert task.plan is plan_before
    assert task.plan.step(1).status is StepStatus.SUCCEEDED  # work after resume
    assert task.current_plan_step().id == 2  # advanced, not restarted
    assert task.status is TaskStatus.WAITING_CONFIRMATION  # parked on next gated step

    while msg.pending_action:
        result = _run(execute_pending_action(msg.pending_action))
        msg = _run(continue_task_after_confirmation(agent, task, result, []))

    assert task.is_complete() is True
    assert task.plan is plan_before  # never regenerated
    assert task.plan_revisions == []  # no re-planning across confirmations


# ---------------------------------------------------------------------------
# J — Failed intermediate step: never reports success
# ---------------------------------------------------------------------------


def test_j_failed_intermediate_step(monkeypatch):
    def flaky(file_path):
        if file_path == "broken.txt":
            return "Error: file not found"
        return "contents"

    monkeypatch.setitem(reg.TOOLS, "read_file", flaky)
    plan = feature_plan(
        goal="Repair the broken file",
        task_type=TaskType.DEBUGGING,
        steps=[
            PlanStep(id=1, description="Inspect the broken file", purpose="Read it",
                     expected_outcome="File read", completion_criteria=["File inspected"]),
            PlanStep(id=2, description="Repair the file", purpose="Fix it",
                     expected_outcome="File repaired", completion_criteria=["File repaired"], dependencies=[1]),
            PlanStep(id=3, description="Verify the repair", purpose="Confirm",
                     expected_outcome="Repair verified", completion_criteria=["Repair verified"], dependencies=[2]),
        ],
        completion_criteria=["File repaired"],
        verification_requirements=["Repair verified"],
    )
    task = TaskState(goal="Repair the broken file", task_type=TaskType.DEBUGGING)
    task.attach_plan(plan)

    engine = FakeEngine(
        [
            _tool("read_file", file_path="broken.txt"),  # fails
            "The file is fixed.",  # false claim
            plan_verify(
                [("File inspected", False)],
                [("File repaired", False), ("Repair verified", False)],
            ),
            _tool("read_file", file_path="fixed.txt"),  # recovery
            "Now it is repaired.",
            plan_verify(
                [("File inspected", True)],
                [("File repaired", False), ("Repair verified", False)],
            ),
            "The file is repaired.",
            plan_verify(
                [("File repaired", True)],
                [("File repaired", True), ("Repair verified", False)],
            ),
            "Repair verified.",
            plan_verify(
                [("Repair verified", True)],
                [("File repaired", True), ("Repair verified", True)],
            ),
        ]
    )
    agent = ReActAgent(engine)
    msg = _run(agent.run("Repair the broken file", task=task))
    task = msg.task_state

    # The failed read was recorded against step 1 (attempt + error), and the
    # false claim was overruled — the task completed only after the recovery
    # rounds, never at the moment of the failed action.
    assert task.plan.step(1).attempts >= 1
    assert task.plan.step(1).error is not None
    assert any(not e.success for e in task.execution_history)
    notes = [m.content for m in task.context if m.name == "task_verification"]
    assert any("is not complete" in n for n in notes)  # false claim rejected
    assert task.is_complete() is True  # completion came only after recovery
    assert [s.status for s in task.plan.steps] == [
        StepStatus.SUCCEEDED,
        StepStatus.SUCCEEDED,
        StepStatus.SUCCEEDED,
    ]


# ---------------------------------------------------------------------------
# K — Retry / recovery within policy
# ---------------------------------------------------------------------------


def test_k_retry_within_policy_recovers(monkeypatch):
    state = {"calls": 0}

    def flaky(file_path):
        state["calls"] += 1
        if state["calls"] == 1:
            return "Error: transient failure"
        return "recovered contents"

    monkeypatch.setitem(reg.TOOLS, "read_file", flaky)
    plan = feature_plan(
        goal="Read the config",
        task_type=TaskType.CONFIGURATION,
        steps=[
            PlanStep(
                id=1,
                description="Read the config file",
                purpose="Load it",
                expected_outcome="Config read",
                completion_criteria=["Config read"],
                failure_strategy=FailureStrategy.RETRY,
                retry_policy=2,
            ),
            PlanStep(id=2, description="Verify the config", purpose="Confirm",
                     expected_outcome="Verified", completion_criteria=["Config verified"], dependencies=[1]),
        ],
        completion_criteria=["Config loaded"],
        verification_requirements=["Config verified"],
    )
    task = TaskState(goal="Read the config", task_type=TaskType.CONFIGURATION)
    task.attach_plan(plan)

    engine = FakeEngine(
        [
            _tool("read_file", file_path="config.txt"),  # fails once
            "I have the config.",
            plan_verify(
                [("Config read", True)],
                [("Config loaded", False), ("Config verified", False)],
            ),
            _tool("read_file", file_path="config.txt"),  # succeeds
            "Config verified.",
            plan_verify(
                [("Config verified", True)],
                [("Config loaded", True), ("Config verified", True)],
            ),
        ]
    )
    agent = ReActAgent(engine)
    msg = _run(agent.run("Read the config", task=task))
    assert msg.pending_action is None
    assert task.is_complete() is True
    # One failed attempt was recorded, recovery succeeded.
    assert task.plan.step(1).attempts == 1
    assert task.plan.step(1).status is StepStatus.SUCCEEDED


def test_k_skip_strategy_cascades_dependents():
    """
    SKIP marks the step SKIPPED and cascades dependents so no step is left
    stranded PENDING (the plan stays consistent). Completion still requires
    the plan satisfied AND overall criteria met — a skipped step alone can
    never complete the task when the model also claims criteria unmet.
    """
    plan = feature_plan(
        goal="Run the pipeline",
        task_type=TaskType.SYSTEM_OPERATION,
        steps=[
            PlanStep(
                id=1,
                description="Build the binary",
                expected_outcome="Built",
                completion_criteria=["Binary built"],
                failure_strategy=FailureStrategy.SKIP,
            ),
            PlanStep(
                id=2,
                description="Deploy the binary",
                expected_outcome="Deployed",
                completion_criteria=["Binary deployed"],
                dependencies=[1],
            ),
            PlanStep(
                id=3,
                description="Smoke-test the deployment",
                expected_outcome="Healthy",
                completion_criteria=["Deployment healthy"],
                dependencies=[2],
            ),
        ],
        completion_criteria=["Pipeline executed"],
        verification_requirements=["Deployment healthy"],
    )
    task = TaskState(goal="Run the pipeline", task_type=TaskType.SYSTEM_OPERATION)
    task.attach_plan(plan)

    engine = FakeEngine(
        [
            "The binary cannot be built here.",
            plan_verify(
                [("Binary built", False)],
                [("Pipeline executed", False), ("Deployment healthy", False)],
                step_failed=True,
                revision_reason="no build toolchain",
            ),
            "Nothing left to do — deployment depends on the skipped build.",
            plan_verify(
                [("Binary deployed", False)],
                [("Pipeline executed", False), ("Deployment healthy", False)],
            ),
        ]
    )
    agent = ReActAgent(engine)
    msg = _run(agent.run("Run the pipeline", task=task))
    task = msg.task_state

    # Step 1 skipped per policy; dependent steps cascaded to SKIPPED so the
    # plan stays consistent (no stranded PENDING step).
    assert task.plan.step(1).status is StepStatus.SKIPPED
    assert task.plan.step(2).status is StepStatus.SKIPPED
    assert task.plan.step(3).status is StepStatus.SKIPPED
    assert task.plan.is_satisfied()  # every step reached a terminal state
    # Overall criteria were never marked satisfied — no false completion.
    assert task.is_complete() is False
    assert task.status is not TaskStatus.TASK_COMPLETED


def test_k_continue_strategy_records_failure_and_keeps_plan_consistent():
    """
    CONTINUE records the failure on the step (as a SKIPPED step with the
    error attached — the plan stays satisfiable) and cascades dependents;
    it never silently completes the failed work or the task.
    """
    plan = feature_plan(
        goal="Deploy the service",
        task_type=TaskType.SYSTEM_OPERATION,
        steps=[
            PlanStep(
                id=1,
                description="Provision the database",
                expected_outcome="DB ready",
                completion_criteria=["Database provisioned"],
                failure_strategy=FailureStrategy.CONTINUE,
            ),
            PlanStep(
                id=2,
                description="Deploy the app",
                expected_outcome="App running",
                completion_criteria=["App deployed"],
                dependencies=[1],
            ),
        ],
        completion_criteria=["Service deployed"],
        verification_requirements=["Service healthy"],
    )
    task = TaskState(goal="Deploy the service", task_type=TaskType.SYSTEM_OPERATION)
    task.attach_plan(plan)

    engine = FakeEngine(
        [
            "Database provisioning failed.",
            plan_verify(
                [("Database provisioned", False)],
                [("Service deployed", False), ("Service healthy", False)],
                step_failed=True,
                revision_reason="quota exceeded",
            ),
            "Nothing further can run — the app depends on the database.",
            plan_verify(
                [("App deployed", False)],
                [("Service deployed", False), ("Service healthy", False)],
            ),
        ]
    )
    agent = ReActAgent(engine)
    msg = _run(agent.run("Deploy the service", task=task))
    task = msg.task_state

    # The failed step was recorded (error attached), its dependent cascaded
    # (never left PENDING), and the task never reported success.
    assert task.plan.step(1).error is not None
    assert task.plan.step(1).status is StepStatus.SKIPPED  # recorded + continued
    assert task.plan.step(2).status is StepStatus.SKIPPED
    assert task.plan.is_satisfied()
    assert task.is_complete() is False
    assert task.status is not TaskStatus.TASK_COMPLETED


def test_k_stranded_pending_step_cannot_complete():
    """
    Regression for the completion-authority hole: a plan with a PENDING step
    whose dependency failed/skipped can never report success — even when all
    requirements are marked complete, mark_complete refuses.
    """
    plan = feature_plan()
    task = TaskState(goal="Create a FastAPI backend")
    task.attach_plan(plan)
    # Simulate the pre-cascade state: step 1 SKIPPED, step 2 left PENDING and
    # unrunnable (its dependency is not SUCCEEDED), step 3 PENDING.
    task.plan.step(1).status = StepStatus.SKIPPED
    for requirement in task.requirements:
        task.mark_requirement_complete(requirement.description)
    assert task.plan.is_satisfied() is False  # step 2 still PENDING
    assert task.is_complete() is False
    with pytest.raises(ValueError, match="plan steps are unfinished"):
        task.mark_complete()
    # Cascade the dependents (what _cascade_skipped does) → plan satisfied,
    # and completion still requires every overall criterion.
    task.plan.step(2).status = StepStatus.SKIPPED
    task.plan.step(3).status = StepStatus.SKIPPED
    assert task.plan.is_satisfied()
    task.mark_complete()
    assert task.is_complete() is True


def test_k_retry_budget_exhausted_terminates():
    plan = feature_plan(
        goal="Read the config",
        task_type=TaskType.CONFIGURATION,
        steps=[
            PlanStep(
                id=1,
                description="Read the config file",
                purpose="Load it",
                expected_outcome="Config read",
                completion_criteria=["Config read"],
                failure_strategy=FailureStrategy.RETRY,
                retry_policy=1,
            ),
        ],
        completion_criteria=["Config loaded"],
        verification_requirements=["Config verified"],
    )
    task = TaskState(goal="Read the config", task_type=TaskType.CONFIGURATION)
    task.attach_plan(plan)
    # Two failed executions exceed retry_policy=1 → verification escalates to STOP.
    task.plan.step(1).attempts = 2

    engine = FakeEngine(
        [
            "The config is read.",  # claim
            plan_verify(
                [("Config read", False)],
                [("Config loaded", False), ("Config verified", False)],
                step_failed=True,
                revision_reason="file is missing",
            ),
        ]
    )
    agent = ReActAgent(engine)
    msg = _run(agent.run("Read the config", task=task))
    assert "incomplete" in msg.content.lower()
    assert task.is_complete() is False
    assert task.status is TaskStatus.TASK_FAILED
    assert task.plan.step(1).status is StepStatus.FAILED


# ---------------------------------------------------------------------------
# L — Adaptive plan modification
# ---------------------------------------------------------------------------

REVISION_STEPS = [
    {
        "id": 3,
        "description": "Repair auth in service layer",
        "purpose": "New evidence points to auth/service.py",
        "expected_outcome": "Service layer repaired",
        "completion_criteria": ["Service layer repaired"],
        "dependencies": [1],  # completed inspection step 1, not removed step 2
        "failure_strategy": "stop",
        "retry_policy": 0,
    },
    {
        "id": 4,
        "description": "Verify the repaired service",
        "purpose": "Confirm the fix",
        "expected_outcome": "Fix verified",
        "completion_criteria": ["Fix verified"],
        "dependencies": [3],
        "failure_strategy": "stop",
        "retry_policy": 0,
    },
]


def test_l_adaptive_plan_modification():
    plan = feature_plan(
        goal="Fix the authentication service",
        task_type=TaskType.DEBUGGING,
        steps=[
            PlanStep(id=1, description="Inspect auth.py", purpose="Look at the file",
                     expected_outcome="Inspected", completion_criteria=["auth.py inspected"]),
            PlanStep(id=2, description="Repair auth.py", purpose="Fix the module",
                     expected_outcome="Repaired", completion_criteria=["auth.py repaired"], dependencies=[1]),
            PlanStep(id=3, description="Verify the fix", purpose="Confirm",
                     expected_outcome="Verified", completion_criteria=["Fix verified"], dependencies=[2]),
        ],
        completion_criteria=["Auth service fixed"],
        verification_requirements=["Tests pass"],
    )
    task = TaskState(goal="Fix the authentication service", task_type=TaskType.DEBUGGING)
    task.attach_plan(plan)

    engine = FakeEngine(
        [
            "The fix is in auth.py.",
            plan_verify(
                [("auth.py inspected", True), ("auth.py repaired", True)],
                [("Auth service fixed", False), ("Tests pass", False)],
                revision=REVISION_STEPS,  # new evidence: repair belongs in service layer
            ),
            "Now repairing auth/service.py.",
            plan_verify(
                [("Service layer repaired", True)],
                [("Auth service fixed", False), ("Tests pass", False)],
            ),
            "Fix verified.",
            plan_verify(
                [("Fix verified", True)],
                [("Auth service fixed", True), ("Tests pass", True)],
            ),
        ]
    )
    agent = ReActAgent(engine)
    msg = _run(agent.run("Fix the authentication service", task=task))
    task = msg.task_state

    assert task.is_complete() is True
    # The revision was applied explicitly: completed step 1 preserved, steps
    # 2 (was pending) replaced by the new remaining work.
    assert [s.id for s in task.plan.steps] == [1, 3, 4]
    assert task.plan.step(1).status is StepStatus.SUCCEEDED
    assert task.plan.step(1).description == "Inspect auth.py"
    assert task.plan_revisions, "an adaptive revision must be recorded"
    assert task.total_steps == 3


def test_l_invalid_plan_revision_rejected():
    plan = feature_plan()
    task = TaskState(goal="Create a FastAPI backend")
    task.attach_plan(plan)
    # Revision with a duplicate id and a cycle — must be rejected.
    bad = [
        {
            "id": 2,
            "description": "Duplicate + cyclic",
            "dependencies": [3],
            "expected_outcome": "x",
            "completion_criteria": ["y"],
        },
        {
            "id": 3,
            "description": "Cycle back",
            "dependencies": [2],
            "expected_outcome": "x",
            "completion_criteria": ["y"],
        },
    ]
    task.plan.step(1).status = StepStatus.SUCCEEDED
    assert task.adapt_plan([PlanStep(**b) for b in bad]) is False
    # Plan unchanged.
    assert [s.id for s in task.plan.steps] == [1, 2, 3]
    assert task.plan_revisions == []


# ---------------------------------------------------------------------------
# M — Final verification gates completion
# ---------------------------------------------------------------------------


def test_m_final_verification_required(sandbox):
    plan = feature_plan()
    task = TaskState(goal="Create a FastAPI backend")
    task.attach_plan(plan)

    # The model claims done, but the final verification step's criteria are
    # never satisfied → the task must NOT complete.
    engine = FakeEngine(
        [
            _tool("run_command", command="mkdir backend"),
            "The app is done.",
            plan_verify(
                [("Project directory exists", True)],
                [("FastAPI backend implemented", True), ("Backend starts without errors", True)],
            ),
            "Really done now.",
            plan_verify(
                [("Backend application files exist", True)],
                [("FastAPI backend implemented", True), ("Backend starts without errors", True)],
            ),
            "Done for real.",
            plan_verify(
                [("Application verified", False)],
                [("FastAPI backend implemented", True), ("Backend starts without errors", False)],
            ),
        ]
    )
    agent = ReActAgent(engine, max_iterations=4)
    msg = _run(agent.run("Create a FastAPI backend", task=task))
    task = msg.task_state
    while msg.pending_action:
        result = _run(execute_pending_action(msg.pending_action))
        msg = _run(continue_task_after_confirmation(agent, task, result, []))
    # Verification never passed → task cannot claim completion.
    assert task.is_complete() is False
    assert task.plan.step(3).status is not StepStatus.SUCCEEDED


# ---------------------------------------------------------------------------
# N — Model falsely claiming completion is overruled
# ---------------------------------------------------------------------------


def test_n_model_false_completion_claim_overruled(sandbox):
    plan = feature_plan()
    task = TaskState(goal="Create a FastAPI backend")
    task.attach_plan(plan)

    engine = FakeEngine(
        [
            "Done! The application has been created.",  # false claim
            plan_verify(
                [("Project directory exists", False)],
                [("FastAPI backend implemented", True), ("Backend starts without errors", True)],
            ),
            _tool("run_command", command="mkdir backend"),
            "The backend is complete.",
            plan_verify(
                [("Project directory exists", True)],
                [("FastAPI backend implemented", False), ("Backend starts without errors", False)],
            ),
            "Backend app files exist.",
            plan_verify(
                [("Backend application files exist", True)],
                [("FastAPI backend implemented", False), ("Backend starts without errors", False)],
            ),
            "Verified.",
            plan_verify(
                [("Application verified", True)],
                [("FastAPI backend implemented", True), ("Backend starts without errors", True)],
            ),
        ]
    )
    agent = ReActAgent(engine)
    msg = _run(agent.run("Create a FastAPI backend", task=task))
    task = msg.task_state
    while msg.pending_action:
        result = _run(execute_pending_action(msg.pending_action))
        msg = _run(continue_task_after_confirmation(agent, task, result, []))
    assert task.is_complete() is True
    # The false claim was rejected: step 1 was NOT complete at claim time and
    # the agent had to actually create the directory before advancing.
    assert task.plan.step(1).status is StepStatus.SUCCEEDED
    assert any(e.tool_name == "run_command" for e in task.execution_history)


# ---------------------------------------------------------------------------
# O — Max iteration protection (plan mode)
# ---------------------------------------------------------------------------


def test_o_max_iterations_plan_mode():
    plan = feature_plan()
    task = TaskState(goal="Create a FastAPI backend")
    task.attach_plan(plan)

    never_done = plan_verify(
        [("Project directory exists", False)],
        [("FastAPI backend implemented", False), ("Backend starts without errors", False)],
    )
    engine = FakeEngine(
        [
            "Almost done.",
            never_done,
            "Almost done.",
            never_done,
        ]
    )
    agent = ReActAgent(engine, max_iterations=2)
    msg = _run(agent.run("Create a FastAPI backend", task=task))
    task = msg.task_state

    assert "incomplete" in msg.content.lower()
    assert "remaining plan steps" in msg.content.lower()
    assert task.is_complete() is False
    assert task.status is TaskStatus.TASK_FAILED
    assert task.plan.step(1).status is StepStatus.FAILED
    assert len(engine.calls) <= 5  # bounded — no infinite loop


# ---------------------------------------------------------------------------
# P — Existing simple commands stay on the fast path
# ---------------------------------------------------------------------------


def test_p_simple_commands_have_no_plan(sandbox):
    task = _run(prepare_task_for_execution("List all Python files in this directory", FakeEngine([])))
    assert task is None  # fast path — no plan, no extra LLM call

    # The plain ReAct flow still works unchanged without a plan.
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setitem(
        reg.TOOLS,
        "read_file",
        lambda file_path: "file contents",
    )
    engine = FakeEngine(
        [
            _tool("read_file", file_path="notes.txt"),
            "The file contains: file contents",
        ]
    )
    agent = ReActAgent(engine)
    msg = _run(agent.run("read the notes file"))
    monkeypatch.undo()
    assert msg.pending_action is None
    assert msg.task_state is None
    assert len(engine.calls) == 2  # no planning/verification overhead


# ---------------------------------------------------------------------------
# Q — Informational requests stay fast
# ---------------------------------------------------------------------------


def test_q_informational_request_has_no_plan():
    engine = FakeEngine([])
    task = _run(prepare_task_for_execution("What is Python?", engine))
    assert task is None
    assert engine.calls == []  # deterministic classification, no LLM call


# ---------------------------------------------------------------------------
# Bootstrap unit tests
# ---------------------------------------------------------------------------


def test_bootstrap_complex_request_builds_planned_task():
    plan_json = json.dumps(
        {
            "steps": [
                {"id": 1, "description": "Establish the workspace",
                 "expected_outcome": "dir exists", "completion_criteria": ["dir exists"]},
                {"id": 2, "description": "Implement the app",
                 "expected_outcome": "app exists", "completion_criteria": ["app exists"],
                 "dependencies": [1]},
            ],
            "completion_criteria": ["app implemented"],
            "verification_requirements": ["app runs"],
        }
    )
    engine = FakeEngine([plan_json])
    task = _run(prepare_task_for_execution("Create a FastAPI backend", engine, cwd="/tmp"))
    assert task is not None
    assert task.plan is not None
    assert task.task_type is TaskType.SOFTWARE_ENGINEERING
    assert task.total_steps == 2
    assert {r.description for r in task.requirements} == {"app implemented", "app runs"}
    assert task.is_complete() is False
    assert task.status is TaskStatus.TASK_STARTED


def test_bootstrap_clarification_blocks_task():
    engine = FakeEngine([])  # deterministic classification: "Deploy this."
    task = _run(prepare_task_for_execution("Deploy this.", engine))
    assert task is not None
    assert task.clarification_required is True
    assert task.status is TaskStatus.TASK_BLOCKED
    assert task.is_complete() is False
    assert task.clarification_questions


def test_bootstrap_falls_back_to_fallback_plan_when_llm_fails():
    engine = FakeEngine(["not json at all"])
    task = _run(prepare_task_for_execution("Create a FastAPI backend", engine, cwd="/tmp"))
    assert task is not None
    assert task.plan is not None
    assert len(task.plan.steps) == 1  # fallback verification step
    assert task.is_complete() is False


# ---------------------------------------------------------------------------
# Step lifecycle unit tests
# ---------------------------------------------------------------------------


def test_plan_step_waiting_confirmation_lifecycle(sandbox):
    task = TaskState(goal="Create a FastAPI backend")
    task.attach_plan(feature_plan())
    engine = FakeEngine(
        [
            _tool("run_command", command="mkdir backend"),  # step 1, gated
            "Workspace ready.",
            plan_verify(
                [("Project directory exists", True)],
                [("FastAPI backend implemented", False), ("Backend starts without errors", False)],
            ),
            _tool("write_file", filename="backend/main.py", content="app = FastAPI()"),  # step 2, gated
            "Backend implemented.",
            plan_verify(
                [("Backend application files exist", True)],
                [("FastAPI backend implemented", False), ("Backend starts without errors", False)],
            ),
            "Verified.",
            plan_verify(
                [("Application verified", True)],
                [("FastAPI backend implemented", True), ("Backend starts without errors", True)],
            ),
        ]
    )
    agent = ReActAgent(engine)

    msg = _run(agent.run("Create a FastAPI backend", task=task))
    task = msg.task_state
    step = task.current_plan_step()
    assert step is not None
    assert step.status is StepStatus.WAITING_CONFIRMATION
    assert task.status is TaskStatus.WAITING_CONFIRMATION

    result = _run(execute_pending_action(msg.pending_action))
    msg = _run(continue_task_after_confirmation(agent, task, result, []))
    # The step resumed after confirmation, completed its criteria, and the
    # task advanced to the next gated step — never a restart from step 1.
    assert task.plan.step(1).status is StepStatus.SUCCEEDED
    assert task.current_plan_step().id == 2
    assert task.plan.step(1).attempts == 0

    while msg.pending_action:
        result = _run(execute_pending_action(msg.pending_action))
        msg = _run(continue_task_after_confirmation(agent, task, result, []))
    assert task.is_complete() is True
    assert [s.status for s in task.plan.steps] == [
        StepStatus.SUCCEEDED,
        StepStatus.SUCCEEDED,
        StepStatus.SUCCEEDED,
    ]


def test_task_plan_adaptive_revision_records_and_preserves_completed():
    plan = feature_plan()
    task = TaskState(goal="Create a FastAPI backend")
    task.attach_plan(plan)
    task.plan.set_step_status(1, StepStatus.SUCCEEDED, result="created")
    assert task.adapt_plan(
        [
            PlanStep(id=2, description="Implement", expected_outcome="x",
                     completion_criteria=["y"], dependencies=[1]),
            PlanStep(id=3, description="Verify", expected_outcome="x",
                     completion_criteria=["y"], dependencies=[2]),
        ]
    ) is True
    assert task.plan.step(1).status is StepStatus.SUCCEEDED  # preserved
    assert [s.id for s in task.plan.steps] == [1, 2, 3]
    assert len(task.plan_revisions) == 1
    assert task.total_steps == 3


def test_task_plan_revision_without_plan_fails():
    task = TaskState(goal="g")
    assert task.adapt_plan([PlanStep(id=1, description="x")]) is False
