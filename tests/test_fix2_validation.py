"""
End-to-end validation of Fix #2 — the GENERAL TASK PLANNING FOUNDATION.

These tests exercise the real pipeline (classification -> goal -> TaskState
-> structured plan -> validation -> plan-aware execution) deterministically,
with scripted engines — no real LLM, no network.  They are NOT about
TodoList creation; they verify that Ultron can understand, plan and execute
a general range of tasks as a foundation for a full coding agent.

Scenarios (T1..T20):

  T1  informational            T11 existing project
  T2  simple action            T12 research + implementation
  T3  multi-step general       T13 confirmation interruption
  T4  new project              T14 failure recovery
  T5  feature implementation   T15 false completion
  T6  bug fix                  T16 invalid plan
  T7  debugging                T17 max iterations
  T8  refactor                 T18 security
  T9  code review              T19 no over-planning
  T10 dependency upgrade       T20 plan adaptation

The architectural invariant under test for every complex task:

  USER -> GOAL -> TASK TYPE -> TASKSTATE -> PLAN -> VALIDATION
  -> CURRENT STEP -> TOOL ACTION -> OBSERVATION -> TASKSTATE
  -> NEXT STEP -> VERIFICATION -> DONE

The plan is the explicit bridge between "what the user wants" and "which
tool to execute next" — never the first tool call, and never the LLM's word.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from ultron.core.agents.react import ReActAgent
from ultron.core.intelligence.plan_validation import validate_plan
from ultron.core.intelligence.task_classification import (
    classify_task_deterministic,
)
from ultron.core.intelligence.task_planning import (
    detect_workspace_kind,
    prepare_task_for_execution,
)
from ultron.core.tools import paths
from ultron.core.tools import registry as reg
from ultron.core.types import (
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


def plan_verify(step_criteria, plan_criteria, step_failed=False, revision=None,
                revision_reason=None) -> str:
    """Builds a plan-mode verification JSON payload (one step per round)."""
    payload = {
        "step_criteria": [{"description": d, "satisfied": s} for d, s in step_criteria],
        "plan_criteria": [{"description": d, "satisfied": s} for d, s in plan_criteria],
        "step_failed": step_failed,
        "plan_revision": revision,
    }
    if revision_reason is not None:
        payload["step_failed_reason"] = revision_reason
    return json.dumps(payload)


def plan_json(steps, completion, verification, **extra) -> str:
    """A valid plan payload for the scripted planner."""
    payload = {
        "steps": steps,
        "completion_criteria": completion,
        "verification_requirements": verification,
        "assumptions": ["scripted for validation"],
        "constraints": [],
        "failure_recovery": "stop on failure unless a step policy says otherwise",
    }
    payload.update(extra)
    return json.dumps(payload)


def step_dict(step_id, description, criteria, deps=(), purpose="p", outcome="o") -> dict:
    return {
        "id": step_id,
        "description": description,
        "purpose": purpose,
        "expected_outcome": outcome,
        "completion_criteria": list(criteria),
        "dependencies": list(deps),
        "failure_strategy": "stop",
        "retry_policy": 0,
    }


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Points cwd + the file-policy allowlist at a temp dir so real tools are safe."""
    monkeypatch.setattr(paths, "ALLOWED_BASE_DIR", tmp_path)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def execute_to_completion(agent, task, msg):
    """Drives the confirmation loop until no pending action remains."""
    while msg.pending_action:
        result = _run(execute_pending_action(msg.pending_action))
        msg = _run(continue_task_after_confirmation(agent, task, result, []))
    return msg


# ---------------------------------------------------------------------------
# T1 — INFORMATIONAL: answer, no plan, no tools, no fs changes
# ---------------------------------------------------------------------------


def test_t1_informational_no_plan_no_tools(sandbox):
    classification = classify_task_deterministic("Explain how dependency injection works.")
    assert classification.task_type is TaskType.INFORMATIONAL

    engine = FakeEngine([])
    task = _run(prepare_task_for_execution("Explain how dependency injection works.", engine))
    assert task is None  # no TaskState, no plan, no extra LLM call
    assert engine.calls == []  # deterministic classification, zero LLM overhead
    # Nothing on disk, nothing executed.
    assert list(sandbox.iterdir()) == []


# ---------------------------------------------------------------------------
# T2 — SIMPLE ACTION: minimal routing, no multi-step workflow
# ---------------------------------------------------------------------------


def test_t2_simple_action_fast_path(sandbox):
    # Single-action file listing is a FILE_OPERATION — explicitly NOT a
    # complex task type, so it stays on the fast path (no plan).
    classification = classify_task_deterministic("List the files in this directory.")
    assert classification.task_type is TaskType.FILE_OPERATION

    engine = FakeEngine([])
    task = _run(prepare_task_for_execution("List the files in this directory.", engine))
    assert task is None
    assert engine.calls == []


# ---------------------------------------------------------------------------
# T3 — MULTI-STEP GENERAL TASK: structured plan, not just tool calls
# ---------------------------------------------------------------------------


def test_t3_multistep_gets_structured_outcome_plan():
    classification = classify_task_deterministic(
        "Find all Python files, count their lines, and save the result to report.txt."
    )
    assert classification.task_type is TaskType.MULTI_STEP

    planner = FakeEngine(
        [
            plan_json(
                steps=[
                    step_dict(1, "Discover Python files", ["Python files enumerated"]),
                    step_dict(2, "Count lines per file", ["Line counts calculated"], deps=[1]),
                    step_dict(3, "Generate the report", ["report.txt written"], deps=[2]),
                    step_dict(4, "Verify the report", ["Report verified"], deps=[3]),
                ],
                completion=["Report produced"],
                verification=["Report contents verified against the files"],
            )
        ]
    )
    task = _run(
        prepare_task_for_execution(
            "Find all Python files, count their lines, and save the result to report.txt.",
            planner,
            cwd="/tmp",
        )
    )
    assert task is not None
    assert task.task_type is TaskType.MULTI_STEP
    assert task.plan is not None
    assert len(task.plan.steps) == 4
    # Steps are OUTCOME-oriented, not tool calls.
    assert "Discover Python files" == task.plan.steps[0].description
    assert task.plan.steps[3].description == "Verify the report"  # final verification
    assert task.plan.steps[1].dependencies == [1]  # explicit ordering
    assert validate_plan(task.plan).valid
    assert task.total_steps == 4
    assert task.is_complete() is False  # a plan alone never completes a task


# ---------------------------------------------------------------------------
# T4 — NEW PROJECT: structured plan + execution must not stop after mkdir
# ---------------------------------------------------------------------------


def test_t4_new_project_does_not_stop_after_mkdir(sandbox):
    classification = classify_task_deterministic("Create a small FastAPI task management API.")
    assert classification.task_type is TaskType.SOFTWARE_ENGINEERING
    assert detect_workspace_kind(str(sandbox)) is WorkspaceKind.NEW_WORKSPACE

    task = TaskState(goal=classification.goal, task_type=TaskType.SOFTWARE_ENGINEERING)
    task.attach_plan(
        TaskPlan(
            goal=classification.goal,
            task_type=TaskType.SOFTWARE_ENGINEERING,
            workspace=WorkspaceKind.NEW_WORKSPACE,
            steps=[
                PlanStep(id=1, description="Establish the workspace",
                         expected_outcome="dir exists", completion_criteria=["Project dir exists"]),
                PlanStep(id=2, description="Implement the API",
                         expected_outcome="app files exist",
                         completion_criteria=["API files exist"], dependencies=[1]),
                PlanStep(id=3, description="Validate the implementation",
                         expected_outcome="app imports cleanly",
                         completion_criteria=["API validated"], dependencies=[2]),
            ],
            completion_criteria=["FastAPI task API implemented"],
            verification_requirements=["API starts without errors"],
        )
    )

    engine = FakeEngine(
        [
            _tool("run_command", command="mkdir taskapi"),  # step 1 — gated
            "Workspace ready.",
            plan_verify([("Project dir exists", True)],
                        [("FastAPI task API implemented", False),
                         ("API starts without errors", False)]),
            _tool("write_file", filename="taskapi/main.py", content="app = FastAPI()"),  # step 2
            "API implemented.",
            plan_verify([("API files exist", True)],
                        [("FastAPI task API implemented", False),
                         ("API starts without errors", False)]),
            _tool("run_command", command="python -c 'import taskapi.main'"),  # step 3
            "API validated.",
            plan_verify([("API validated", True)],
                        [("FastAPI task API implemented", True),
                         ("API starts without errors", True)]),
        ]
    )
    agent = ReActAgent(engine)
    msg = _run(agent.run("Create a small FastAPI task management API.", task=task))
    msg = execute_to_completion(agent, task, msg)

    assert task.is_complete() is True
    assert task.status is TaskStatus.TASK_COMPLETED
    # The task continued past the mkdir: all steps succeeded and files exist.
    assert [s.status for s in task.plan.steps] == [
        StepStatus.SUCCEEDED,
        StepStatus.SUCCEEDED,
        StepStatus.SUCCEEDED,
    ]
    assert (sandbox / "taskapi" / "main.py").exists()
    assert [e.tool_name for e in task.execution_history] == [
        "run_command", "write_file", "run_command",
    ]


# ---------------------------------------------------------------------------
# T5 — FEATURE IMPLEMENTATION in an EXISTING project
# ---------------------------------------------------------------------------


def test_t5_feature_implementation_in_existing_project(sandbox):
    # An existing repo: planner must detect it and start by inspecting it.
    (sandbox / "pyproject.toml").write_text("[project]\nname = 'api'\n")
    (sandbox / "src").mkdir()
    (sandbox / "src" / "main.py").write_text("def app(): ...\n")
    assert detect_workspace_kind(str(sandbox)) is WorkspaceKind.EXISTING_PROJECT

    classification = classify_task_deterministic("Add JWT authentication to this API.")
    assert classification.task_type is TaskType.SOFTWARE_ENGINEERING

    planner = FakeEngine(
        [
            plan_json(
                steps=[
                    step_dict(1, "Inspect the existing architecture", ["Architecture understood"]),
                    step_dict(2, "Identify the authentication flow", ["Auth flow located"], deps=[1]),
                    step_dict(3, "Implement JWT authentication", ["JWT integrated"], deps=[2]),
                    step_dict(4, "Update dependencies and configuration", ["Dependencies updated"], deps=[3]),
                    step_dict(5, "Test and verify", ["Tests pass"], deps=[4]),
                ],
                completion=["JWT authentication implemented"],
                verification=["Login flow works with a valid token"],
            )
        ]
    )
    task = _run(
        prepare_task_for_execution("Add JWT authentication to this API.", planner, cwd=str(sandbox))
    )
    assert task is not None
    assert task.plan.workspace is WorkspaceKind.EXISTING_PROJECT
    # The plan begins with inspection, never a blind first tool call.
    assert task.plan.steps[0].description.startswith("Inspect")
    assert len(task.plan.steps) == 5
    assert task.plan.steps[-1].description.startswith("Test")


# ---------------------------------------------------------------------------
# T6 — BUG FIX: reproduce -> inspect -> fix -> regression -> verify
# ---------------------------------------------------------------------------


def test_t6_bug_fix_plan_and_execution(sandbox, monkeypatch):
    classification = classify_task_deterministic("Fix the failing login tests.")
    assert classification.task_type is TaskType.DEBUGGING
    assert classification.goal == "Make the login tests pass."

    monkeypatch.setitem(
        reg.TOOLS, "read_file",
        lambda file_path: "def login(u, p):\n    return u == 'admin'",
    )
    task = TaskState(goal=classification.goal, task_type=TaskType.DEBUGGING)
    task.attach_plan(
        TaskPlan(
            goal=classification.goal,
            task_type=TaskType.DEBUGGING,
            workspace=WorkspaceKind.EXISTING_PROJECT,
            steps=[
                PlanStep(id=1, description="Reproduce the failure",
                         expected_outcome="failure observed",
                         completion_criteria=["Failure reproduced"]),
                PlanStep(id=2, description="Locate the root cause",
                         expected_outcome="cause identified",
                         completion_criteria=["Root cause located"], dependencies=[1]),
                PlanStep(id=3, description="Fix and rerun tests",
                         expected_outcome="tests pass",
                         completion_criteria=["Login tests pass"], dependencies=[2]),
            ],
            completion_criteria=["Login tests pass"],
            verification_requirements=["Test suite green"],
        )
    )
    engine = FakeEngine(
        [
            _tool("run_command", command="pytest tests/test_login.py"),
            "Failure reproduced.",
            plan_verify([("Failure reproduced", True)],
                        [("Login tests pass", False), ("Test suite green", False)]),
            _tool("read_file", file_path="auth.py"),
            "Root cause: hardcoded comparison.",
            plan_verify([("Root cause located", True)],
                        [("Login tests pass", False), ("Test suite green", False)]),
            _tool("run_command", command="pytest tests/test_login.py"),
            "All tests pass.",
            plan_verify([("Login tests pass", True)],
                        [("Login tests pass", True), ("Test suite green", True)]),
        ]
    )
    agent = ReActAgent(engine)
    msg = _run(agent.run("Fix the failing login tests.", task=task))
    msg = execute_to_completion(agent, task, msg)
    assert task.is_complete() is True
    assert [s.status for s in task.plan.steps] == [StepStatus.SUCCEEDED] * 3


# ---------------------------------------------------------------------------
# T7 — DEBUGGING: reproduce -> inspect logs -> trace -> fix -> rerun -> verify
# ---------------------------------------------------------------------------


def test_t7_debugging_startup_crash(sandbox, monkeypatch):
    classification = classify_task_deterministic(
        "Find out why the application crashes during startup and fix it."
    )
    assert classification.task_type is TaskType.DEBUGGING

    monkeypatch.setitem(
        reg.TOOLS, "read_file",
        lambda file_path: "Traceback ... KeyError: 'DATABASE_URL'",
    )
    task = TaskState(goal=classification.goal, task_type=TaskType.DEBUGGING)
    task.attach_plan(
        TaskPlan(
            goal=classification.goal,
            task_type=TaskType.DEBUGGING,
            workspace=WorkspaceKind.EXISTING_PROJECT,
            steps=[
                PlanStep(id=1, description="Reproduce the startup crash",
                         expected_outcome="crash observed",
                         completion_criteria=["Crash reproduced"]),
                PlanStep(id=2, description="Inspect logs and trace the root cause",
                         expected_outcome="cause traced",
                         completion_criteria=["Root cause traced"], dependencies=[1]),
                PlanStep(id=3, description="Implement the fix",
                         expected_outcome="fix applied",
                         completion_criteria=["Fix applied"], dependencies=[2]),
                PlanStep(id=4, description="Rerun and verify startup",
                         expected_outcome="app starts",
                         completion_criteria=["Startup verified"], dependencies=[3]),
            ],
            completion_criteria=["Application starts without crashing"],
            verification_requirements=["Startup smoke test passes"],
        )
    )
    engine = FakeEngine(
        [
            _tool("run_command", command="python -m app"),
            "Crashes with KeyError: 'DATABASE_URL'.",
            plan_verify([("Crash reproduced", True)],
                        [("Application starts without crashing", False),
                         ("Startup smoke test passes", False)]),
            _tool("read_file", file_path="config.py"),
            "Root cause: missing DATABASE_URL default.",
            plan_verify([("Root cause traced", True)],
                        [("Application starts without crashing", False),
                         ("Startup smoke test passes", False)]),
            _tool("write_file", filename="config.py", content="DATABASE_URL = os.getenv(...)"),
            "Fix applied.",
            plan_verify([("Fix applied", True)],
                        [("Application starts without crashing", False),
                         ("Startup smoke test passes", False)]),
            _tool("run_command", command="python -m app"),
            "Starts cleanly.",
            plan_verify([("Startup verified", True)],
                        [("Application starts without crashing", True),
                         ("Startup smoke test passes", True)]),
        ]
    )
    agent = ReActAgent(engine)
    msg = _run(agent.run(classification.goal, task=task))
    msg = execute_to_completion(agent, task, msg)
    assert task.is_complete() is True
    assert [e.tool_name for e in task.execution_history] == [
        "run_command", "read_file", "write_file", "run_command",
    ]


# ---------------------------------------------------------------------------
# T8 — REFACTOR: inspect -> identify duplication -> design -> modify -> verify
# ---------------------------------------------------------------------------


def test_t8_refactor_plan(sandbox):
    classification = classify_task_deterministic(
        "Refactor this authentication module to remove duplicated logic."
    )
    assert classification.task_type is TaskType.SOFTWARE_ENGINEERING

    planner = FakeEngine(
        [
            plan_json(
                steps=[
                    step_dict(1, "Inspect the authentication implementation", ["Implementation read"]),
                    step_dict(2, "Identify duplicated logic", ["Duplication mapped"], deps=[1]),
                    step_dict(3, "Design the target structure", ["Target structure designed"], deps=[2]),
                    step_dict(4, "Apply the refactor", ["Refactor applied"], deps=[3]),
                    step_dict(5, "Run tests and verify behavior", ["Behavior unchanged"], deps=[4]),
                ],
                completion=["Duplicated logic removed"],
                verification=["Tests pass and behavior is unchanged"],
            )
        ]
    )
    task = _run(prepare_task_for_execution(
        "Refactor this authentication module to remove duplicated logic.",
        planner, cwd="/tmp"))
    assert task is not None
    assert task.plan.steps[0].description.startswith("Inspect")
    assert task.plan.steps[1].description.startswith("Identify duplicated")
    assert task.plan.steps[-1].description.startswith("Run tests")
    assert validate_plan(task.plan).valid


# ---------------------------------------------------------------------------
# T9 — CODE REVIEW: read-only, must not modify source files
# ---------------------------------------------------------------------------


def test_t9_code_review_is_read_only(sandbox, monkeypatch):
    classification = classify_task_deterministic("Review this repository for security issues.")
    assert classification.task_type is TaskType.CODE_REVIEW

    monkeypatch.setitem(reg.TOOLS, "read_file", lambda file_path: "code under review")
    task = TaskState(goal=classification.goal, task_type=TaskType.CODE_REVIEW)
    task.attach_plan(
        TaskPlan(
            goal=classification.goal,
            task_type=TaskType.CODE_REVIEW,
            workspace=WorkspaceKind.EXISTING_PROJECT,
            steps=[
                PlanStep(id=1, description="Inspect the code", expected_outcome="code read",
                         completion_criteria=["Code inspected"]),
                PlanStep(id=2, description="Gather evidence and identify findings",
                         expected_outcome="findings listed",
                         completion_criteria=["Findings identified"], dependencies=[1]),
                PlanStep(id=3, description="Produce the review report",
                         expected_outcome="report written",
                         completion_criteria=["Report produced"], dependencies=[2]),
            ],
            completion_criteria=["Security review complete"],
            verification_requirements=["Findings validated against evidence"],
        )
    )
    engine = FakeEngine(
        [
            _tool("read_file", file_path="app.py"),
            "Code inspected.",
            plan_verify([("Code inspected", True)],
                        [("Security review complete", False),
                         ("Findings validated against evidence", False)]),
            "Findings: hardcoded secret, missing rate limit.",
            plan_verify([("Findings identified", True)],
                        [("Security review complete", False),
                         ("Findings validated against evidence", False)]),
            "Final report.",
            plan_verify([("Report produced", True)],
                        [("Security review complete", True),
                         ("Findings validated against evidence", True)]),
        ]
    )
    agent = ReActAgent(engine)
    msg = _run(agent.run(classification.goal, task=task))
    assert msg.pending_action is None  # nothing state-changing was requested
    assert task.is_complete() is True
    # A review may only READ — no write/command executions, no files written.
    assert all(e.tool_name == "read_file" for e in task.execution_history)
    assert not any(e.tool_name in ("write_file", "run_command") for e in task.execution_history)
    assert list(sandbox.iterdir()) == []


# ---------------------------------------------------------------------------
# T10 — DEPENDENCY UPGRADE
# ---------------------------------------------------------------------------


def test_t10_dependency_upgrade_plan(sandbox):
    classification = classify_task_deterministic(
        "Upgrade the project's FastAPI dependency and fix any compatibility issues."
    )
    assert classification.task_type is TaskType.SOFTWARE_ENGINEERING

    (sandbox / "pyproject.toml").write_text("[project]\ndependencies = ['fastapi==0.100']\n")
    planner = FakeEngine(
        [
            plan_json(
                steps=[
                    step_dict(1, "Inspect the current dependency configuration", ["Config inspected"]),
                    step_dict(2, "Modify the dependency", ["Version bumped"], deps=[1]),
                    step_dict(3, "Run tests and repair compatibility issues", ["Compatibility fixed"], deps=[2]),
                    step_dict(4, "Verify the upgrade", ["Upgrade verified"], deps=[3]),
                ],
                completion=["FastAPI upgraded"],
                verification=["Tests pass on the new version"],
            )
        ]
    )
    task = _run(prepare_task_for_execution(
        "Upgrade the project's FastAPI dependency and fix any compatibility issues.",
        planner, cwd=str(sandbox)))
    assert task is not None
    assert task.plan.workspace is WorkspaceKind.EXISTING_PROJECT
    assert task.plan.steps[0].description.startswith("Inspect")
    assert task.plan.steps[-1].description.startswith("Verify")
    assert validate_plan(task.plan).valid


# ---------------------------------------------------------------------------
# T11 — EXISTING PROJECT: add an endpoint without creating a new project
# ---------------------------------------------------------------------------


def test_t11_existing_project_endpoint(sandbox):
    # A realistic fake project.
    (sandbox / "pyproject.toml").write_text("[project]\n")
    (sandbox / "src").mkdir()
    (sandbox / "src" / "app.py").write_text("from fastapi import FastAPI\napp = FastAPI()\n")
    (sandbox / "tests").mkdir()
    (sandbox / "tests" / "test_app.py").write_text("def test_health():\n    pass\n")
    (sandbox / "README.md").write_text("# api\n")

    classification = classify_task_deterministic("Add a health-check endpoint.")
    assert classification.task_type is TaskType.SOFTWARE_ENGINEERING
    assert detect_workspace_kind(str(sandbox)) is WorkspaceKind.EXISTING_PROJECT

    planner = FakeEngine(
        [
            plan_json(
                steps=[
                    step_dict(1, "Inspect the repository and framework", ["Structure understood"]),
                    step_dict(2, "Locate the application entrypoint", ["Entrypoint located"], deps=[1]),
                    step_dict(3, "Implement the health-check endpoint", ["Endpoint implemented"], deps=[2]),
                    step_dict(4, "Add or update tests", ["Tests updated"], deps=[3]),
                    step_dict(5, "Run tests and verify", ["Tests pass"], deps=[4]),
                ],
                completion=["Health-check endpoint added"],
                verification=["GET /health returns 200"],
            )
        ]
    )
    task = _run(prepare_task_for_execution("Add a health-check endpoint.", planner, cwd=str(sandbox)))
    assert task is not None
    assert task.plan.workspace is WorkspaceKind.EXISTING_PROJECT
    # The plan respects the existing structure: it starts by inspecting the
    # repo and ends with running the project's tests — no new-project scaffold.
    assert task.plan.steps[0].description == "Inspect the repository and framework"
    assert task.plan.steps[-1].description.startswith("Run tests")
    assert all(step.dependencies or step.id == 1 for step in task.plan.steps)
    assert validate_plan(task.plan).valid


# ---------------------------------------------------------------------------
# T12 — RESEARCH + IMPLEMENTATION (compound task)
# ---------------------------------------------------------------------------


def test_t12_research_then_implementation(sandbox, monkeypatch):
    classification = classify_task_deterministic(
        "Research how this library's current API should be used and update my code to use the recommended API."
    )
    # Compound: research first, then implementation — classified RESEARCH.
    assert classification.task_type is TaskType.RESEARCH

    monkeypatch.setitem(reg.TOOLS, "read_file", lambda file_path: "old-style API call")
    task = TaskState(goal=classification.goal, task_type=TaskType.RESEARCH)
    task.attach_plan(
        TaskPlan(
            goal=classification.goal,
            task_type=TaskType.RESEARCH,
            workspace=WorkspaceKind.EXISTING_PROJECT,
            steps=[
                PlanStep(id=1, description="Research the recommended API", expected_outcome="API known",
                         completion_criteria=["Recommended API researched"]),
                PlanStep(id=2, description="Inspect the existing usage", expected_outcome="usage known",
                         completion_criteria=["Existing usage located"], dependencies=[1]),
                PlanStep(id=3, description="Update the code to the recommended API",
                         expected_outcome="code updated",
                         completion_criteria=["Code updated"], dependencies=[2]),
                PlanStep(id=4, description="Test and verify", expected_outcome="verified",
                         completion_criteria=["Verified"], dependencies=[3]),
            ],
            completion_criteria=["Code uses the recommended API"],
            verification_requirements=["Tests pass with the new API"],
        )
    )
    engine = FakeEngine(
        [
            "Research: use the new `client()` entrypoint.",
            plan_verify([("Recommended API researched", True)],
                        [("Code uses the recommended API", False),
                         ("Tests pass with the new API", False)]),
            _tool("read_file", file_path="src/client.py"),
            "Old usage found.",
            plan_verify([("Existing usage located", True)],
                        [("Code uses the recommended API", False),
                         ("Tests pass with the new API", False)]),
            _tool("write_file", filename="src/client.py", content="client()"),
            "Code updated.",
            plan_verify([("Code updated", True)],
                        [("Code uses the recommended API", False),
                         ("Tests pass with the new API", False)]),
            "Verified with the new API.",
            plan_verify([("Verified", True)],
                        [("Code uses the recommended API", True),
                         ("Tests pass with the new API", True)]),
        ]
    )
    agent = ReActAgent(engine)
    msg = _run(agent.run(classification.goal, task=task))
    msg = execute_to_completion(agent, task, msg)
    assert task.is_complete() is True
    # Research (read) preceded implementation (write) — compound order kept.
    assert [e.tool_name for e in task.execution_history] == ["read_file", "write_file"]


# ---------------------------------------------------------------------------
# T13 — CONFIRMATION INTERRUPTION: nothing lost across the boundary
# ---------------------------------------------------------------------------


def test_t13_confirmation_interruption_preserves_everything(sandbox, monkeypatch):
    monkeypatch.setitem(reg.TOOLS, "read_file", lambda file_path: "from fastapi import FastAPI")
    task = TaskState(goal="Add a health-check endpoint")
    task.attach_plan(
        TaskPlan(
            goal=task.goal,
            task_type=TaskType.SOFTWARE_ENGINEERING,
            workspace=WorkspaceKind.EXISTING_PROJECT,
            steps=[
                PlanStep(id=1, description="Inspect the entrypoint", expected_outcome="read",
                         completion_criteria=["Entrypoint inspected"]),
                PlanStep(id=2, description="Implement the endpoint", expected_outcome="written",
                         completion_criteria=["Endpoint implemented"], dependencies=[1]),
            ],
            completion_criteria=["Endpoint added"],
            verification_requirements=["Endpoint verified"],
        )
    )
    engine = FakeEngine(
        [
            _tool("read_file", file_path="src/app.py"),  # step 1 (read-only)
            "Entrypoint inspected.",
            plan_verify([("Entrypoint inspected", True)],
                        [("Endpoint added", False), ("Endpoint verified", False)]),
            _tool("write_file", filename="src/app.py", content="app = FastAPI()"),  # step 2 — gated
            "Endpoint implemented.",
            plan_verify([("Endpoint implemented", True)],
                        [("Endpoint added", True), ("Endpoint verified", True)]),
        ]
    )
    agent = ReActAgent(engine)
    msg = _run(agent.run("Add a health-check endpoint", task=task))
    task = msg.task_state

    # Step 1 completed; step 2 is now gated and parked in WAITING_CONFIRMATION.
    assert msg.pending_action is not None
    assert msg.pending_action.action_type == "write_file"
    assert task.plan.step(1).status is StepStatus.SUCCEEDED  # work retained
    assert task.current_plan_step().id == 2
    assert task.current_plan_step().status is StepStatus.WAITING_CONFIRMATION
    assert task.status is TaskStatus.WAITING_CONFIRMATION

    goal_before = task.goal
    plan_before = task.plan
    history_len_before = len(task.execution_history)

    result = _run(execute_pending_action(msg.pending_action))
    msg = _run(continue_task_after_confirmation(agent, task, result, []))
    task = msg.task_state

    # After approval: goal, plan object, and completed work all survive; the
    # step resumed and no work was repeated or lost.
    assert task.goal == goal_before
    assert task.plan is plan_before  # plan never regenerated
    assert task.plan.step(1).status is StepStatus.SUCCEEDED  # not restarted
    assert task.plan.step(1).attempts == 0
    assert task.plan.step(2).status is StepStatus.RUNNING or task.is_complete()
    assert len(task.execution_history) > history_len_before  # observation recorded
    assert task.status is not TaskStatus.TASK_STARTED  # not restarted from scratch

    msg = execute_to_completion(agent, task, msg)
    assert task.is_complete() is True
    assert (sandbox / "src" / "app.py").exists()


# ---------------------------------------------------------------------------
# T14 — FAILURE RECOVERY: failure recorded, recovery, final verification
# ---------------------------------------------------------------------------


def test_t14_failed_intermediate_step_recovers(sandbox, monkeypatch):
    state = {"calls": 0}

    def flaky(file_path):
        state["calls"] += 1
        if state["calls"] == 1:
            return "Error: transient read failure"
        return "healthy config"

    monkeypatch.setitem(reg.TOOLS, "read_file", flaky)
    task = TaskState(goal="Repair the service config")
    task.attach_plan(
        TaskPlan(
            goal=task.goal,
            task_type=TaskType.DEBUGGING,
            workspace=WorkspaceKind.EXISTING_PROJECT,
            steps=[
                PlanStep(id=1, description="Read the config", expected_outcome="read",
                         completion_criteria=["Config read"]),
                PlanStep(id=2, description="Verify the config", expected_outcome="verified",
                         completion_criteria=["Config verified"], dependencies=[1]),
            ],
            completion_criteria=["Config repaired"],
            verification_requirements=["Config verified"],
        )
    )
    engine = FakeEngine(
        [
            _tool("read_file", file_path="config.toml"),  # fails once
            "The config is fixed.",  # false claim while step 1 unmet
            plan_verify([("Config read", False)],
                        [("Config repaired", False), ("Config verified", False)]),
            _tool("read_file", file_path="config.toml"),  # retry succeeds
            "Config read.",
            plan_verify([("Config read", True)],
                        [("Config repaired", False), ("Config verified", False)]),
            "Config verified.",
            plan_verify([("Config verified", True)],
                        [("Config repaired", True), ("Config verified", True)]),
        ]
    )
    agent = ReActAgent(engine)
    msg = _run(agent.run("Repair the service config", task=task))
    task = msg.task_state
    # The failure was recorded against the step, and the false claim overruled.
    assert task.plan.step(1).attempts >= 1
    assert task.plan.step(1).error is not None
    assert any(not e.success for e in task.execution_history)
    notes = [m.content for m in task.context if m.name == "task_verification"]
    assert any("is not complete" in n for n in notes)

    msg = execute_to_completion(agent, task, msg)
    assert task.is_complete() is True  # recovered and verified
    assert [s.status for s in task.plan.steps] == [StepStatus.SUCCEEDED] * 2


def test_t14_fatal_step_failure_marks_task_failed():
    task = TaskState(goal="Deploy the service")
    task.attach_plan(
        TaskPlan(
            goal=task.goal,
            task_type=TaskType.SYSTEM_OPERATION,
            workspace=WorkspaceKind.EXISTING_PROJECT,
            steps=[
                PlanStep(id=1, description="Deploy", expected_outcome="deployed",
                         completion_criteria=["Deployment succeeded"]),
            ],
            completion_criteria=["Service deployed"],
            verification_requirements=["Service healthy"],
        )
    )
    engine = FakeEngine(
        [
            "The deployment is done.",
            plan_verify([("Deployment succeeded", False)],
                        [("Service deployed", False), ("Service healthy", False)],
                        step_failed=True, revision_reason="no such target"),
        ]
    )
    agent = ReActAgent(engine)
    msg = _run(agent.run("Deploy the service", task=task))
    task = msg.task_state
    # STOP strategy on the failed step: the task is marked failed, never
    # complete, and the failure is reported.
    assert "incomplete" in msg.content.lower()
    assert task.is_complete() is False
    assert task.status is TaskStatus.TASK_FAILED
    assert task.plan.step(1).status is StepStatus.FAILED


# ---------------------------------------------------------------------------
# T15 — FALSE COMPLETION: \"Done\" while steps remain is rejected
# ---------------------------------------------------------------------------


def test_t15_false_completion_rejected(sandbox):
    task = TaskState(goal="Create a FastAPI backend")
    task.attach_plan(
        TaskPlan(
            goal=task.goal,
            task_type=TaskType.SOFTWARE_ENGINEERING,
            workspace=WorkspaceKind.NEW_WORKSPACE,
            steps=[
                PlanStep(id=1, description="Establish the workspace", expected_outcome="dir",
                         completion_criteria=["Project directory exists"]),
                PlanStep(id=2, description="Implement the app", expected_outcome="app",
                         completion_criteria=["Backend files exist"], dependencies=[1]),
                PlanStep(id=3, description="Verify", expected_outcome="verified",
                         completion_criteria=["Application verified"], dependencies=[2]),
            ],
            completion_criteria=["Backend implemented"],
            verification_requirements=["Backend starts"],
        )
    )
    engine = FakeEngine(
        [
            "Done, everything is complete.",  # false claim — steps 1..3 remain
            plan_verify([("Project directory exists", False)],
                        [("Backend implemented", True), ("Backend starts", True)]),
            _tool("run_command", command="mkdir backend"),
            "Workspace ready.",
            plan_verify([("Project directory exists", True)],
                        [("Backend implemented", False), ("Backend starts", False)]),
            "Backend app files written.",
            plan_verify([("Backend files exist", True)],
                        [("Backend implemented", False), ("Backend starts", False)]),
            "Verified.",
            plan_verify([("Application verified", True)],
                        [("Backend implemented", True), ("Backend starts", True)]),
        ]
    )
    agent = ReActAgent(engine)
    msg = _run(agent.run("Create a FastAPI backend", task=task))
    msg = execute_to_completion(agent, task, msg)
    # The false claim was overruled: the directory had to actually exist
    # before any step advanced, and completion came only after step 3.
    assert task.is_complete() is True
    assert [s.status for s in task.plan.steps] == [StepStatus.SUCCEEDED] * 3
    assert any(e.tool_name == "run_command" for e in task.execution_history)
    # The claim was explicitly rejected before any work was done.
    notes = [m.content for m in task.context if m.name == "task_verification"]
    assert any("is not complete" in n for n in notes)


# ---------------------------------------------------------------------------
# T16 — INVALID PLAN (circular dependency) rejected before execution
# ---------------------------------------------------------------------------


def test_t16_circular_dependency_plan_rejected():
    circular = TaskPlan(
        goal="g",
        task_type=TaskType.SOFTWARE_ENGINEERING,
        workspace=WorkspaceKind.NEW_WORKSPACE,
        steps=[
            PlanStep(id=1, description="a", expected_outcome="o", completion_criteria=["c"], dependencies=[2]),
            PlanStep(id=2, description="b", expected_outcome="o", completion_criteria=["c"], dependencies=[1]),
        ],
        completion_criteria=["done"],
        verification_requirements=["verified"],
    )
    report = validate_plan(circular)
    assert report.valid is False  # invalid plans must never execute
    assert report.circular_dependencies  # the cycle is reported


def test_t16_invalid_llm_plan_falls_back(sandbox):
    # The LLM proposes a cyclic plan: it must be rejected and replaced by the
    # fallback plan — execution never runs an invalid plan.
    planner = FakeEngine(
        [
            plan_json(
                steps=[
                    step_dict(1, "a", ["c"], deps=[2]),
                    step_dict(2, "b", ["c"], deps=[1]),
                ],
                completion=["done"],
                verification=["verified"],
            )
        ]
    )
    task = _run(
        prepare_task_for_execution("Create a FastAPI backend", planner, cwd=str(sandbox))
    )
    assert task is not None
    assert len(task.plan.steps) == 1  # fallback verification plan, not the invalid one
    assert task.plan.steps[0].description == "Verify the final user goal"
    assert validate_plan(task.plan).valid


# ---------------------------------------------------------------------------
# T17 — MAX ITERATIONS: stops safely, marks incomplete, reports remaining
# ---------------------------------------------------------------------------


def test_t17_max_iterations_stops_safely():
    task = TaskState(goal="Create a FastAPI backend")
    task.attach_plan(
        TaskPlan(
            goal=task.goal,
            task_type=TaskType.SOFTWARE_ENGINEERING,
            workspace=WorkspaceKind.NEW_WORKSPACE,
            steps=[
                PlanStep(id=1, description="Establish the workspace", expected_outcome="dir",
                         completion_criteria=["Project directory exists"]),
                PlanStep(id=2, description="Implement", expected_outcome="app",
                         completion_criteria=["Backend files exist"], dependencies=[1]),
            ],
            completion_criteria=["Backend implemented"],
            verification_requirements=["Backend starts"],
        )
    )
    never_done = plan_verify(
        [("Project directory exists", False)],
        [("Backend implemented", False), ("Backend starts", False)],
    )
    engine = FakeEngine(["Almost done.", never_done, "Almost done.", never_done])
    agent = ReActAgent(engine, max_iterations=2)
    msg = _run(agent.run("Create a FastAPI backend", task=task))
    task = msg.task_state

    assert "incomplete" in msg.content.lower()
    assert "remaining plan steps" in msg.content.lower()  # remaining work reported
    assert task.is_complete() is False
    assert task.status is TaskStatus.TASK_FAILED
    assert task.plan.step(1).status is StepStatus.FAILED
    assert len(engine.calls) <= 5  # bounded — no infinite loop


# ---------------------------------------------------------------------------
# T18 — SECURITY: confirmation stays mandatory inside a multi-step task
# ---------------------------------------------------------------------------


def test_t18_state_changing_action_still_requires_confirmation(sandbox):
    task = TaskState(goal="Set up a test directory")
    task.attach_plan(
        TaskPlan(
            goal=task.goal,
            task_type=TaskType.SYSTEM_OPERATION,
            workspace=WorkspaceKind.NEW_WORKSPACE,
            steps=[
                PlanStep(id=1, description="Create the directory", expected_outcome="dir",
                         completion_criteria=["Directory exists"]),
            ],
            completion_criteria=["Directory set up"],
            verification_requirements=["Directory verified"],
        )
    )
    engine = FakeEngine([_tool("run_command", command="mkdir TestDir")])
    agent = ReActAgent(engine)
    msg = _run(agent.run("Set up a test directory", task=task))

    # The planner cannot bypass security: the state-changing action is still
    # gated, the task parks in WAITING_CONFIRMATION, and nothing executed.
    assert msg.pending_action is not None
    assert msg.pending_action.action_type == "run_command"
    assert task.status is TaskStatus.WAITING_CONFIRMATION
    assert task.current_plan_step().status is StepStatus.WAITING_CONFIRMATION
    assert not (sandbox / "TestDir").exists()  # nothing ran without approval


def test_t18_guardrail_blocked_action_stays_blocked_mid_task(sandbox):
    task = TaskState(goal="Write project notes")
    task.attach_plan(
        TaskPlan(
            goal=task.goal,
            task_type=TaskType.FILE_OPERATION,
            workspace=WorkspaceKind.NEW_WORKSPACE,
            steps=[
                PlanStep(id=1, description="Write the notes file", expected_outcome="written",
                         completion_criteria=["Notes written"]),
            ],
            completion_criteria=["Notes file created"],
            verification_requirements=["Notes verified"],
        )
    )
    # A guardrail hard-block (secret exfiltration) must never execute.
    engine = FakeEngine(
        [
            _tool("write_file", filename="leak.txt", content="aws key AKIA1234567890ABCDEF"),
            "I could not write it.",
        ]
    )
    agent = ReActAgent(engine)
    msg = _run(agent.run("Write project notes", task=task))
    assert not (sandbox / "leak.txt").exists()  # blocked action never ran
    # The blocked observation was fed back into the task context — the agent
    # saw "Blocked by security", not a silent pass or a pending confirmation.
    blocked = [m for m in msg.task_state.context if m.name == "write_file"]
    assert blocked, "the blocked write must be recorded in the task context"
    assert any("Blocked by security" in m.content for m in blocked)


# ---------------------------------------------------------------------------
# T19 — NO OVER-PLANNING: the classifier distinguishes complexity
# ---------------------------------------------------------------------------


def test_t19_no_over_planning():
    # Informational: fast, no plan, no LLM.
    assert classify_task_deterministic("What is a linked list?").task_type is TaskType.INFORMATIONAL
    engine = FakeEngine([])
    assert _run(prepare_task_for_execution("What is a linked list?", engine)) is None
    assert engine.calls == []

    # Simple action: fast, no plan.
    assert classify_task_deterministic("Run git status.").task_type is TaskType.SIMPLE_ACTION
    engine = FakeEngine([])
    assert _run(prepare_task_for_execution("Run git status.", engine)) is None
    assert engine.calls == []

    # Complex compound request: full planning path.
    classification = classify_task_deterministic(
        "Fix the failing tests and refactor the authentication module."
    )
    assert classification.task_type in (
        TaskType.DEBUGGING,
        TaskType.SOFTWARE_ENGINEERING,
        TaskType.MULTI_STEP,
    )
    planner = FakeEngine(
        [
            plan_json(
                steps=[
                    step_dict(1, "Inspect the codebase", ["Codebase inspected"]),
                    step_dict(2, "Fix the failing tests", ["Tests pass"], deps=[1]),
                    step_dict(3, "Refactor the authentication module", ["Refactor applied"], deps=[2]),
                    step_dict(4, "Verify", ["Verified"], deps=[3]),
                ],
                completion=["Tests pass and module refactored"],
                verification=["Full suite green"],
            )
        ]
    )
    task = _run(prepare_task_for_execution(
        "Fix the failing tests and refactor the authentication module.", planner, cwd="/tmp"))
    assert task is not None
    assert task.plan is not None
    assert len(task.plan.steps) >= 3


# ---------------------------------------------------------------------------
# T20 — PLAN ADAPTATION: the plan changes when new information appears
# ---------------------------------------------------------------------------


REVISION_STEPS = [
    {
        "id": 3,
        "description": "Identify the missing configuration",
        "purpose": "Inspection revealed a missing environment variable",
        "expected_outcome": "Missing config identified",
        "completion_criteria": ["Missing config identified"],
        "dependencies": [1],
        "failure_strategy": "stop",
        "retry_policy": 0,
    },
    {
        "id": 4,
        "description": "Configure the test environment",
        "purpose": "Provide the missing variable",
        "expected_outcome": "Environment configured",
        "completion_criteria": ["Environment configured"],
        "dependencies": [3],
        "failure_strategy": "stop",
        "retry_policy": 0,
    },
    {
        "id": 5,
        "description": "Rerun tests and verify",
        "purpose": "Confirm the fix",
        "expected_outcome": "Tests pass",
        "completion_criteria": ["Tests pass"],
        "dependencies": [4],
        "failure_strategy": "stop",
        "retry_policy": 0,
    },
]


def test_t20_plan_adapts_to_new_evidence(sandbox):
    task = TaskState(goal="Make the failing tests pass")
    task.attach_plan(
        TaskPlan(
            goal=task.goal,
            task_type=TaskType.DEBUGGING,
            workspace=WorkspaceKind.EXISTING_PROJECT,
            steps=[
                PlanStep(id=1, description="Inspect the test setup", expected_outcome="understood",
                         completion_criteria=["Test setup inspected"]),
                PlanStep(id=2, description="Run the failing tests", expected_outcome="failure seen",
                         completion_criteria=["Failure reproduced"], dependencies=[1]),
                PlanStep(id=3, description="Fix the implementation", expected_outcome="fixed",
                         completion_criteria=["Implementation fixed"], dependencies=[2]),
                PlanStep(id=4, description="Verify", expected_outcome="verified",
                         completion_criteria=["Tests pass"], dependencies=[3]),
            ],
            completion_criteria=["Tests pass"],
            verification_requirements=["Full suite green"],
        )
    )
    engine = FakeEngine(
        [
            "The tests fail because DATABASE_URL is missing.",
            plan_verify(
                [("Test setup inspected", True)],
                [("Tests pass", False), ("Full suite green", False)],
                revision=REVISION_STEPS,  # new evidence: missing env var
            ),
            "The missing variable is DATABASE_URL.",
            plan_verify([("Missing config identified", True)],
                        [("Tests pass", False), ("Full suite green", False)]),
            "Environment configured.",
            plan_verify([("Environment configured", True)],
                        [("Tests pass", False), ("Full suite green", False)]),
            "Tests pass now.",
            plan_verify([("Tests pass", True)],
                        [("Tests pass", True), ("Full suite green", True)]),
        ]
    )
    agent = ReActAgent(engine)
    msg = _run(agent.run("Make the failing tests pass", task=task))
    task = msg.task_state

    assert task.is_complete() is True
    # The plan adapted: completed step 1 preserved; steps 2-4 replaced by the
    # configuration-focused remaining work (3, 4, 5).
    assert [s.id for s in task.plan.steps] == [1, 3, 4, 5]
    assert task.plan.step(1).status is StepStatus.SUCCEEDED
    assert task.plan.step(3).description == "Identify the missing configuration"
    assert task.plan_revisions, "an adaptive revision must be recorded"
    assert task.total_steps == 4
