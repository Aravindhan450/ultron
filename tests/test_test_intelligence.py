"""
FIX #5 — Test Intelligence: deterministic tests.

Covers the four areas of the Fix #5 audit:

A. TEST DISCOVERY         — discover_test_files() builds a deterministic
                            multi-language test-file inventory.
B. AFFECTED-TEST SELECTION — select_affected_tests() maps changed source
                            files to likely-affected tests by convention
                            (mirror / sibling / module / node variants) and
                            via the code-intelligence dependents hook.
C. FAILURE LOCALIZATION   — localize_failure() extracts file/line/test_name
                            from pytest/traceback/go/jest output, and
                            classify_failure() carries the location.
D. FALSE-COMPLETION REJECTION — a model claiming "done" while tests are
                            failing is rejected: the localized failure is
                            recorded as evidence, the task refuses
                            completion, and execution continues until the
                            tests actually pass.

Scenarios D are driven with a scripted FakeEngine (no real LLM) against the
REAL ReAct loop and REAL pytest confined to temporary repositories — the
Ultron repository itself is never modified.
"""

import asyncio
import json
import sys

import pytest

from ultron.core.agents.react import ReActAgent
from ultron.core.coding.context import CodeContext
from ultron.core.coding.executor import (
    FailureAnalysis,
    FailureCategory,
    classify_failure,
    localize_failure,
)
from ultron.core.coding.intelligence.facade import CodeIntelligence
from ultron.core.coding.test_selection import (
    discover_test_files,
    select_affected_tests,
)
from ultron.core.coding.workspace import discover_workspace
from ultron.core.tools import paths as tools_paths
from ultron.core.types import (
    FailureStrategy,
    PlanStep,
    TaskPlan,
    TaskState,
    TaskType,
    WorkspaceKind,
)
from ultron.main import (
    continue_task_after_confirmation,
    execute_pending_action,
)

PYTHON = sys.executable


class FakeEngine:
    """Scripted engine — deterministic responses, no real LLM."""

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


def _tool_call(tool, **arguments):
    return f"```json\n{json.dumps({'tool': tool, 'arguments': arguments})}\n```"


def _pytest_call(project_file: str = "test_math.py") -> str:
    # -B (never write bytecode) avoids stale __pycache__ making pytest see
    # pre-edit code after a file edit.
    return _tool_call(
        "run_command",
        command=f"{PYTHON} -B -m pytest -q -p no:cacheprovider {project_file}",
    )


def _all_satisfied(step_criteria, plan_criteria) -> str:
    payload = {
        "step_criteria": [
            {"description": c, "satisfied": True} for c in step_criteria
        ],
        "plan_criteria": [
            {"description": c, "satisfied": True} for c in plan_criteria
        ],
        "step_failed": False,
        "plan_revision": None,
    }
    return json.dumps(payload)


def _make_task(
    goal: str,
    task_type: TaskType,
    steps: list[PlanStep],
    plan_criteria: list[str],
    verification: list[str],
    workspace: WorkspaceKind,
    cwd: str,
) -> TaskState:
    """Hand-builds a planned TaskState with a CodeContext (no planner LLM)."""
    task = TaskState(goal=goal, task_type=task_type)
    task.attach_plan(
        TaskPlan(
            goal=goal,
            task_type=task_type,
            workspace=workspace,
            steps=steps,
            completion_criteria=plan_criteria,
            verification_requirements=verification,
        )
    )
    task.code_context = CodeContext(workspace=discover_workspace(cwd))
    task.code_context.attach_task(task)
    return task


def _step(
    step_id: int,
    description: str,
    criteria: list[str],
    deps: list[int] | None = None,
) -> PlanStep:
    return PlanStep(
        id=step_id,
        description=description,
        purpose=f"Purpose of {description}",
        expected_outcome=f"Outcome of {description}",
        completion_criteria=criteria,
        dependencies=deps or [],
        failure_strategy=FailureStrategy.STOP,
    )


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(tools_paths, "ALLOWED_BASE_DIR", tmp_path)
    monkeypatch.chdir(tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# A. TEST DISCOVERY
# ---------------------------------------------------------------------------


def test_discover_python_test_files(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_math.py").write_text("", encoding="utf-8")
    (tmp_path / "tests" / "auth").mkdir()
    (tmp_path / "tests" / "auth" / "test_service.py").write_text("", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("", encoding="utf-8")
    (tmp_path / "README.md").write_text("", encoding="utf-8")

    found = discover_test_files(tmp_path)
    assert found == ["tests/auth/test_service.py", "tests/test_math.py"]


def test_discover_node_test_files(tmp_path):
    (tmp_path / "__tests__").mkdir()
    (tmp_path / "__tests__" / "login.test.ts").write_text("", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "util.spec.js").write_text("", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "index.ts").write_text("", encoding="utf-8")

    found = discover_test_files(tmp_path)
    assert "__tests__/login.test.ts" in found
    assert "tests/util.spec.js" in found
    assert "src/index.ts" not in found


def test_discover_go_rust_java_test_files(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "service_test.go").write_text("", encoding="utf-8")
    (tmp_path / "tests" / "service_test.rs").write_text("", encoding="utf-8")
    (tmp_path / "tests" / "ServiceTest.java").write_text("", encoding="utf-8")
    (tmp_path / "tests" / "ServiceTests.java").write_text("", encoding="utf-8")
    (tmp_path / "tests" / "Helper.java").write_text("", encoding="utf-8")

    found = discover_test_files(tmp_path)
    assert "tests/service_test.go" in found
    assert "tests/service_test.rs" in found
    assert "tests/ServiceTest.java" in found
    assert "tests/ServiceTests.java" in found
    assert "tests/Helper.java" not in found


def test_discover_ignores_vendored_dirs(tmp_path):
    (tmp_path / "node_modules" / "pkg").mkdir(parents=True)
    (tmp_path / "node_modules" / "pkg" / "x.test.js").write_text("", encoding="utf-8")
    (tmp_path / ".venv" / "lib").mkdir(parents=True)
    (tmp_path / ".venv" / "lib" / "y_test.py").write_text("", encoding="utf-8")
    (tmp_path / "__pycache__").mkdir(parents=True)
    (tmp_path / "__pycache__" / "z_test.py").write_text("", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "real_test.py").write_text("", encoding="utf-8")

    found = discover_test_files(tmp_path)
    assert found == ["tests/real_test.py"]


def test_discover_empty_repo(tmp_path):
    assert discover_test_files(tmp_path) == []


def test_discover_bounded(tmp_path):
    (tmp_path / "tests").mkdir()
    for i in range(50):
        (tmp_path / "tests" / f"test_{i:03}.py").write_text("", encoding="utf-8")
    found = discover_test_files(tmp_path, max_files=10)
    assert len(found) == 10


def test_discover_explicit_test_dirs(tmp_path):
    (tmp_path / "spec").mkdir()
    (tmp_path / "spec" / "a_test.py").write_text("", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "b_test.py").write_text("", encoding="utf-8")

    found = discover_test_files(tmp_path, test_dirs=["spec"])
    assert found == ["spec/a_test.py"]


# ---------------------------------------------------------------------------
# B. AFFECTED-TEST SELECTION
# ---------------------------------------------------------------------------


def _write_files(root, paths):
    for rel in paths:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("", encoding="utf-8")


def test_affected_mirror_convention(tmp_path):
    _write_files(
        tmp_path,
        [
            "src/auth/service.py",
            "tests/auth/test_service.py",
            "tests/auth/test_other.py",
        ],
    )
    selected = select_affected_tests(["src/auth/service.py"], tmp_path)
    assert selected == ["tests/auth/test_service.py"]


def test_affected_module_level_and_sibling(tmp_path):
    _write_files(
        tmp_path,
        [
            "src/auth/service.py",
            "tests/auth/service_test.py",
            "tests/test_auth.py",
        ],
    )
    selected = select_affected_tests(["src/auth/service.py"], tmp_path)
    assert "tests/auth/service_test.py" in selected
    assert "tests/test_auth.py" in selected


def test_affected_only_existing_tests(tmp_path):
    # Candidate names are generated for all conventions, but only files that
    # actually exist on disk are returned.
    _write_files(
        tmp_path,
        ["src/auth/service.py", "tests/auth/test_service.py"],
    )
    selected = select_affected_tests(["src/auth/service.py"], tmp_path)
    assert selected == ["tests/auth/test_service.py"]


def test_affected_multiple_changed_files(tmp_path):
    _write_files(
        tmp_path,
        [
            "src/auth/service.py",
            "src/billing/invoice.py",
            "tests/auth/test_service.py",
            "tests/test_billing.py",
        ],
    )
    selected = select_affected_tests(
        ["src/auth/service.py", "src/billing/invoice.py"], tmp_path
    )
    assert selected == ["tests/auth/test_service.py", "tests/test_billing.py"]


def test_affected_node_variants(tmp_path):
    _write_files(
        tmp_path,
        [
            "src/auth/service.ts",
            "tests/auth/service.test.ts",
            "tests/auth/service.spec.ts",
        ],
    )
    selected = select_affected_tests(["src/auth/service.ts"], tmp_path)
    assert "tests/auth/service.test.ts" in selected
    assert "tests/auth/service.spec.ts" in selected


def test_affected_changed_test_selects_itself(tmp_path):
    _write_files(tmp_path, ["tests/auth/test_service.py"])
    selected = select_affected_tests(["tests/auth/test_service.py"], tmp_path)
    assert selected == ["tests/auth/test_service.py"]


def test_affected_no_tests(tmp_path):
    _write_files(tmp_path, ["src/auth/service.py"])
    assert select_affected_tests(["src/auth/service.py"], tmp_path) == []


def test_affected_deterministic(tmp_path):
    _write_files(
        tmp_path,
        ["src/auth/service.py", "tests/auth/test_service.py"],
    )
    first = select_affected_tests(["src/auth/service.py"], tmp_path)
    second = select_affected_tests(["src/auth/service.py"], tmp_path)
    assert first == second == ["tests/auth/test_service.py"]


def test_affected_never_returns_non_test_files(tmp_path):
    # A near-miss test-looking name that is NOT a test file must never leak
    # into the selection, even when its stem matches a changed module.
    _write_files(
        tmp_path,
        [
            "src/auth/service.py",
            "tests/auth/test_service_helpers.py",  # helper, not a test
            "tests/auth/test_service.py",
        ],
    )
    selected = select_affected_tests(["src/auth/service.py"], tmp_path)
    assert selected == ["tests/auth/test_service.py"]
    assert "tests/auth/test_service_helpers.py" not in selected


def test_affected_go_in_place_discovery(tmp_path):
    # Go convention: service_test.go sits NEXT TO the source with no tests/
    # dir — the whole-tree fallback finds it and by_target matches it.
    _write_files(tmp_path, ["pkg/service.go", "pkg/service_test.go"])
    selected = select_affected_tests(["pkg/service.go"], tmp_path)
    assert selected == ["pkg/service_test.go"]


def test_affected_max_results_cap(tmp_path):
    # 20 changed files each mapping to their own test file -> 20 selections,
    # capped at max_results.
    _write_files(
        tmp_path,
        [f"src/mod_{i}.py" for i in range(20)]
        + [f"tests/test_mod_{i}.py" for i in range(20)],
    )
    changed = [f"src/mod_{i}.py" for i in range(20)]
    selected = select_affected_tests(changed, tmp_path, max_results=5)
    assert len(selected) == 5


def test_affected_via_intelligence_dependents(sandbox):
    # The test file has an UNUSUAL name that no convention maps to — only the
    # code-intelligence dependents relationship (FIX #4) can find it.
    (sandbox / "src").mkdir(parents=True)
    (sandbox / "src" / "auth").mkdir(parents=True)
    (sandbox / "src" / "auth" / "__init__.py").write_text("", encoding="utf-8")
    (sandbox / "src" / "auth" / "service.py").write_text(
        "def authenticate(token):\n    return token == 'tok-1'\n",
        encoding="utf-8",
    )
    (sandbox / "tests").mkdir()
    # A valid test file whose name matches NO convention for service.py
    # (module would need test_service / test_auth) — only the dependents
    # relationship can find it.
    (sandbox / "tests" / "test_integration.py").write_text(
        "from auth.service import authenticate\n",
        encoding="utf-8",
    )

    ci = CodeIntelligence(root=str(sandbox))
    ci.refresh()
    try:
        selected = select_affected_tests(
            ["src/auth/service.py"], sandbox, intelligence=ci
        )
        # test_integration.py imports auth.service, so dependents select it
        # even though no naming convention maps to it.
        assert "tests/test_integration.py" in selected
    finally:
        ci.close()


# ---------------------------------------------------------------------------
# C. FAILURE LOCALIZATION
# ---------------------------------------------------------------------------


def test_localize_pytest_failed_line():
    text = (
        "FAILED tests/test_auth.py::test_login - "
        "AssertionError: expected 5, got 4"
    )
    loc = localize_failure(stdout=text)
    assert loc.test_name == "tests/test_auth.py::test_login"
    assert loc.file == "tests/test_auth.py"
    assert loc.line is None


def test_localize_traceback_frame():
    text = 'File "src/auth/service.py", line 42, in authenticate\n    return token'
    loc = localize_failure(stdout=text)
    assert loc.file == "src/auth/service.py"
    assert loc.line == 42
    assert loc.test_name is None


def test_localize_short_pytest_node():
    text = "tests/test_auth.py::test_login_ok AssertionError: expected 5, got 4"
    loc = localize_failure(stdout=text)
    assert loc.test_name == "tests/test_auth.py::test_login_ok"
    assert loc.file == "tests/test_auth.py"


def test_localize_py_in_frame():
    text = "app.py:12: in <module>\n    raise ValueError('boom')"
    loc = localize_failure(stdout=text)
    assert loc.file == "app.py"
    assert loc.line == 12


def test_localize_go_failure():
    text = "--- FAIL: TestAuthenticate (0.01s)\n    service_test.go:31: got 'no'"
    loc = localize_failure(stdout=text)
    assert loc.test_name == "TestAuthenticate"
    assert loc.file == "service_test.go"
    assert loc.line == 31


def test_localize_jest_failure():
    text = "FAIL tests/auth.test.js\n  ● login › rejects invalid token"
    loc = localize_failure(stdout=text)
    assert loc.file == "tests/auth.test.js"


def test_localize_no_match():
    loc = localize_failure(stdout="weird unclassifiable noise")
    assert loc.file is None
    assert loc.line is None
    assert loc.test_name is None


def test_localize_never_raises():
    for junk in (None, "", "  ", "Exit code: 1"):
        loc = localize_failure(stdout=junk or "", stderr="")
        assert loc is not None


def test_classify_failure_carries_location():
    output = (
        "Exit code: 1\n"
        "Output:\n"
        "FAILED tests/test_auth.py::test_login - AssertionError: expected 5, got 4\n"
        '  File "src/auth/service.py", line 42, in authenticate\n'
        "    return token == 'tok-1'\n"
        "1 failed"
    )
    analysis = classify_failure("pytest -q", exit_code=1, stdout=output)
    assert analysis.category is FailureCategory.TEST_ASSERTION
    # The implementation frame is the repair target; the test node names the test.
    assert analysis.file == "src/auth/service.py"
    assert analysis.line == 42
    assert analysis.test_name == "tests/test_auth.py::test_login"


def test_failure_analysis_serializes_location():
    analysis = FailureAnalysis(
        category=FailureCategory.TEST_ASSERTION,
        command="pytest",
        summary="1 failed",
        file="src/auth/service.py",
        line=42,
        test_name="tests/test_auth.py::test_login",
    )
    restored = FailureAnalysis.model_validate(analysis.model_dump())
    assert restored.file == "src/auth/service.py"
    assert restored.line == 42
    assert restored.test_name == "tests/test_auth.py::test_login"
    assert restored.category is FailureCategory.TEST_ASSERTION


def test_failure_analysis_prompt_line_includes_location():
    analysis = FailureAnalysis(
        category=FailureCategory.TEST_ASSERTION,
        command="pytest",
        summary="1 failed",
        file="src/auth/service.py",
        line=42,
        test_name="tests/test_auth.py::test_login",
    )
    line = analysis.to_prompt_line()
    assert "[test_assertion]" in line
    assert "src/auth/service.py:42" in line
    assert "tests/test_auth.py::test_login" in line


# ---------------------------------------------------------------------------
# D. FALSE-COMPLETION REJECTION (FIX #5 flavor)
# ---------------------------------------------------------------------------


def test_done_claim_rejected_while_test_failure_localized(sandbox):
    # Real pytest fails; the model claims completion; the localized failure
    # becomes evidence; the task refuses completion and execution continues.
    (sandbox / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\ntestpaths=['.']\n", encoding="utf-8"
    )
    (sandbox / "math_util.py").write_text(
        "def add(a, b):\n    return a - b\n", encoding="utf-8"
    )
    (sandbox / "test_math.py").write_text(
        "from math_util import add\n\ndef test_add():\n    assert add(2, 3) == 5\n",
        encoding="utf-8",
    )
    not_done = json.dumps(
        {
            "step_criteria": [
                {"description": "tests pass", "satisfied": False}
            ],
            "plan_criteria": [
                {"description": "tests pass", "satisfied": False}
            ],
            "step_failed": False,
            "plan_revision": None,
        }
    )
    engine = FakeEngine(
        [
            _pytest_call(),
            "Done, everything works!",  # false claim while tests still fail
            not_done,  # verifier: not satisfied
            _tool_call(
                "replace_in_file",
                file_path="math_util.py",
                old="return a - b",
                new="return a + b",
            ),
            _pytest_call(),
            "Tests pass now.",
            _all_satisfied(["tests pass"], ["tests pass"]),
        ]
    )
    task = _make_task(
        "Fix the failing tests",
        TaskType.DEBUGGING,
        [_step(1, "Fix the failing tests", ["tests pass"])],
        ["tests pass"],
        ["tests pass"],
        WorkspaceKind.EXISTING_PROJECT,
        str(sandbox),
    )
    agent = ReActAgent(engine, max_iterations=8)
    msg = _run(agent.run(task.goal, [], task=task))

    # First confirmed action is the failing pytest run.
    assert msg.pending_action is not None
    result = _run(execute_pending_action(msg.pending_action))
    assert "Exit code: 1" in result

    msg = _run(continue_task_after_confirmation(agent, task, result, []))
    # The false "Done" claim was rejected: the agent kept working (its next
    # action is the edit) and the task is NOT complete.
    assert msg.pending_action is not None
    assert msg.pending_action.action_type == "replace_in_file"
    assert task.is_complete() is False
    # The task correctly sits in WAITING_CONFIRMATION for the next gated
    # action — a live task, never completed, never restarted.
    assert task.status.value == "waiting_confirmation"

    # The localized failure was recorded as verification evidence. pytest -q
    # truncates the traceback, so localization recovers the FAILED node:
    # test file + test name (line stays None for -q output — the honest
    # deterministic result).
    failures = task.code_context.executor.failures
    assert failures, "the failing pytest must be recorded as a classified failure"
    failure = failures[0]
    assert failure.category is FailureCategory.TEST_ASSERTION
    assert failure.test_name == "test_math.py::test_add"
    assert failure.file == "test_math.py"
    assert failure.line is None  # no frame in -q output — nothing invented
    evidence = task.code_context.executor.verification_evidence(task)
    assert "test_math.py::test_add" in evidence

    # Repair: execute the edit, rerun pytest (passes), verify → completes.
    result = _run(execute_pending_action(msg.pending_action))
    msg = _run(continue_task_after_confirmation(agent, task, result, []))

    assert msg.pending_action is not None  # the second pytest run
    result = _run(execute_pending_action(msg.pending_action))
    assert "Exit code: 0" in result
    msg = _run(continue_task_after_confirmation(agent, task, result, []))

    assert msg.pending_action is None
    assert task.is_complete() is True
    assert "a + b" in (sandbox / "math_util.py").read_text(encoding="utf-8")


def test_executor_records_failed_pytest_with_location(tmp_path):
    task = TaskState(goal="Fix tests")
    task.code_context = CodeContext(workspace=discover_workspace(str(tmp_path)))
    task.code_context.attach_task(task)

    task.code_context.executor.record_observation(
        "run_command",
        {"command": "pytest"},
        "Exit code: 1\nOutput:\n"
        "FAILED tests/test_auth.py::test_login - AssertionError: expected 5, got 4\n"
        "1 failed",
        succeeded=False,
    )
    failures = task.code_context.executor.failures
    assert len(failures) == 1
    assert failures[0].category is FailureCategory.TEST_ASSERTION
    assert failures[0].test_name == "tests/test_auth.py::test_login"
    assert failures[0].file == "tests/test_auth.py"
    # The repair budget counts the failure — no blind repetition.
    assert task.code_context.executor.budget.failure_count == 1


def test_verification_evidence_includes_localized_failure(tmp_path):
    task = TaskState(goal="Fix tests")
    task.add_requirements(["tests pass"])
    task.code_context = CodeContext(workspace=discover_workspace(str(tmp_path)))
    task.code_context.attach_task(task)
    task.code_context.executor.record_observation(
        "run_command",
        {"command": "pytest"},
        "Exit code: 1\nOutput:\nFAILED test_math.py::test_add - AssertionError\n"
        '  File "math_util.py", line 2, in add\n1 failed',
        succeeded=False,
    )
    evidence = task.code_context.executor.verification_evidence(task)
    assert "[test_assertion]" in evidence
    assert "test_math.py::test_add" in evidence
    # A failing test means the requirement cannot be complete.
    assert task.is_complete() is False
    with pytest.raises(ValueError):
        task.mark_complete()


def test_non_plan_done_claim_rejected_with_localized_evidence(sandbox):
    # Requirements-only task (no structured plan): the model claims "done"
    # while a failing pytest observation is recorded. TaskState rejects the
    # claim and the loop continues to a repair.
    (sandbox / "calc.py").write_text(
        "def mul(a, b):\n    return a + b\n", encoding="utf-8"
    )
    (sandbox / "test_calc.py").write_text(
        "from calc import mul\n\ndef test_mul():\n    assert mul(2, 3) == 6\n",
        encoding="utf-8",
    )
    not_done = json.dumps(
        [{"description": "tests pass", "satisfied": False}]
    )
    all_done = json.dumps(
        [{"description": "tests pass", "satisfied": True}]
    )
    engine = FakeEngine(
        [
            _pytest_call("test_calc.py"),
            "Done, everything works!",
            not_done,
            _tool_call(
                "replace_in_file",
                file_path="calc.py",
                old="return a + b",
                new="return a * b",
            ),
            _pytest_call("test_calc.py"),
            "Tests pass now.",
            all_done,
        ]
    )
    task = TaskState(goal="Make the tests pass")
    task.add_requirements(["tests pass"])
    task.code_context = CodeContext(workspace=discover_workspace(str(sandbox)))
    task.code_context.attach_task(task)

    agent = ReActAgent(engine, max_iterations=8)
    msg = _run(agent.run(task.goal, [], task=task))

    result = _run(execute_pending_action(msg.pending_action))
    assert "Exit code: 1" in result
    msg = _run(continue_task_after_confirmation(agent, task, result, []))

    # False claim rejected — the agent proceeded to the repair edit.
    assert msg.pending_action is not None
    assert msg.pending_action.action_type == "replace_in_file"
    assert task.is_complete() is False
    assert task.remaining_requirements()  # "tests pass" still open

    result = _run(execute_pending_action(msg.pending_action))
    msg = _run(continue_task_after_confirmation(agent, task, result, []))

    result = _run(execute_pending_action(msg.pending_action))
    assert "Exit code: 0" in result
    msg = _run(continue_task_after_confirmation(agent, task, result, []))

    assert msg.pending_action is None
    assert task.is_complete() is True
    assert task.remaining_requirements() == []
    assert "a * b" in (sandbox / "calc.py").read_text(encoding="utf-8")
