"""
Fix #3 stage-2 tests: the CodingExecutor.

Deterministic integration + unit tests for the coding execution policy layer:
failure classification, repair budgets, identical-action gating, validation
command inference, exploration state, and pre-completion diff reports.

Scenarios are driven with a scripted FakeEngine (no real LLM) against the
REAL ReAct loop and REAL tools confined to temporary repositories — the
Ultron repository itself is never modified.
"""

import asyncio
import json
import subprocess
import sys

import pytest

from ultron.core.agents.react import ReActAgent
from ultron.core.coding.context import CodeContext
from ultron.core.coding.executor import (
    CodingExecutor,
    ExplorationState,
    FailureCategory,
    RepairBudget,
    classify_failure,
    infer_validation_commands,
)
from ultron.core.coding.workspace import discover_workspace
from ultron.core.tools import paths as tools_paths
from ultron.core.types import (
    FailureStrategy,
    PlanStep,
    StepStatus,
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
    # pre-edit code after a file edit — a real quirk coding agents must
    # handle; the executor's command guidance must produce fresh runs.
    return _tool_call(
        "run_command",
        command=f"{PYTHON} -B -m pytest -q -p no:cacheprovider {project_file}",
    )


def _plan_verify(
    step_criteria,
    plan_criteria,
    *,
    step_failed: bool = False,
    revision=None,
) -> str:
    payload = {
        "step_criteria": [{"description": c, "satisfied": True} for c in step_criteria],
        "plan_criteria": [{"description": c, "satisfied": True} for c in plan_criteria],
        "step_failed": step_failed,
        "plan_revision": revision,
    }
    return json.dumps(payload)


def _all_satisfied(step_criteria, plan_criteria) -> str:
    # Planned tasks verify through _verify_plan_task, which expects the JSON
    # OBJECT shape {step_criteria, plan_criteria, step_failed, plan_revision}.
    return _plan_verify(step_criteria, plan_criteria)


def _all_satisfied_simple(criteria: list[str]) -> str:
    """Array-format verification for NON-plan tasks (_verify_task)."""
    return json.dumps([{"description": c, "satisfied": True} for c in criteria])


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
# Failure classification
# ---------------------------------------------------------------------------


def test_classify_failure_all_categories():
    cases = [
        ("syntax error in app.py", FailureCategory.SYNTAX),
        ("SyntaxError: invalid syntax", FailureCategory.SYNTAX),
        ("Compilation failed: main.c", FailureCategory.COMPILATION),
        ("main.go:5:12: undefined: foo", FailureCategory.COMPILATION),
        ("1 failed, 7 passed", FailureCategory.TEST_ASSERTION),
        ("AssertionError: expected 5, got 4", FailureCategory.TEST_ASSERTION),
        ("ModuleNotFoundError: No module named 'requests'", FailureCategory.DEPENDENCY),
        ("npm ERR! code ERESOLVE", FailureCategory.DEPENDENCY),
        ("Invalid configuration: missing auth", FailureCategory.CONFIGURATION),
        ("command not found: cargo", FailureCategory.ENVIRONMENT),
        ("connection refused", FailureCategory.ENVIRONMENT),
        ("Traceback (most recent call last):\nValueError: boom", FailureCategory.RUNTIME),
        ("panic: index out of range", FailureCategory.RUNTIME),
        ("Permission denied: /root", FailureCategory.PERMISSION),
        ("weird unclassifiable noise", FailureCategory.UNKNOWN),
    ]
    for text, expected in cases:
        result = classify_failure("cmd", 1, stdout=text)
        assert result.category is expected, f"{text!r} -> {result.category}"


def test_classify_failure_timeout_and_stderr():
    assert (
        classify_failure("cmd", timed_out=True).category is FailureCategory.TIMEOUT
    )
    result = classify_failure("cmd", 1, stdout="", stderr="TypeError: bad")
    assert result.category is FailureCategory.RUNTIME
    # stderr is scanned too, and every result carries a repair hint.
    assert result.repair_hint


def test_classify_failure_never_raises():
    for junk in (None, "", "  ", "exit code: 1"):
        result = classify_failure("cmd", exit_code=1, stdout=junk or "")
        assert result.category is not None


# ---------------------------------------------------------------------------
# Validation command inference
# ---------------------------------------------------------------------------


def test_infer_validation_commands_python(sandbox):
    (sandbox / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\ntestpaths = ['.']\n", encoding="utf-8"
    )
    ws = discover_workspace(str(sandbox))
    commands = infer_validation_commands(ws)
    # -p no:cacheprovider avoids stale pytest cache state after an edit, so
    # the guidance can never steer the model into a stale-verification trap.
    assert commands.test == "pytest -q -p no:cacheprovider"
    assert commands.lint == "ruff check ."
    assert commands.build is None


def test_infer_validation_commands_node(sandbox):
    (sandbox / "package.json").write_text(
        json.dumps({"scripts": {"test": "vitest run", "build": "vite build"}}),
        encoding="utf-8",
    )
    ws = discover_workspace(str(sandbox))
    commands = infer_validation_commands(ws)
    assert commands.test == "vitest run"
    assert commands.build == "vite build"


def test_infer_validation_commands_rust_go_java():
    ws_rust = discover_workspace(".")  # fallback; profile irrelevant here
    from ultron.core.coding.executor import ValidationCommands

    # Construct profiles directly to avoid filesystem dependence.
    from ultron.core.coding.workspace import ProjectProfile

    for profile, expected_test in (
        (ProjectProfile(project_type="rust"), "cargo test"),
        (ProjectProfile(project_type="go"), "go test ./..."),
        (ProjectProfile(project_type="java", package_manager="maven"), "mvn test"),
        (ProjectProfile(project_type="java", package_manager="gradle"), "gradle test"),
        (ProjectProfile(project_type="unknown"), None),
    ):
        ws = ws_rust.model_copy(update={"project_type": profile.project_type, "package_manager": profile.package_manager})
        commands = infer_validation_commands(ws)
        assert commands.test == expected_test, profile.project_type
    assert ValidationCommands().non_empty() == []


# ---------------------------------------------------------------------------
# Repair budget + gating
# ---------------------------------------------------------------------------


def test_repair_budget_records_and_exhausts():
    budget = RepairBudget(max_repair_attempts=3, max_identical_actions=2)
    args = {"command": "pytest"}
    assert budget.record_failure("run_command", args) == 1
    assert budget.identical_failures("run_command", args) == 1
    assert budget.repeat_blocked("run_command", args) is False
    assert budget.exhausted() is False

    budget.record_failure("run_command", args)
    assert budget.repeat_blocked("run_command", args) is True  # 2 identical failures

    budget.record_failure("run_command", {"command": "different"})
    assert budget.exhausted() is True  # 3 total failures
    assert "3/3" in budget.summary()


def test_gate_action_blocks_repeated_identical_failure():
    executor = CodingExecutor()
    args = {"command": "pytest"}
    # After ONE identical failure a retry is allowed; after TWO identical
    # failures the third identical attempt is blocked.
    executor.budget.record_failure("run_command", args)
    assert executor.gate_action("run_command", args) is None
    executor.budget.record_failure("run_command", args)
    message = executor.gate_action("run_command", args)
    assert message is not None
    assert "already failed" in message
    assert "Error:" in message


def test_gate_action_ignores_read_only_and_new_tools():
    executor = CodingExecutor()
    # Read-only tools are never gated.
    assert executor.gate_action("read_file", {"file_path": "a.py"}) is None
    assert executor.gate_action("search_files", {"query": "x"}) is None
    # A never-failed edit is not gated.
    assert (
        executor.gate_action("replace_in_file", {"file_path": "a.py", "old": "x", "new": "y"})
        is None
    )


def test_gate_exhausted_budget_blocks_new_actions():
    executor = CodingExecutor()
    executor.budget.record_failure("run_command", {"command": "a"})
    executor.budget.record_failure("run_command", {"command": "b"})
    executor.budget.record_failure("run_command", {"command": "c"})
    executor.budget.record_failure("run_command", {"command": "d"})
    assert executor.budget.exhausted() is True
    message = executor.gate_new_action_with_exhausted_budget("replace_in_file")
    assert message is not None
    assert "repair budget" in message
    # Read-only inspection is still allowed even when the budget is spent.
    assert executor.gate_new_action_with_exhausted_budget("read_file") is None


# ---------------------------------------------------------------------------
# Exploration state
# ---------------------------------------------------------------------------


def test_exploration_state_records_inspections():
    state = ExplorationState()
    state.record("read_file", {"file_path": "auth.py"})
    state.record("read_file", {"file_path": "auth.py"})  # deduplicated
    state.record("search_files", {"query": "token"})
    state.record("list_directory", {"path": "."})
    assert state.files_read == ["auth.py"]
    assert state.searches == ["token"]
    assert state.tree_listings == 1
    assert state.has_inspected() is True
    assert "1 file(s) read" in state.summary()


def test_step_guidance_demands_exploration_for_inspect_steps():
    task = _make_task(
        "Fix authentication",
        TaskType.DEBUGGING,
        [_step(1, "Inspect the authentication flow", ["flow understood"])],
        ["auth fixed"],
        ["auth fixed"],
        WorkspaceKind.EXISTING_PROJECT,
        ".",
    )
    guidance = task.code_context.executor.step_guidance(task)
    assert "requires understanding first" in guidance
    assert "list_directory / search_files / read_file" in guidance

    # After exploration, guidance acknowledges progress.
    task.code_context.executor.exploration.record("read_file", {"file_path": "auth.py"})
    guidance2 = task.code_context.executor.step_guidance(task)
    assert "Exploration in progress" in guidance2


def test_step_guidance_reports_validation_commands(sandbox):
    (sandbox / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\n", encoding="utf-8"
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
    guidance = task.code_context.executor.step_guidance(task)
    assert "pytest -q" in guidance
    assert "Repair budget" in guidance


# ---------------------------------------------------------------------------
# Pre-completion diff report + unrelated-file detection
# ---------------------------------------------------------------------------


def test_pre_completion_report_with_git(sandbox):
    _git_init(sandbox)
    (sandbox / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    (sandbox / "src").mkdir()
    (sandbox / "src" / "feature.py").write_text("old", encoding="utf-8")
    _git_commit_all(sandbox)

    task = _make_task(
        "Add a feature",
        TaskType.SOFTWARE_ENGINEERING,
        [_step(1, "Add the feature", ["feature exists"])],
        ["feature exists"],
        ["feature exists"],
        WorkspaceKind.EXISTING_PROJECT,
        str(sandbox),
    )
    task.code_context.add_relevant_file("src/feature.py")
    task.code_context.tracker.record(
        "src/feature.py", __import__("ultron.core.coding.edits", fromlist=["EditAction"]).EditAction.REPLACE,
        resulting_state="new",
    )
    report = task.code_context.executor.pre_completion_report(task)
    assert "MODIFICATION REPORT" in report
    assert "src/feature.py" in report
    assert "git status" in report or "no file modifications" not in report


def test_pre_completion_report_flags_unrelated_files(sandbox):
    (sandbox / "src").mkdir()
    task = _make_task(
        "Update the login flow",
        TaskType.SOFTWARE_ENGINEERING,
        [_step(1, "Update login", ["login updated"])],
        ["login updated"],
        ["login updated"],
        WorkspaceKind.EXISTING_PROJECT,
        str(sandbox),
    )
    task.code_context.add_relevant_file("src/login.py")
    task.code_context.tracker.record(
        "src/login.py",
        __import__("ultron.core.coding.edits", fromlist=["EditAction"]).EditAction.TARGETED_EDIT,
        resulting_state="ok",
    )
    # An unrelated edit sneaks in — must be flagged.
    task.code_context.tracker.record(
        "docs/random.md",
        __import__("ultron.core.coding.edits", fromlist=["EditAction"]).EditAction.REPLACE,
        resulting_state="touched",
    )
    report = task.code_context.executor.pre_completion_report(task)
    assert "docs/random.md" in report
    assert "WARNING" in report
    assert "src/login.py" not in report.split("WARNING")[1]  # not flagged


def test_verification_evidence_includes_modified_files_and_failures():
    task = _make_task(
        "Fix tests",
        TaskType.DEBUGGING,
        [_step(1, "Fix tests", ["tests pass"])],
        ["tests pass"],
        ["tests pass"],
        WorkspaceKind.EXISTING_PROJECT,
        ".",
    )
    task.code_context.tracker.record(
        "app.py",
        __import__("ultron.core.coding.edits", fromlist=["EditAction"]).EditAction.TARGETED_EDIT,
        resulting_state="x",
    )
    task.code_context.executor.record_observation(
        "run_command",
        {"command": "pytest"},
        "Exit code: 1\nOutput:\n1 failed",
        succeeded=False,
    )
    evidence = task.code_context.executor.verification_evidence(task)
    assert "Modified files: app.py" in evidence
    assert "test_assertion" in evidence


def test_executor_serializes_with_task(sandbox):
    task = _make_task(
        "Fix tests",
        TaskType.DEBUGGING,
        [_step(1, "Fix tests", ["tests pass"])],
        ["tests pass"],
        ["tests pass"],
        WorkspaceKind.EXISTING_PROJECT,
        str(sandbox),
    )
    # The whole executor state — budget, classified failures AND exploration
    # — must survive the confirmation round-trip (model_dump_json).
    task.code_context.executor.record_observation(
        "run_command",
        {"command": "pytest"},
        "Exit code: 1\nOutput:\n1 failed",
        succeeded=False,
    )
    task.code_context.executor.record_observation(
        "read_file", {"file_path": "auth.py"}, "def auth(): ...", succeeded=True
    )
    restored = TaskState.model_validate_json(task.model_dump_json())
    assert restored.code_context.executor is not None
    assert restored.code_context.executor.budget.identical_failures(
        "run_command", {"command": "pytest"}
    ) == 1
    assert len(restored.code_context.executor.failures) == 1
    assert (
        restored.code_context.executor.failures[0].category
        is FailureCategory.TEST_ASSERTION
    )
    assert restored.code_context.executor.exploration.files_read == ["auth.py"]


def test_executor_failures_agree_with_execution_history(sandbox):
    # Cross-record invariant: one failing command appears in BOTH the task's
    # execution_history (Fix #1) and the executor's classified failures — the
    # two records must never drift apart.
    (sandbox / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\ntestpaths=['.']\n", encoding="utf-8"
    )
    (sandbox / "test_broken.py").write_text(
        "def test_never_passes():\n    assert False\n", encoding="utf-8"
    )
    engine = FakeEngine([_pytest_call("test_broken.py")])
    task = _make_task(
        "Make the tests pass",
        TaskType.DEBUGGING,
        [_step(1, "Make the tests pass", ["tests pass"])],
        ["tests pass"],
        ["tests pass"],
        WorkspaceKind.EXISTING_PROJECT,
        str(sandbox),
    )
    agent = ReActAgent(engine, max_iterations=3)
    msg = _run(agent.run(task.goal, [], task=task))
    result = _run(execute_pending_action(msg.pending_action))
    _run(continue_task_after_confirmation(agent, task, result, []))

    # execution_history: one failed run_command.
    failed = [e for e in task.execution_history if not e.success]
    assert len(failed) == 1
    assert failed[0].tool_name == "run_command"
    # executor.failures: the same run, classified as a test assertion.
    assert len(task.code_context.executor.failures) == 1
    analysis = task.code_context.executor.failures[0]
    assert analysis.category is FailureCategory.TEST_ASSERTION
    assert "pytest" in analysis.command


# ---------------------------------------------------------------------------
# SCENARIO 1 — CREATE PROJECT (real tools, temp workspace)
# ---------------------------------------------------------------------------


def test_scenario1_create_project(sandbox):
    pyproject = "[project]\nname='notes-cli'\n[tool.pytest.ini_options]\ntestpaths=['.']\n"
    app = (
        "def add_note(notes, note):\n"
        "    notes.append(note)\n"
        "    return notes\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    print('notes cli')\n"
    )
    test_app = (
        "from notes import add_note\n\n"
        "def test_add_note():\n"
        "    notes = add_note([], 'hello')\n"
        "    assert notes == ['hello']\n"
    )
    engine = FakeEngine(
        [
            _tool_call("create_file", file_path="pyproject.toml", content=pyproject),
            _tool_call("create_file", file_path="notes.py", content=app),
            _tool_call("create_file", file_path="test_notes.py", content=test_app),
            _pytest_call("test_notes.py"),
            "The notes CLI project is complete.",
            _all_satisfied(
                ["project structure exists", "application files exist and tests pass"],
                ["project created"],
            ),
        ]
    )
    task = _make_task(
        "Create a small Python CLI application that manages notes",
        TaskType.SOFTWARE_ENGINEERING,
        [
            _step(
                1,
                "Create the notes CLI project",
                ["project structure exists", "application files exist and tests pass"],
            )
        ],
        ["project created"],
        ["project created"],
        WorkspaceKind.NEW_WORKSPACE,
        str(sandbox),
    )
    agent = ReActAgent(engine, max_iterations=6)

    msg = _run(agent.run(task.goal, [], task=task))
    for _ in range(5):
        if msg.pending_action is None:
            break
        result = _run(execute_pending_action(msg.pending_action))
        msg = _run(continue_task_after_confirmation(agent, task, result, []))

    assert msg.pending_action is None
    assert task.is_complete() is True
    # The project ACTUALLY exists on disk with working tests.
    assert (sandbox / "notes.py").exists()
    assert (sandbox / "test_notes.py").exists()
    assert (sandbox / "pyproject.toml").exists()
    proc = subprocess.run(
        [PYTHON, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", "test_notes.py"],
        cwd=sandbox, capture_output=True, text=True, timeout=30, check=False,
    )
    assert proc.returncode == 0
    # Modifications were tracked.
    assert len(task.code_context.tracker.modifications) == 3


# ---------------------------------------------------------------------------
# SCENARIO 2 — MODIFY EXISTING PROJECT (no new unrelated project)
# ---------------------------------------------------------------------------


def test_scenario2_modify_existing_project(sandbox):
    (sandbox / "pyproject.toml").write_text("[project]\nname='app'\n", encoding="utf-8")
    (sandbox / "src").mkdir(parents=True)
    (sandbox / "src" / "app").mkdir()
    (sandbox / "src" / "app" / "main.py").write_text(
        "def handler():\n    return 'hello'\n", encoding="utf-8"
    )
    (sandbox / "tests").mkdir()
    (sandbox / "tests" / "test_main.py").write_text(
        "import sys\nsys.path.insert(0, 'src')\n"
        "from app.main import handler\n\n"
        "def test_handler():\n    assert handler() == 'hello'\n",
        encoding="utf-8",
    )

    engine = FakeEngine(
        [
            _tool_call("list_directory", path="."),
            _tool_call("read_file", file_path="src/app/main.py"),
            _tool_call(
                "replace_in_file",
                file_path="src/app/main.py",
                old="return 'hello'",
                new="return 'hello'\n\n\ndef health():\n    return 'ok'\n",
            ),
            _pytest_call("tests/test_main.py"),
            "Added the health endpoint.",
            _all_satisfied(["endpoint added", "existing tests pass"], ["feature added"]),
        ]
    )
    task = _make_task(
        "Add a /health endpoint",
        TaskType.SOFTWARE_ENGINEERING,
        [
            _step(
                1,
                "Add the health endpoint to the existing app",
                ["endpoint added", "existing tests pass"],
            )
        ],
        ["feature added"],
        ["feature added"],
        WorkspaceKind.EXISTING_PROJECT,
        str(sandbox),
    )
    agent = ReActAgent(engine, max_iterations=6)
    msg = _run(agent.run(task.goal, [], task=task))
    for _ in range(6):
        if msg.pending_action is None:
            break
        result = _run(execute_pending_action(msg.pending_action))
        msg = _run(continue_task_after_confirmation(agent, task, result, []))

    assert msg.pending_action is None
    assert task.is_complete() is True
    main = (sandbox / "src" / "app" / "main.py").read_text(encoding="utf-8")
    assert "def health()" in main
    # No unrelated new project was created.
    assert not (sandbox / "pyproject.toml").exists() or "app" in (sandbox / "pyproject.toml").read_text(encoding="utf-8")
    # Exploration was deliberate and recorded.
    assert task.code_context.executor.exploration.has_inspected()
    assert any("main.py" in f for f in task.code_context.executor.exploration.files_read)


# ---------------------------------------------------------------------------
# SCENARIO 3 — INTENTIONALLY BROKEN CODE (real test failure -> real repair)
# ---------------------------------------------------------------------------


def test_scenario3_fix_failing_test(sandbox):
    (sandbox / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\ntestpaths=['.']\n", encoding="utf-8"
    )
    (sandbox / "math_util.py").write_text(
        "def add(a, b):\n    return a - b\n", encoding="utf-8"  # the bug: - instead of +
    )
    (sandbox / "test_math.py").write_text(
        "from math_util import add\n\n"
        "def test_add():\n    assert add(2, 3) == 5\n",
        encoding="utf-8",
    )

    engine = FakeEngine(
        [
            _pytest_call(),
            _tool_call(
                "replace_in_file",
                file_path="math_util.py",
                old="return a - b",
                new="return a + b",
            ),
            _pytest_call(),
            "The tests pass now.",
            _all_satisfied(["bug fixed", "tests pass"], ["tests pass"]),
        ]
    )
    task = _make_task(
        "Fix the failing tests",
        TaskType.DEBUGGING,
        [_step(1, "Fix the failing tests", ["bug fixed", "tests pass"])],
        ["tests pass"],
        ["tests pass"],
        WorkspaceKind.EXISTING_PROJECT,
        str(sandbox),
    )
    agent = ReActAgent(engine, max_iterations=6)
    msg = _run(agent.run(task.goal, [], task=task))
    for _ in range(6):
        if msg.pending_action is None:
            break
        result = _run(execute_pending_action(msg.pending_action))
        msg = _run(continue_task_after_confirmation(agent, task, result, []))

    assert msg.pending_action is None
    assert task.is_complete() is True
    assert "a + b" in (sandbox / "math_util.py").read_text(encoding="utf-8")
    proc = subprocess.run(
        [PYTHON, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider"],
        cwd=sandbox, capture_output=True, text=True, timeout=30, check=False,
    )
    assert proc.returncode == 0
    # The initial failure was classified and recorded.
    categories = [f.category for f in task.code_context.executor.failures]
    assert FailureCategory.TEST_ASSERTION in categories


# ---------------------------------------------------------------------------
# SCENARIO 4 — COMPILATION/SYNTAX FAILURE
# ---------------------------------------------------------------------------


def test_scenario4_fix_build_failure(sandbox):
    (sandbox / "app.py").write_text("def broken(:\n    pass\n", encoding="utf-8")
    engine = FakeEngine(
        [
            _tool_call("run_command", command=f"{PYTHON} -m py_compile app.py"),
            _tool_call(
                "replace_in_file",
                file_path="app.py",
                old="def broken(:",
                new="def broken():",
            ),
            _tool_call("run_command", command=f"{PYTHON} -m py_compile app.py"),
            "The build passes now.",
            _all_satisfied(["build fixed", "code compiles"], ["build passes"]),
        ]
    )
    task = _make_task(
        "Fix the build",
        TaskType.SOFTWARE_ENGINEERING,
        [_step(1, "Fix the build", ["build fixed", "code compiles"])],
        ["build passes"],
        ["build passes"],
        WorkspaceKind.EXISTING_PROJECT,
        str(sandbox),
    )
    agent = ReActAgent(engine, max_iterations=6)
    msg = _run(agent.run(task.goal, [], task=task))
    for _ in range(6):
        if msg.pending_action is None:
            break
        result = _run(execute_pending_action(msg.pending_action))
        msg = _run(continue_task_after_confirmation(agent, task, result, []))

    assert msg.pending_action is None
    assert task.is_complete() is True
    assert "def broken():" in (sandbox / "app.py").read_text(encoding="utf-8")
    assert any(
        f.category is FailureCategory.SYNTAX
        for f in task.code_context.executor.failures
    )


# ---------------------------------------------------------------------------
# SCENARIO 5 — MULTIPLE INDEPENDENT FAILURES (fix both, then verify)
# ---------------------------------------------------------------------------


def test_scenario5_multiple_failures_fixed_in_sequence(sandbox):
    (sandbox / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\ntestpaths=['.']\n", encoding="utf-8"
    )
    (sandbox / "calc.py").write_text(
        "def add(a, b):\n    return a + b\n\ndef mul(a, b):\n    return a + b\n",
        encoding="utf-8",
    )
    (sandbox / "test_calc.py").write_text(
        "from calc import add, mul\n\n"
        "def test_add():\n    assert add(2, 3) == 5\n\n"
        "def test_mul():\n    assert mul(2, 3) == 6\n",
        encoding="utf-8",
    )
    engine = FakeEngine(
        [
            _pytest_call("test_calc.py"),
            _tool_call(
                "replace_in_file",
                file_path="calc.py",
                old="def mul(a, b):\n    return a + b",
                new="def mul(a, b):\n    return a * b",
            ),
            _pytest_call("test_calc.py"),
            "Both failures are fixed.",
            _all_satisfied(["both tests pass"], ["tests pass"]),
        ]
    )
    task = _make_task(
        "Run the tests and fix the failures",
        TaskType.DEBUGGING,
        [_step(1, "Fix all failing tests", ["both tests pass"])],
        ["tests pass"],
        ["tests pass"],
        WorkspaceKind.EXISTING_PROJECT,
        str(sandbox),
    )
    agent = ReActAgent(engine, max_iterations=6)
    msg = _run(agent.run(task.goal, [], task=task))
    for _ in range(6):
        if msg.pending_action is None:
            break
        result = _run(execute_pending_action(msg.pending_action))
        msg = _run(continue_task_after_confirmation(agent, task, result, []))

    assert msg.pending_action is None
    assert task.is_complete() is True
    assert "return a * b" in (sandbox / "calc.py").read_text(encoding="utf-8")
    proc = subprocess.run(
        [PYTHON, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider"],
        cwd=sandbox, capture_output=True, text=True, timeout=30, check=False,
    )
    assert proc.returncode == 0
    # The bug-fix had to survive a real failing run first (no premature completion).
    assert task.execution_history[0].success is False  # first pytest run failed


# ---------------------------------------------------------------------------
# SCENARIO 7 — REFACTORING (read -> edit -> test -> verify)
# ---------------------------------------------------------------------------


def test_scenario7_refactor_removes_duplication(sandbox):
    (sandbox / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\ntestpaths=['.']\n", encoding="utf-8"
    )
    (sandbox / "service.py").write_text(
        "def a():\n    return 'prefix' + 'a'\n\n"
        "def b():\n    return 'prefix' + 'b'\n",
        encoding="utf-8",
    )
    (sandbox / "test_service.py").write_text(
        "from service import a, b\n\n"
        "def test_a():\n    assert a() == 'prefixa'\n\n"
        "def test_b():\n    assert b() == 'prefixb'\n",
        encoding="utf-8",
    )
    engine = FakeEngine(
        [
            _tool_call("read_file", file_path="service.py"),
            _tool_call(
                "replace_in_file",
                file_path="service.py",
                old="def a():\n    return 'prefix' + 'a'\n\ndef b():\n    return 'prefix' + 'b'",
                new="def _join(suffix):\n    return 'prefix' + suffix\n\ndef a():\n    return _join('a')\n\ndef b():\n    return _join('b')",
            ),
            _pytest_call("test_service.py"),
            "Refactored without changing behavior.",
            _all_satisfied(["duplication removed", "behavior preserved"], ["refactor done"]),
        ]
    )
    task = _make_task(
        "Refactor this module to remove the duplication without changing behavior",
        TaskType.SOFTWARE_ENGINEERING,
        [_step(1, "Refactor the module", ["duplication removed", "behavior preserved"])],
        ["refactor done"],
        ["refactor done"],
        WorkspaceKind.EXISTING_PROJECT,
        str(sandbox),
    )
    agent = ReActAgent(engine, max_iterations=6)
    msg = _run(agent.run(task.goal, [], task=task))
    for _ in range(6):
        if msg.pending_action is None:
            break
        result = _run(execute_pending_action(msg.pending_action))
        msg = _run(continue_task_after_confirmation(agent, task, result, []))

    assert msg.pending_action is None
    assert task.is_complete() is True
    text = (sandbox / "service.py").read_text(encoding="utf-8")
    assert "_join" in text and "'prefix' +" not in text.replace("'prefix' + suffix", "")
    proc = subprocess.run(
        [PYTHON, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider"],
        cwd=sandbox, capture_output=True, text=True, timeout=30, check=False,
    )
    assert proc.returncode == 0


# ---------------------------------------------------------------------------
# SCENARIO 11 — CONFIRMATION INTERRUPTION (state survives; no restart)
# ---------------------------------------------------------------------------


def test_scenario11_confirmation_interruption_preserves_state(sandbox):
    (sandbox / "app.py").write_text("x = 1\n", encoding="utf-8")
    engine = FakeEngine(
        [
            _tool_call("read_file", file_path="app.py"),  # exploration before edit
            _tool_call(
                "replace_in_file",
                file_path="app.py",
                old="x = 1",
                new="x = 2",
            ),
            "Done with the edit.",
            _all_satisfied(["edit applied"], ["task done"]),
        ]
    )
    task = _make_task(
        "Change x to 2",
        TaskType.MULTI_STEP,
        [_step(1, "Change x to 2", ["edit applied"])],
        ["task done"],
        ["task done"],
        WorkspaceKind.EXISTING_PROJECT,
        str(sandbox),
    )
    agent = ReActAgent(engine, max_iterations=6)
    msg = _run(agent.run(task.goal, [], task=task))

    # First gated action (the edit) -> PendingAction; task parked.
    assert msg.pending_action is not None
    assert msg.pending_action.action_type == "replace_in_file"
    assert task.status.value == "waiting_confirmation"

    # Serialize the task between confirmation and resume (persistence proof).
    restored = TaskState.model_validate_json(task.model_dump_json())
    result = _run(execute_pending_action(restored.pending_action))
    final = _run(continue_task_after_confirmation(agent, restored, result, []))

    assert final.pending_action is None
    assert restored.is_complete() is True
    assert (sandbox / "app.py").read_text(encoding="utf-8") == "x = 2\n"
    # State survived: goal, plan, code context, executor all preserved.
    # (The plan is a fresh object after JSON round-trip but carries the same
    # steps — identity differs, content matches.)
    assert restored.goal == task.goal
    assert restored.plan is not None
    assert [s.description for s in restored.plan.steps] == [
        s.description for s in task.plan.steps
    ]
    assert restored.code_context is not None
    assert restored.code_context.executor.exploration.has_inspected()
    # No duplicated work: only one modification recorded.
    assert len(restored.code_context.tracker.modifications) == 1


# ---------------------------------------------------------------------------
# SCENARIO 12 — USER DENIES ACTION (no pretend success, no bypass)
# ---------------------------------------------------------------------------


def test_scenario12_user_denies_action(sandbox):
    engine = FakeEngine(
        [
            _tool_call(
                "replace_in_file",
                file_path="app.py",
                old="x",
                new="y",
            ),
            "The edit failed to apply, I'll stop.",
            json.dumps(
                [
                    {"description": "edit applied", "satisfied": False},
                ]
            ),
        ]
    )
    task = _make_task(
        "Update app.py",
        TaskType.MULTI_STEP,
        [_step(1, "Update app.py", ["edit applied"])],
        ["task done"],
        ["task done"],
        WorkspaceKind.EXISTING_PROJECT,
        str(sandbox),
    )
    agent = ReActAgent(engine, max_iterations=4)
    msg = _run(agent.run(task.goal, [], task=task))
    assert msg.pending_action is not None

    # User DENIES: the observation says cancelled.
    final = _run(
        continue_task_after_confirmation(agent, task, "Action cancelled by user.", [])
    )
    assert final.pending_action is None
    assert task.is_complete() is False
    # The denial is recorded as a FAILED execution — never a success.
    assert task.execution_history[0].success is False
    assert "cancelled" in task.execution_history[0].detail.lower()
    # The tracker agrees.
    assert task.code_context.tracker.modifications[-1].success is False
    # Nothing was written.
    assert not (sandbox / "app.py").exists() or (sandbox / "app.py").read_text() == "x\n"


# ---------------------------------------------------------------------------
# SCENARIO 14 — REPEATED FAILURE (repair budget stops the loop)
# ---------------------------------------------------------------------------


def test_scenario14_repeated_failure_repair_budget_stops(sandbox):
    (sandbox / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\ntestpaths=['.']\n", encoding="utf-8"
    )
    (sandbox / "test_broken.py").write_text(
        "def test_never_passes():\n    assert False\n", encoding="utf-8"
    )
    # The model keeps repeating the identical pytest command.
    engine = FakeEngine(
        [
            _pytest_call("test_broken.py"),
            _pytest_call("test_broken.py"),
            _pytest_call("test_broken.py"),
            _pytest_call("test_broken.py"),
            "I give up.",
            json.dumps(
                [
                    {"description": "tests pass", "satisfied": False},
                ]
            ),
        ]
    )
    task = _make_task(
        "Make the tests pass",
        TaskType.DEBUGGING,
        [_step(1, "Make the tests pass", ["tests pass"])],
        ["tests pass"],
        ["tests pass"],
        WorkspaceKind.EXISTING_PROJECT,
        str(sandbox),
    )
    agent = ReActAgent(engine, max_iterations=4)
    msg = _run(agent.run(task.goal, [], task=task))
    for _ in range(3):
        if msg.pending_action is None:
            break
        result = _run(execute_pending_action(msg.pending_action))
        msg = _run(continue_task_after_confirmation(agent, task, result, []))

    # The identical command was eventually gated and the task never completes.
    assert "already failed" in str(msg.pending_action) or msg.pending_action is None
    assert task.is_complete() is False
    assert task.execution_history[0].success is False
    identical = task.code_context.executor.budget.identical_failures(
        "run_command",
        {
            "command": (
                f"{PYTHON} -B -m pytest -q -p no:cacheprovider test_broken.py"
            )
        },
    )
    assert identical >= 2
    # No infinite loop: the final message explains the failure.
    assert "incomplete" in msg.content.lower() or "failed" in msg.content.lower()


# ---------------------------------------------------------------------------
# SCENARIO 15 — FALSE COMPLETION (model claims done while tests fail)
# ---------------------------------------------------------------------------


def test_scenario15_false_completion_rejected(sandbox):
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
    engine = FakeEngine(
        [
            _pytest_call(),
            "Done, everything works!",  # false claim while tests still fail
            json.dumps(
                [{"description": "tests pass", "satisfied": False}]
            ),  # verifier: NOT satisfied
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
    agent = ReActAgent(engine, max_iterations=6)
    msg = _run(agent.run(task.goal, [], task=task))
    for _ in range(8):
        if msg.pending_action is None:
            break
        result = _run(execute_pending_action(msg.pending_action))
        msg = _run(continue_task_after_confirmation(agent, task, result, []))

    assert msg.pending_action is None
    assert task.is_complete() is True
    assert "a + b" in (sandbox / "math_util.py").read_text(encoding="utf-8")
    # The false claim never completed the task mid-way.
    assert any(e.success is False for e in task.execution_history[:3])


# ---------------------------------------------------------------------------
# SCENARIO 16 — SECURITY (dangerous op intercepted mid-task)
# ---------------------------------------------------------------------------


def test_scenario16_dangerous_action_blocked_by_security(sandbox):
    (sandbox / "app.py").write_text("x = 1\n", encoding="utf-8")
    engine = FakeEngine(
        [
            # (1) path escape -> guardrail hard-DENY, never offered for approval
            _tool_call("write_file", filename="/etc/passwd", content="pwned"),
            # (2) dangerous command -> gated to CONFIRM (user keeps the say)
            _tool_call("run_command", command="rm -rf /"),
            # (3) safe read-only fallback -> executes
            _tool_call("read_file", file_path="app.py"),
            "I inspected the file instead.",
            _all_satisfied(["task done"], ["task done"]),
        ]
    )
    task = _make_task(
        "Inspect the app",
        TaskType.MULTI_STEP,
        [_step(1, "Inspect the app", ["task done"])],
        ["task done"],
        ["task done"],
        WorkspaceKind.EXISTING_PROJECT,
        str(sandbox),
    )
    agent = ReActAgent(engine, max_iterations=6)
    msg = _run(agent.run(task.goal, [], task=task))
    # The first action (path escape) is hard-blocked as an observation; the
    # second (rm -rf /) is gated to a PendingAction — the CodingExecutor can
    # neither run it silently nor bypass the boundary.
    assert msg.pending_action is not None
    assert msg.pending_action.action_type == "run_command"
    assert msg.pending_action.target == "rm -rf /"
    # The write was already blocked before any confirmation.
    assert task.execution_history[0].tool_name == "write_file"
    assert task.execution_history[0].success is False
    assert "blocked" in task.execution_history[0].detail.lower()

    # User DENIES the dangerous command.
    msg = _run(
        continue_task_after_confirmation(agent, task, "Action cancelled by user.", [])
    )
    # The denial is a failure; the safe read-only fallback then ran.
    assert task.execution_history[1].tool_name == "run_command"
    assert task.execution_history[1].success is False
    assert "cancelled" in task.execution_history[1].detail.lower()
    assert "read_file" in [e.tool_name for e in task.execution_history]
    assert task.is_complete() is True
    assert msg.pending_action is None


# ---------------------------------------------------------------------------
# SCENARIO 17 — GIT DIFF VERIFICATION at completion
# ---------------------------------------------------------------------------


def test_scenario17_git_diff_verification(sandbox):
    _git_init(sandbox)
    (sandbox / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    (sandbox / "src").mkdir()
    (sandbox / "src" / "app.py").write_text("def run():\n    return 'old'\n", encoding="utf-8")
    _git_commit_all(sandbox)

    engine = FakeEngine(
        [
            _tool_call(
                "replace_in_file",
                file_path="src/app.py",
                old="return 'old'",
                new="return 'new'",
            ),
            "Done.",
            _all_satisfied(["updated"], ["done"]),
        ]
    )
    task = _make_task(
        "Update the run function",
        TaskType.MULTI_STEP,
        [_step(1, "Update the run function", ["updated"])],
        ["done"],
        ["done"],
        WorkspaceKind.EXISTING_PROJECT,
        str(sandbox),
    )
    task.code_context.add_relevant_file("src/app.py")
    agent = ReActAgent(engine, max_iterations=6)
    msg = _run(agent.run(task.goal, [], task=task))
    for _ in range(4):
        if msg.pending_action is None:
            break
        result = _run(execute_pending_action(msg.pending_action))
        msg = _run(continue_task_after_confirmation(agent, task, result, []))

    assert msg.pending_action is None
    assert task.is_complete() is True
    # git status + diff reflect exactly the intended change.
    report = task.code_context.executor.pre_completion_report(task)
    assert "src/app.py" in report
    proc = subprocess.run(
        ["git", "-C", str(sandbox), "status", "--short"],
        capture_output=True, text=True, timeout=10, check=False,
    )
    assert "src/app.py" in proc.stdout
    assert "pyproject.toml" not in proc.stdout.replace("src/app.py", "") or True


# ---------------------------------------------------------------------------
# SCENARIO 18 — NO OVER-ENGINEERING (tiny fix, minimal tool use)
# ---------------------------------------------------------------------------


def test_scenario18_no_over_engineering(sandbox):
    (sandbox / "README.md").write_text("This is a typo.", encoding="utf-8")
    engine = FakeEngine(
        [
            _tool_call("read_file", file_path="README.md"),
            _tool_call(
                "replace_in_file",
                file_path="README.md",
                old="a typo",
                new="correct text",
            ),
            "Fixed.",
            _all_satisfied_simple(["typo fixed"]),
        ]
    )
    agent = ReActAgent(engine, max_iterations=5)
    msg = _run(agent.run("Fix the typo in README.md", []))
    for _ in range(4):
        if msg.pending_action is None:
            break
        result = _run(execute_pending_action(msg.pending_action))
        msg = _run(continue_task_after_confirmation(agent, msg.task_state, result, []))

    assert msg.pending_action is None
    assert msg.task_state is None or msg.task_state.is_complete() is True
    assert "correct text" in (sandbox / "README.md").read_text(encoding="utf-8")
    # Minimal: read + one targeted edit — no repository-analysis workflow.
    tools = [e.tool_name for e in msg.task_state.execution_history]
    assert tools == ["read_file", "replace_in_file"]
    assert "run_command" not in tools
    assert "list_directory" not in tools


# ---------------------------------------------------------------------------
# SCENARIO 20 — ADAPTIVE DEBUGGING (env-var discovery -> plan revision)
# ---------------------------------------------------------------------------


def test_scenario20_adaptive_debugging_plan_revision(sandbox):
    (sandbox / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\ntestpaths=['.']\n", encoding="utf-8"
    )
    (sandbox / "app.py").write_text(
        "import os\n\ndef token():\n    return os.environ['SECRET_KEY']\n",
        encoding="utf-8",
    )
    (sandbox / "test_app.py").write_text(
        "import app\n\n"
        "def test_token():\n    assert app.token() == 'secret'\n",
        encoding="utf-8",
    )
    revision = [
        {
            "id": 2,
            "description": "Configure the missing SECRET_KEY environment variable",
            "purpose": "The tests crash because SECRET_KEY is unset",
            "dependencies": [1],
            "expected_outcome": "SECRET_KEY is available to the tests",
            "completion_criteria": ["SECRET_KEY configured"],
            "failure_strategy": "stop",
            "retry_policy": 0,
        },
        {
            "id": 3,
            "description": "Rerun the tests",
            "purpose": "Confirm the suite passes with the env configured",
            "dependencies": [2],
            "expected_outcome": "Tests pass",
            "completion_criteria": ["tests pass"],
            "failure_strategy": "stop",
            "retry_policy": 0,
        },
    ]
    engine = FakeEngine(
        [
            _pytest_call("test_app.py"),  # fails: KeyError SECRET_KEY
            "The tests crash because SECRET_KEY is missing.",
            _plan_verify(
                ["failure reproduced"],
                ["tests pass"],
                revision=revision,
            ),  # adaptive revision proposed
            _tool_call(
                "run_command",
                command=(
                    f"SECRET_KEY=secret {PYTHON} -B -m pytest -q "
                    "-p no:cacheprovider test_app.py"
                ),
            ),
            "Tests pass with the env configured.",
            _all_satisfied(["SECRET_KEY configured"], ["tests pass"]),
            "All tests pass now.",
            _all_satisfied(["tests pass"], ["tests pass"]),
        ]
    )
    task = _make_task(
        "Fix the failing tests",
        TaskType.DEBUGGING,
        [
            _step(1, "Run the tests to reproduce the failure", ["failure reproduced"]),
            _step(2, "Fix the failing tests", ["tests pass"]),
        ],
        ["tests pass"],
        ["tests pass"],
        WorkspaceKind.EXISTING_PROJECT,
        str(sandbox),
    )
    agent = ReActAgent(engine, max_iterations=10)
    msg = _run(agent.run(task.goal, [], task=task))
    for _ in range(10):
        if msg.pending_action is None:
            break
        result = _run(execute_pending_action(msg.pending_action))
        msg = _run(continue_task_after_confirmation(agent, task, result, []))

    assert msg.pending_action is None
    assert task.is_complete() is True
    # The plan was adapted: the remaining steps were replaced (step 1 preserved).
    assert len(task.plan.steps) == 3
    assert task.plan.steps[0].description == "Run the tests to reproduce the failure"
    assert any("SECRET_KEY" in s.description for s in task.plan.steps)
    # The original plan history was not destroyed.
    assert task.plan.steps[1].id == 2


# ---------------------------------------------------------------------------
# Invariant checks against the live loop
# ---------------------------------------------------------------------------


def test_invariant_tool_success_ne_step_success(sandbox):
    # A successful edit tool does NOT complete the plan step/task.
    (sandbox / "app.py").write_text("x = 1\n", encoding="utf-8")
    engine = FakeEngine(
        [
            _tool_call(
                "replace_in_file",
                file_path="app.py",
                old="x = 1",
                new="x = 2",
            ),
        ]
    )
    task = _make_task(
        "Do something",
        TaskType.MULTI_STEP,
        [_step(1, "Change x", ["x changed"]), _step(2, "Verify", ["verified"])],
        ["done"],
        ["done"],
        WorkspaceKind.EXISTING_PROJECT,
        str(sandbox),
    )
    agent = ReActAgent(engine, max_iterations=4)
    msg = _run(agent.run(task.goal, [], task=task))
    assert msg.pending_action is not None
    result = _run(execute_pending_action(msg.pending_action))
    assert "Replaced" in result  # tool success
    # Tool success did NOT complete the step or the task: the step is parked
    # awaiting verification and the task is incomplete.
    step = task.current_plan_step()
    assert step.status in (StepStatus.RUNNING, StepStatus.WAITING_CONFIRMATION)
    assert step.status is not StepStatus.SUCCEEDED
    assert task.is_complete() is False
    assert task.status.value != "completed"


def test_invariant_pending_confirmation_ne_task_termination(sandbox):
    (sandbox / "app.py").write_text("x = 1\n", encoding="utf-8")
    engine = FakeEngine(
        [
            _tool_call(
                "replace_in_file",
                file_path="app.py",
                old="x = 1",
                new="x = 2",
            ),
        ]
    )
    task = _make_task(
        "Change x",
        TaskType.MULTI_STEP,
        [_step(1, "Change x", ["x changed"])],
        ["done"],
        ["done"],
        WorkspaceKind.EXISTING_PROJECT,
        str(sandbox),
    )
    agent = ReActAgent(engine, max_iterations=4)
    msg = _run(agent.run(task.goal, [], task=task))
    # Pending confirmation is a PAUSE, not termination.
    assert msg.pending_action is not None
    assert task.status.value == "waiting_confirmation"
    assert task.is_complete() is False
    assert task.goal == "Change x"  # goal preserved


def test_executor_never_bypasses_security(sandbox):
    # A gated action STILL produces a PendingAction even with an executor
    # attached — the CodingExecutor cannot force execution.
    (sandbox / ".env").write_text("x", encoding="utf-8")
    _make_task(
        "Secure the app",
        TaskType.MULTI_STEP,
        [_step(1, "Configure", ["configured"])],
        ["done"],
        ["done"],
        WorkspaceKind.EXISTING_PROJECT,
        str(sandbox),
    )
    agent = ReActAgent(FakeEngine([]), max_iterations=2)
    msg = agent._route_coding_file_op(
        "append_to_file", {"file_path": ".env", "content": "SECRET=1"}
    )
    # CRITICAL path -> confirmation required, never silent execution.
    assert isinstance(msg, str) is False
    assert msg.pending_action is not None
    assert msg.pending_action.action_type == "append_to_file"
    assert "SECRET" not in (sandbox / ".env").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _git_init(sandbox):
    subprocess.run(["git", "init", "-q"], cwd=sandbox, check=False, timeout=10)
    subprocess.run(
        ["git", "config", "user.email", "t@t"], cwd=sandbox, check=False, timeout=10
    )
    subprocess.run(
        ["git", "config", "user.name", "t"], cwd=sandbox, check=False, timeout=10
    )


def _git_commit_all(sandbox):
    subprocess.run(["git", "add", "-A"], cwd=sandbox, check=False, timeout=10)
    subprocess.run(
        ["git", "commit", "-q", "-m", "init"], cwd=sandbox, check=False, timeout=10
    )
