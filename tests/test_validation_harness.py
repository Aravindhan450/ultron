"""Deterministic tests for the hardened Ultron Agent Validation Harness.

Tests:
1. Workspace confinement (pass vs escape failure)
2. Dependency ordering (pass vs premature consumption failure)
3. Failure thrashing (repeated identical failure vs meaningful retry)
4. Invalid dependency declaration (stdlib modules vs valid external)
5. State progress analysis (progress vs no progress vs regression)
6. Execution success vs Objective success separation
7. Verification evidence (missing verification on completion claim vs verified evidence)
8. Expense manager regression scenario structure & validation
9. Anti-cheating / deliberate harness break tests
"""

from __future__ import annotations

from ultron.core.tools.definitions import ToolCapability
from ultron.validation.evaluate import evaluate_trace
from ultron.validation.invariants import (
    check_dependency_ordering,
    check_failure_thrashing,
    check_invalid_dependency_declarations,
    check_workspace_confinement,
)
from ultron.validation.model import (
    AcceptanceCriterion,
    CapabilityTestCase,
    EnvironmentSnapshot,
    FailureKind,
    TaskTrace,
    Verdict,
    VerificationEvidence,
)
from ultron.validation.probes import (
    WorkspaceProbe,
)
from ultron.validation.progress import ProgressKind, StateProgressAnalyzer
from ultron.validation.report import build_real_world_task_report
from ultron.validation.scenarios import build_expense_manager_scenario
from ultron.validation.validators import (
    validate_verification_evidence_criterion,
)

# ===========================================================================
# 1. WORKSPACE CONFINEMENT PROBE & INVARIANT (INV-001)
# ===========================================================================


def test_workspace_confinement_pass(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    probe = WorkspaceProbe(workspace_root=ws)
    
    file1 = ws / "app.py"
    file2 = ws / "src" / "util.py"
    
    is_confined, escaped = probe.check_confinement([file1, file2])
    assert is_confined is True
    assert escaped == []

    env = EnvironmentSnapshot(
        workspace_root=str(ws),
        files_created=[str(file1), str(file2)],
    )
    res = check_workspace_confinement(env, workspace_root=ws)
    assert res.passed is True
    assert res.invariant_id == "INV-001"


def test_workspace_confinement_fail_on_escape(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    outside = tmp_path / "outside.py"
    
    probe = WorkspaceProbe(workspace_root=ws)
    is_confined, escaped = probe.check_confinement([ws / "app.py", outside])
    assert is_confined is False
    assert str(outside) in escaped

    env = EnvironmentSnapshot(
        workspace_root=str(ws),
        files_created=[str(ws / "app.py"), str(outside)],
    )
    res = check_workspace_confinement(env, workspace_root=ws)
    assert res.passed is False
    assert "outside" in res.failure_reason


# ===========================================================================
# 2. DEPENDENCY ORDERING (INV-002)
# ===========================================================================


def test_dependency_ordering_pass(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    req_file = ws / "requirements.txt"
    req_file.write_text("requests\n", encoding="utf-8")

    history = [
        {"tool": "create_file", "target": str(req_file), "exit_code": 0, "success": True},
        {"tool": "run_command", "command": "pip install -r requirements.txt", "cwd": str(ws), "exit_code": 0, "success": True},
    ]

    res = check_dependency_ordering(history, workspace_root=ws)
    assert res.passed is True


def test_dependency_ordering_fail_missing_target(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    # requirements.txt does not exist

    history = [
        {
            "tool": "run_command",
            "command": "pip install -r requirements.txt",
            "cwd": str(ws),
            "exit_code": 1,
            "output_snippet": "ERROR: Could not open requirements file: [Errno 2] No such file or directory: 'requirements.txt'",
        }
    ]

    res = check_dependency_ordering(history, workspace_root=ws)
    assert res.passed is False
    assert "executed at step 1 before required target" in res.failure_reason


# ===========================================================================
# 3. FAILURE THRASHING (INV-003)
# ===========================================================================


def test_failure_thrashing_detected():
    history = [
        {"tool": "run_command", "command": "pip install -r requirements.txt", "exit_code": 1, "success": False},
        {"tool": "run_command", "command": "pip install -r requirements.txt", "exit_code": 1, "success": False},
    ]
    res = check_failure_thrashing(history, max_identical_retries=2)
    assert res.passed is False
    assert "Repeated identical failure thrashing detected" in res.failure_reason


def test_failure_thrashing_not_triggered_on_meaningful_retry():
    history = [
        {"tool": "run_command", "command": "pip install -r requirements.txt", "exit_code": 1, "success": False},
        {"tool": "create_file", "target": "requirements.txt", "exit_code": 0, "success": True},
        {"tool": "run_command", "command": "pip install -r requirements.txt", "exit_code": 0, "success": True},
    ]
    res = check_failure_thrashing(history, max_identical_retries=2)
    assert res.passed is True


# ===========================================================================
# 4. INVALID DEPENDENCY DECLARATION (INV-004)
# ===========================================================================


def test_invalid_python_stdlib_dependency_detected(tmp_path):
    req_file = tmp_path / "requirements.txt"
    req_file.write_text("tkinter\npandas\nsqlite3\n", encoding="utf-8")

    res = check_invalid_dependency_declarations(files=[req_file])
    assert res.passed is False
    assert "tkinter" in res.failure_reason
    assert "sqlite3" in res.failure_reason


def test_valid_external_dependencies_pass(tmp_path):
    req_file = tmp_path / "requirements.txt"
    req_file.write_text("pandas>=2.0\nfastapi\nuvicorn\n", encoding="utf-8")

    res = check_invalid_dependency_declarations(files=[req_file])
    assert res.passed is True


# ===========================================================================
# 5. STATE PROGRESS ANALYSIS
# ===========================================================================


def test_state_progress_analysis_progress():
    s_before = {"files": ["main.py"]}
    s_after = {"files": ["main.py", "requirements.txt"]}
    trans = StateProgressAnalyzer.analyze_step(1, "create_file", "requirements.txt", s_before, s_after, success=True)
    assert trans.kind == ProgressKind.PROGRESS
    assert "requirements.txt" in trans.delta["new_files"]


def test_state_progress_analysis_no_progress():
    s_before = {"files": ["main.py"], "last_error": "No such file"}
    s_after = {"files": ["main.py"], "last_error": "No such file"}
    trans = StateProgressAnalyzer.analyze_step(2, "run_command", "pip install", s_before, s_after, success=False, error="No such file")
    assert trans.kind == ProgressKind.NO_PROGRESS


def test_state_progress_analysis_regression():
    s_before = {"files": ["main.py", "database.py"]}
    s_after = {"files": ["main.py"]}
    trans = StateProgressAnalyzer.analyze_step(3, "run_command", "cleanup", s_before, s_after, success=True)
    assert trans.kind == ProgressKind.REGRESSION


# ===========================================================================
# 6. EXECUTION SUCCESS VS OBJECTIVE SUCCESS SEPARATION
# ===========================================================================


def test_tool_execution_succeeded_but_objective_failed(tmp_path):
    # Scenario: Tool executed (create_file), but requirements.txt has invalid stdlib packages
    # and acceptance criteria fails.
    req = tmp_path / "requirements.txt"
    req.write_text("tkinter\n", encoding="utf-8")

    case = CapabilityTestCase(
        case_id="TEST_CASE_OBJ_FAIL",
        capability=ToolCapability.BUILD,
        expected_capability=ToolCapability.BUILD,
        task="Build an application",
        subject="App",
        acceptance_criteria=[
            AcceptanceCriterion(
                criterion_id="AC-03",
                description="Valid dependencies",
                validator_type="dependency_validity",
                required=True,
            )
        ],
    )

    env = EnvironmentSnapshot(
        workspace_root=str(tmp_path),
        files_created=[str(req)],
        commands_executed=[{"tool": "create_file", "target": str(req), "exit_code": 0, "success": True}],
    )

    trace = TaskTrace(
        case=case,
        transcript="Created file requirements.txt successfully. Done.",
        latency_s=1.0,
        detected_tool_hint="create_file",
        environment=env,
    )

    eval_res = evaluate_trace(trace, repo_root=tmp_path)

    # Invariants should fail on INV-004
    assert eval_res.invariants_verdict == Verdict.FAIL
    # Objective should fail
    assert eval_res.objective_verdict == Verdict.FAIL
    # Overall must be FAIL
    assert eval_res.overall == Verdict.FAIL
    assert eval_res.failure_kind == FailureKind.INVALID_DEPENDENCY_FAILURE


def test_tool_execution_succeeded_and_objective_succeeded(tmp_path):
    app = tmp_path / "app.py"
    app.write_text("import sqlite3\nimport csv\nprint('hello')\n", encoding="utf-8")
    req = tmp_path / "requirements.txt"
    req.write_text("pandas\n", encoding="utf-8")

    case = CapabilityTestCase(
        case_id="TEST_CASE_OBJ_PASS",
        capability=ToolCapability.BUILD,
        expected_capability=ToolCapability.BUILD,
        task="Build an expense app",
        subject="Expense",
        acceptance_criteria=[
            AcceptanceCriterion(
                criterion_id="AC-01",
                description="Workspace confinement",
                validator_type="workspace_confinement",
                required=True,
            ),
            AcceptanceCriterion(
                criterion_id="AC-02",
                description="Artifacts exist",
                validator_type="artifact_exists",
                required=True,
                details={"filenames": ["app.py", "requirements.txt"]},
            ),
            AcceptanceCriterion(
                criterion_id="AC-03",
                description="Valid dependencies",
                validator_type="dependency_validity",
                required=True,
            ),
        ],
    )


    env = EnvironmentSnapshot(
        workspace_root=str(tmp_path),
        files_created=[str(app), str(req)],
        commands_executed=[{"tool": "create_file", "target": str(app), "exit_code": 0, "success": True}],
    )

    trace = TaskTrace(
        case=case,
        transcript="Created app.py and requirements.txt. Ran pytest test suite with verified results.",
        latency_s=1.0,
        detected_tool_hint="create_file",
        environment=env,
        verification_evidence=VerificationEvidence(
            model_claimed_success=True,
            independent_evidence_produced=True,
        ),
    )

    eval_res = evaluate_trace(trace, repo_root=tmp_path)
    assert eval_res.invariants_verdict == Verdict.PASS
    assert eval_res.objective_verdict == Verdict.PASS


# ===========================================================================
# 7. VERIFICATION EVIDENCE (MISSING VS VERIFIED)
# ===========================================================================


def test_verification_evidence_missing_declared_completion():
    crit = AcceptanceCriterion(
        criterion_id="AC-09",
        description="Verification evidence required",
        validator_type="verification_evidence",
    )
    # Model claims completed but has zero independent verification
    res = validate_verification_evidence_criterion(
        crit,
        evidence=VerificationEvidence(model_claimed_success=True, independent_evidence_produced=False),
        transcript="I have finished building the app. The task is completed successfully.",
    )
    assert res.status == Verdict.FAIL
    assert "without independent verification evidence" in res.error


def test_verification_evidence_valid_independent_tests():
    crit = AcceptanceCriterion(
        criterion_id="AC-09",
        description="Verification evidence required",
        validator_type="verification_evidence",
    )
    res = validate_verification_evidence_criterion(
        crit,
        evidence=VerificationEvidence(model_claimed_success=True, independent_evidence_produced=True),
        transcript="Ran pytest: 5 passed in 0.2s. Verified persistence and CSV export.",
    )
    assert res.status == Verdict.PASS


# ===========================================================================
# 8. EXPENSE MANAGER REGRESSION SCENARIO DEFINITION & REPORT
# ===========================================================================


def test_expense_manager_scenario_definition():
    scenario = build_expense_manager_scenario()
    assert scenario.case_id == "EXPENSE_MANAGER_REAL_WORLD_01"
    assert len(scenario.acceptance_criteria) == 9
    crit_ids = [c.criterion_id for c in scenario.acceptance_criteria]
    assert "AC-01" in crit_ids
    assert "AC-09" in crit_ids


def test_real_world_task_report_formatting(tmp_path):
    scenario = build_expense_manager_scenario()
    trace = TaskTrace(
        case=scenario,
        transcript="Execution transcript...",
        latency_s=2.5,
    )
    evaluation = evaluate_trace(trace, repo_root=tmp_path)
    report = build_real_world_task_report(trace, evaluation)

    assert "# ULTRON REAL-WORLD TASK VALIDATION" in report
    assert "Workspace confinement:" in report
    assert "Dependency ordering:" in report
    assert "Failure thrashing:" in report
    assert "Dependency validity:" in report
    assert "Objective achieved:" in report
