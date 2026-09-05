"""
Phase 3.3: Model-in-the-Loop Reliability Benchmark Validation Suite.

Explicitly distinguishes:
- 6 Real MITL Coding Scenarios (Model-in-the-loop autonomous coding with real engine)
- 4 Deterministic Safety / Contract Scenarios (Proving invariants deterministically)
- 10 Total Validation Scenarios
"""

from __future__ import annotations

import json
import sys
from typing import Any

import pytest

from tests.model_in_loop.conftest import make_mitl_sandbox
from tests.model_in_loop.harness.grader import (
    DETERMINISTIC_CONTRACT_SCENARIO_COUNT,
    REAL_MITL_SCENARIO_COUNT,
    TOTAL_VALIDATION_SCENARIO_COUNT,
    BenchmarkMatrix,
    FailureTaxonomy,
    GradingReport,
    MITLGrader,
)
from tests.model_in_loop.scenarios import (
    CalculatorBugFixScenario,
    ConfigServiceScenario,
    DurationFormatterScenario,
    PriceCalculatorScenario,
    SlugifyRepairScenario,
    SyntaxImportRecoveryScenario,
)

PYTHON = sys.executable


class ScriptedEngine:
    """Deterministic scripted engine returning predefined responses."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[Any] = []

    async def generate(self, messages, **kwargs) -> str:
        self.calls.append(messages)
        return self._responses.pop(0) if self._responses else ""

    async def stream(self, messages, **kwargs):
        yield ""


def _tool_call(tool: str, **arguments) -> str:
    return f"```json\n{json.dumps({'tool': tool, 'arguments': arguments})}\n```"


# ===========================================================================
# 1. Real MITL Scenarios Contract Validation (Scenarios 1 - 6)
# ===========================================================================


@pytest.mark.parametrize(
    "scenario_cls",
    [
        CalculatorBugFixScenario,
        SlugifyRepairScenario,
        DurationFormatterScenario,
        ConfigServiceScenario,
        SyntaxImportRecoveryScenario,
        PriceCalculatorScenario,
    ],
)
def test_real_mitl_scenarios_structure_and_sandbox(tmp_path, monkeypatch, scenario_cls):
    """Verifies that each of the 6 real MITL scenarios cleanly initializes its sandbox and files."""
    scenario = scenario_cls()
    sandbox = make_mitl_sandbox(tmp_path, monkeypatch, scenario)

    for filename in scenario.initial_files:
        assert sandbox.file_exists(filename), f"Missing initial file: {filename}"

    # Verify target file(s) exist
    if hasattr(scenario, "target_files"):
        for tf in scenario.target_files:
            assert sandbox.file_exists(tf)
    else:
        assert sandbox.file_exists(scenario.target_file)

    # Initial implementation should fail validation
    valid, _ = scenario.validate_implementation(sandbox)
    assert not valid, f"Initial implementation for {scenario.name} should fail validation before fix"


# ===========================================================================
# 2. Multi-File Tracking & Grading Enforcement
# ===========================================================================


def test_grader_multi_file_tracking(tmp_path, monkeypatch):
    """Verifies MITLGrader evaluates multi-file target scenarios correctly."""
    scenario = ConfigServiceScenario()
    sandbox = make_mitl_sandbox(tmp_path, monkeypatch, scenario)

    # Modify only config.py
    sandbox.write_file(
        "config.py",
        "from dataclasses import dataclass\n@dataclass\nclass ServerConfig:\n    host: str\n    port: int = 8080\n    timeout_seconds: int = 30\n",
    )

    report = MITLGrader.grade(scenario=scenario, sandbox=sandbox, trace=None)
    assert not report.correct_file_modified, "Expected failure when client.py was not modified"
    assert not report.is_success

    # Modify both config.py and client.py
    sandbox.write_file(
        "client.py",
        "from config import ServerConfig\nclass APIClient:\n    def __init__(self, config: ServerConfig) -> None:\n        self.config = config\n        self.base_url = f\"http://{config.host}:{config.port}/api/v1\"\n        self.timeout = config.timeout_seconds\n    def get_endpoint(self, path: str) -> str:\n        return f\"{self.base_url}/{path.lstrip('/')}\"\n",
    )
    valid, err = scenario.validate_implementation(sandbox)
    assert valid, f"Validation failed: {err}"


# ===========================================================================
# 3. Deterministic Contract Scenario 7: Repair Budget Exhaustion
# ===========================================================================


def test_deterministic_contract_budget_exhaustion(tmp_path, monkeypatch):
    """
    Deterministic Contract Scenario 7: Exhausted repair budget leads to graceful termination
    and failed grading report with REPAIR_BUDGET_EXHAUSTED failure taxonomy.
    """
    scenario = CalculatorBugFixScenario()
    sandbox = make_mitl_sandbox(tmp_path, monkeypatch, scenario)

    class DummyTrace:
        duration_seconds = 12.5
        iterations = 10
        total_tokens = 2500
        error = "Repair budget exhausted: max attempts reached"
        final_response = "Unable to fix bug within allocated budget."

    trace = DummyTrace()
    report = MITLGrader.grade(
        scenario=scenario,
        sandbox=sandbox,
        trace=trace,
        scenario_type="contract",
    )

    assert not report.is_success
    assert report.failure_taxonomy in (
        FailureTaxonomy.REPAIR_BUDGET_EXHAUSTED,
        FailureTaxonomy.TESTS_FAILED,
    )


# ===========================================================================
# 4. Deterministic Contract Scenario 8: Cancellation Handling
# ===========================================================================


def test_deterministic_contract_cancellation_handling(tmp_path, monkeypatch):
    """
    Deterministic Contract Scenario 8: Mid-execution cancellation sets CANCELLED taxonomy
    and produces a clean report without hanging or leaking resources.
    """
    scenario = CalculatorBugFixScenario()
    sandbox = make_mitl_sandbox(tmp_path, monkeypatch, scenario)

    class CancelledTrace:
        duration_seconds = 2.1
        iterations = 2
        total_tokens = 400
        error = "Task cancelled by user"
        final_response = "Operation cancelled."

    report = MITLGrader.grade(
        scenario=scenario,
        sandbox=sandbox,
        trace=CancelledTrace(),
        scenario_type="contract",
    )
    assert not report.is_success
    assert report.failure_taxonomy == FailureTaxonomy.CANCELLED


# ===========================================================================
# 5. Deterministic Contract Scenario 9: Out-of-Scope Modification Rejection
# ===========================================================================


def test_deterministic_contract_scope_rejection(tmp_path, monkeypatch):
    """
    Deterministic Contract Scenario 9: Modifying files outside the scenario scope
    is flagged by no_unrelated_changes and assigns OUT_OF_SCOPE_MODIFICATION taxonomy.
    """
    scenario = CalculatorBugFixScenario()
    sandbox = make_mitl_sandbox(tmp_path, monkeypatch, scenario)

    # Correct fix in calculator.py
    sandbox.write_file(
        "calculator.py",
        "def add(a, b):\n    return a + b\n\ndef multiply(a, b):\n    return a * b\n",
    )
    # Unrelated modification
    sandbox.write_file("unrelated_module.py", "# Malicious or out of scope change\nx = 1\n")

    report = MITLGrader.grade(
        scenario=scenario,
        sandbox=sandbox,
        trace=None,
        scenario_type="contract",
    )
    assert not report.no_unrelated_changes
    assert not report.is_success
    assert report.failure_taxonomy == FailureTaxonomy.OUT_OF_SCOPE_MODIFICATION


# ===========================================================================
# 6. Deterministic Contract Scenario 10: Hard Verification Enforcement
# ===========================================================================


def test_deterministic_contract_verification_enforcement():
    """
    Deterministic Contract Scenario 10: If tests are not executed, verification_executed
    is False, grading is rejected, and failure taxonomy is VERIFICATION_FAILED.
    """
    report = GradingReport(
        scenario_name="verification_enforcement_test",
        scenario_type="contract",
        correct_file_modified=True,
        expected_implementation=True,
        tests_pass=True,
        no_unrelated_changes=True,
        non_empty_diff=True,
        verification_executed=False,  # <--- Agent omitted verification
        budget_respected=True,
    )
    assert not report.is_success
    assert report.failure_taxonomy == FailureTaxonomy.VERIFICATION_FAILED


# ===========================================================================
# 7. Benchmark Matrix Aggregator & Separation Validation
# ===========================================================================


def test_benchmark_matrix_aggregation_and_separation():
    """
    Verifies that BenchmarkMatrix strictly distinguishes:
    - Real MITL Scenarios (6)
    - Deterministic Contract Scenarios (4)
    - Total Validation Scenarios (10)
    """
    assert REAL_MITL_SCENARIO_COUNT == 6
    assert DETERMINISTIC_CONTRACT_SCENARIO_COUNT == 4
    assert TOTAL_VALIDATION_SCENARIO_COUNT == 10

    # 6 Real MITL reports (5 pass, 1 fail)
    real_reports = [
        GradingReport(
            scenario_name=f"real_scen_{i}",
            scenario_type="real",
            correct_file_modified=True,
            expected_implementation=True,
            tests_pass=(i != 6),
            no_unrelated_changes=True,
            non_empty_diff=True,
            verification_executed=True,
            budget_respected=True,
        )
        for i in range(1, 7)
    ]

    # 4 Deterministic Contract reports (all 4 pass)
    contract_reports = [
        GradingReport(
            scenario_name=name,
            scenario_type="contract",
            correct_file_modified=True,
            expected_implementation=True,
            tests_pass=True,
            no_unrelated_changes=True,
            non_empty_diff=True,
            verification_executed=True,
            budget_respected=True,
        )
        for name in [
            "Budget Exhaustion",
            "Cancellation",
            "Scope Enforcement",
            "Verification Gate",
        ]
    ]

    matrix = BenchmarkMatrix(
        real_mitl_reports=real_reports,
        contract_reports=contract_reports,
    )

    assert matrix.real_mitl_passed == 5
    assert matrix.real_mitl_total == 6
    assert matrix.contract_passed == 4
    assert matrix.contract_total == 4
    assert matrix.total_validation_passed == 9
    assert matrix.total_validation_count == 10

    summary = matrix.summary()
    assert "Real MITL Coding Scenarios" in summary
    assert "Real MITL Result: 5/6" in summary
    assert "Deterministic Contract Scenarios" in summary
    assert "Contract Result: 4/4" in summary
    assert "Total Validation: 9/10" in summary
