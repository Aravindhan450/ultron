"""
Phase 3.3: Model-in-the-Loop Reliability Benchmark Contract Tests.

Validates the full benchmark matrix and contract behaviors:
1. Scenario definition and contract validation for all 6 core scenarios
2. Multi-file tracking and allowed file enforcement in MITLGrader
3. Scenario 7: Repair Budget Exhaustion
4. Scenario 8: Mid-execution Cancellation Handling
5. Scenario 9: Out-of-scope modification rejection
6. Scenario 10: Hard verification enforcement
7. Benchmark result matrix aggregation and taxonomy reporting
"""

from __future__ import annotations

import json
import sys
from typing import Any

import pytest

from tests.model_in_loop.conftest import make_mitl_sandbox
from tests.model_in_loop.harness.grader import (
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
# 1. Scenarios Contract Validation
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
def test_all_benchmark_scenarios_structure_and_sandbox(tmp_path, monkeypatch, scenario_cls):
    """Verifies that each scenario cleanly initializes its sandbox and files."""
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
# 3. Scenario 7: Repair Budget Exhaustion Contract
# ===========================================================================


def test_benchmark_scenario7_budget_exhaustion(tmp_path, monkeypatch):
    """
    Scenario 7 Contract: Exhausted repair budget leads to graceful termination
    and failed grading report with REPAIR_BUDGET_EXHAUSTED failure taxonomy.
    """
    scenario = CalculatorBugFixScenario()
    sandbox = make_mitl_sandbox(tmp_path, monkeypatch, scenario)

    # Simulate a trace where budget was exceeded
    class DummyTrace:
        duration_seconds = 12.5
        iterations = 10
        total_tokens = 2500
        error = "Repair budget exhausted: max attempts reached"
        final_response = "Unable to fix bug within allocated budget."

    trace = DummyTrace()
    report = MITLGrader.grade(scenario=scenario, sandbox=sandbox, trace=trace)

    assert not report.is_success
    assert report.failure_taxonomy in (
        FailureTaxonomy.REPAIR_BUDGET_EXHAUSTED,
        FailureTaxonomy.TESTS_FAILED,
    )


# ===========================================================================
# 4. Scenario 8: Cancellation Contract
# ===========================================================================


def test_benchmark_scenario8_cancellation_handling(tmp_path, monkeypatch):
    """
    Scenario 8 Contract: Mid-execution cancellation sets CANCELLED taxonomy
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

    report = MITLGrader.grade(scenario=scenario, sandbox=sandbox, trace=CancelledTrace())
    assert not report.is_success
    assert report.failure_taxonomy == FailureTaxonomy.CANCELLED


# ===========================================================================
# 5. Scenario 9: Out-of-Scope Modification Rejection
# ===========================================================================


def test_benchmark_scenario9_scope_rejection(tmp_path, monkeypatch):
    """
    Scenario 9 Contract: Modifying files outside the scenario scope
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

    report = MITLGrader.grade(scenario=scenario, sandbox=sandbox, trace=None)
    assert not report.no_unrelated_changes
    assert not report.is_success
    assert report.failure_taxonomy == FailureTaxonomy.OUT_OF_SCOPE_MODIFICATION


# ===========================================================================
# 6. Scenario 10: Hard Verification Enforcement
# ===========================================================================


def test_benchmark_scenario10_verification_enforcement(tmp_path, monkeypatch):
    """
    Scenario 10 Contract: If tests are not executed or pass, verification_executed
    is False, grading is rejected, and failure taxonomy is VERIFICATION_FAILED.
    """
    report = GradingReport(
        scenario_name="verification_enforcement_test",
        correct_file_modified=True,
        expected_implementation=True,
        tests_pass=True,
        no_unrelated_changes=True,
        non_empty_diff=True,
        verification_executed=False,  # <--- Agent didn't run verification
        budget_respected=True,
    )
    assert not report.is_success
    assert report.failure_taxonomy == FailureTaxonomy.VERIFICATION_FAILED


# ===========================================================================
# 7. Benchmark Matrix Aggregator
# ===========================================================================


def test_benchmark_matrix_aggregation():
    """Verifies that BenchmarkResult computes pass rates and aggregates reports correctly."""
    r1 = GradingReport(
        scenario_name="scen_1",
        correct_file_modified=True,
        expected_implementation=True,
        tests_pass=True,
        no_unrelated_changes=True,
        non_empty_diff=True,
        verification_executed=True,
        budget_respected=True,
    )
    r2 = GradingReport(
        scenario_name="scen_2",
        correct_file_modified=True,
        expected_implementation=False,
        tests_pass=False,
        no_unrelated_changes=True,
        non_empty_diff=True,
        verification_executed=True,
        budget_respected=True,
    )

    benchmark = BenchmarkMatrix(total_scenarios=2, reports=[r1, r2])
    assert benchmark.passed_scenarios == 1
    assert benchmark.failed_scenarios == 1
    assert benchmark.pass_rate == 0.5
    summary = benchmark.summary()
    assert "Passed: 1/2 (50.0%)" in summary
    assert "scen_1: PASS" in summary
    assert "scen_2: FAIL" in summary
