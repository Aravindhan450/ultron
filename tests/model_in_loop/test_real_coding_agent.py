"""
Model-in-the-Loop test: Real autonomous LLM coding agent validation across benchmark matrix.
Exercises real LlamaCppEngine + real ReActAgent + real CodingExecutor on isolated sandboxes.
"""

from __future__ import annotations

import asyncio

import pytest

from tests.model_in_loop.conftest import make_mitl_sandbox
from tests.model_in_loop.harness.grader import MITLGrader
from tests.model_in_loop.harness.runner import MITLRunner
from tests.model_in_loop.harness.sandbox import MITLSandbox
from tests.model_in_loop.scenarios import (
    CalculatorBugFixScenario,
    ConfigServiceScenario,
    DurationFormatterScenario,
    PriceCalculatorScenario,
    SlugifyRepairScenario,
    SyntaxImportRecoveryScenario,
)


@pytest.mark.mitl
def test_real_model_fixes_calculator_bug(mitl_server, mitl_sandbox: MITLSandbox):
    """
    Scenario 1: Simple Bug Fix
    Validates that the real Ultron LLM agent autonomously:
    1. Inspects the repository
    2. Identifies the defect in calculator.py
    3. Edits the file to fix the bug
    4. Runs/verifies the tests
    5. Completes within resource budgets
    """
    scenario = CalculatorBugFixScenario()
    runner = MITLRunner(engine=mitl_server, max_iterations=15, timeout=180.0)

    trace = asyncio.run(runner.run(scenario=scenario, sandbox=mitl_sandbox))
    report = MITLGrader.grade(scenario=scenario, sandbox=mitl_sandbox, trace=trace)

    print("\n" + report.summary())
    print("\nExecution Trace:\n" + trace.summary())

    assert report.is_success, (
        f"MITL validation failed for {scenario.name}:\n"
        f"{report.summary()}\n\n"
        f"Execution Trace:\n{trace.summary()}\n"
        f"Final Model Response:\n{trace.final_response}"
    )


@pytest.mark.mitl
def test_real_model_repair_slugify_bug(mitl_server, tmp_path, monkeypatch):
    """
    Scenario 2: Autonomous Repair
    Validates that the real Ultron LLM agent autonomously:
    1. Inspects the string-util repository
    2. Diagnoses the failure across multiple test cases in test_string_util.py
    3. Repairs string_util.py (lowercasing, punctuation replacement, stripping)
    4. Retests via pytest and verifies all cases pass
    5. Terminates within bounded budget with complete objective verification
    """
    scenario = SlugifyRepairScenario()
    sandbox = make_mitl_sandbox(tmp_path, monkeypatch, scenario)
    runner = MITLRunner(engine=mitl_server, max_iterations=15, timeout=180.0)

    trace = asyncio.run(runner.run(scenario=scenario, sandbox=sandbox))
    report = MITLGrader.grade(scenario=scenario, sandbox=sandbox, trace=trace)

    print("\n" + report.summary())
    print("\nExecution Trace:\n" + trace.summary())

    assert report.is_success, (
        f"MITL repair validation failed for {scenario.name}:\n"
        f"{report.summary()}\n\n"
        f"Execution Trace:\n{trace.summary()}\n"
        f"Final Model Response:\n{trace.final_response}"
    )


@pytest.mark.mitl
def test_real_model_multi_case_duration(mitl_server, tmp_path, monkeypatch):
    """
    Scenario 3: Multi-Case Bug
    Validates handling multiple distinct input and duration division cases.
    """
    scenario = DurationFormatterScenario()
    sandbox = make_mitl_sandbox(tmp_path, monkeypatch, scenario)
    runner = MITLRunner(engine=mitl_server, max_iterations=15, timeout=180.0)

    trace = asyncio.run(runner.run(scenario=scenario, sandbox=sandbox))
    report = MITLGrader.grade(scenario=scenario, sandbox=sandbox, trace=trace)

    print("\n" + report.summary())
    print("\nExecution Trace:\n" + trace.summary())

    assert report.is_success, (
        f"MITL multi-case validation failed for {scenario.name}:\n"
        f"{report.summary()}\n\n"
        f"Execution Trace:\n{trace.summary()}\n"
        f"Final Model Response:\n{trace.final_response}"
    )


@pytest.mark.mitl
def test_real_model_multi_file_config(mitl_server, tmp_path, monkeypatch):
    """
    Scenario 4: Multi-File Change
    Validates coordinated changes across config.py and client.py.
    """
    scenario = ConfigServiceScenario()
    sandbox = make_mitl_sandbox(tmp_path, monkeypatch, scenario)
    runner = MITLRunner(engine=mitl_server, max_iterations=15, timeout=180.0)

    trace = asyncio.run(runner.run(scenario=scenario, sandbox=sandbox))
    report = MITLGrader.grade(scenario=scenario, sandbox=sandbox, trace=trace)

    print("\n" + report.summary())
    print("\nExecution Trace:\n" + trace.summary())

    assert report.is_success, (
        f"MITL multi-file validation failed for {scenario.name}:\n"
        f"{report.summary()}\n\n"
        f"Execution Trace:\n{trace.summary()}\n"
        f"Final Model Response:\n{trace.final_response}"
    )


@pytest.mark.mitl
def test_real_model_syntax_import_recovery(mitl_server, tmp_path, monkeypatch):
    """
    Scenario 5: Syntax / Import Recovery
    Validates diagnosing and recovering from broken import and syntax error.
    """
    scenario = SyntaxImportRecoveryScenario()
    sandbox = make_mitl_sandbox(tmp_path, monkeypatch, scenario)
    runner = MITLRunner(engine=mitl_server, max_iterations=15, timeout=180.0)

    trace = asyncio.run(runner.run(scenario=scenario, sandbox=sandbox))
    report = MITLGrader.grade(scenario=scenario, sandbox=sandbox, trace=trace)

    print("\n" + report.summary())
    print("\nExecution Trace:\n" + trace.summary())

    assert report.is_success, (
        f"MITL syntax/import validation failed for {scenario.name}:\n"
        f"{report.summary()}\n\n"
        f"Execution Trace:\n{trace.summary()}\n"
        f"Final Model Response:\n{trace.final_response}"
    )


@pytest.mark.mitl
def test_real_model_regression_prevention(mitl_server, tmp_path, monkeypatch):
    """
    Scenario 6: Regression Prevention
    Validates adding new coupon rules while preserving existing volume discount behavior.
    """
    scenario = PriceCalculatorScenario()
    sandbox = make_mitl_sandbox(tmp_path, monkeypatch, scenario)
    runner = MITLRunner(engine=mitl_server, max_iterations=15, timeout=180.0)

    trace = asyncio.run(runner.run(scenario=scenario, sandbox=sandbox))
    report = MITLGrader.grade(scenario=scenario, sandbox=sandbox, trace=trace)

    print("\n" + report.summary())
    print("\nExecution Trace:\n" + trace.summary())

    assert report.is_success, (
        f"MITL regression validation failed for {scenario.name}:\n"
        f"{report.summary()}\n\n"
        f"Execution Trace:\n{trace.summary()}\n"
        f"Final Model Response:\n{trace.final_response}"
    )
