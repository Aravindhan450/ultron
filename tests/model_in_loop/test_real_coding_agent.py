"""
Model-in-the-Loop test: Real autonomous LLM coding agent validation.
Exercises real LlamaCppEngine + real ReActAgent + real CodingExecutor on an isolated sandbox.
"""

from __future__ import annotations

import asyncio

import pytest

from tests.model_in_loop.harness.grader import MITLGrader
from tests.model_in_loop.harness.runner import MITLRunner
from tests.model_in_loop.harness.sandbox import MITLSandbox
from tests.model_in_loop.scenarios.bug_fix import CalculatorBugFixScenario


@pytest.mark.mitl
def test_real_model_fixes_calculator_bug(mitl_server, mitl_sandbox: MITLSandbox):
    """
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
