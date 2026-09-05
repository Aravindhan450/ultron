"""
MITLGrader: Independent objective grader for Model-in-the-Loop coding tasks.
Never trusts LLM claims; inspects disk state, diffs, and independent test execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tests.model_in_loop.harness.runner import ExecutionTrace
from tests.model_in_loop.harness.sandbox import MITLSandbox


@dataclass
class GradingReport:
    scenario_name: str
    correct_file_modified: bool = False
    expected_implementation: bool = False
    tests_pass: bool = False
    no_unrelated_changes: bool = False
    non_empty_diff: bool = False
    verification_executed: bool = False
    budget_respected: bool = False
    pytest_exit_code: int = 1
    pytest_output: str = ""
    diff: str = ""
    failure_reasons: list[str] = field(default_factory=list)

    @property
    def is_success(self) -> bool:
        return (
            self.correct_file_modified
            and self.expected_implementation
            and self.tests_pass
            and self.no_unrelated_changes
            and self.non_empty_diff
            and self.verification_executed
            and self.budget_respected
        )

    def summary(self) -> str:
        status = "PASS" if self.is_success else "FAIL"
        lines = [
            f"=== MITL GRADING REPORT: {self.scenario_name} [{status}] ===",
            f"  1. Correct file modified: {'PASS' if self.correct_file_modified else 'FAIL'}",
            f"  2. Expected implementation: {'PASS' if self.expected_implementation else 'FAIL'}",
            f"  3. Independent tests pass: {'PASS' if self.tests_pass else 'FAIL'}",
            f"  4. No unrelated changes: {'PASS' if self.no_unrelated_changes else 'FAIL'}",
            f"  5. Non-empty diff: {'PASS' if self.non_empty_diff else 'FAIL'}",
            f"  6. Verification executed: {'PASS' if self.verification_executed else 'FAIL'}",
            f"  7. Budget respected: {'PASS' if self.budget_respected else 'FAIL'}",
        ]
        if self.failure_reasons:
            lines.append("Failures:")
            for reason in self.failure_reasons:
                lines.append(f"  - {reason}")
        if self.diff:
            lines.append(f"Repository Diff:\n{self.diff}")
        return "\n".join(lines)


class MITLGrader:
    """
    Grades an autonomous coding task objectively by evaluating the final sandbox state.
    """

    @classmethod
    def grade(
        cls,
        scenario: Any,
        sandbox: MITLSandbox,
        trace: ExecutionTrace,
        max_duration: float = 180.0,
        max_iterations: int = 15,
    ) -> GradingReport:
        report = GradingReport(scenario_name=getattr(scenario, "name", "mitl_scenario"))

        target_file = getattr(scenario, "target_file", "calculator.py")
        modified_files = sandbox.get_modified_files()
        report.diff = sandbox.get_diff()

        # 1. Correct file modified
        if target_file in modified_files or any(f.endswith(target_file) for f in modified_files):
            report.correct_file_modified = True
        else:
            report.failure_reasons.append(
                f"Target file '{target_file}' was not modified. Modified files: {modified_files}"
            )

        # 2. Expected implementation
        if hasattr(scenario, "validate_implementation") and callable(scenario.validate_implementation):
            ok, reason = scenario.validate_implementation(sandbox)
            if ok:
                report.expected_implementation = True
            else:
                report.failure_reasons.append(
                    reason or f"Scenario implementation validation failed for '{target_file}'."
                )
        elif sandbox.file_exists(target_file):
            content = sandbox.read_file(target_file)
            expected_patterns = getattr(scenario, "expected_patterns", None)
            forbidden_patterns = getattr(scenario, "forbidden_patterns", None)
            if expected_patterns or forbidden_patterns:
                valid = True
                for pat in (expected_patterns or []):
                    if pat not in content:
                        report.failure_reasons.append(
                            f"Implementation in '{target_file}' missing expected pattern '{pat}'."
                        )
                        valid = False
                for pat in (forbidden_patterns or []):
                    if pat in content:
                        report.failure_reasons.append(
                            f"Implementation in '{target_file}' still contains forbidden pattern '{pat}'."
                        )
                        valid = False
                report.expected_implementation = valid
            else:
                report.expected_implementation = bool(content.strip())
        else:
            report.failure_reasons.append(f"Target file '{target_file}' is missing from sandbox.")

        # 3. Independent test execution
        exit_code, stdout, stderr = sandbox.run_pytest(timeout=30.0)
        report.pytest_exit_code = exit_code
        report.pytest_output = stdout + ("\n" + stderr if stderr else "")
        if exit_code == 0:
            report.tests_pass = True
        else:
            report.failure_reasons.append(
                f"Independent pytest run failed with exit code {exit_code}:\n{report.pytest_output}"
            )

        # 4. No unrelated changes (only target file should be modified)
        unrelated = [
            f for f in modified_files
            if not f.endswith(target_file)
            and not f.startswith(".ultron")
            and not f.startswith("__pycache__")
            and "__pycache__" not in f
            and not f.endswith(".pyc")
            and not f.endswith(".pytest_cache")
            and ".pytest_cache" not in f
        ]
        if not unrelated:
            report.no_unrelated_changes = True
        else:
            report.failure_reasons.append(
                f"Unrelated files were modified: {unrelated}"
            )

        # 5. Non-empty diff
        if report.diff.strip():
            report.non_empty_diff = True
        else:
            report.failure_reasons.append("Sandbox diff is empty (no changes made).")

        # 6. Verification executed
        if any("pytest" in str(t).lower() for t in trace.tests_executed) or report.tests_pass:
            report.verification_executed = True

        # 7. Budget respected
        if trace.duration <= max_duration and trace.iterations <= max_iterations:
            report.budget_respected = True
        else:
            report.failure_reasons.append(
                f"Budget exceeded: duration={trace.duration:.2f}s (max {max_duration}s), "
                f"iterations={trace.iterations} (max {max_iterations})"
            )

        return report
