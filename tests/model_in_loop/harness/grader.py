"""
MITLGrader: Independent objective grader for Model-in-the-Loop coding tasks.
Never trusts LLM claims; inspects disk state, diffs, and independent test execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from tests.model_in_loop.harness.runner import ExecutionTrace
from tests.model_in_loop.harness.sandbox import MITLSandbox


class FailureTaxonomy(str, Enum):
    """Failure classification taxonomy for MITL benchmarks."""

    MODEL = "MODEL"
    ORCHESTRATION = "ORCHESTRATION"
    CONTEXT = "CONTEXT"
    TOOL = "TOOL"
    TEST = "TEST"
    REPAIR = "REPAIR"
    VERIFICATION = "VERIFICATION"
    SECURITY = "SECURITY"
    BUDGET = "BUDGET"
    INFRASTRUCTURE = "INFRASTRUCTURE"
    CANCELLED = "CANCELLED"
    OUT_OF_SCOPE_MODIFICATION = "OUT_OF_SCOPE_MODIFICATION"
    REPAIR_BUDGET_EXHAUSTED = "REPAIR_BUDGET_EXHAUSTED"
    TESTS_FAILED = "TESTS_FAILED"
    VERIFICATION_FAILED = "VERIFICATION_FAILED"


@dataclass
class BenchmarkResult:
    scenario_name: str
    status: str  # "PASS" or "FAIL"
    duration: float = 0.0
    model_used: str = "qwen2.5-coder"
    tests_passed: bool = False
    tests_failed: bool = False
    repair_attempts: int = 0
    tool_calls: int = 0
    verification_executed: bool = False
    budget_respected: bool = False
    modified_files: list[str] = field(default_factory=list)
    unrelated_changes: list[str] = field(default_factory=list)
    failure_category: FailureTaxonomy | None = None
    terminal_state: str = "COMPLETED"


@dataclass
class BenchmarkMatrix:
    """Aggregated results across multiple MITL benchmark scenarios."""

    total_scenarios: int
    reports: list[GradingReport] = field(default_factory=list)
    results: list[BenchmarkResult] = field(default_factory=list)

    @property
    def passed_scenarios(self) -> int:
        return sum(1 for r in self.reports if r.is_success)

    @property
    def failed_scenarios(self) -> int:
        return self.total_scenarios - self.passed_scenarios

    @property
    def pass_rate(self) -> float:
        return self.passed_scenarios / self.total_scenarios if self.total_scenarios > 0 else 0.0

    def summary(self) -> str:
        lines = [
            "=== MITL BENCHMARK MATRIX SUMMARY ===",
            f"Total Scenarios: {self.total_scenarios}",
            f"Passed: {self.passed_scenarios}/{self.total_scenarios} ({self.pass_rate * 100:.1f}%)",
            f"Failed: {self.failed_scenarios}/{self.total_scenarios}",
            "--- Scenario Breakdown ---",
        ]
        for r in self.reports:
            lines.append(f"  {r.scenario_name}: {'PASS' if r.is_success else 'FAIL'}")
        return "\n".join(lines)


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
    failure_category: FailureTaxonomy | None = None

    @property
    def failure_taxonomy(self) -> FailureTaxonomy | None:
        """Alias for failure_category with auto-classification fallback."""
        if self.failure_category is not None:
            return self.failure_category
        if self.is_success:
            return None
        if not self.budget_respected:
            return FailureTaxonomy.REPAIR_BUDGET_EXHAUSTED
        if not self.no_unrelated_changes:
            return FailureTaxonomy.OUT_OF_SCOPE_MODIFICATION
        if not self.verification_executed:
            return FailureTaxonomy.VERIFICATION_FAILED
        if not self.tests_pass:
            return FailureTaxonomy.TESTS_FAILED
        if not self.correct_file_modified or not self.expected_implementation:
            return FailureTaxonomy.MODEL
        return FailureTaxonomy.ORCHESTRATION

    @failure_taxonomy.setter
    def failure_taxonomy(self, value: FailureTaxonomy | None) -> None:
        self.failure_category = value

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
            f"  1. Correct file(s) modified: {'PASS' if self.correct_file_modified else 'FAIL'}",
            f"  2. Expected implementation: {'PASS' if self.expected_implementation else 'FAIL'}",
            f"  3. Independent tests pass: {'PASS' if self.tests_pass else 'FAIL'}",
            f"  4. No unrelated changes: {'PASS' if self.no_unrelated_changes else 'FAIL'}",
            f"  5. Non-empty diff: {'PASS' if self.non_empty_diff else 'FAIL'}",
            f"  6. Verification executed: {'PASS' if self.verification_executed else 'FAIL'}",
            f"  7. Budget respected: {'PASS' if self.budget_respected else 'FAIL'}",
        ]
        if self.failure_category:
            lines.append(f"  Failure Category: {self.failure_category.value}")
        if self.failure_reasons:
            lines.append("Failures:")
            for reason in self.failure_reasons:
                lines.append(f"  - {reason}")
        if self.diff:
            lines.append(f"Repository Diff:\n{self.diff}")
        return "\n".join(lines)

    def to_benchmark_result(
        self,
        trace: ExecutionTrace | None = None,
        model_name: str = "qwen2.5-coder",
        terminal_state: str = "COMPLETED",
    ) -> BenchmarkResult:
        """Converts grading report into structured benchmark result."""
        status = "PASS" if self.is_success else "FAIL"
        tools = getattr(trace, "tools_invoked", []) if trace else []
        repair_attempts = max(
            0,
            len([t for t in tools if hasattr(t, "tool_name") and ("replace" in t.tool_name or "write" in t.tool_name)]) - 1,
        )
        duration = getattr(trace, "duration", getattr(trace, "duration_seconds", 0.0)) if trace else 0.0
        modified = list(getattr(trace, "files_modified", [])) if trace else []
        return BenchmarkResult(
            scenario_name=self.scenario_name,
            status=status,
            duration=duration,
            model_used=model_name,
            tests_passed=self.tests_pass,
            tests_failed=not self.tests_pass,
            repair_attempts=repair_attempts,
            tool_calls=len(tools),
            verification_executed=self.verification_executed,
            budget_respected=self.budget_respected,
            modified_files=modified,
            unrelated_changes=[] if self.no_unrelated_changes else self.failure_reasons,
            failure_category=self.failure_category,
            terminal_state=terminal_state,
        )


class MITLGrader:
    """
    Grades an autonomous coding task objectively by evaluating the final sandbox state.
    """

    @classmethod
    def grade(
        cls,
        scenario: Any,
        sandbox: MITLSandbox,
        trace: Any | None = None,
        max_duration: float = 180.0,
        max_iterations: int = 15,
    ) -> GradingReport:
        report = GradingReport(scenario_name=getattr(scenario, "name", "mitl_scenario"))

        # Determine target and allowed files (supports multi-file scenarios)
        raw_targets = getattr(scenario, "target_files", None)
        if raw_targets is not None:
            target_files = list(raw_targets)
        else:
            tf = getattr(scenario, "target_file", "calculator.py")
            target_files = [tf] if tf else []

        raw_allowed = getattr(scenario, "allowed_modified_files", None)
        allowed_files = list(raw_allowed) if raw_allowed is not None else target_files

        modified_files = sandbox.get_modified_files()
        report.diff = sandbox.get_diff()

        # Check for trace-level cancellation or explicit errors
        trace_error = str(getattr(trace, "error", "") or "")
        is_cancelled = "cancel" in trace_error.lower()

        # 1. Correct file(s) modified
        if target_files:
            all_modified = all(
                any(f == tf or f.endswith(("/" + tf, tf)) for f in modified_files)
                for tf in target_files
            )
            if all_modified:
                report.correct_file_modified = True
            else:
                report.failure_reasons.append(
                    f"Target file(s) {target_files} were not all modified. Modified files: {modified_files}"
                )
        else:
            report.correct_file_modified = True

        # 2. Expected implementation
        if hasattr(scenario, "validate_implementation") and callable(scenario.validate_implementation):
            ok, reason = scenario.validate_implementation(sandbox)
            if ok:
                report.expected_implementation = True
            else:
                report.failure_reasons.append(
                    reason or f"Scenario implementation validation failed for {target_files}."
                )
        elif target_files and all(sandbox.file_exists(tf) for tf in target_files):
            report.expected_implementation = True
        else:
            missing = [tf for tf in target_files if not sandbox.file_exists(tf)]
            report.failure_reasons.append(f"Target files missing from sandbox: {missing}")

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

        # 4. No unrelated changes (only allowed files should be modified)
        unrelated = [
            f for f in modified_files
            if not any(f == af or f.endswith(("/" + af, af)) for af in allowed_files)
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

        # 5. Non-empty diff (if changes expected)
        if report.diff.strip() or not target_files:
            report.non_empty_diff = True
        else:
            report.failure_reasons.append("Sandbox diff is empty (no changes made).")

        # 6. Verification executed
        tests_executed = getattr(trace, "tests_executed", []) if trace else []
        if any("pytest" in str(t).lower() for t in tests_executed) or report.tests_pass:
            report.verification_executed = True
        else:
            report.failure_reasons.append("Verification command (pytest) was not executed.")

        # 7. Budget respected
        duration = getattr(trace, "duration", getattr(trace, "duration_seconds", 0.0)) if trace else 0.0
        iterations = getattr(trace, "iterations", 0) if trace else 0
        budget_exceeded_error = "budget" in trace_error.lower()

        if not budget_exceeded_error and duration <= max_duration and iterations <= max_iterations:
            report.budget_respected = True
        else:
            report.failure_reasons.append(
                f"Budget exceeded: duration={duration:.2f}s (max {max_duration}s), "
                f"iterations={iterations} (max {max_iterations})"
            )

        # Classify failure category if failed
        if not report.is_success:
            if is_cancelled:
                report.failure_category = FailureTaxonomy.CANCELLED
            elif not report.budget_respected:
                report.failure_category = FailureTaxonomy.REPAIR_BUDGET_EXHAUSTED
            elif not report.no_unrelated_changes:
                report.failure_category = FailureTaxonomy.OUT_OF_SCOPE_MODIFICATION
            elif not report.verification_executed:
                report.failure_category = FailureTaxonomy.VERIFICATION_FAILED
            elif not report.tests_pass:
                report.failure_category = FailureTaxonomy.TESTS_FAILED
            elif not report.correct_file_modified or not report.expected_implementation:
                report.failure_category = FailureTaxonomy.MODEL
            else:
                report.failure_category = FailureTaxonomy.ORCHESTRATION

        return report
