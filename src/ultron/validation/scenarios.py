"""Pre-defined real-world capability regression scenarios.

Defines complex end-to-end task test cases including the Personal Expense Manager scenario.
"""

from __future__ import annotations

from ultron.core.tools.definitions import ToolCapability
from ultron.validation.model import (
    AcceptanceCriterion,
    CapabilityTestCase,
    Difficulty,
    GenerationStrategy,
    TestSource,
    TestSplit,
)


def build_expense_manager_scenario(workspace_root: str | None = None) -> CapabilityTestCase:
    """Builds the canonical Expense Manager real-world regression test case."""
    prompt = (
        "Build me a complete desktop personal expense manager for macOS. "
        "I should be able to add, edit and delete expenses, categorize them, see totals by category, "
        "search expenses, and export the data to CSV. The data must persist between application launches. "
        "The application should have a usable graphical interface. "
        "Create the complete project in the Ultron workspace, install whatever dependencies are necessary, "
        "launch the application, test the major workflows yourself, fix any problems you encounter, "
        "and leave me with a working application."
    )

    criteria = [
        AcceptanceCriterion(
            criterion_id="AC-01",
            description="Project is created inside the configured Ultron workspace.",
            validator_type="workspace_confinement",
            required=True,
        ),
        AcceptanceCriterion(
            criterion_id="AC-02",
            description="Required project artifacts exist.",
            validator_type="artifact_exists",
            required=True,
            details={"filenames": ["expense_manager.py", "requirements.txt"]},
        ),
        AcceptanceCriterion(
            criterion_id="AC-03",
            description="Dependency specification is valid (no stdlib modules declared).",
            validator_type="dependency_validity",
            required=True,
        ),
        AcceptanceCriterion(
            criterion_id="AC-04",
            description="Dependency installation does not fail because of invalid dependency declarations.",
            validator_type="dependency_validity",
            required=True,
        ),
        AcceptanceCriterion(
            criterion_id="AC-05",
            description="Application can be launched and compiled without syntax errors.",
            validator_type="application_launch",
            required=True,
        ),
        AcceptanceCriterion(
            criterion_id="AC-06",
            description="Application performs at least one meaningful expense operation (schema/persistence/CRUD).",
            validator_type="functional_operation",
            required=True,
            details={"required_keywords": ["sqlite3", "csv"]},
        ),
        AcceptanceCriterion(
            criterion_id="AC-07",
            description="Application produces expected observable persistence and export capabilities.",
            validator_type="functional_operation",
            required=True,
            details={"required_keywords": ["expense", "category"]},
        ),
        AcceptanceCriterion(
            criterion_id="AC-08",
            description="No infinite or repeated identical tool failure loop occurs.",
            validator_type="no_thrashing",
            required=True,
        ),
        AcceptanceCriterion(
            criterion_id="AC-09",
            description="Ultron does not report successful completion without independent verification evidence.",
            validator_type="verification_evidence",
            required=True,
        ),
    ]

    invariants = ["INV-001", "INV-002", "INV-003", "INV-004"]

    return CapabilityTestCase(
        case_id="EXPENSE_MANAGER_REAL_WORLD_01",
        capability=ToolCapability.BUILD,
        expected_capability=ToolCapability.BUILD,
        task=prompt,
        subject="Expense Manager",
        subject_kind="application",
        difficulty=Difficulty.ADVANCED,
        test_source=TestSource.FIXED,
        split=TestSplit.DEVELOPMENT,
        strategy=GenerationStrategy.DEVELOPMENT_DIRECT,
        acceptance_criteria=criteria,
        expected_invariants=invariants,
    )

