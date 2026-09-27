"""ultron.core.intelligence.plan_validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PLAN VALIDATION — the gate before any plan may be executed.

Deterministic checks:

- unique step ids
- valid dependencies (no self-dependency, no unknown step ids)
- no circular dependencies
- every step reachable from a root step
- every step has a description, expected outcome and completion criteria
- the plan has completion criteria and verification requirements for
  complex task types
- the plan is non-empty for action task types

Invalid plans are rejected before execution; a plan that cannot be safely
executed must never run half-correct.
"""

from __future__ import annotations

import re

from ultron.core.types import (
    ExecutionPolicy,
    PlanValidationIssue,
    PlanValidationReport,
    TaskPlan,
    TaskType,
)

# Task types where a plan is expected to decompose the goal and verify it.
# Also the set of task types that take the full planning/execution path
# (informational / simple-action / file-operation stay on the fast path).
COMPLEX_TASK_TYPES = frozenset(
    {
        TaskType.MULTI_STEP,
        TaskType.SOFTWARE_ENGINEERING,
        TaskType.DEBUGGING,
        TaskType.CODE_REVIEW,
        TaskType.RESEARCH,
        TaskType.SYSTEM_OPERATION,
        TaskType.CONFIGURATION,
        TaskType.DATA_OPERATION,
    }
)


def _find_cycles(steps: list) -> list[list[int]]:
    """Detects dependency cycles (as sorted id-lists) via DFS back-edge scan."""
    by_id = {step.id: step for step in steps}
    cycles: set[tuple[int, ...]] = set()

    def dfs(node: int, path: list[int], visited: set[int]) -> None:
        if node in path:
            start = path.index(node)
            cycle = tuple(sorted(path[start:]))
            if len(cycle) > 1:
                cycles.add(cycle)
            return
        if node in visited:
            return
        visited.add(node)
        step = by_id.get(node)
        if step is not None:
            for dep in step.dependencies:
                dfs(dep, [*path, node], visited)

    for step in steps:
        dfs(step.id, [], set())
    return [list(cycle) for cycle in sorted(cycles)]


def _unreachable_steps(steps: list) -> list[int]:
    """
    Steps that cannot be reached from any root (a step with no
    dependencies). A step with an unknown dependency is unreachable too.
    """
    reached = {step.id for step in steps if not step.dependencies}
    changed = True
    while changed:
        changed = False
        for step in steps:
            if step.id in reached:
                continue
            if any(dep in reached for dep in step.dependencies):
                reached.add(step.id)
                changed = True
    return sorted(step.id for step in steps if step.id not in reached)


_MUTATION_STEP_RE = re.compile(
    r"\b(modify\s+(?:file|code|source)|edit\s+(?:file|code|source)|write\s+(?:file|code|source)|"
    r"create\s+(?:file|source)|delete\s+(?:file|code)|overwrite\s+file|"
    r"apply\s+(?:a\s+)?(?:targeted\s+)?(?:bug\s+)?fix|apply\s+patch|apply\s+changes|"
    r"scaffold\s+(?:project|application)|code\s+modification|"
    r"implement\s+(?:application|fix|code))\b",
    re.IGNORECASE,
)
_EXECUTION_STEP_RE = re.compile(
    r"\b(run\s+(?:regression\s+)?tests?|execute\s+command|launch\s+application|"
    r"start\s+(?:process|server|service)|interact\s+with\s+application)\b",
    re.IGNORECASE,
)


def validate_plan_policy(
    plan: TaskPlan,
    policy: ExecutionPolicy | None = None,
) -> list[PlanValidationIssue]:
    """
    Validates that the steps of a TaskPlan comply with the authoritative ExecutionPolicy.
    """
    active_policy = policy or plan.execution_policy
    if active_policy is None:
        return []

    issues: list[PlanValidationIssue] = []

    for step in plan.steps:
        text = f"{step.description} {step.purpose} {step.expected_outcome}"
        # Check mutation violations
        if not active_policy.can_mutate and _MUTATION_STEP_RE.search(text):
            issues.append(
                PlanValidationIssue(
                    code="policy_violation_mutation",
                    message=(
                        f"Step {step.id} requires file mutation ('{step.description}') "
                        f"which is forbidden by active policy: {active_policy.mutation_authority.value}"
                    ),
                    step_id=step.id,
                )
            )
        # Check execution violations
        if not active_policy.can_execute and _EXECUTION_STEP_RE.search(text):
            issues.append(
                PlanValidationIssue(
                    code="policy_violation_execution",
                    message=(
                        f"Step {step.id} requires process execution ('{step.description}') "
                        f"which is forbidden by active policy: {active_policy.execution_authority.value}"
                    ),
                    step_id=step.id,
                )
            )

    return issues


def validate_plan(
    plan: TaskPlan,
    policy: ExecutionPolicy | None = None,
) -> PlanValidationReport:
    """Validates a TaskPlan; returns a report that must be ``valid`` to run."""
    issues: list[PlanValidationIssue] = []

    ids = [step.id for step in plan.steps]
    id_set = set(ids)

    # 1. Unique step ids.
    for dup in sorted({step_id for step_id in ids if ids.count(step_id) > 1}):
        issues.append(
            PlanValidationIssue(
                code="duplicate_step_id",
                message=f"Duplicate step id: {dup}",
                step_id=dup,
            )
        )

    # 2. Dependency sanity (self + unknown).
    for step in plan.steps:
        for dep in step.dependencies:
            if dep == step.id:
                issues.append(
                    PlanValidationIssue(
                        code="self_dependency",
                        message=f"Step {step.id} depends on itself",
                        step_id=step.id,
                    )
                )
            elif dep not in id_set:
                issues.append(
                    PlanValidationIssue(
                        code="unknown_dependency",
                        message=f"Step {step.id} depends on unknown step {dep}",
                        step_id=step.id,
                    )
                )

    # 3. Circular dependencies.
    cycles = _find_cycles(plan.steps)
    for cycle in cycles:
        issues.append(
            PlanValidationIssue(
                code="circular_dependency",
                message="Circular dependency: " + " -> ".join(map(str, cycle)),
                step_id=cycle[0],
            )
        )

    # 4. Reachability.
    unreachable = _unreachable_steps(plan.steps)
    for step_id in unreachable:
        issues.append(
            PlanValidationIssue(
                code="unreachable_step",
                message=f"Step {step_id} is not reachable from any root step",
                step_id=step_id,
            )
        )

    # 5. Step completeness.
    for step in plan.steps:
        if not step.description:
            issues.append(
                PlanValidationIssue(
                    code="missing_description",
                    message=f"Step {step.id} has no description",
                    step_id=step.id,
                )
            )
        if not step.expected_outcome:
            issues.append(
                PlanValidationIssue(
                    code="missing_expected_outcome",
                    message=f"Step {step.id} has no expected outcome",
                    step_id=step.id,
                )
            )
        if not step.completion_criteria:
            issues.append(
                PlanValidationIssue(
                    code="missing_step_criteria",
                    message=f"Step {step.id} has no completion criteria",
                    step_id=step.id,
                )
            )

    # 6. Plan-level completeness for action task types.
    if plan.task_type in COMPLEX_TASK_TYPES:
        if not plan.steps:
            issues.append(
                PlanValidationIssue(
                    code="no_steps",
                    message=f"Plan for {plan.task_type.value} has no steps",
                )
            )
        if not plan.completion_criteria:
            issues.append(
                PlanValidationIssue(
                    code="missing_plan_criteria",
                    message="Plan has no overall completion criteria",
                )
            )
        if not plan.verification_requirements:
            issues.append(
                PlanValidationIssue(
                    code="missing_verification",
                    message="Plan has no verification requirements",
                )
            )
    elif plan.task_type is TaskType.SIMPLE_ACTION and not plan.steps:
        issues.append(
            PlanValidationIssue(
                code="no_steps",
                message=f"Plan for {plan.task_type.value} has no steps",
            )
        )

    # 7. Policy compliance.
    policy_issues = validate_plan_policy(plan, policy)
    issues.extend(policy_issues)

    return PlanValidationReport(
        valid=not issues,
        issues=issues,
        circular_dependencies=cycles,
        unreachable_steps=unreachable,
    )
