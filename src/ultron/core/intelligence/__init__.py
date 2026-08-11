"""ultron.core.intelligence
~~~~~~~~~~~~~~~~~~~~~~~~~~

Intelligence helpers for the general task pipeline:

    USER REQUEST -> GOAL UNDERSTANDING -> TASK CLASSIFICATION -> PLANNING
    -> PLAN VALIDATION -> EXECUTION -> TOOLS

- :mod:`task_classification` — goal extraction + task-type classification
- :mod:`task_planning` — structured, outcome-oriented plan generation
- :mod:`plan_validation` — deterministic pre-execution plan checks
- :mod:`planning` — security preflight for plan steps (consent aggregation)
- plus hardware awareness, model catalog, structured output, etc.
"""

from ultron.core.intelligence.plan_validation import (
    COMPLEX_TASK_TYPES,
    validate_plan,
)
from ultron.core.intelligence.task_classification import (
    classify_task,
    classify_task_deterministic,
    extract_goal,
)
from ultron.core.intelligence.task_planning import (
    build_planning_prompt,
    detect_workspace_kind,
    fallback_plan,
    generate_task_plan,
    parse_plan_json,
    prepare_task_for_execution,
    probe_working_context,
)

__all__ = [
    "COMPLEX_TASK_TYPES",
    "build_planning_prompt",
    "classify_task",
    "classify_task_deterministic",
    "detect_workspace_kind",
    "extract_goal",
    "fallback_plan",
    "generate_task_plan",
    "parse_plan_json",
    "prepare_task_for_execution",
    "probe_working_context",
    "validate_plan",
]
