"""ultron.core.intelligence.task_planning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

STRUCTURED PLANNING — the third layer of the general task pipeline:

    USER REQUEST -> GOAL UNDERSTANDING -> TASK CLASSIFICATION -> PLANNING
    -> PLAN VALIDATION -> EXECUTION -> TOOLS

Plans are outcome-oriented: each step states *what must be accomplished*
(expected outcome, completion criteria) rather than a bare tool call —
tools are implementation details the executor fills in later.  Dependencies
are explicit step ids, so ordering never depends on the LLM's memory.

The planner is read-only: it may probe the filesystem for project markers
(workspace detection) and call the LLM, but it NEVER executes action tools.
The generated :class:`~ultron.core.types.TaskPlan` is a real object that
callers attach to the TaskState, so it survives LLM turns, tool calls,
confirmations, failures, and agent continuation — it is never kept only
inside an LLM prompt.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from ultron.core.intelligence.plan_validation import (
    COMPLEX_TASK_TYPES,
    validate_plan,
)
from ultron.core.intelligence.task_classification import classify_task
from ultron.core.logging import get_logger
from ultron.core.types import (
    AcceptanceCriterion,
    AcceptanceCriterionStatus,
    EvidenceLevel,
    FailureStrategy,
    PlanStep,
    TaskPlan,
    TaskState,
    TaskType,
    WorkspaceKind,
)

logger = get_logger("ultron.intelligence.planning")

# Files/dirs that mark a directory as an existing software project.
MANIFEST_MARKERS = (
    "pyproject.toml",
    "package.json",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "Cargo.toml",
    "go.mod",
    "requirements.txt",
    "setup.py",
    "setup.cfg",
    "Pipfile",
    "poetry.lock",
    "Gemfile",
    "composer.json",
    "pom.xml",
    "build.gradle",
    "build.gradle.kts",
    "mix.exs",
    "CMakeLists.txt",
    "Makefile",
    "Dockerfile",
    ".git",
)

# Planning-pattern guidance shown to the LLM.  These are shapes, never
# hardcoded workflows: the model adapts them to the specific request.
PLANNING_GUIDANCE = """PATTERN GUIDANCE (shapes only - adapt to the specific request):
- feature implementation: understand codebase -> identify affected components
  -> design change -> implement -> test -> verify
- debugging / bug: reproduce -> inspect failure -> locate root cause -> modify
  -> test -> regression test -> verify
- refactoring: inspect -> identify affected dependencies -> define target
  structure -> refactor -> run tests -> verify behavior
- code review: inspect -> analyze -> identify findings -> validate findings
  -> produce report
- dependency upgrade: inspect dependencies -> determine compatibility -> modify
  dependency configuration -> install/update -> run tests -> repair issues
  -> verify
- repository analysis: inspect structure -> read key files -> trace relevant
  paths -> explain -> cite evidence
- system / configuration: inspect current setup -> determine required changes
  -> apply -> verify
- NEW workspace: establish workspace -> establish project structure ->
  implement -> validate -> verify final user goal
- EXISTING project: respect the current structure, package manager, build
  system and test framework; do not blindly create files before understanding
  the project"""

_PLAN_SCHEMA = """Respond with ONLY a JSON object, no markdown fences:
{
  "assumptions": ["...what the plan assumes..."],
  "constraints": ["...limits the plan respects..."],
  "steps": [
    {
      "id": 1,
      "description": "Outcome-oriented step name",
      "purpose": "Why this step exists",
      "expected_outcome": "What must be true after this step",
      "completion_criteria": ["evidence this step succeeded"],
      "dependencies": [],
      "failure_strategy": "stop | retry | skip | continue",
      "retry_policy": 0
    }
  ],
  "completion_criteria": ["...criteria the whole task must satisfy..."],
  "verification_requirements": ["...how the final goal will be verified..."],
  "failure_recovery": "Overall policy when a step fails"
}
Rules:
- Steps describe OUTCOMES, not tool calls.  A step may mention the action it
  implies, but the plan must be understandable without any tools.
- "dependencies" are ids of steps that must succeed first.
- Every step needs at least one completion_criteria; the plan needs overall
  completion_criteria and verification_requirements (complex tasks).
- Use at least one step for a final verification of the user goal."""


def detect_workspace_kind(cwd: str | None = None) -> WorkspaceKind:
    """
    Distinguishes a NEW WORKSPACE from an EXISTING PROJECT by probing the
    current directory for project markers (manifests, lock files, VCS).

    Pure filesystem inspection — no tools are executed.
    """
    base = Path(cwd) if cwd else Path.cwd()
    if not base.exists() or not base.is_dir():
        return WorkspaceKind.UNKNOWN
    if any((base / marker).exists() for marker in MANIFEST_MARKERS):
        return WorkspaceKind.EXISTING_PROJECT
    return WorkspaceKind.NEW_WORKSPACE


def probe_working_context(cwd: str | None = None) -> str:
    """
    Summarizes the current directory for the planner prompt: detected
    project markers, top-level directories and files.

    Pure filesystem inspection — no tools are executed.
    """
    base = Path(cwd) if cwd else Path.cwd()
    try:
        entries = list(base.iterdir())
    except OSError:
        return ""
    manifests = sorted(m for m in MANIFEST_MARKERS if (base / m).exists())
    dirs = sorted(
        p.name for p in entries if p.is_dir() and not p.name.startswith(".")
    )[:12]
    files = sorted(p.name for p in entries if p.is_file())[:12]
    parts = []
    if manifests:
        parts.append("manifests: " + ", ".join(manifests))
    if dirs:
        parts.append("directories: " + ", ".join(dirs))
    if files:
        parts.append("files: " + ", ".join(files))
    return "; ".join(parts) or "empty directory"


def build_planning_prompt(
    goal: str,
    task_type: TaskType,
    workspace: WorkspaceKind,
    working_context: str,
) -> str:
    """Builds the planner prompt for a goal, task type and workspace."""
    workspace_label = (
        "NEW WORKSPACE"
        if workspace is WorkspaceKind.NEW_WORKSPACE
        else "EXISTING PROJECT"
        if workspace is WorkspaceKind.EXISTING_PROJECT
        else "UNKNOWN"
    )
    return (
        "You are the structured planner of a local AI coding assistant.\n"
        "Create an actionable, step-by-step engineering plan to accomplish the user's goal.\n\n"
        f"GOAL: {goal}\n"
        f"TASK TYPE: {task_type.value}\n"
        f"WORKSPACE: {workspace_label}\n"
        f"WORKING CONTEXT: {working_context}\n\n"
        f"{PLANNING_GUIDANCE}\n\n"
        f"{_PLAN_SCHEMA}"
    )


def _strip_fences(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[\w]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    return raw.strip()


def _extract_json_payload(raw: str) -> dict | None:
    """Extracts a JSON dictionary from model output tolerating markdown, thoughts, and prose."""
    cleaned = _strip_fences(raw)
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, TypeError):
        pass

    # Balanced brace scan for outermost {...}
    first = raw.find("{")
    last = raw.rfind("}")
    if first != -1 and last > first:
        candidate = raw[first : last + 1]
        # Clean trailing commas: ,} -> } and ,] -> ]
        candidate_clean = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            data = json.loads(candidate_clean)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
    return None


def parse_plan_json(
    raw: str,
    goal: str,
    task_type: TaskType,
    workspace: WorkspaceKind,
    working_context: str,
) -> TaskPlan | None:
    """
    Parses the LLM's plan JSON into a TaskPlan.

    ``goal`` / ``task_type`` / ``workspace`` / ``working_context`` are
    authoritative and always come from the caller — the model never
    dictates the goal or task type.  Returns None when the payload cannot
    be parsed into a structurally valid plan.
    """
    payload = _extract_json_payload(raw)
    if payload is None or not isinstance(payload.get("steps"), list):
        return None

    raw_steps = payload["steps"]
    if not raw_steps:
        return None

    steps: list[PlanStep] = []

    for idx, item in enumerate(raw_steps, start=1):
        if not isinstance(item, dict):
            continue
        raw_id = item.get("id")
        step_id = raw_id if isinstance(raw_id, int) else idx

        description = str(item.get("description") or "").strip()
        purpose = str(item.get("purpose") or "").strip()
        expected_outcome = str(item.get("expected_outcome") or "").strip()

        criteria_list = [
            str(c).strip() for c in item.get("completion_criteria", []) if str(c).strip()
        ]

        strategy = str(item.get("failure_strategy", "stop")).lower()
        try:
            failure_strategy = FailureStrategy(strategy)
        except ValueError:
            failure_strategy = FailureStrategy.STOP

        deps = [
            int(d)
            for d in item.get("dependencies", [])
            if isinstance(d, int)
        ]

        steps.append(
            PlanStep(
                id=step_id,
                description=description,
                purpose=purpose,
                dependencies=deps,
                expected_outcome=expected_outcome,
                completion_criteria=criteria_list,
                failure_strategy=failure_strategy,
                retry_policy=int(item.get("retry_policy", 0) or 0),
            )
        )

    if not steps:
        return None

    completion_criteria = [
        str(c).strip() for c in payload.get("completion_criteria", []) if str(c).strip()
    ]
    if not completion_criteria:
        completion_criteria = [goal]

    verification_requirements = [
        str(v).strip()
        for v in payload.get("verification_requirements", [])
        if str(v).strip()
    ]
    if not verification_requirements:
        verification_requirements = [f"The user goal is satisfied: {goal}"]

    return TaskPlan(
        goal=goal,
        task_type=task_type,
        workspace=workspace,
        working_context=working_context,
        assumptions=[str(a).strip() for a in payload.get("assumptions", []) if str(a).strip()],
        constraints=[str(c).strip() for c in payload.get("constraints", []) if str(c).strip()],
        steps=steps,
        completion_criteria=completion_criteria,
        verification_requirements=verification_requirements,
        failure_recovery=str(payload.get("failure_recovery", "")).strip(),
        needs_clarification=bool(payload.get("needs_clarification", False)),
        clarification_questions=[
            str(q).strip()
            for q in payload.get("clarification_questions", [])
            if str(q).strip()
        ],
    )


def _slugify(goal: str) -> str:
    """Generates a clean directory slug for an artifact project from a goal."""
    clean = re.sub(r"[^a-zA-Z0-9]+", "-", (goal or "").lower()).strip("-")
    prefixes = (
        "build-a-", "build-an-", "build-", "create-a-", "create-an-", "create-",
        "make-a-", "make-an-", "make-", "write-a-", "write-an-", "write-",
    )
    for p in prefixes:
        if clean.startswith(p):
            clean = clean[len(p):]
            break
    clean = clean.strip("-")
    return (clean or "artifact-project")[:40]


def build_software_engineering_plan(
    goal: str,
    workspace: WorkspaceKind = WorkspaceKind.UNKNOWN,
    cwd: str | None = None,
    project_dir: str | Path | None = None,
) -> TaskPlan:
    """
    Deterministic, fully executable plan for software engineering and application creation.
    Enforces: Scaffolding -> Dependencies -> Execution -> Interaction -> Repair -> Verification.
    """
    from ultron.core.intelligence.intent_understanding import understand_user_intent
    from ultron.core.tools.paths import (
        ALLOWED_BASE_DIR,
        get_configured_workspace,
        set_active_project_dir,
    )

    intent = understand_user_intent(goal, cwd=cwd)

    resolved_project_dir: Path
    if project_dir is not None:
        resolved_project_dir = Path(project_dir).expanduser().resolve()
    else:
        slug = _slugify(goal)
        ws = get_configured_workspace()
        if ws is not None:
            resolved_project_dir = (ws / slug).resolve()
        else:
            resolved_project_dir = (ALLOWED_BASE_DIR / slug).resolve()

    try:
        resolved_project_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

    set_active_project_dir(resolved_project_dir)

    steps = [
        PlanStep(
            id=1,
            description="Implement application files and scaffold project",
            purpose=f"Create required source code, configuration, and data files in {resolved_project_dir}",
            expected_outcome=f"All necessary application files exist in {resolved_project_dir}",
            completion_criteria=["Source files created on disk", "Dependencies/entrypoint defined"],
            failure_strategy=FailureStrategy.RETRY,
            retry_policy=2,
        ),
        PlanStep(
            id=2,
            description="Launch application and observe runtime readiness",
            purpose="Run the application or server and establish that the process/window/service is active and ready",
            expected_outcome="Application launches and enters active/ready state without crashes",
            dependencies=[1],
            completion_criteria=["Application process/service started and running"],
            failure_strategy=FailureStrategy.RETRY,
            retry_policy=2,
        ),
        PlanStep(
            id=3,
            description="Interact with application and test user behavior",
            purpose="Exercise actual application functionality (query UI/endpoints/database/CLI) and repair any discovered issues",
            expected_outcome="Application responds correctly to inputs and workflows",
            dependencies=[1, 2],
            completion_criteria=["Core user workflows executed and verified"],
            failure_strategy=FailureStrategy.RETRY,
            retry_policy=2,
        ),
        PlanStep(
            id=4,
            description="Verify the final user goal with independent evidence",
            purpose="Ensure the original user request is fully satisfied and backed by verifiable output/reference comparison",
            expected_outcome="The original user goal is satisfied with collected evidence",
            dependencies=[1, 2, 3],
            completion_criteria=[goal, "Verified completion with empirical evidence"],
            failure_strategy=FailureStrategy.STOP,
        ),
    ]

    acceptance_criteria = list(intent.acceptance_criteria) if intent else [
        AcceptanceCriterion(
            id="workspace_isolation",
            description=f"Project is isolated under dedicated workspace directory: {resolved_project_dir}",
            verification_method="file_check",
            required=True,
            status=AcceptanceCriterionStatus.PENDING,
            evidence_level=EvidenceLevel.LEVEL_1_CREATED,
        ),
        AcceptanceCriterion(
            id="dependencies_and_code",
            description="Application code is implemented and runnable",
            verification_method="file_check",
            required=True,
            status=AcceptanceCriterionStatus.PENDING,
            evidence_level=EvidenceLevel.LEVEL_1_CREATED,
        ),
        AcceptanceCriterion(
            id="application_execution",
            description="Application launches and runs without crashing",
            verification_method="process_check",
            required=True,
            status=AcceptanceCriterionStatus.PENDING,
            evidence_level=EvidenceLevel.LEVEL_3_LAUNCHED,
        ),
        AcceptanceCriterion(
            id="behavior_interaction",
            description="Application behavior and core features are exercised and interacted with",
            verification_method="execution",
            required=True,
            status=AcceptanceCriterionStatus.PENDING,
            evidence_level=EvidenceLevel.LEVEL_4_INTERACTED,
        ),
        AcceptanceCriterion(
            id="verified_completion",
            description=f"Goal '{goal}' is verified with independent evidence",
            verification_method="reference_comparison",
            required=True,
            status=AcceptanceCriterionStatus.PENDING,
            evidence_level=EvidenceLevel.LEVEL_5_VERIFIED,
        ),
    ]

    return TaskPlan(
        goal=goal,
        task_type=TaskType.SOFTWARE_ENGINEERING,
        workspace=workspace,
        steps=steps,
        completion_criteria=[goal, "All application files created, launched, interacted with, and verified"],
        verification_requirements=[f"The user goal is satisfied: {goal}"],
        acceptance_criteria=acceptance_criteria,
        user_intent=intent,
        failure_recovery="Diagnose and classify failures; apply repair and retest; do not report completion without evidence.",
        project_dir=str(resolved_project_dir),
    )


# Alias build_artifact_plan for backwards compatibility
build_artifact_plan = build_software_engineering_plan


def build_debugging_plan(
    goal: str,
    workspace: WorkspaceKind = WorkspaceKind.UNKNOWN,
) -> TaskPlan:
    """Deterministic executable plan for debugging and defect repair."""
    from ultron.core.intelligence.intent_understanding import understand_user_intent

    intent = understand_user_intent(goal)
    steps = [
        PlanStep(
            id=1,
            description="Reproduce failure and inspect error state",
            purpose="Run failing test or command to reproduce the defect and capture stack trace",
            expected_outcome="Defect is reproduced and error signature is observed",
            completion_criteria=["Failure reproduced or located"],
            failure_strategy=FailureStrategy.RETRY,
            retry_policy=2,
        ),
        PlanStep(
            id=2,
            description="Diagnose root cause in relevant source files",
            purpose="Inspect code context, trace variable flow, and identify exact root cause",
            expected_outcome="Root cause of the defect is identified in code",
            dependencies=[1],
            completion_criteria=["Root cause identified"],
            failure_strategy=FailureStrategy.RETRY,
            retry_policy=2,
        ),
        PlanStep(
            id=3,
            description="Apply targeted bug fix to affected files",
            purpose="Modify code to resolve the defect without introducing regressions",
            expected_outcome="Bug fix is implemented in source files",
            dependencies=[1, 2],
            completion_criteria=["Code modification applied"],
            failure_strategy=FailureStrategy.RETRY,
            retry_policy=2,
        ),
        PlanStep(
            id=4,
            description="Run regression tests and validate fix",
            purpose="Execute test suite or reproduction command to verify resolution",
            expected_outcome="Tests pass and original failure is resolved",
            dependencies=[1, 2, 3],
            completion_criteria=["Regression tests pass"],
            failure_strategy=FailureStrategy.RETRY,
            retry_policy=2,
        ),
        PlanStep(
            id=5,
            description="Verify the final user goal",
            purpose="Confirm the original problem is resolved and system is in working order",
            expected_outcome="The user goal is satisfied with verified fix evidence",
            dependencies=[1, 2, 3, 4],
            completion_criteria=[goal],
            failure_strategy=FailureStrategy.STOP,
        ),
    ]
    return TaskPlan(
        goal=goal,
        task_type=TaskType.DEBUGGING,
        workspace=workspace,
        steps=steps,
        completion_criteria=[goal, "Bug resolved and regression tests pass"],
        verification_requirements=[f"The user goal is satisfied: {goal}"],
        acceptance_criteria=list(intent.acceptance_criteria) if intent else [],
        user_intent=intent,
        failure_recovery="Diagnose root cause; apply targeted fix; re-test before completing.",
    )


def build_research_plan(
    goal: str,
    workspace: WorkspaceKind = WorkspaceKind.UNKNOWN,
) -> TaskPlan:
    """Deterministic executable plan for research and evidence synthesis."""
    from ultron.core.intelligence.intent_understanding import understand_user_intent

    intent = understand_user_intent(goal)
    steps = [
        PlanStep(
            id=1,
            description="Identify information requirements and search targets",
            purpose="Deconstruct research goal into key technical questions and search targets",
            expected_outcome="Core research questions and query strategy defined",
            completion_criteria=["Key questions identified"],
            failure_strategy=FailureStrategy.RETRY,
            retry_policy=1,
        ),
        PlanStep(
            id=2,
            description="Gather evidence across relevant files and sources",
            purpose="Search codebase, documentation, or authoritative sources for evidence",
            expected_outcome="Raw facts and findings collected",
            dependencies=[1],
            completion_criteria=["Evidence gathered from authoritative sources"],
            failure_strategy=FailureStrategy.RETRY,
            retry_policy=2,
        ),
        PlanStep(
            id=3,
            description="Evaluate and normalize findings",
            purpose="Assess evidence quality, freshness, and resolve conflicting claims",
            expected_outcome="Validated, consistent findings organized by topic",
            dependencies=[1, 2],
            completion_criteria=["Evidence evaluated and verified"],
            failure_strategy=FailureStrategy.CONTINUE,
        ),
        PlanStep(
            id=4,
            description="Synthesize response grounded in verified evidence",
            purpose="Formulate clear, comprehensive answer with explicit citations",
            expected_outcome="Cited, authoritative answer prepared for user",
            dependencies=[1, 2, 3],
            completion_criteria=["Comprehensive cited answer produced"],
            failure_strategy=FailureStrategy.RETRY,
            retry_policy=1,
        ),
        PlanStep(
            id=5,
            description="Verify research goal completeness",
            purpose="Ensure all aspects of original research question are answered with evidence",
            expected_outcome="Research goal satisfied completely",
            dependencies=[1, 2, 3, 4],
            completion_criteria=[goal],
            failure_strategy=FailureStrategy.STOP,
        ),
    ]
    return TaskPlan(
        goal=goal,
        task_type=TaskType.RESEARCH,
        workspace=workspace,
        steps=steps,
        completion_criteria=[goal, "All research questions answered with verified citations"],
        verification_requirements=[f"The user goal is satisfied: {goal}"],
        acceptance_criteria=list(intent.acceptance_criteria) if intent else [],
        user_intent=intent,
        failure_recovery="Gather multi-source evidence; cross-reference findings before completing.",
    )


def build_system_operation_plan(
    goal: str,
    task_type: TaskType = TaskType.SYSTEM_OPERATION,
    workspace: WorkspaceKind = WorkspaceKind.UNKNOWN,
) -> TaskPlan:
    """Deterministic executable plan for system, configuration, and data operations."""
    from ultron.core.intelligence.intent_understanding import understand_user_intent

    intent = understand_user_intent(goal)
    steps = [
        PlanStep(
            id=1,
            description="Inspect current system state and configuration",
            purpose="Inspect existing system settings, environment, or resources",
            expected_outcome="Current system state is documented",
            completion_criteria=["Current configuration observed"],
            failure_strategy=FailureStrategy.RETRY,
            retry_policy=2,
        ),
        PlanStep(
            id=2,
            description="Determine required configuration changes",
            purpose="Formulate safe command sequence and configuration modifications",
            expected_outcome="Modification plan is ready for execution",
            dependencies=[1],
            completion_criteria=["Change sequence defined"],
            failure_strategy=FailureStrategy.RETRY,
            retry_policy=1,
        ),
        PlanStep(
            id=3,
            description="Apply changes through authorized system operations",
            purpose="Execute state-modifying operations through security boundary",
            expected_outcome="Configuration changes applied successfully",
            dependencies=[1, 2],
            completion_criteria=["Operations executed successfully"],
            failure_strategy=FailureStrategy.RETRY,
            retry_policy=2,
        ),
        PlanStep(
            id=4,
            description="Verify resulting state against user requirements",
            purpose="Confirm that the applied changes produced the expected system state",
            expected_outcome="Target system state is confirmed with evidence",
            dependencies=[1, 2, 3],
            completion_criteria=[goal],
            failure_strategy=FailureStrategy.STOP,
        ),
    ]
    return TaskPlan(
        goal=goal,
        task_type=TaskType.SYSTEM_OPERATION,
        workspace=workspace,
        steps=steps,
        completion_criteria=[goal, "System operations applied and verified"],
        verification_requirements=[f"The user goal is satisfied: {goal}"],
        acceptance_criteria=list(intent.acceptance_criteria) if intent else [],
        user_intent=intent,
        failure_recovery="Inspect current state; apply changes safely; verify outcome.",
    )


def fallback_plan(
    goal: str,
    task_type: TaskType,
    workspace: WorkspaceKind = WorkspaceKind.UNKNOWN,
    cwd: str | None = None,
) -> TaskPlan:
    """
    Constructs a deterministic, structurally valid, and fully executable fallback plan
    tailored to the task type when LLM planning is unavailable or fails.
    """
    if task_type in (TaskType.SOFTWARE_ENGINEERING, TaskType.MULTI_STEP):
        return build_software_engineering_plan(goal, workspace, cwd=cwd)
    if task_type == TaskType.DEBUGGING:
        return build_debugging_plan(goal, workspace)
    if task_type in (TaskType.RESEARCH, TaskType.CODE_REVIEW):
        return build_research_plan(goal, workspace)
    if task_type in (TaskType.SYSTEM_OPERATION, TaskType.CONFIGURATION, TaskType.DATA_OPERATION):
        return build_system_operation_plan(goal, task_type, workspace)
    return build_software_engineering_plan(goal, workspace, cwd=cwd)


def _bind_user_intent_to_plan(plan: TaskPlan, goal: str, cwd: str | None = None) -> None:
    """Attaches UserIntent and acceptance criteria to a plan if not already bound."""
    from ultron.core.intelligence.intent_understanding import understand_user_intent

    if plan.user_intent is None:
        plan.user_intent = understand_user_intent(goal, cwd=cwd)
    if not plan.acceptance_criteria and plan.user_intent:
        plan.acceptance_criteria = list(plan.user_intent.acceptance_criteria)


async def generate_task_plan(
    goal: str,
    task_type: TaskType,
    engine,
    workspace: WorkspaceKind | None = None,
    working_context: str | None = None,
    cwd: str | None = None,
) -> TaskPlan | None:
    """
    Generates a validated, outcome-oriented TaskPlan for a goal with bounded recovery.

    - Informational requests never get a plan (returns None).
    - If the LLM generates a valid plan, it is used directly.
    - If the LLM returns invalid JSON or fails structural validation, an automated
      re-plan recovery turn is attempted with validation feedback.
    - If LLM planning fails completely, a validated deterministic fallback plan is used.
    """
    if task_type is TaskType.INFORMATIONAL:
        return None

    ws = workspace or detect_workspace_kind(cwd)
    context = (
        working_context if working_context is not None else probe_working_context(cwd)
    )
    prompt = build_planning_prompt(goal, task_type, ws, context)

    logger.info(
        "[PLAN_START] task_type=%s, workspace=%s, goal_len=%d",
        task_type.value,
        ws.value,
        len(goal),
    )

    plan: TaskPlan | None = None
    if engine is not None:
        try:
            raw = await engine.generate([{"role": "user", "content": prompt}])
            logger.info("[PLAN_GENERATION] received %d chars from engine", len(raw or ""))
            plan = parse_plan_json(raw, goal, task_type, ws, context)
            if plan is not None:
                report = validate_plan(plan)
                if report.valid:
                    logger.info("[PLAN_VALIDATION] valid=True (%d steps)", len(plan.steps))
                    logger.info("[PLAN_SELECTED] normal")
                    _bind_user_intent_to_plan(plan, goal, cwd)
                    return plan

                issue_msgs = [i.message for i in report.issues]
                logger.warning(
                    "[PLAN_VALIDATION] valid=False issues: %s",
                    "; ".join(issue_msgs),
                )
                # Attempt 1 bounded re-plan recovery with validation feedback
                recovery_prompt = (
                    f"{prompt}\n\n"
                    f"IMPORTANT CORRECTION: Your previous plan was invalid ({'; '.join(issue_msgs[:3])}). "
                    "Please correct the issues and output ONLY a valid JSON object matching the plan schema."
                )
                retry_raw = await engine.generate([{"role": "user", "content": recovery_prompt}])
                retry_plan = parse_plan_json(retry_raw, goal, task_type, ws, context)
                if retry_plan is not None and validate_plan(retry_plan).valid:
                    logger.info(
                        "[PLAN_RECOVERY] bounded recovery succeeded (%d steps)",
                        len(retry_plan.steps),
                    )
                    logger.info("[PLAN_SELECTED] recovered")
                    _bind_user_intent_to_plan(retry_plan, goal, cwd)
                    return retry_plan
        except Exception as exc:  # noqa: BLE001
            logger.warning("[PLAN_GENERATION_FAILED] engine call failed: %s", exc)

    return None


async def prepare_task_for_execution(
    user_input: str,
    engine,
    cwd: str | None = None,
) -> TaskState | None:
    """
    GOAL UNDERSTANDING + TASK CLASSIFICATION + PLANNING for one request,
    returning a TaskState ready for plan-aware execution.
    """
    classification = await classify_task(user_input, engine)
    if classification.task_type not in COMPLEX_TASK_TYPES:
        return None

    task = TaskState(goal=classification.goal, task_type=classification.task_type)
    if classification.clarification_required:
        task.require_clarification(classification.clarification_questions)
        return task

    ws = detect_workspace_kind(cwd)
    plan = await generate_task_plan(
        classification.goal,
        classification.task_type,
        engine,
        cwd=cwd,
    )
    if plan is None:
        logger.info("[PLAN_RECOVERY] using deterministic fallback plan for task_type=%s", classification.task_type.value)
        plan = fallback_plan(
            classification.goal,
            classification.task_type,
            ws,
            cwd=cwd,
        )
        plan.failure_recovery = (
            f"Initial LLM planner failed or was unavailable; recovered using deterministic {classification.task_type.value} execution plan."
        )
        report = validate_plan(plan)
        if report.valid:
            logger.info("[PLAN_SELECTED] deterministic_fallback (valid=True, %d steps)", len(plan.steps))
            _bind_user_intent_to_plan(plan, classification.goal, cwd)
        else:
            logger.error("[PLAN_VALIDATION_ERROR] deterministic fallback plan failed validation: %s", report.issues)

    task.attach_plan(plan)
    _attach_coding_context(task, cwd)
    return task


def _attach_coding_context(task: TaskState, cwd: str | None) -> None:
    """
    Attaches a Fix #3 CodeContext (workspace awareness) to a prepared task.
    """
    from ultron.core.coding.context import CodeContext
    from ultron.core.coding.workspace import discover_workspace

    target_dir = cwd
    if task.plan and getattr(task.plan, "project_dir", None):
        target_dir = task.plan.project_dir
    try:
        workspace = discover_workspace(target_dir)
    except (OSError, ValueError):
        workspace = None
    task.code_context = CodeContext(workspace=workspace)
    task.code_context.attach_task(task)
