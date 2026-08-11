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
from ultron.core.types import (
    FailureStrategy,
    PlanStep,
    TaskPlan,
    TaskState,
    TaskType,
    WorkspaceKind,
)

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
        "Create a plan to accomplish the user's goal.\n\n"
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
    try:
        payload = json.loads(_strip_fences(raw))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("steps"), list):
        return None

    steps: list[PlanStep] = []
    for item in payload["steps"]:
        if not isinstance(item, dict):
            return None
        step_id = item.get("id")
        description = item.get("description")
        if not isinstance(step_id, int) or not isinstance(description, str) or not description.strip():
            return None
        strategy = str(item.get("failure_strategy", "stop")).lower()
        try:
            failure_strategy = FailureStrategy(strategy)
        except ValueError:
            failure_strategy = FailureStrategy.STOP
        steps.append(
            PlanStep(
                id=step_id,
                description=description.strip(),
                purpose=str(item.get("purpose", "")).strip(),
                dependencies=[
                    int(d) for d in item.get("dependencies", []) if isinstance(d, int)
                ],
                expected_outcome=str(item.get("expected_outcome", "")).strip(),
                completion_criteria=[
                    str(c).strip() for c in item.get("completion_criteria", []) if str(c).strip()
                ],
                failure_strategy=failure_strategy,
                retry_policy=int(item.get("retry_policy", 0) or 0),
            )
        )

    return TaskPlan(
        goal=goal,
        task_type=task_type,
        workspace=workspace,
        working_context=working_context,
        assumptions=[str(a).strip() for a in payload.get("assumptions", []) if str(a).strip()],
        constraints=[str(c).strip() for c in payload.get("constraints", []) if str(c).strip()],
        steps=steps,
        completion_criteria=[
            str(c).strip() for c in payload.get("completion_criteria", []) if str(c).strip()
        ],
        verification_requirements=[
            str(v).strip()
            for v in payload.get("verification_requirements", [])
            if str(v).strip()
        ],
        failure_recovery=str(payload.get("failure_recovery", "")).strip(),
        needs_clarification=bool(payload.get("needs_clarification", False)),
        clarification_questions=[
            str(q).strip()
            for q in payload.get("clarification_questions", [])
            if str(q).strip()
        ],
    )


def fallback_plan(
    goal: str,
    task_type: TaskType,
    workspace: WorkspaceKind = WorkspaceKind.UNKNOWN,
) -> TaskPlan:
    """
    A minimal, always-valid plan used when the LLM plan cannot be produced.

    It contains a single verification step so the completion policy stays
    intact: the task must be verified against the goal before it can ever
    be reported complete.
    """
    return TaskPlan(
        goal=goal,
        task_type=task_type,
        workspace=workspace,
        steps=[
            PlanStep(
                id=1,
                description="Verify the final user goal",
                purpose=(
                    "Ensure the original request is actually satisfied before "
                    "reporting completion"
                ),
                expected_outcome="The original user goal is satisfied",
                completion_criteria=[goal],
                failure_strategy=FailureStrategy.STOP,
            )
        ],
        completion_criteria=[goal],
        verification_requirements=[f"The user goal is satisfied: {goal}"],
        failure_recovery=(
            "Stop on the first failure; the task must not report completion "
            "until the goal has been verified."
        ),
    )


async def generate_task_plan(
    goal: str,
    task_type: TaskType,
    engine,
    workspace: WorkspaceKind | None = None,
    working_context: str | None = None,
    cwd: str | None = None,
) -> TaskPlan | None:
    """
    Generates a validated, outcome-oriented TaskPlan for a goal.

    - Informational requests never get a plan (returns None).
    - The workspace is probed from ``cwd`` (or the process CWD) unless
      explicitly provided.
    - The generated plan is validated; invalid plans are rejected (returns
      None) so callers can fall back to :func:`fallback_plan`.
    """
    if task_type is TaskType.INFORMATIONAL:
        return None

    ws = workspace or detect_workspace_kind(cwd)
    context = (
        working_context if working_context is not None else probe_working_context(cwd)
    )
    prompt = build_planning_prompt(goal, task_type, ws, context)

    try:
        raw = await engine.generate([{"role": "user", "content": prompt}])
    except Exception:  # noqa: BLE001 — planning failures fall back to the fallback plan
        return None

    plan = parse_plan_json(raw, goal, task_type, ws, context)
    if plan is None:
        return None
    if not validate_plan(plan).valid:
        return None
    return plan


async def prepare_task_for_execution(
    user_input: str,
    engine,
    cwd: str | None = None,
) -> TaskState | None:
    """
    GOAL UNDERSTANDING + TASK CLASSIFICATION + PLANNING for one request,
    returning a TaskState ready for plan-aware execution.

    - Informational / simple-action / file-operation requests return None —
      the fast path stays fast (no plan, no extra LLM call).
    - Requests that genuinely need clarification return a BLOCKED TaskState
      whose questions the UI should surface before executing.
    - Everything else (multi-step, software engineering, debugging, code
      review, research, system/config/data work) gets a validated structured
      plan attached to a TaskState — the plan is the source of truth the
      executor follows.

    The plan is generated by the LLM planner (or the deterministic fallback
    plan) and validated; a plan that fails validation falls back to the
    fallback plan so execution never runs unplanned.
    """
    classification = await classify_task(user_input, engine)
    if classification.task_type not in COMPLEX_TASK_TYPES:
        return None

    task = TaskState(goal=classification.goal, task_type=classification.task_type)
    if classification.clarification_required:
        task.require_clarification(classification.clarification_questions)
        return task

    plan = await generate_task_plan(
        classification.goal,
        classification.task_type,
        engine,
        cwd=cwd,
    )
    if plan is None:
        plan = fallback_plan(
            classification.goal,
            classification.task_type,
            detect_workspace_kind(cwd),
        )
    task.attach_plan(plan)
    return task
