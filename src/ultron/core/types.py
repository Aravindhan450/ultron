from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import Enum, unique
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Role(str, Enum):
    """
    Represents the sender's role in a conversation.
    """
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"

class PendingAction(BaseModel):
    """
    Represents an action requiring user interactive confirmation (e.g. running a command or overwriting a file).
    
    Design Choice:
    Instead of string matching on user messages back-and-forth, we attach a `pending_action` object to ChatMessage.
    This clearly signals to the CLI interface (main.py) that interactive confirmation via questionary is required.
    """
    # When adding a new tool that uses PendingAction, add its action_type string here too.
    action_type: Literal[
        "run_command",
        "run_parallel",
        "overwrite_file",
        "read_file",
        "write_file",
        "web_search",
        "fetch_page",
        "db_query",
        "execute_plan",
        # Fix #3 coding-edit actions (state-changing, gated like write_file)
        "create_file",
        "replace_file",
        "replace_in_file",
        "append_to_file",
        "delete_file",
        "rename_file",
        "discover_workspace_summary",
        "list_directory",
        "search_files",
    ]
    target: str          # The command string OR the filename/query/URL to act upon
    content: str | None = None  # Content to write if action_type is "write_file" or "overwrite_file"

class ChatMessage(BaseModel):
    """
    Represents a structured chat message for the Ultron assistant.
    """
    role: Role
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    pending_action: PendingAction | None = None  # Optional interactive confirmation request payload
    task_state: TaskState | None = None  # Optional task this message belongs to (survives confirmation)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def to_openai_format(self) -> dict[str, Any]:
        """
        Convert the ChatMessage to standard OpenAI compatible dict structure.
        """
        payload: dict[str, Any] = {
            "role": self.role.value,
            "content": self.content,
        }
        if self.name is not None:
            payload["name"] = self.name
        if self.tool_call_id is not None:
            payload["tool_call_id"] = self.tool_call_id
        return payload

class TaskStatus(str, Enum):
    """
    Lifecycle state of a user's overall task.

    Completion is explicit: a task only reaches ``TASK_COMPLETED`` via
    :meth:`TaskState.mark_complete` — never as a side effect of a tool
    succeeding, a command exiting 0, or the LLM producing a final-looking
    reply.
    """

    TASK_STARTED = "task_started"
    TASK_RUNNING = "task_running"
    WAITING_CONFIRMATION = "waiting_confirmation"
    TASK_FAILED = "task_failed"
    TASK_BLOCKED = "task_blocked"
    TASK_COMPLETED = "task_completed"

class TaskType(str, Enum):
    """
    High-level category of *what the user wants to accomplish*.

    Classification is goal-oriented ("what outcome does the user want?")
    rather than tool-oriented ("which tool should I call?"). A task type
    may map to many tools, and a tool may serve many task types — the
    taxonomy is about intent, never about the registry.
    """

    INFORMATIONAL = "informational"  # answer / explain; no actions required
    SIMPLE_ACTION = "simple_action"  # one straightforward action
    MULTI_STEP = "multi_step"  # general sequenced multi-action work
    SOFTWARE_ENGINEERING = (
        "software_engineering"  # build / implement / refactor / upgrade code
    )
    DEBUGGING = "debugging"  # diagnose and fix failures
    CODE_REVIEW = "code_review"  # inspect code and report findings
    RESEARCH = "research"  # investigate / understand / analyze
    SYSTEM_OPERATION = "system_operation"  # deploy / run / manage services
    FILE_OPERATION = "file_operation"  # file-system operations
    CONFIGURATION = "configuration"  # settings / env / config changes
    DATA_OPERATION = "data_operation"  # database / query / schema / data work

    @property
    def requires_actions(self) -> bool:
        """Whether accomplishing this task needs tool actions at all."""
        return self is not TaskType.INFORMATIONAL

class WorkspaceKind(str, Enum):
    """Whether a task targets a brand-new workspace or an existing project."""

    NEW_WORKSPACE = "new_workspace"
    EXISTING_PROJECT = "existing_project"
    UNKNOWN = "unknown"

class StepStatus(str, Enum):
    """Lifecycle state of a single plan step."""

    PENDING = "pending"
    RUNNING = "running"
    WAITING_CONFIRMATION = "waiting_confirmation"  # a gated action awaits approval
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"  # a dependency failed, so this step cannot run
    BLOCKED = "blocked"  # needs clarification / approval before running

class FailureStrategy(str, Enum):
    """How the executor should react when a plan step fails."""

    STOP = "stop"  # halt the plan on failure
    RETRY = "retry"  # retry up to ``retry_policy`` attempts
    SKIP = "skip"  # mark SKIPPED and continue with independent steps
    CONTINUE = "continue"  # record the failure but keep going

class EvidenceLevel(str, Enum):
    """
    Evidence level backing task acceptance and verification.

    LEVEL 0 — Generated: Code/files exist.
    LEVEL 1 — Created: Project structure exists in workspace.
    LEVEL 2 — Executed: Program actually ran (exit code 0 / invocation started).
    LEVEL 3 — Launched: Application became available (process running, port listening, GUI initialized).
    LEVEL 4 — Interacted: Real application behavior exercised with inputs/events.
    LEVEL 5 — Verified: Checked against an independent or deterministic source.
    LEVEL 6 — Repaired: Failure diagnosed, repair applied, and verification re-tested and passed.
    """

    LEVEL_0_GENERATED = "level_0_generated"
    LEVEL_1_CREATED = "level_1_created"
    LEVEL_2_EXECUTED = "level_2_executed"
    LEVEL_3_LAUNCHED = "level_3_launched"
    LEVEL_4_INTERACTED = "level_4_interacted"
    LEVEL_5_VERIFIED = "level_5_verified"
    LEVEL_6_REPAIRED = "level_6_repaired"


class AcceptanceCriterionStatus(str, Enum):
    """Lifecycle status of an individual acceptance criterion."""

    PENDING = "pending"
    RUNNING = "running"
    VERIFIED = "verified"
    FAILED = "failed"
    SKIPPED = "skipped"


class AcceptanceCriterion(BaseModel):
    """
    Explicit, evidence-backed acceptance criterion for autonomous task execution.
    """

    id: str
    description: str
    verification_method: str = "execution"
    required: bool = True
    status: AcceptanceCriterionStatus = AcceptanceCriterionStatus.PENDING
    evidence: str = ""
    evidence_level: EvidenceLevel = EvidenceLevel.LEVEL_0_GENERATED
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ApplicationLifecycleState(str, Enum):
    """
    Observable lifecycle state of an application created or managed by Ultron.
    """

    CREATED = "created"
    EXECUTED = "executed"
    STARTED = "started"
    READY = "ready"
    INTERACTED = "interacted"
    VERIFIED = "verified"


class ProductType(str, Enum):
    """
    Type of product / software artifact being engineered.
    """

    DESKTOP_GUI = "desktop_gui"
    WEB_APP = "web_app"
    CLI_TOOL = "cli_tool"
    REST_API = "rest_api"
    DATABASE_APP = "database_app"
    LIBRARY = "library"
    AUTOMATION_SCRIPT = "automation_script"
    DATA_PIPELINE = "data_pipeline"
    DOCUMENTATION = "documentation"
    GENERAL_SOFTWARE = "general_software"


class UserIntent(BaseModel):
    """
    Deep intent representation: translating natural language requests into
    concrete engineering goals, explicit & inferred requirements, and verification strategy.
    """

    raw_prompt: str = ""
    goal: str = ""
    product_type: ProductType = ProductType.GENERAL_SOFTWARE
    explicit_requirements: list[str] = Field(default_factory=list)
    inferred_necessary_requirements: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    ambiguities: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    acceptance_criteria: list[AcceptanceCriterion] = Field(default_factory=list)
    verification_strategy: str = ""


class TaskRequirement(BaseModel):
    """
    One explicit completion criterion for a task.

    Requirements are generic — not tied to any specific goal. Callers
    (agents / planners) decide what counts as "done" for a given request,
    e.g. "application can run" or "TodoList directory exists".
    """

    description: str
    completed: bool = False


class ToolExecution(BaseModel):
    """
    A single tool execution recorded in a task's execution history.

    Recording a successful tool execution never changes the task status on
    its own — an intermediate action is not task completion (see TaskState).
    """

    tool_name: str
    target: str = ""
    success: bool = True
    detail: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class TaskError(BaseModel):
    """A failure or blocking error recorded against a task."""

    message: str
    step: int | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class TaskClassification(BaseModel):
    """
    Result of GOAL UNDERSTANDING + TASK CLASSIFICATION.

    ``goal`` is the desired *outcome*, expressed independently of any tool
    call — for "Fix the failing authentication tests." the goal is "Make
    the authentication test suite pass." The first tool call is never the
    goal. ``task_type`` answers "what does the user want to accomplish?",
    not "which tool should I call?".
    """

    task_type: TaskType
    goal: str
    summary: str = ""
    clarification_required: bool = False
    clarification_questions: list[str] = Field(default_factory=list)
    user_intent: UserIntent | None = None

    @property
    def requires_actions(self) -> bool:
        """True unless this is a pure informational request."""
        return self.task_type.requires_actions

class PlanStep(BaseModel):
    """
    One outcome-oriented step in a TaskPlan.

    Steps describe *what must be accomplished* (description, purpose,
    expected outcome, completion criteria); the tool action is an
    implementation detail the executor may fill in later. Dependencies are
    explicit step ids — ordering is never left to the LLM's memory.
    """

    id: int
    description: str
    purpose: str = ""
    dependencies: list[int] = Field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    expected_outcome: str = ""
    completion_criteria: list[str] = Field(default_factory=list)
    failure_strategy: FailureStrategy = FailureStrategy.STOP
    retry_policy: int = 0
    attempts: int = 0  # failed tool executions recorded against this step
    action: str | None = None  # optional tool-action template (executor detail)
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: str | None = None
    error: str | None = None

class TaskPlan(BaseModel):
    """
    A structured, outcome-oriented decomposition of a user's goal.

    A plan is a real object owned by the TaskState — never kept only inside
    an LLM prompt. Because it is persisted with the task, it survives LLM
    turns, tool calls, confirmations, failures, retries, and agent
    continuation.
    """

    goal: str
    task_type: TaskType
    workspace: WorkspaceKind = WorkspaceKind.UNKNOWN
    working_context: str = ""
    assumptions: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    steps: list[PlanStep] = Field(default_factory=list)
    completion_criteria: list[str] = Field(default_factory=list)
    verification_requirements: list[str] = Field(default_factory=list)
    acceptance_criteria: list[AcceptanceCriterion] = Field(default_factory=list)
    user_intent: UserIntent | None = None
    failure_recovery: str = ""
    project_dir: str | None = None
    needs_clarification: bool = False
    clarification_questions: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # ------------------------------------------------------------------
    # Step queries
    # ------------------------------------------------------------------

    def step(self, step_id: int) -> PlanStep | None:
        """Returns the step with the given id, or None."""
        for step in self.steps:
            if step.id == step_id:
                return step
        return None

    def remaining_steps(self) -> list[PlanStep]:
        """Steps that still need work (pending / running / waiting / blocked)."""
        return [
            s
            for s in self.steps
            if s.status
            in (
                StepStatus.PENDING,
                StepStatus.RUNNING,
                StepStatus.WAITING_CONFIRMATION,
                StepStatus.BLOCKED,
            )
        ]

    def completed_steps(self) -> list[PlanStep]:
        """Steps that succeeded."""
        return [s for s in self.steps if s.status is StepStatus.SUCCEEDED]

    def failed_steps(self) -> list[PlanStep]:
        """Steps that failed."""
        return [s for s in self.steps if s.status is StepStatus.FAILED]

    def blocked_steps(self) -> list[PlanStep]:
        """Steps blocked pending clarification / approval."""
        return [s for s in self.steps if s.status is StepStatus.BLOCKED]

    def next_step(self) -> PlanStep | None:
        """
        The first PENDING step whose dependencies are all satisfied, or
        None when no step can run yet / nothing remains.
        """
        for step in self.steps:
            if step.status is not StepStatus.PENDING:
                continue
            if all(
                (dep := self.step(d)) is not None and dep.status is StepStatus.SUCCEEDED
                for d in step.dependencies
            ):
                return step
        return None

    def active_step(self) -> PlanStep | None:
        """
        The step currently being worked on: the RUNNING (or confirmation-
        waiting) step if there is one, otherwise the next PENDING step whose
        dependencies are satisfied.
        """
        for step in self.steps:
            if step.status in (
                StepStatus.RUNNING,
                StepStatus.WAITING_CONFIRMATION,
            ):
                return step
        return self.next_step()

    def set_step_status(
        self,
        step_id: int,
        status: StepStatus,
        result: str | None = None,
        error: str | None = None,
    ) -> PlanStep:
        """Updates a step's status (and optional result/error); raises if unknown."""
        step = self.step(step_id)
        if step is None:
            raise ValueError(f"Unknown plan step id: {step_id}")
        step.status = status
        if result is not None:
            step.result = result
        if error is not None:
            step.error = error
        self._touch()
        return step

    def is_satisfied(self) -> bool:
        """True when every step reached a non-failed terminal state."""
        return bool(self.steps) and all(
            s.status in (StepStatus.SUCCEEDED, StepStatus.SKIPPED) for s in self.steps
        )

    def revise_steps(self, new_steps: list[PlanStep]) -> bool:
        """
        ADAPTIVE PLANNING: replaces the REMAINING (non-terminal) steps with
        ``new_steps`` while preserving completed work.

        Completed steps (SUCCEEDED / SKIPPED) are kept exactly as recorded;
        only pending / running / failed work may be replaced. The revised
        step list is sanity-checked (unique ids, no self/unknown
        dependencies, no cycles, all steps reachable) and rejected — leaving
        the plan unchanged — when it is structurally invalid.

        Returns True when the plan was revised, False otherwise.
        """
        kept = [
            s
            for s in self.steps
            if s.status in (StepStatus.SUCCEEDED, StepStatus.SKIPPED)
        ]
        candidate = kept + list(new_steps)
        if not self._revision_is_valid(candidate):
            return False
        self.steps = candidate
        self._touch()
        return True

    @staticmethod
    def _revision_is_valid(steps: list[PlanStep]) -> bool:
        """Structural sanity for a step list: unique ids, sane deps, acyclic,
        reachable. Mirrors plan_validation's core checks so adaptive
        revisions cannot introduce an unrunnable plan."""
        if not steps:
            return False
        ids = [s.id for s in steps]
        if len(ids) != len(set(ids)):
            return False
        by_id = {s.id: s for s in steps}
        for step in steps:
            if any(d == step.id for d in step.dependencies):
                return False
            if any(d not in by_id for d in step.dependencies):
                return False
        # Cycle detection via DFS back-edge scan.
        visited: set[int] = set()
        path: list[int] = []

        def dfs(node: int) -> bool:
            if node in path:
                return True
            if node in visited:
                return False
            visited.add(node)
            path.append(node)
            for dep in by_id.get(node).dependencies if node in by_id else []:
                if dfs(dep):
                    return True
            path.pop()
            return False

        for step in steps:
            if dfs(step.id):
                return False
        # Reachability: every step must be reachable from a root.
        reached = {s.id for s in steps if not s.dependencies}
        changed = True
        while changed:
            changed = False
            for step in steps:
                if step.id in reached:
                    continue
                if any(d in reached for d in step.dependencies):
                    reached.add(step.id)
                    changed = True
        return len(reached) == len(steps)

    def _touch(self) -> None:
        self.updated_at = datetime.now(UTC)

class PlanValidationIssue(BaseModel):
    """One problem found while validating a TaskPlan."""

    code: str  # e.g. duplicate_step_id, circular_dependency, missing_verification
    message: str
    step_id: int | None = None

class PlanValidationReport(BaseModel):
    """
    Result of validating a TaskPlan before it may be executed.

    ``valid`` must be True for a plan to be executed; invalid plans are
    rejected rather than run half-correct.
    """

    valid: bool
    issues: list[PlanValidationIssue] = Field(default_factory=list)
    circular_dependencies: list[list[int]] = Field(default_factory=list)
    unreachable_steps: list[int] = Field(default_factory=list)

# ---------------------------------------------------------------------------
# Authoritative Canonical TaskState & Finite Lifecycle
# ---------------------------------------------------------------------------


@unique
class TaskLifecycleStatus(str, Enum):
    """
    Authoritative finite lifecycle states for an Ultron task.

    Progressive pipeline:
        CREATED -> UNDERSTANDING -> PLANNING -> READY -> EXECUTING -> VERIFYING -> COMPLETED
                                                            │             ▲
                                                            ▼             │
                                                        REPAIRING ────────┘

    Special non-terminal state:
        WAITING_CONFIRMATION (paused for interactive user approval)

    Terminal states:
        COMPLETED (successful goal satisfaction)
        FAILED (unrecoverable execution/planning error)
        BLOCKED (security or policy hard-stop)
        CANCELLED (user or cooperative cancellation)
    """

    CREATED = "created"
    UNDERSTANDING = "understanding"
    PLANNING = "planning"
    READY = "ready"
    EXECUTING = "executing"
    WAITING_CONFIRMATION = "waiting_confirmation"
    VERIFYING = "verifying"
    REPAIRING = "repairing"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        """True if the state is terminal (cannot transition out)."""
        return self in TERMINAL_LIFECYCLE_STATUSES

    @property
    def is_active(self) -> bool:
        """True if the task is actively progressing or paused waiting."""
        return self not in TERMINAL_LIFECYCLE_STATUSES


TERMINAL_LIFECYCLE_STATUSES: frozenset[TaskLifecycleStatus] = frozenset(
    {
        TaskLifecycleStatus.COMPLETED,
        TaskLifecycleStatus.CANCELLED,
    }
)


TASK_TRANSITIONS: dict[TaskLifecycleStatus, frozenset[TaskLifecycleStatus]] = {
    TaskLifecycleStatus.CREATED: frozenset(
        {
            TaskLifecycleStatus.UNDERSTANDING,
            TaskLifecycleStatus.PLANNING,
            TaskLifecycleStatus.READY,
            TaskLifecycleStatus.EXECUTING,
            TaskLifecycleStatus.WAITING_CONFIRMATION,
            TaskLifecycleStatus.COMPLETED,
            TaskLifecycleStatus.FAILED,
            TaskLifecycleStatus.CANCELLED,
            TaskLifecycleStatus.BLOCKED,
        }
    ),
    TaskLifecycleStatus.UNDERSTANDING: frozenset(
        {
            TaskLifecycleStatus.PLANNING,
            TaskLifecycleStatus.READY,
            TaskLifecycleStatus.EXECUTING,
            TaskLifecycleStatus.WAITING_CONFIRMATION,
            TaskLifecycleStatus.FAILED,
            TaskLifecycleStatus.CANCELLED,
            TaskLifecycleStatus.BLOCKED,
        }
    ),
    TaskLifecycleStatus.PLANNING: frozenset(
        {
            TaskLifecycleStatus.READY,
            TaskLifecycleStatus.EXECUTING,
            TaskLifecycleStatus.PLANNING,  # adaptive replanning
            TaskLifecycleStatus.WAITING_CONFIRMATION,
            TaskLifecycleStatus.COMPLETED,
            TaskLifecycleStatus.FAILED,
            TaskLifecycleStatus.CANCELLED,
            TaskLifecycleStatus.BLOCKED,
        }
    ),
    TaskLifecycleStatus.READY: frozenset(
        {
            TaskLifecycleStatus.EXECUTING,
            TaskLifecycleStatus.PLANNING,
            TaskLifecycleStatus.WAITING_CONFIRMATION,
            TaskLifecycleStatus.COMPLETED,
            TaskLifecycleStatus.FAILED,
            TaskLifecycleStatus.CANCELLED,
            TaskLifecycleStatus.BLOCKED,
        }
    ),
    TaskLifecycleStatus.EXECUTING: frozenset(
        {
            TaskLifecycleStatus.WAITING_CONFIRMATION,
            TaskLifecycleStatus.VERIFYING,
            TaskLifecycleStatus.REPAIRING,
            TaskLifecycleStatus.COMPLETED,
            TaskLifecycleStatus.EXECUTING,  # step-to-step loop
            TaskLifecycleStatus.FAILED,
            TaskLifecycleStatus.CANCELLED,
            TaskLifecycleStatus.BLOCKED,
        }
    ),
    TaskLifecycleStatus.WAITING_CONFIRMATION: frozenset(
        {
            TaskLifecycleStatus.EXECUTING,
            TaskLifecycleStatus.FAILED,
            TaskLifecycleStatus.CANCELLED,
            TaskLifecycleStatus.BLOCKED,
        }
    ),
    TaskLifecycleStatus.VERIFYING: frozenset(
        {
            TaskLifecycleStatus.COMPLETED,
            TaskLifecycleStatus.REPAIRING,
            TaskLifecycleStatus.EXECUTING,
            TaskLifecycleStatus.WAITING_CONFIRMATION,
            TaskLifecycleStatus.FAILED,
            TaskLifecycleStatus.CANCELLED,
            TaskLifecycleStatus.BLOCKED,
        }
    ),
    TaskLifecycleStatus.REPAIRING: frozenset(
        {
            TaskLifecycleStatus.EXECUTING,
            TaskLifecycleStatus.PLANNING,
            TaskLifecycleStatus.VERIFYING,
            TaskLifecycleStatus.WAITING_CONFIRMATION,
            TaskLifecycleStatus.FAILED,
            TaskLifecycleStatus.CANCELLED,
            TaskLifecycleStatus.BLOCKED,
        }
    ),
    # Terminal / error states
    TaskLifecycleStatus.COMPLETED: frozenset(),
    TaskLifecycleStatus.CANCELLED: frozenset(),
    # FAILED and BLOCKED can transition to REPAIRING on explicit repair request or CANCELLED
    TaskLifecycleStatus.FAILED: frozenset(
        {
            TaskLifecycleStatus.REPAIRING,
            TaskLifecycleStatus.CANCELLED,
        }
    ),
    TaskLifecycleStatus.BLOCKED: frozenset(
        {
            TaskLifecycleStatus.REPAIRING,
            TaskLifecycleStatus.CANCELLED,
        }
    ),
}


class InvalidStateTransitionError(ValueError):
    """Raised when an illegal lifecycle transition is attempted."""

    def __init__(
        self, current: TaskLifecycleStatus, target: TaskLifecycleStatus, reason: str = ""
    ) -> None:
        self.current = current
        self.target = target
        self.reason = reason
        allowed = [s.value for s in TASK_TRANSITIONS.get(current, frozenset())]
        msg = (
            f"Illegal task state transition from '{current.value}' to '{target.value}'."
            f" Allowed transitions: {allowed}"
        )
        if reason:
            msg += f" (reason: {reason})"
        super().__init__(msg)


def assert_task_transition(
    current: TaskLifecycleStatus, target: TaskLifecycleStatus, reason: str = ""
) -> None:
    """
    Validates that a transition from `current` to `target` is legal.
    Raises InvalidStateTransitionError on invalid transitions.
    """
    allowed = TASK_TRANSITIONS.get(current, frozenset())
    if target not in allowed:
        raise InvalidStateTransitionError(current, target, reason=reason)


class CanonicalTaskState(BaseModel):
    """
    The ONE authoritative runtime state model for an Ultron task.

    Unifies identity, lifecycle status, planning decomposition, execution history,
    verification evidence, repair state, and recovery references.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Identity
    task_id: str = Field(default_factory=lambda: f"task_{uuid.uuid4().hex[:8]}")
    parent_task_id: str | None = None
    session_id: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # Lifecycle & status
    lifecycle_status: TaskLifecycleStatus = TaskLifecycleStatus.CREATED
    lifecycle_history: list[str] = Field(
        default_factory=lambda: [TaskLifecycleStatus.CREATED.value]
    )
    status: TaskStatus = TaskStatus.TASK_STARTED  # Backwards compatibility view
    current_phase: str = "init"
    current_step: int = 0
    total_steps: int = 0

    def __init__(
        self,
        goal: str = "",
        task_id: str | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if goal and "goal" not in kwargs:
            kwargs["goal"] = goal
        if task_id is not None:
            kwargs["task_id"] = task_id
        tid = kwargs.get("task_id")
        g = kwargs.get("goal", "")
        # Enforce stable task_id format and prevent task_id from equaling goal
        if not tid or tid == g or not str(tid).startswith("task_"):
            kwargs["task_id"] = f"task_{uuid.uuid4().hex[:8]}"

        super().__init__(**kwargs)
        if "lifecycle_history" not in kwargs:
            self.lifecycle_history = [self.lifecycle_status.value]

    # Planning
    goal: str
    task_type: TaskType | None = None
    plan: TaskPlan | None = None
    current_plan_step_id: int | None = None
    completed_step_ids: list[int] = Field(default_factory=list)
    blocked_step_ids: list[int] = Field(default_factory=list)
    requirements: list[TaskRequirement] = Field(default_factory=list)
    acceptance_criteria: list[AcceptanceCriterion] = Field(default_factory=list)
    clarification_required: bool = False
    clarification_questions: list[str] = Field(default_factory=list)
    plan_revisions: list[str] = Field(default_factory=list)
    user_intent: UserIntent | None = None

    # Execution tracking
    active_action_ids: list[str] = Field(default_factory=list)
    completed_action_ids: list[str] = Field(default_factory=list)
    failed_action_ids: list[str] = Field(default_factory=list)
    execution_history: list[ToolExecution] = Field(default_factory=list)
    context: list[ChatMessage] = Field(default_factory=list)
    pending_action: PendingAction | None = None
    last_observation: str | None = None
    app_lifecycle_state: ApplicationLifecycleState | None = None

    # Verification
    requires_verification: bool = False
    verification_status: str | None = None  # None, "started", "passed", "failed"
    last_verification_result: dict[str, Any] | None = None

    # Repair
    repair_attempts: int = 0
    max_repair_attempts: int = 5
    repair_budget_state: dict[str, Any] = Field(default_factory=dict)

    # Model & Routing
    selected_model: str | None = None
    routing_decision: dict[str, Any] | None = None

    # Repository & Environment
    workspace_root: str | None = None
    relevant_files: list[str] = Field(default_factory=list)
    code_context: CodeContext | None = None

    # Error & Failure
    errors: list[TaskError] = Field(default_factory=list)
    failure_classification: str | None = None

    # Recovery
    checkpoint_id: str | None = None
    recovery_state: dict[str, Any] | None = None

    def _touch(self) -> None:
        self.updated_at = datetime.now(UTC)

    # ------------------------------------------------------------------
    # Explicit Lifecycle Transitions
    # ------------------------------------------------------------------

    def transition_to(
        self, target: TaskLifecycleStatus, reason: str = ""
    ) -> TaskLifecycleStatus:
        """
        Validates and transitions the task into a target lifecycle status.
        Synchronizes legacy `status` view for backwards compatibility.
        """
        assert_task_transition(self.lifecycle_status, target, reason=reason)
        self.lifecycle_status = target
        if not self.lifecycle_history or self.lifecycle_history[-1] != target.value:
            self.lifecycle_history.append(target.value)

        # Synchronize backward-compatible TaskStatus
        if target in (
            TaskLifecycleStatus.CREATED,
            TaskLifecycleStatus.UNDERSTANDING,
            TaskLifecycleStatus.PLANNING,
            TaskLifecycleStatus.READY,
        ):
            self.status = TaskStatus.TASK_STARTED
        elif target in (
            TaskLifecycleStatus.EXECUTING,
            TaskLifecycleStatus.VERIFYING,
            TaskLifecycleStatus.REPAIRING,
        ):
            self.status = TaskStatus.TASK_RUNNING
        elif target == TaskLifecycleStatus.WAITING_CONFIRMATION:
            self.status = TaskStatus.WAITING_CONFIRMATION
        elif target == TaskLifecycleStatus.COMPLETED:
            self.status = TaskStatus.TASK_COMPLETED
        elif target in (TaskLifecycleStatus.FAILED, TaskLifecycleStatus.CANCELLED):
            self.status = TaskStatus.TASK_FAILED
        elif target == TaskLifecycleStatus.BLOCKED:
            self.status = TaskStatus.TASK_BLOCKED

        self._touch()
        return self.lifecycle_status

    # ------------------------------------------------------------------
    # Requirements
    # ------------------------------------------------------------------

    def add_requirement(self, description: str) -> TaskRequirement:
        """Adds one completion criterion to the task."""
        if any(r.description == description for r in self.requirements):
            raise ValueError(f"Requirement already exists: '{description}'")
        requirement = TaskRequirement(description=description)
        self.requirements.append(requirement)
        self._touch()
        return requirement

    def add_requirements(self, descriptions: list[str]) -> None:
        """Adds several completion criteria in one call."""
        for description in descriptions:
            self.add_requirement(description)

    def mark_requirement_complete(self, description: str) -> TaskRequirement:
        """Marks a completion criterion as satisfied; raises if unknown."""
        requirement = self._find_requirement(description)
        requirement.completed = True
        self._touch()
        return requirement

    def mark_requirement_incomplete(self, description: str) -> TaskRequirement:
        """Reverts a completion criterion back to unsatisfied; raises if unknown."""
        requirement = self._find_requirement(description)
        requirement.completed = False
        self._touch()
        return requirement

    def _find_requirement(self, description: str) -> TaskRequirement:
        for requirement in self.requirements:
            if requirement.description == description:
                return requirement
        raise ValueError(f"Unknown requirement: '{description}'")

    @property
    def completed_requirements(self) -> list[TaskRequirement]:
        return [r for r in self.requirements if r.completed]

    def remaining_requirements(self) -> list[TaskRequirement]:
        return [r for r in self.requirements if not r.completed]

    # ------------------------------------------------------------------
    # Acceptance Criteria
    # ------------------------------------------------------------------

    def add_acceptance_criterion(
        self,
        id: str,
        description: str,
        verification_method: str = "execution",
        required: bool = True,
    ) -> AcceptanceCriterion:
        for crit in self.acceptance_criteria:
            if crit.id == id:
                crit.description = description
                crit.verification_method = verification_method
                crit.required = required
                self._touch()
                return crit
        criterion = AcceptanceCriterion(
            id=id,
            description=description,
            verification_method=verification_method,
            required=required,
        )
        self.acceptance_criteria.append(criterion)
        self._touch()
        return criterion

    def verify_acceptance_criterion(
        self,
        id: str,
        evidence: str,
        level: EvidenceLevel = EvidenceLevel.LEVEL_4_INTERACTED,
    ) -> AcceptanceCriterion:
        for crit in self.acceptance_criteria:
            if crit.id == id:
                crit.status = AcceptanceCriterionStatus.VERIFIED
                crit.evidence = evidence
                crit.evidence_level = level
                crit.timestamp = datetime.now(UTC)
                self._touch()
                return crit
        crit = AcceptanceCriterion(
            id=id,
            description=id,
            status=AcceptanceCriterionStatus.VERIFIED,
            evidence=evidence,
            evidence_level=level,
        )
        self.acceptance_criteria.append(crit)
        self._touch()
        return crit

    def fail_acceptance_criterion(
        self, id: str, error: str
    ) -> AcceptanceCriterion:
        for crit in self.acceptance_criteria:
            if crit.id == id:
                crit.status = AcceptanceCriterionStatus.FAILED
                crit.evidence = error
                self._touch()
                return crit
        crit = AcceptanceCriterion(
            id=id,
            description=id,
            status=AcceptanceCriterionStatus.FAILED,
            evidence=error,
        )
        self.acceptance_criteria.append(crit)
        self._touch()
        return crit

    def all_required_criteria_satisfied(self) -> bool:
        return all(
            crit.status == AcceptanceCriterionStatus.VERIFIED
            for crit in self.acceptance_criteria
            if crit.required
        )

    def remaining_acceptance_criteria(self) -> list[AcceptanceCriterion]:
        return [
            crit
            for crit in self.acceptance_criteria
            if crit.required and crit.status != AcceptanceCriterionStatus.VERIFIED
        ]

    # ------------------------------------------------------------------
    # Step & Plan Tracking
    # ------------------------------------------------------------------

    def set_current_step(self, step: int) -> None:
        if step < 0:
            raise ValueError(f"step must be >= 0, got {step}")
        self.current_step = step
        self.current_plan_step_id = step
        self._touch()

    def set_total_steps(self, total: int) -> None:
        if total < 0:
            raise ValueError(f"total must be >= 0, got {total}")
        self.total_steps = total
        self._touch()


    # ------------------------------------------------------------------
    # Execution History & Actions
    # ------------------------------------------------------------------

    def record_tool_execution(
        self,
        tool_name: str,
        target: str = "",
        success: bool = True,
        detail: str = "",
    ) -> ToolExecution:
        entry = ToolExecution(
            tool_name=tool_name, target=target, success=success, detail=detail
        )
        self.execution_history.append(entry)
        self._touch()
        return entry

    def record_failure(self, message: str, step: int | None = None) -> TaskError:
        error = TaskError(message=message, step=step)
        self.errors.append(error)
        if self.lifecycle_status not in (
            TaskLifecycleStatus.COMPLETED,
            TaskLifecycleStatus.BLOCKED,
        ):
            self.transition_to(TaskLifecycleStatus.FAILED, reason=message)
        self._touch()
        return error

    def wait_for_confirmation(self) -> None:
        self.transition_to(
            TaskLifecycleStatus.WAITING_CONFIRMATION, reason="waiting user confirmation"
        )

    def resume(self) -> None:
        self.transition_to(TaskLifecycleStatus.EXECUTING, reason="resumed from confirmation")

    def transition_to_repair(self) -> None:
        self.repair_attempts += 1
        self.transition_to(TaskLifecycleStatus.REPAIRING, reason="starting repair attempt")
        if self.plan is not None:
            for step in self.plan.steps:
                if step.status is StepStatus.FAILED:
                    step.status = StepStatus.RUNNING
                    self.set_current_step(step.id)

    def block(self, message: str | None = None) -> None:
        if message:
            self.errors.append(TaskError(message=message))
        self.transition_to(TaskLifecycleStatus.BLOCKED, reason=message or "blocked by policy")

    def mark_complete(self) -> None:
        """
        Explicitly completes the task — validates completion criteria and transitions to COMPLETED.
        """
        if self.lifecycle_status in (
            TaskLifecycleStatus.FAILED,
            TaskLifecycleStatus.BLOCKED,
            TaskLifecycleStatus.CANCELLED,
        ):
            raise ValueError(
                f"Cannot complete a task in state '{self.lifecycle_status.value}'"
            )
        if self.plan is not None and not self.plan.is_satisfied():
            pending = self.plan.remaining_steps()
            names = ", ".join(f"step {s.id}" for s in pending)
            raise ValueError(
                f"Cannot complete task while plan steps are unfinished: {names}"
            )
        remaining = self.remaining_requirements()
        if remaining:
            missing = ", ".join(f"'{r.description}'" for r in remaining)
            raise ValueError(
                f"Cannot complete task with incomplete requirements: {missing}"
            )
        if not self.all_required_criteria_satisfied():
            unmet = self.remaining_acceptance_criteria()
            names = ", ".join(f"{c.id} ('{c.description}')" for c in unmet)
            raise ValueError(
                f"Cannot complete task with unsatisfied acceptance criteria: {names}"
            )

        self.transition_to(TaskLifecycleStatus.COMPLETED, reason="all criteria satisfied")

    def is_complete(self) -> bool:
        return (
            (self.lifecycle_status == TaskLifecycleStatus.COMPLETED or self.status == TaskStatus.TASK_COMPLETED)
            and not self.remaining_requirements()
            and self.all_required_criteria_satisfied()
        )

    @property
    def started_at(self) -> datetime:
        return self.created_at

    @property
    def is_blocked(self) -> bool:
        return self.lifecycle_status == TaskLifecycleStatus.BLOCKED or self.status == TaskStatus.TASK_BLOCKED

    @property
    def is_waiting_confirmation(self) -> bool:
        """True when the task is paused for user approval."""
        return (
            self.lifecycle_status == TaskLifecycleStatus.WAITING_CONFIRMATION
            or self.status == TaskStatus.WAITING_CONFIRMATION
        )

    def summary(self) -> str:
        """Compact one-line description for logs and debugging."""
        done = len(self.completed_requirements)
        total = len(self.requirements)
        crit_done = len(
            [c for c in self.acceptance_criteria if c.status == AcceptanceCriterionStatus.VERIFIED]
        )
        crit_total = len(self.acceptance_criteria)
        return (
            f"TaskState(goal='{self.goal}', status={self.status.value}, "
            f"lifecycle_status={self.lifecycle_status.value}, "
            f"requirements={done}/{total}, criteria={crit_done}/{crit_total}, "
            f"step={self.current_step}/{self.total_steps}, "
            f"tools={len(self.execution_history)}, errors={len(self.errors)})"
        )

    # ------------------------------------------------------------------
    # Structured plan integration (Fix #2)
    # ------------------------------------------------------------------

    def attach_plan(self, plan: TaskPlan) -> None:
        """
        Attaches a structured plan to the task and seeds its completion
        criteria from the plan (plan-level criteria + verification
        requirements + explicit acceptance criteria).
        """
        self.plan = plan
        self.task_type = plan.task_type
        self.user_intent = plan.user_intent
        self.set_total_steps(len(plan.steps))
        seen: set[str] = set()
        for description in [*plan.completion_criteria, *plan.verification_requirements]:
            description = description.strip()
            if description and description not in seen:
                seen.add(description)
                self.add_requirement(description)
        if plan.acceptance_criteria:
            for crit in plan.acceptance_criteria:
                self.add_acceptance_criterion(
                    id=crit.id,
                    description=crit.description,
                    verification_method=crit.verification_method,
                    required=crit.required,
                )
        if self.lifecycle_status in (
            TaskLifecycleStatus.CREATED,
            TaskLifecycleStatus.UNDERSTANDING,
        ):
            self.transition_to(TaskLifecycleStatus.PLANNING, reason="plan attached")
        if plan.needs_clarification:
            self.require_clarification(plan.clarification_questions)
        self._touch()

    def require_clarification(self, questions: list[str] | None = None) -> None:
        """
        Blocks the task pending user clarification. Moves the task to
        BLOCKED so it cannot proceed without answers.
        """
        self.clarification_required = True
        if questions:
            self.clarification_questions = list(
                dict.fromkeys(q for q in questions if q)
            )
        if self.lifecycle_status != TaskLifecycleStatus.BLOCKED:
            self.block("Task requires clarification before it can proceed.")
        self._touch()

    def adapt_plan(self, new_steps: list[Any]) -> bool:
        """
        ADAPTIVE PLANNING: replaces remaining steps, preserving completed work.
        """
        if self.plan is None:
            return False
        total_before = len(self.plan.steps)
        if not self.plan.revise_steps(new_steps):
            return False
        completed = len(self.plan.completed_steps())
        self.set_total_steps(len(self.plan.steps))
        self.record_plan_revision(
            f"Remaining plan steps revised ({completed}/{total_before} completed "
            f"steps preserved; {len(new_steps)} replacement step(s))."
        )
        return True

    def current_plan_step(self) -> Any | None:
        """The step being worked on (RUNNING) or the next step that can run."""
        return self.plan.active_step() if self.plan else None

    def remaining_steps(self) -> list[Any]:
        return self.plan.remaining_steps() if self.plan else []

    def completed_steps(self) -> list[Any]:
        return self.plan.completed_steps() if self.plan else []

    def failed_steps(self) -> list[Any]:
        return self.plan.failed_steps() if self.plan else []

    def blocked_steps(self) -> list[Any]:
        return self.plan.blocked_steps() if self.plan else []

    plan_completed_steps = completed_steps
    plan_failed_steps = failed_steps
    plan_blocked_steps = blocked_steps

    def record_plan_revision(self, note: str) -> None:
        self.plan_revisions.append(note)
        self._touch()


# Primary Canonical Alias
TaskState = CanonicalTaskState


def truncate_history(history: list[ChatMessage], max_messages: int = 10) -> list[ChatMessage]:
    """
    Truncates conversation history to stay within message limits.
    
    Preserves all leading SYSTEM messages at the start of history, 
    and keeps up to the last `max_messages` non-system messages.
    """
    leading_system: list[ChatMessage] = []
    index = 0
    # Collect all consecutive system messages at the start of the list
    while index < len(history) and history[index].role == Role.SYSTEM:
        leading_system.append(history[index])
        index += 1

    # Extract all non-system messages that follow
    non_system = [msg for msg in history[index:] if msg.role != Role.SYSTEM]
    truncated_non_system = non_system[-max_messages:]

    return leading_system + truncated_non_system

def history_to_openai_format(history: list[ChatMessage]) -> list[dict[str, Any]]:
    """
    Map history list of ChatMessage instances to OpenAI compatible dictionary list.
    """
    return [msg.to_openai_format() for msg in history]


# Late import: TaskState.code_context references the coding CodeContext, which
# lives in the coding package. Importing it here (after all model classes are
# defined) and rebuilding the model resolves the forward reference without a
# circular import (coding modules never import ultron.core.types at runtime).
from ultron.core.coding.context import CodeContext

TaskState.model_rebuild()
