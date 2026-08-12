from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


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
    failure_recovery: str = ""
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

class TaskState(BaseModel):
    """
    Explicit representation of a user's overall task and its completion state.

    This is the structured alternative to letting an intermediate tool action
    (mkdir, a file write, exit code 0) implicitly end a request. A TaskState
    owns:

    - the original user goal (``goal``),
    - the lifecycle status (``status``),
    - explicit completion criteria (``requirements``),
    - step tracking (``current_step`` / ``total_steps``),
    - an execution history (``execution_history``),
    - failures / blocking errors (``errors``).

    Completion is explicit: :meth:`mark_complete` is the only path to
    ``TASK_COMPLETED``, and it refuses to complete a task that still has
    incomplete requirements, is blocked, or has failed.

    Instances are pydantic models, so they serialize losslessly via
    ``model_dump()`` / ``model_dump_json()`` for logging and debugging. A
    TaskState is created per task — no global mutable state is involved.
    """

    goal: str
    status: TaskStatus = TaskStatus.TASK_STARTED
    requirements: list[TaskRequirement] = Field(default_factory=list)
    current_step: int = 0
    total_steps: int = 0
    execution_history: list[ToolExecution] = Field(default_factory=list)
    errors: list[TaskError] = Field(default_factory=list)
    # --- ReAct task-continuation state (populated by the agent) ---
    context: list[ChatMessage] = Field(
        default_factory=list
    )  # task transcript: goal + assistant/tool pairs
    pending_action: PendingAction | None = None  # action awaiting confirmation
    last_observation: str | None = None  # latest confirmed tool result, fed back on resume
    requires_verification: bool = (
        False  # set once a state-changing action is gated on this task
    )
    # --- Task understanding / structured plan (Fix #2) ---
    task_type: TaskType | None = None  # what the user wants to accomplish
    plan: TaskPlan | None = None  # structured, outcome-oriented decomposition
    clarification_required: bool = False  # planner could not proceed safely
    clarification_questions: list[str] = Field(default_factory=list)
    plan_revisions: list[str] = Field(default_factory=list)  # adaptive-plan audit trail
    # --- Coding workspace / execution context (Fix #3 stage 1) ---
    # Structured, coding-specific context (workspace, relevant files,
    # observations, modifications) — separate from the raw transcript.
    code_context: CodeContext | None = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # ------------------------------------------------------------------
    # Requirements (completion criteria)
    # ------------------------------------------------------------------

    def add_requirement(self, description: str) -> TaskRequirement:
        """
        Adds one completion criterion to the task.

        Descriptions must be unique — a duplicate would make
        mark_requirement_complete() ambiguous, so it is rejected up front.
        """
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
        """All completion criteria satisfied so far."""
        return [r for r in self.requirements if r.completed]

    def remaining_requirements(self) -> list[TaskRequirement]:
        """Completion criteria not yet satisfied."""
        return [r for r in self.requirements if not r.completed]

    # ------------------------------------------------------------------
    # Step tracking
    # ------------------------------------------------------------------

    def set_current_step(self, step: int) -> None:
        """Records the step the task is currently working on (1-based; 0 = none)."""
        if step < 0:
            raise ValueError(f"step must be >= 0, got {step}")
        self.current_step = step
        self._touch()

    def set_total_steps(self, total: int) -> None:
        """Records how many steps the task is expected to have in total."""
        if total < 0:
            raise ValueError(f"total must be >= 0, got {total}")
        self.total_steps = total
        self._touch()

    # ------------------------------------------------------------------
    # Execution history + failures
    # ------------------------------------------------------------------

    def record_tool_execution(
        self,
        tool_name: str,
        target: str = "",
        success: bool = True,
        detail: str = "",
    ) -> ToolExecution:
        """
        Appends one tool execution to the task history.

        This never changes the task status: a tool succeeding is an
        intermediate event, not task completion. Use mark_complete() to
        finish the task explicitly. A failed tool run should be recorded
        here with success=False for history, but if the failure should
        also move the task to TASK_FAILED, call record_failure() explicitly.
        """
        entry = ToolExecution(
            tool_name=tool_name, target=target, success=success, detail=detail
        )
        self.execution_history.append(entry)
        self._touch()
        return entry

    def record_failure(self, message: str, step: int | None = None) -> TaskError:
        """
        Records a failure and moves the task to TASK_FAILED.

        A completed task is never downgraded to failed; a blocked task stays
        blocked (blocking is the stronger condition).
        """
        error = TaskError(message=message, step=step)
        self.errors.append(error)
        if self.status not in (
            TaskStatus.TASK_COMPLETED,
            TaskStatus.TASK_BLOCKED,
        ):
            self.status = TaskStatus.TASK_FAILED
        self._touch()
        return error

    # ------------------------------------------------------------------
    # Lifecycle transitions
    # ------------------------------------------------------------------

    def wait_for_confirmation(self) -> None:
        """
        Marks the task as waiting on the user to approve an action.

        Raises if the task is already in a terminal state.
        """
        if self.status in (
            TaskStatus.TASK_COMPLETED,
            TaskStatus.TASK_FAILED,
            TaskStatus.TASK_BLOCKED,
        ):
            raise ValueError(
                f"Cannot wait for confirmation from terminal state '{self.status.value}'"
            )
        self.status = TaskStatus.WAITING_CONFIRMATION
        self._touch()

    def resume(self) -> None:
        """
        Resumes a task after the user confirmed (or a pause).

        Allowed from TASK_STARTED / TASK_RUNNING / WAITING_CONFIRMATION;
        terminal states (completed, failed, blocked) cannot resume.
        """
        if self.status in (
            TaskStatus.TASK_COMPLETED,
            TaskStatus.TASK_FAILED,
            TaskStatus.TASK_BLOCKED,
        ):
            raise ValueError(
                f"Cannot resume a task in terminal state '{self.status.value}'"
            )
        self.status = TaskStatus.TASK_RUNNING
        self._touch()

    def block(self, message: str | None = None) -> None:
        """
        Hard-stops the task (e.g. a security block) and moves it to
        TASK_BLOCKED. A completed task cannot be blocked retroactively.
        """
        if self.status == TaskStatus.TASK_COMPLETED:
            raise ValueError("Cannot block a completed task")
        self.status = TaskStatus.TASK_BLOCKED
        if message:
            self.errors.append(TaskError(message=message))
        self._touch()

    def mark_complete(self) -> None:
        """
        Explicitly completes the task — the only path to TASK_COMPLETED.

        Refuses when requirements are still incomplete, the task has failed /
        been blocked, or (with a structured plan attached) any plan step is
        still pending / running / failed — a task whose plan is not fully
        satisfied must never report success.
        """
        if self.status in (TaskStatus.TASK_FAILED, TaskStatus.TASK_BLOCKED):
            raise ValueError(f"Cannot complete a task in state '{self.status.value}'")
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
        self.status = TaskStatus.TASK_COMPLETED
        self._touch()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def is_complete(self) -> bool:
        """
        True only when the task has been explicitly marked complete and no
        requirements remain unsatisfied (defense-in-depth for deserialized
        or corrupt state — the normal path is guarded by mark_complete()).
        """
        return self.status == TaskStatus.TASK_COMPLETED and not self.remaining_requirements()

    @property
    def is_blocked(self) -> bool:
        """True when the task has been hard-stopped (security block etc.)."""
        return self.status == TaskStatus.TASK_BLOCKED

    @property
    def is_waiting_confirmation(self) -> bool:
        """True when the task is paused for user approval."""
        return self.status == TaskStatus.WAITING_CONFIRMATION

    def summary(self) -> str:
        """Compact one-line description for logs and debugging."""
        done = len(self.completed_requirements)
        total = len(self.requirements)
        return (
            f"TaskState(goal='{self.goal}', status={self.status.value}, "
            f"requirements={done}/{total}, step={self.current_step}/{self.total_steps}, "
            f"tools={len(self.execution_history)}, errors={len(self.errors)})"
        )

    # ------------------------------------------------------------------
    # Structured plan integration (Fix #2)
    # ------------------------------------------------------------------

    def attach_plan(self, plan: TaskPlan) -> None:
        """
        Attaches a structured plan to the task and seeds its completion
        criteria from the plan (plan-level criteria + verification
        requirements).

        TaskState remains the runtime source of truth; the plan is
        persisted with it, so it survives LLM turns, tool calls,
        confirmations, failures, and agent continuation.
        """
        self.plan = plan
        self.task_type = plan.task_type
        self.set_total_steps(len(plan.steps))
        seen: set[str] = set()
        for description in [*plan.completion_criteria, *plan.verification_requirements]:
            description = description.strip()
            if description and description not in seen:
                seen.add(description)
                self.add_requirement(description)
        if plan.needs_clarification:
            self.require_clarification(plan.clarification_questions)
        self._touch()

    def require_clarification(self, questions: list[str] | None = None) -> None:
        """
        Blocks the task pending user clarification (e.g. a deployment
        request with no target). Moves the task to TASK_BLOCKED so it can
        never be reported complete while unanswered.
        """
        self.clarification_required = True
        if questions:
            self.clarification_questions = list(
                dict.fromkeys(q for q in questions if q)
            )
        if self.status != TaskStatus.TASK_BLOCKED:
            self.block("Task requires clarification before it can proceed.")
        self._touch()

    def adapt_plan(self, new_steps: list[PlanStep]) -> bool:
        """
        ADAPTIVE PLANNING: replaces the task's remaining plan steps with
        ``new_steps``, preserving completed work.

        The revision must be structurally valid (unique ids, sane
        dependencies, acyclic, reachable) and is recorded explicitly in
        ``plan_revisions`` so plan changes are auditable and the TaskState
        stays the consistent source of truth. Returns True when applied.
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

    def remaining_steps(self) -> list[PlanStep]:
        """Plan steps that still need work (empty when no plan is attached)."""
        return self.plan.remaining_steps() if self.plan else []

    def completed_steps(self) -> list[PlanStep]:
        """Plan steps that succeeded (empty when no plan is attached)."""
        return self.plan.completed_steps() if self.plan else []

    def failed_steps(self) -> list[PlanStep]:
        """Plan steps that failed (empty when no plan is attached)."""
        return self.plan.failed_steps() if self.plan else []

    def blocked_steps(self) -> list[PlanStep]:
        """Plan steps blocked pending clarification (empty when no plan)."""
        return self.plan.blocked_steps() if self.plan else []

    def current_plan_step(self) -> PlanStep | None:
        """The step being worked on (RUNNING) or the next step that can run."""
        return self.plan.active_step() if self.plan else None

    def record_plan_revision(self, note: str) -> None:
        """
        Explicitly records an adaptive plan revision (audit trail).

        The note describes what changed and why; completed steps are never
        rewritten by a revision — only the remaining work is replaced.
        """
        self.plan_revisions.append(note)
        self._touch()

    def _touch(self) -> None:
        """Bumps the updated_at marker after any mutation."""
        self.updated_at = datetime.now(UTC)

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
