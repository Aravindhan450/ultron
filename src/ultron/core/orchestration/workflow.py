"""
ultron.core.orchestration.workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Workflow engine + sequential execution (Fix #7, section 7.6).

A :class:`Workflow` is a deterministic, ordered sequence of specialist
delegations sitting ABOVE the existing Supervisor / Delegation / Agent /
AgentResult / Artifact / TaskState / Validation systems:

    USER
      |
    TASKSTATE
      |
    WORKFLOW  (owned by the engine)
      |
    WORKFLOW STEP  (agent_type + objective + dependencies)
      |
    DELEGATION  (Supervisor.create_delegation -> dispatch)
      |
    AGENT
      |
    AGENT RESULT
      |
    ORCHESTRATION VALIDATOR  (deterministic, read-only)
      |
    WORKFLOW STEP RESULT
      |
    NEXT WORKFLOW STEP
      |
    FINAL VERIFICATION
      |
    TASKSTATE

CRITICAL ARCHITECTURAL RULE: the engine does NOT replace TaskState and does
NOT create a second task-state system. TaskState remains the authoritative
task state (goal, requirements, plan, task status, task errors). The
workflow owns only orchestration structure: step ordering, dependencies,
step execution, step results, transition decisions and workflow status.

State machines are explicit and transition-validated (mirroring
:mod:`ultron.core.orchestration.lifecycle`):

    Workflow:       PENDING -> RUNNING -> PAUSED -> RUNNING -> COMPLETED
                    (WAITING while a step awaits input; FAILED / BLOCKED /
                    CANCELLED as terminal stops)
    WorkflowStep:   PENDING -> READY -> RUNNING -> WAITING -> RUNNING
                    -> COMPLETED  (FAILED / BLOCKED / CANCELLED terminal)

BLOCKED is a terminal workflow state BY DESIGN, mirroring
:meth:`TaskState.block` (a blocked task cannot complete and recovery is a
new task): an agent BLOCKED stops the workflow and preserves all produced
state/artifacts, but a blocked workflow is NOT resumable — resume is
supported for PAUSED (caller freeze) and WAITING (NEEDS_INPUT) workflows
only. A workflow in any terminal state must be recreated to run again.

Dependency validation rejects missing/unknown/self/circular/cross-workflow
dependencies and duplicate ids BEFORE any step may execute — an invalid
workflow never starts.

The engine never instantiates agents itself: it routes every step through
the Supervisor (which selects the specialist, builds the scoped
ExecutionContext and enforces timeouts/cancellation), then gates each
AgentResult through the OrchestrationValidator. Agent claims are NOT proof:
a step is successful only when the agent succeeded AND the result is valid
AND required evidence exists. Workflow completion then invokes the existing
TaskState completion enforcement (:meth:`TaskState.mark_complete`) — it
never forces ``task.status`` directly.

TOOL SUCCESS != AGENT SUCCESS != STEP SUCCESS != WORKFLOW SUCCESS != TASK
SUCCESS — this module enforces that chain deterministically.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from ultron.core.logging import get_logger
from ultron.core.orchestration.artifacts import (
    AgentArtifact,
    TestResult,
    task_key,
)
from ultron.core.orchestration.delegation import DelegationRequest, Supervisor
from ultron.core.orchestration.lifecycle import AgentStatus
from ultron.core.orchestration.models import AgentResult, AgentType
from ultron.core.orchestration.validation import (
    OrchestrationValidator,
    ValidationContext,
    ValidationResult,
)
from ultron.core.types import TaskError, TaskState

logger = get_logger("ultron.orchestration.workflow")

# ---------------------------------------------------------------------------
# Status enums — distinct from AgentStatus: the workflow has READY/PAUSED and
# step ordering semantics the agent lifecycle does not express.
# ---------------------------------------------------------------------------


class WorkflowStatus(str, Enum):
    """Lifecycle state of one workflow."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    WAITING = "waiting"  # a step awaits input (NEEDS_INPUT)
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"
    COMPLETED = "completed"

    @property
    def is_terminal(self) -> bool:
        return self in WORKFLOW_TERMINAL_STATUSES


class WorkflowStepStatus(str, Enum):
    """Lifecycle state of one workflow step."""

    PENDING = "pending"
    READY = "ready"  # dependencies satisfied, may start
    RUNNING = "running"
    WAITING = "waiting"  # awaiting input (delegation NEEDS_INPUT)
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in STEP_TERMINAL_STATUSES


# The only legal transitions, keyed by source state. Anything not listed is
# invalid; terminal states accept nothing.
WORKFLOW_TRANSITIONS: dict[WorkflowStatus, frozenset[WorkflowStatus]] = {
    WorkflowStatus.PENDING: frozenset(
        {WorkflowStatus.RUNNING, WorkflowStatus.PAUSED, WorkflowStatus.CANCELLED}
    ),
    WorkflowStatus.RUNNING: frozenset(
        {
            WorkflowStatus.PAUSED,
            WorkflowStatus.WAITING,
            WorkflowStatus.FAILED,
            WorkflowStatus.BLOCKED,
            WorkflowStatus.CANCELLED,
            WorkflowStatus.COMPLETED,
        }
    ),
    WorkflowStatus.PAUSED: frozenset(
        {WorkflowStatus.RUNNING, WorkflowStatus.CANCELLED}
    ),
    WorkflowStatus.WAITING: frozenset(
        {
            WorkflowStatus.RUNNING,
            WorkflowStatus.FAILED,
            WorkflowStatus.BLOCKED,
            WorkflowStatus.CANCELLED,
        }
    ),
    WorkflowStatus.FAILED: frozenset(),
    WorkflowStatus.BLOCKED: frozenset(),
    WorkflowStatus.CANCELLED: frozenset(),
    WorkflowStatus.COMPLETED: frozenset(),
}

WORKFLOW_TERMINAL_STATUSES: frozenset[WorkflowStatus] = frozenset(
    {
        WorkflowStatus.FAILED,
        WorkflowStatus.BLOCKED,
        WorkflowStatus.CANCELLED,
        WorkflowStatus.COMPLETED,
    }
)

STEP_TRANSITIONS: dict[WorkflowStepStatus, frozenset[WorkflowStepStatus]] = {
    WorkflowStepStatus.PENDING: frozenset(
        {WorkflowStepStatus.READY, WorkflowStepStatus.CANCELLED}
    ),
    WorkflowStepStatus.READY: frozenset(
        {WorkflowStepStatus.RUNNING, WorkflowStepStatus.CANCELLED}
    ),
    WorkflowStepStatus.RUNNING: frozenset(
        {
            WorkflowStepStatus.WAITING,
            WorkflowStepStatus.COMPLETED,
            WorkflowStepStatus.FAILED,
            WorkflowStepStatus.BLOCKED,
            WorkflowStepStatus.CANCELLED,
        }
    ),
    WorkflowStepStatus.WAITING: frozenset(
        {
            WorkflowStepStatus.RUNNING,
            WorkflowStepStatus.FAILED,
            WorkflowStepStatus.BLOCKED,
            WorkflowStepStatus.CANCELLED,
        }
    ),
    WorkflowStepStatus.COMPLETED: frozenset(),
    WorkflowStepStatus.FAILED: frozenset(),
    WorkflowStepStatus.BLOCKED: frozenset(),
    WorkflowStepStatus.CANCELLED: frozenset(),
}

STEP_TERMINAL_STATUSES: frozenset[WorkflowStepStatus] = frozenset(
    {
        WorkflowStepStatus.COMPLETED,
        WorkflowStepStatus.FAILED,
        WorkflowStepStatus.BLOCKED,
        WorkflowStepStatus.CANCELLED,
    }
)


def _assert_step_transition(
    current: WorkflowStepStatus, target: WorkflowStepStatus, context: str
) -> None:
    if target not in STEP_TRANSITIONS.get(current, frozenset()):
        raise ValueError(
            f"Invalid {context} workflow-step transition: "
            f"'{current.value}' -> '{target.value}'"
        )


def _assert_workflow_transition(
    current: WorkflowStatus, target: WorkflowStatus, context: str
) -> None:
    if target not in WORKFLOW_TRANSITIONS.get(current, frozenset()):
        raise ValueError(
            f"Invalid {context} workflow transition: "
            f"'{current.value}' -> '{target.value}'"
        )


# ---------------------------------------------------------------------------
# Workflow event — observability record (traceable via workflow/task/step/
# delegation/agent ids)
# ---------------------------------------------------------------------------


class WorkflowEvent(BaseModel):
    """One recorded workflow event (also logged via get_logger)."""

    event: str
    at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    step_id: str = ""
    delegation_id: str = ""
    message: str = ""


# ---------------------------------------------------------------------------
# Workflow step
# ---------------------------------------------------------------------------


class WorkflowStep(BaseModel):
    """One ordered unit of delegated work inside a workflow.

    - ``step_id`` / ``name`` — identity and human label
    - ``agent_type`` — which specialist performs the work (registry type)
    - ``objective`` — the concrete goal handed to the specialist
    - ``input_artifacts`` — artifact ids this step may consume (engine fills
      these from completed dependency steps, filtered by
      ``input_artifact_types`` when set)
    - ``dependencies`` — step ids that must COMPLETE before this step starts
    - ``expected_output`` / ``constraints`` — handed to the DelegationRequest
    - ``required_evidence`` — deterministic evidence gates (artifact /
      changed_files / tests / verification) the step must satisfy
    - ``claims_completion`` — when True, a SUCCESS result is treated as a
      task-completion claim and validated against the task's requirements
    - ``allowed_scope`` — workspace-relative path prefixes the step may touch
    - ``timeout_seconds`` — per-step delegation timeout (None = spec default)
    """

    step_id: str
    workflow_id: str = ""
    name: str
    description: str = ""
    agent_type: AgentType | str
    objective: str
    input_artifacts: list[str] = Field(default_factory=list)
    input_artifact_types: list[str] = Field(default_factory=list)
    expected_output: str = ""
    dependencies: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    required_evidence: list[str] = Field(default_factory=list)
    allowed_scope: list[str] = Field(default_factory=list)
    claims_completion: bool = False
    timeout_seconds: int | None = None
    status: WorkflowStepStatus = WorkflowStepStatus.PENDING
    delegation_id: str | None = None
    result_artifact: str | None = None
    attempts: int = 0
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None
    validation: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    # -- identity ------------------------------------------------------------

    @property
    def resolved_type(self) -> AgentType | None:
        """The AgentType, or None when ``agent_type`` is not a valid type."""
        if isinstance(self.agent_type, AgentType):
            return self.agent_type
        try:
            return AgentType(self.agent_type)
        except ValueError:
            return None

    @property
    def agent_type_value(self) -> str:
        resolved = self.resolved_type
        return resolved.value if resolved is not None else str(self.agent_type)

    @property
    def is_terminal(self) -> bool:
        return self.status in STEP_TERMINAL_STATUSES

    @property
    def is_active(self) -> bool:
        return self.status not in STEP_TERMINAL_STATUSES

    def label(self) -> str:
        return f"step:{self.step_id}"

    # -- lifecycle -----------------------------------------------------------

    def _transition(self, target: WorkflowStepStatus, reason: str) -> None:
        _assert_step_transition(self.status, target, self.label())
        previous = self.status.value
        self.status = target
        self.metadata.setdefault("history", []).append(
            {"from": previous, "to": target.value, "reason": reason}
        )

    def mark_ready(self, reason: str = "dependencies satisfied") -> None:
        """PENDING -> READY: all dependencies completed; may start."""
        self._transition(WorkflowStepStatus.READY, reason)

    def start(self, reason: str = "started") -> None:
        """READY -> RUNNING: begin executing."""
        self._transition(WorkflowStepStatus.RUNNING, reason)

    def resume(self, reason: str = "resumed") -> None:
        """WAITING -> RUNNING: continue after the required input."""
        self._transition(WorkflowStepStatus.RUNNING, reason)

    def wait(self, reason: str = "awaiting input") -> None:
        """RUNNING -> WAITING: the delegation paused for input."""
        self._transition(WorkflowStepStatus.WAITING, reason)

    def complete(
        self,
        *,
        result_artifact: str | None = None,
        validation: ValidationResult | None = None,
        reason: str = "completed",
    ) -> None:
        """RUNNING -> COMPLETED: agent succeeded, validation passed."""
        self.result_artifact = result_artifact or self.result_artifact
        if validation is not None:
            self.validation = {
                "status": validation.status.value,
                "violations": [v.code.value for v in validation.violations],
                "warnings": len(validation.warnings),
            }
        self.completed_at = datetime.now(UTC)
        self._transition(WorkflowStepStatus.COMPLETED, reason)

    def fail(self, reason: str = "failed", validation: ValidationResult | None = None) -> None:
        """Any active state -> FAILED: the step's delegation or validation failed."""
        self.error = reason
        if validation is not None:
            self.validation = {
                "status": validation.status.value,
                "violations": [v.code.value for v in validation.violations],
                "warnings": len(validation.warnings),
            }
        self.completed_at = datetime.now(UTC)
        self._transition(WorkflowStepStatus.FAILED, reason)

    def block(self, reason: str = "blocked") -> None:
        """Any active state -> BLOCKED: hard-stopped (security/policy)."""
        self.error = reason
        self.completed_at = datetime.now(UTC)
        self._transition(WorkflowStepStatus.BLOCKED, reason)

    def cancel(self, reason: str = "cancelled") -> None:
        """Any active state -> CANCELLED."""
        self.error = reason
        self._transition(WorkflowStepStatus.CANCELLED, reason)

    def summary(self) -> str:
        return (
            f"WorkflowStep({self.step_id}, type={self.agent_type_value}, "
            f"status={self.status.value})"
        )


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------


class Workflow(BaseModel):
    """A deterministic, ordered sequence of specialist delegations.

    The workflow owns orchestration only (step ordering, dependencies, step
    execution, step results, transition decisions, workflow status). The
    TaskState owns the overall task and remains authoritative — completing
    a workflow invokes TaskState's own completion enforcement; it never
    overwrites ``task.status`` directly.
    """

    workflow_id: str
    task_id: str
    name: str = ""
    description: str = ""
    steps: list[WorkflowStep] = Field(default_factory=list)
    status: WorkflowStatus = WorkflowStatus.PENDING
    current_step: str | None = None
    events: list[WorkflowEvent] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, context: Any) -> None:
        for step in self.steps:
            if not step.workflow_id:
                step.workflow_id = self.workflow_id

    # -- lookup --------------------------------------------------------------

    def get_step(self, step_id: str) -> WorkflowStep | None:
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None

    def step_ids(self) -> list[str]:
        return [step.step_id for step in self.steps]

    # -- lifecycle -----------------------------------------------------------

    def _transition(self, target: WorkflowStatus, reason: str) -> None:
        _assert_workflow_transition(self.status, target, f"workflow {self.workflow_id}")
        previous = self.status.value
        self.status = target
        self.updated_at = datetime.now(UTC)
        self.metadata.setdefault("history", []).append(
            {"from": previous, "to": target.value, "reason": reason}
        )

    def start(self, reason: str = "started") -> None:
        """PENDING -> RUNNING: begin execution."""
        self._transition(WorkflowStatus.RUNNING, reason)

    def pause(self, reason: str = "paused") -> None:
        """PENDING or RUNNING -> PAUSED: freeze execution (state preserved)."""
        self._transition(WorkflowStatus.PAUSED, reason)

    def resume(self, reason: str = "resumed") -> None:
        """PAUSED or WAITING -> RUNNING: continue from the incomplete step."""
        self._transition(WorkflowStatus.RUNNING, reason)

    def wait(self, reason: str = "awaiting input") -> None:
        """RUNNING -> WAITING: a step awaits input (NEEDS_INPUT)."""
        self._transition(WorkflowStatus.WAITING, reason)

    def fail(self, reason: str = "failed") -> None:
        """Any active state -> FAILED."""
        self.metadata["failure"] = reason
        self._transition(WorkflowStatus.FAILED, reason)

    def block(self, reason: str = "blocked") -> None:
        """Any active state -> BLOCKED (not a failure — a hard stop)."""
        self.metadata["blocked"] = reason
        self._transition(WorkflowStatus.BLOCKED, reason)

    def cancel(self, reason: str = "cancelled") -> None:
        """Any active state -> CANCELLED (artifacts are preserved)."""
        self.metadata["cancelled"] = reason
        self._transition(WorkflowStatus.CANCELLED, reason)

    def complete(self, reason: str = "all steps completed") -> None:
        """RUNNING -> COMPLETED. A workflow may only complete when every
        step is COMPLETED — completing early would be state corruption."""
        unfinished = [
            s.step_id for s in self.steps if s.status is not WorkflowStepStatus.COMPLETED
        ]
        if unfinished:
            raise ValueError(
                f"cannot complete workflow '{self.workflow_id}' with unfinished "
                f"steps: {', '.join(unfinished)}"
            )
        self._transition(WorkflowStatus.COMPLETED, reason)

    @property
    def is_terminal(self) -> bool:
        return self.status in WORKFLOW_TERMINAL_STATUSES

    def summary(self) -> str:
        states = ", ".join(f"{s.step_id}={s.status.value}" for s in self.steps)
        return (
            f"Workflow({self.workflow_id}, task={self.task_id}, "
            f"status={self.status.value}, steps=[{states}])"
        )


# ---------------------------------------------------------------------------
# Workflow engine — deterministic, sequential, Supervisor-routed
# ---------------------------------------------------------------------------


class WorkflowEngine:
    """Executes a sequence of agent delegations as a validated workflow.

    - creates + validates workflows (invalid construction raises ValueError),
    - starts / pauses / resumes / cancels workflows,
    - executes one ready step at a time through the Supervisor,
    - gates every AgentResult through the OrchestrationValidator,
    - tracks step results, artifacts and events for traceability,
    - invokes TaskState's own completion enforcement at workflow completion.

    The engine has NO LLM and never instantiates agents — it routes, gates
    and records. It is sequential only (no parallel execution, no dynamic
    routing, no retries — later sections).
    """

    def __init__(
        self,
        supervisor: Supervisor,
        validator: OrchestrationValidator | None = None,
    ) -> None:
        self.supervisor = supervisor
        self.validator = validator or OrchestrationValidator()
        self._workflows: dict[str, Workflow] = {}
        self._task_states: dict[str, TaskState] = {}
        # Fallback artifact index when the supervisor has no ArtifactStore
        # (resolved artifact ids -> AgentArtifact objects).
        self._artifact_cache: dict[str, AgentArtifact] = {}

    # ------------------------------------------------------------------
    # Creation + validation
    # ------------------------------------------------------------------

    def create_workflow(
        self,
        *,
        steps: list[WorkflowStep],
        task_state: TaskState | None = None,
        task_id: str | None = None,
        workflow_id: str | None = None,
        name: str = "",
        description: str = "",
    ) -> Workflow:
        """Builds, validates and registers a workflow.

        Raises ``ValueError`` when the task id is missing, the workflow id
        is already registered, or the workflow is structurally invalid
        (duplicate ids, unknown/self/circular dependencies, unregistered
        agent type, empty steps) — an invalid workflow never starts.
        """
        if task_id is None:
            task_id = task_key(task_state) if task_state is not None else ""
        if not task_id:
            raise ValueError(
                "cannot create workflow: missing task_id (pass task_id or task_state)"
            )
        wid = workflow_id or f"wf:{task_id}:{uuid.uuid4().hex[:6]}"
        if wid in self._workflows:
            raise ValueError(f"duplicate workflow_id: '{wid}'")

        workflow = Workflow(
            workflow_id=wid,
            task_id=task_id,
            name=name,
            description=description,
            steps=list(steps),
        )
        problems = self.validate_workflow(workflow)
        if problems:
            raise ValueError(
                f"invalid workflow '{wid}': " + "; ".join(problems)
            )
        self._workflows[wid] = workflow
        if task_state is not None:
            self._task_states[wid] = task_state
        self._log(workflow, "WORKFLOW_CREATED")
        logger.info(
            "workflow engine: created %s (task=%s, %d step(s))",
            wid,
            task_id,
            len(steps),
        )
        return workflow

    def validate_workflow(self, workflow: Workflow) -> list[str]:
        """Structural problems with a workflow (empty = valid).

        Detects: empty step list, duplicate step ids, unknown/missing
        dependencies, self-dependencies, circular dependencies, and
        unregistered agent types. Cross-workflow dependencies are caught by
        the unknown-dependency check (a foreign step id is not present).
        """
        problems: list[str] = []
        if not workflow.steps:
            problems.append("workflow has no steps")
            return problems

        ids = workflow.step_ids()
        seen: set[str] = set()
        for step in workflow.steps:
            if step.step_id in seen:
                problems.append(f"duplicate step id: '{step.step_id}'")
            seen.add(step.step_id)
            if not step.objective:
                problems.append(f"step '{step.step_id}' has no objective")
            # Registered agent type (fail safely on unknown specialists).
            try:
                self.supervisor.registry.get(step.agent_type)
            except KeyError:
                problems.append(
                    f"step '{step.step_id}' references unregistered agent type "
                    f"'{step.agent_type_value}'"
                )
            for dep in step.dependencies:
                if dep not in ids:
                    problems.append(
                        f"step '{step.step_id}' depends on unknown step '{dep}'"
                    )
                elif dep == step.step_id:
                    problems.append(f"step '{step.step_id}' depends on itself")

        for cycle in self._cycles(workflow):
            problems.append(f"circular dependency: {' -> '.join(cycle)}")
        return problems

    @staticmethod
    def _cycles(workflow: Workflow) -> list[list[str]]:
        """Simple cycle report via DFS over the step dependency graph."""
        index = {step.step_id: step for step in workflow.steps}
        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[str, int] = {sid: WHITE for sid in index}
        stack: list[str] = []
        cycles: list[list[str]] = []

        def visit(node: str) -> None:
            color[node] = GRAY
            stack.append(node)
            for dep in index[node].dependencies:
                if color.get(dep) is GRAY:
                    cut = stack[stack.index(dep):]
                    cycles.append([*cut, dep])
                elif color.get(dep) is WHITE:
                    visit(dep)
            stack.pop()
            color[node] = BLACK

        for sid in index:
            if color[sid] is WHITE:
                visit(sid)
        return cycles

    def get_workflow(self, workflow_id: str) -> Workflow | None:
        return self._workflows.get(workflow_id)

    def get_status(self, workflow: Workflow) -> WorkflowStatus:
        return workflow.status

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    async def start(
        self,
        workflow: Workflow,
        workspace: str = "",
        task_state: TaskState | None = None,
    ) -> Workflow:
        """PENDING -> RUNNING and marks dependency-satisfied steps READY."""
        if workflow.status is not WorkflowStatus.PENDING:
            raise ValueError(
                f"cannot start workflow '{workflow.workflow_id}': status is "
                f"'{workflow.status.value}'"
            )
        workflow.start()
        self._log(workflow, "WORKFLOW_STARTED")
        self._mark_ready(workflow)
        task_state = task_state or self._task_states.get(workflow.workflow_id)
        if task_state is not None:
            self._task_states[workflow.workflow_id] = task_state
        return workflow

    def pause(self, workflow: Workflow, reason: str = "paused by caller") -> Workflow:
        """Freezes the workflow; the current state (including any completed
        steps, artifacts and the active step) is preserved. Allowed before
        the first step (PENDING) and between steps (RUNNING)."""
        if workflow.status not in (WorkflowStatus.PENDING, WorkflowStatus.RUNNING):
            raise ValueError(
                f"cannot pause workflow '{workflow.workflow_id}': status is "
                f"'{workflow.status.value}'"
            )
        workflow.pause(reason=reason)
        self._log(workflow, "WORKFLOW_PAUSED", message=reason)
        return workflow

    async def resume(
        self,
        workflow: Workflow,
        workspace: str = "",
        task_state: TaskState | None = None,
    ) -> Workflow:
        """Continues a paused or waiting workflow from its incomplete step.

        Completed steps are never repeated. A workflow saved while WAITING
        (a step awaits input) re-enters RUNNING and the waiting step is
        re-dispatched by the next :meth:`execute_next_step`.
        """
        already_running = workflow.status is WorkflowStatus.RUNNING
        if workflow.status is WorkflowStatus.PAUSED:
            workflow.resume(reason="resumed")
        elif workflow.status is WorkflowStatus.WAITING:
            workflow.resume(reason="input received")
        elif workflow.status is not WorkflowStatus.RUNNING:
            raise ValueError(
                f"cannot resume workflow '{workflow.workflow_id}': status is "
                f"'{workflow.status.value}'"
            )
        if not already_running:
            self._log(workflow, "WORKFLOW_RESUMED")
        task_state = task_state or self._task_states.get(workflow.workflow_id)
        if task_state is not None:
            self._task_states[workflow.workflow_id] = task_state
        return workflow

    def cancel(
        self, workflow: Workflow, reason: str = "cancelled by caller"
    ) -> Workflow:
        """Stops the workflow: no future steps start, the active delegation
        is cancelled where supported, and already-produced artifacts are
        preserved. Never marks the task complete."""
        if workflow.is_terminal:
            raise ValueError(
                f"cannot cancel workflow '{workflow.workflow_id}': already "
                f"'{workflow.status.value}'"
            )
        for step in workflow.steps:
            if not step.is_terminal:
                if (
                    step.status is WorkflowStepStatus.WAITING
                    and step.delegation_id is not None
                ):
                    self._cancel_parked_delegation(step.delegation_id)
                step.cancel(reason=reason)
        workflow.cancel(reason=reason)
        self._log(workflow, "WORKFLOW_CANCELLED", message=reason)
        return workflow

    def _cancel_parked_delegation(self, delegation_id: str) -> None:
        request = self.supervisor.get_delegation(delegation_id)
        if request is None or request.is_terminal:
            return
        try:
            self.supervisor.cancel_delegation(request, reason="workflow cancelled")
        except ValueError:  # already terminal — nothing to cancel
            pass

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def execute_next_step(
        self,
        workflow: Workflow,
        workspace: str = "",
        task_state: TaskState | None = None,
    ) -> WorkflowStep | None:
        """Executes the next runnable step (or the waiting one) and returns it.

        Returns ``None`` when the workflow is not RUNNING or no step is
        ready (dependencies unresolved) — the caller can poll again later.
        """
        if workflow.status is not WorkflowStatus.RUNNING:
            logger.info(
                "workflow engine: %s execute_next_step skipped (status=%s)",
                workflow.workflow_id,
                workflow.status.value,
            )
            return None
        task_state = task_state or self._task_states.get(workflow.workflow_id)
        step = self._next_step(workflow)
        if step is None:
            return None
        await self._run_step(workflow, step, workspace, task_state)
        return step

    async def execute_until_blocked(
        self,
        workflow: Workflow,
        workspace: str = "",
        task_state: TaskState | None = None,
        max_steps: int | None = None,
    ) -> int:
        """Runs steps until none is ready, the workflow stops, or ``max_steps``
        executions happen. Returns the number of steps executed.

        A PENDING workflow is started automatically (convenience); granular
        callers may instead call :meth:`start` explicitly and then drive
        :meth:`execute_next_step` one step at a time (note the asymmetry:
        ``execute_next_step`` on a PENDING workflow is a no-op that returns
        None — start first when driving step-by-step)."""
        if workflow.status is WorkflowStatus.PENDING:
            await self.start(workflow, workspace, task_state)
        executed = 0
        while workflow.status is WorkflowStatus.RUNNING:
            step = await self.execute_next_step(workflow, workspace, task_state)
            if step is None:
                break
            executed += 1
            if max_steps is not None and executed >= max_steps:
                break
        return executed

    def _next_step(self, workflow: Workflow) -> WorkflowStep | None:
        """The step to run: first a WAITING step whose input has been
        provided, then the first READY/PENDING step with all dependencies
        completed (step list order)."""
        for step in workflow.steps:
            if (
                step.status is WorkflowStepStatus.WAITING
                and self._deps_met(workflow, step)
            ):
                return step
        for step in workflow.steps:
            if (
                step.status in (WorkflowStepStatus.PENDING, WorkflowStepStatus.READY)
                and self._deps_met(workflow, step)
            ):
                return step
        return None

    @staticmethod
    def _deps_met(workflow: Workflow, step: WorkflowStep) -> bool:
        return all(
            (dep := workflow.get_step(dep_id)) is not None
            and dep.status is WorkflowStepStatus.COMPLETED
            for dep_id in step.dependencies
        )

    def _mark_ready(self, workflow: Workflow) -> None:
        """Promotes PENDING steps whose dependencies are all complete."""
        for step in workflow.steps:
            if step.status is WorkflowStepStatus.PENDING and self._deps_met(workflow, step):
                step.mark_ready()
                self._log(workflow, "STEP_READY", step)

    def _wire_input_artifacts(self, workflow: Workflow, step: WorkflowStep) -> None:
        """Fills ``step.input_artifacts`` with the result artifacts of the
        step's completed dependencies (the only cross-agent channel). Type
        filtering (``input_artifact_types``) happens at resolution time."""
        ids: list[str] = []
        for dep_id in step.dependencies:
            dep = workflow.get_step(dep_id)
            if dep is not None and dep.result_artifact:
                ids.append(dep.result_artifact)
        step.input_artifacts = ids

    # -- one step ------------------------------------------------------------

    async def _run_step(
        self,
        workflow: Workflow,
        step: WorkflowStep,
        workspace: str,
        task_state: TaskState | None,
    ) -> None:
        """Dispatches one step via the Supervisor and gates its result."""
        self._wire_input_artifacts(workflow, step)
        artifacts = self._resolve_artifacts(
            step.input_artifacts, types=set(step.input_artifact_types) or None
        )

        if step.status is WorkflowStepStatus.WAITING:
            step.resume(reason="input provided")
        elif step.status is WorkflowStepStatus.PENDING:
            step.mark_ready(reason="selected by engine")
            step.start(reason="step started")
        else:  # READY
            step.start(reason="step started")
        step.attempts += 1
        step.started_at = datetime.now(UTC)
        workflow.current_step = step.step_id
        workflow.updated_at = datetime.now(UTC)

        request = self._request_for(workflow, step, artifacts, task_state)
        if request is None:
            request = self.supervisor.create_delegation(
                task_state,
                step.agent_type,
                step.objective,
                input_artifacts=artifacts,
                constraints=list(step.constraints),
                expected_output=step.expected_output,
                timeout_seconds=step.timeout_seconds,
            )
        step.delegation_id = request.delegation_id
        self._log(
            workflow,
            "STEP_STARTED",
            step,
            delegation_id=request.delegation_id,
            message=step.objective,
        )

        await self.supervisor.dispatch(request, workspace=workspace, task_state=task_state)
        self._finalize_step(workflow, step, request, task_state, workspace)

    def _request_for(
        self,
        workflow: Workflow,
        step: WorkflowStep,
        artifacts: list[AgentArtifact],
        task_state: TaskState | None,
    ) -> DelegationRequest | None:
        """The existing WAITING delegation to resume (if any), else None."""
        if step.delegation_id is None:
            return None
        request = self.supervisor.get_delegation(step.delegation_id)
        if request is None or request.is_terminal:
            return None
        if request.status is not AgentStatus.WAITING:
            return None
        return request

    # -- step outcome --------------------------------------------------------

    def _finalize_step(
        self,
        workflow: Workflow,
        step: WorkflowStep,
        request: DelegationRequest,
        task_state: TaskState | None,
        workspace: str,
    ) -> None:
        result = request.result
        run = self.supervisor.get_run(request.delegation_id)
        artifacts: list[AgentArtifact] = []
        if result is not None and result.artifact is not None:
            artifacts.append(result.artifact)
            self._artifact_cache[result.artifact.artifact_id] = result.artifact

        test_results = [
            a for a in artifacts if isinstance(a, TestResult)
        ] + self._resolve_artifacts(step.input_artifacts, types={"test_result"})

        vresult = self.validator.validate(
            ValidationContext(
                agent_state=run,
                result=result,
                delegation=request,
                task_state=task_state,
                artifacts=artifacts,
                workspace=workspace,
                allowed_scope=list(step.allowed_scope),
                required_evidence=list(step.required_evidence),
                test_results=test_results,
                claims_completion=step.claims_completion,
            )
        )
        step.validation = {
            "status": vresult.status.value,
            "violations": [v.code.value for v in vresult.violations],
            "warnings": len(vresult.warnings),
        }

        if request.status.value == "completed":
            missing = self._missing_evidence(step, result)
            test_failed = self._test_artifact_failed(result)
            if vresult.valid and not missing and not test_failed:
                step.complete(
                    result_artifact=(
                        result.artifact.artifact_id if result.artifact is not None else None
                    ),
                    validation=vresult,
                )
                self._log(workflow, "STEP_COMPLETED", step, request.delegation_id or "")
                self._mark_ready(workflow)
                self._maybe_complete_workflow(workflow, task_state)
            else:
                reasons: list[str] = [v.message for v in vresult.violations]
                reasons.extend(f"missing evidence: {m}" for m in missing)
                if test_failed:
                    reasons.append("test artifact reports failures")
                detail = "; ".join(reasons) or "step validation failed"
                step.fail(reason=detail, validation=vresult)
                workflow.fail(reason=f"step '{step.step_id}' failed: {detail}")
                self._log(workflow, "STEP_FAILED", step, request.delegation_id or "", detail)
                self._record_task_error(task_state, f"[workflow:{step.step_id}] {detail}")
        elif request.status.value == "waiting":
            step.wait()
            workflow.wait(reason=f"step '{step.step_id}' awaits input")
            self._log(workflow, "STEP_WAITING", step, request.delegation_id or "")
        elif request.status.value == "blocked":
            detail = (result.summary if result else "") or (request.error or "blocked")
            step.block(reason=detail)
            workflow.block(reason=f"step '{step.step_id}' blocked: {detail}")
            self._log(workflow, "STEP_BLOCKED", step, request.delegation_id or "", detail)
            self._record_task_error(task_state, f"[workflow:{step.step_id}] blocked: {detail}")
        elif request.status.value == "cancelled":
            step.cancel(reason=request.error or "delegation cancelled")
            workflow.cancel(reason=f"step '{step.step_id}' cancelled")
            self._log(workflow, "STEP_CANCELLED", step, request.delegation_id or "")
        else:  # failed
            detail = (result.summary if result else "") or (request.error or "failed")
            step.fail(reason=detail, validation=vresult)
            workflow.fail(reason=f"step '{step.step_id}' failed: {detail}")
            self._log(workflow, "STEP_FAILED", step, request.delegation_id or "", detail)
            self._record_task_error(task_state, f"[workflow:{step.step_id}] {detail}")

    @staticmethod
    def _missing_evidence(step: WorkflowStep, result: AgentResult | None) -> list[str]:
        """Deterministic evidence gates — the step's own requirements."""
        artifact = result.artifact if result is not None else None
        missing: list[str] = []
        for req in step.required_evidence:
            if req == "artifact":
                ok = artifact is not None
            elif req == "changed_files":
                ok = bool(
                    (result is not None and result.changed_files)
                    or (
                        artifact is not None
                        and getattr(artifact, "changed_files", None)
                    )
                )
            elif req == "tests":
                ok = bool(result is not None and result.tests) or isinstance(
                    artifact, TestResult
                )
            elif req == "verification":
                ok = bool(
                    result is not None
                    and (result.metadata.get("verified") is True or result.metadata.get("verification") is True)
                )
            else:
                ok = bool(
                    result is not None
                    and any(req in e for e in result.evidence)
                )
            if not ok:
                missing.append(req)
        return missing

    @staticmethod
    def _test_artifact_failed(result: AgentResult | None) -> bool:
        """A TestResult artifact reporting failures fails its step."""
        artifact = result.artifact if result is not None else None
        if isinstance(artifact, TestResult):
            return artifact.failed > 0 or artifact.timed_out
        return False

    @staticmethod
    def _record_task_error(task_state: TaskState | None, message: str) -> None:
        """Records a failure on the TaskState via its existing mechanism.

        Also overwrites the observation surface so the failure is the latest
        recorded state (the supervisor records a success observation when a
        delegation lifecycle completes even if the workflow then rejects the
        result — the workflow's failure note must win)."""
        if task_state is None:
            return
        task_state.last_observation = message
        task_state.errors.append(
            TaskError(message=message, step=task_state.current_step)
        )
        task_state.updated_at = datetime.now(UTC)

    def _maybe_complete_workflow(
        self, workflow: Workflow, task_state: TaskState | None
    ) -> None:
        """Workflow COMPLETED only when every step is complete; then invokes
        the TaskState's own completion enforcement (never forces the task)."""
        if workflow.status is not WorkflowStatus.RUNNING:
            return
        if not all(
            step.status is WorkflowStepStatus.COMPLETED for step in workflow.steps
        ):
            return
        workflow.complete(reason="all steps completed")
        self._log(workflow, "WORKFLOW_COMPLETED")
        if task_state is None:
            return
        try:
            task_state.mark_complete()
        except ValueError as exc:
            workflow.metadata["task_completion"] = f"not_completed: {exc}"
            logger.info(
                "workflow %s: orchestration complete but task not completed: %s",
                workflow.workflow_id,
                exc,
            )
        else:
            workflow.metadata["task_completion"] = "completed"
            logger.info(
                "workflow %s: task marked complete via TaskState.mark_complete()",
                workflow.workflow_id,
            )

    # ------------------------------------------------------------------
    # Artifacts + observability
    # ------------------------------------------------------------------

    def _resolve_artifacts(
        self,
        artifact_ids: list[str],
        types: set[str] | None = None,
    ) -> list[AgentArtifact]:
        """Resolves artifact ids to artifacts (store first, then cache)."""
        resolved: list[AgentArtifact] = []
        for aid in artifact_ids:
            artifact = None
            if self.supervisor.store is not None:
                artifact = self.supervisor.store.get(aid)
            if artifact is None:
                artifact = self._artifact_cache.get(aid)
            if artifact is None:
                logger.warning(
                    "workflow engine: artifact '%s' not found (store/cache)", aid
                )
                continue
            if types is None or artifact.artifact_type.value in types:
                resolved.append(artifact)
        return resolved

    def _log(
        self,
        workflow: Workflow,
        event: str,
        step: WorkflowStep | None = None,
        delegation_id: str = "",
        message: str = "",
    ) -> None:
        record = WorkflowEvent(
            event=event,
            step_id=step.step_id if step is not None else "",
            delegation_id=delegation_id,
            message=message,
        )
        workflow.events.append(record)
        workflow.updated_at = datetime.now(UTC)
        logger.info(
            "workflow %s: %s step=%s delegation=%s %s",
            workflow.workflow_id,
            event,
            record.step_id,
            record.delegation_id,
            message,
        )

    def trace_rows(self, workflow: Workflow) -> list[dict[str, Any]]:
        """One row per step — the full workflow execution is reconstructable:
        workflow_id -> task_id -> step_id -> delegation_id -> agent_id ->
        result artifact / error."""
        rows: list[dict[str, Any]] = []
        for step in workflow.steps:
            request = (
                self.supervisor.get_delegation(step.delegation_id)
                if step.delegation_id is not None
                else None
            )
            rows.append(
                {
                    "workflow_id": workflow.workflow_id,
                    "task_id": workflow.task_id,
                    "step_id": step.step_id,
                    "name": step.name,
                    "agent_type": step.agent_type_value,
                    "status": step.status.value,
                    "delegation_id": step.delegation_id,
                    "agent_id": request.run_agent_id if request is not None else None,
                    "result_artifact": step.result_artifact,
                    "error": step.error,
                }
            )
        return rows
