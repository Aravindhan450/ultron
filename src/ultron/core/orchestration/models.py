"""
ultron.core.orchestration.models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Data models for the multi-agent orchestration layer (Fix #7, section 7.1).

Section 7.1 scope — the agent contract and lifecycle ONLY:

- :class:`AgentType` / :class:`AgentIdentity` — who the agent is
- :class:`AgentResultStatus` / :class:`AgentResult` — what the agent produced
  (structured, never arbitrary strings as the primary protocol)
- :class:`ExecutionBudget` — scoped limits for one agent run
- :class:`ExecutionContext` — the agent's scoped view of its task
- :class:`AgentMetadata` — execution/audit metadata (timing, status history)
- :class:`AgentState` — the runtime record: identity + objective + context
  + lifecycle status + result + metadata, with transition-validated
  lifecycle methods.

Architectural rule (section 7.1): ``ExecutionContext`` is a *scoped view* —
it references the task by id and never duplicates TaskState. TaskState
remains the authoritative runtime state for the overall task; completing an
agent run never completes a task (see :meth:`AgentState.complete` — it only
touches the agent record).

The supervisor, delegation, parallel execution and workflows are NOT part
of this section and are intentionally not implemented here.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from ultron.core.orchestration.artifacts import AgentArtifactUnion

# Late import to avoid a cycle at module load: permissions.py imports only
# ultron.security, never orchestration.models.
from ultron.core.orchestration.lifecycle import (
    ACTIVE_STATUSES,
    TERMINAL_STATUSES,
    AgentStatus,
    assert_transition,
)
from ultron.core.orchestration.permissions import AgentPermissions
from ultron.security import Decision

# ---------------------------------------------------------------------------
# Agent identity
# ---------------------------------------------------------------------------


class AgentType(str, Enum):
    """
    Standard agent roles for orchestration.

    Extensible: new roles can be added by appending members — nothing in the
    contract hardcodes a fixed set. ``AGENT_TYPES`` is the canonical tuple
    for CLI/discovery.
    """

    SUPERVISOR = "supervisor"
    RESEARCH = "research"
    CODING = "coding"
    TEST_QA = "test_qa"
    REVIEWER = "reviewer"
    SECURITY = "security"


AGENT_TYPES: tuple[str, ...] = tuple(member.value for member in AgentType)


class AgentIdentity(BaseModel):
    """Who an orchestrated agent is."""

    agent_id: str
    agent_type: AgentType
    display_name: str = ""

    @property
    def label(self) -> str:
        """Compact 'type:agent_id' label for logs and prompts."""
        return f"{self.agent_type.value}:{self.agent_id}"

    def to_prompt_line(self) -> str:
        name = self.display_name or self.agent_type.value
        return f"{name} (agent_id={self.agent_id}, type={self.agent_type.value})"


# ---------------------------------------------------------------------------
# Agent result
# ---------------------------------------------------------------------------


class AgentResultStatus(str, Enum):
    """Outcome of an agent run — distinct from the lifecycle status."""

    SUCCESS = "success"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"
    NEEDS_INPUT = "needs_input"


class AgentResult(BaseModel):
    """
    Structured result of an agent run.

    This is the primary communication protocol between agents and the
    (future) supervisor — never arbitrary strings. Fields:

    - ``summary`` — one-paragraph human/LLM-readable outcome
    - ``artifacts`` — files/objects the agent produced (paths, ids)
    - ``evidence`` — observations/commands that back the result
    - ``changed_files`` — files the agent modified
    - ``tests`` — tests the agent ran (node ids / commands + outcome)
    - ``blockers`` — things that prevented completion
    - ``recommendations`` — suggested next actions for the supervisor
    - ``metadata`` — free-form structured extras
    - ``artifact`` — optional section-7.3 structured artifact payload
      (a discriminated union over :class:`AgentArtifact` subclasses, keyed
      by ``artifact_type``); agents communicate through artifacts, never
      raw trajectories. JSON round-trips restore the concrete artifact type.
    """

    status: AgentResultStatus
    summary: str = ""
    artifacts: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    changed_files: list[str] = Field(default_factory=list)
    tests: list[str] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    artifact: AgentArtifactUnion | None = None

    @property
    def is_success(self) -> bool:
        return self.status is AgentResultStatus.SUCCESS

    @property
    def is_terminal(self) -> bool:
        """True for final outcomes; NEEDS_INPUT is a pause, not an end."""
        return self.status is not AgentResultStatus.NEEDS_INPUT

    def to_prompt_line(self, max_len: int = 240) -> str:
        head = f"[{self.status.value}] {self.summary or 'no summary'}"
        extras = []
        if self.changed_files:
            extras.append(f"{len(self.changed_files)} file(s) changed")
        if self.tests:
            extras.append(f"{len(self.tests)} test(s)")
        if self.blockers:
            extras.append(f"{len(self.blockers)} blocker(s)")
        if extras:
            head = f"{head} ({', '.join(extras)})"
        return head[:max_len]


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------


class ExecutionBudget(BaseModel):
    """
    Scoped limits for one agent run.

    - ``max_steps`` — max reasoning/action iterations
    - ``max_tool_calls`` — max tool invocations
    - ``timeout_seconds`` — wall-clock limit (None = no time limit)

    The agent records usage via :meth:`record_step` / :meth:`record_tool_call`
    so the loop can stop safely instead of looping forever. Budgets are
    per-agent-run and serialized on the ExecutionContext, so they survive
    waits and confirmations.

    NOTE (section 7.1 scope): enforcement is the AGENT's responsibility —
    the contract exposes :meth:`is_exhausted` / :meth:`timed_out` and the
    agent must check them at its checkpoints. No supervisor loop exists yet
    to enforce them externally (that is a later section).
    """

    max_steps: int = 20
    max_tool_calls: int = 50
    timeout_seconds: int | None = None
    steps_used: int = 0
    tool_calls_used: int = 0
    started_at: datetime | None = None

    def record_step(self, count: int = 1) -> None:
        """Records one reasoning/action step (or ``count``)."""
        self.steps_used += max(count, 0)

    def record_tool_call(self, count: int = 1) -> None:
        """Records one tool invocation (or ``count``)."""
        self.tool_calls_used += max(count, 0)

    def steps_remaining(self) -> int:
        return max(self.max_steps - self.steps_used, 0)

    def tool_calls_remaining(self) -> int:
        return max(self.max_tool_calls - self.tool_calls_used, 0)

    def is_exhausted(self) -> bool:
        """True when any hard limit is hit — the run must stop safely."""
        return (
            self.steps_used >= self.max_steps
            or self.tool_calls_used >= self.max_tool_calls
        )

    def timed_out(self, now: datetime | None = None) -> bool:
        """True when the wall-clock limit has been exceeded."""
        if self.timeout_seconds is None or self.started_at is None:
            return False
        now = now or datetime.now(UTC)
        return (now - self.started_at).total_seconds() > self.timeout_seconds

    def summary(self) -> str:
        return (
            f"steps {self.steps_used}/{self.max_steps}, "
            f"tools {self.tool_calls_used}/{self.max_tool_calls}"
            + (
                f", timeout {self.timeout_seconds}s"
                if self.timeout_seconds is not None
                else ""
            )
        )


# ---------------------------------------------------------------------------
# Execution context (the agent's scoped view — never a TaskState copy)
# ---------------------------------------------------------------------------


class ExecutionContext(BaseModel):
    """
    The agent's scoped view of its task.

    Contains everything an agent needs to act WITHOUT granting it the whole
    TaskState: task reference, workspace, allowed tools, permission profile,
    budget, the current plan step, relevant context snippets, and a
    cancellation flag.

    ``allowed_tools`` is deny-by-default: an empty list permits NO tools.
    ``permissions`` is a scoped profile (e.g. ``{"security_mode":
    "interactive"}``) the agent must respect — the security boundary remains
    outside the LLM's authority regardless.

    ``agent_permissions`` (section 7.2) is the agent TYPE's frozen
    :class:`~ultron.core.orchestration.permissions.AgentPermissions` — the
    runtime-controlled profile. :meth:`check_action` consults it and, for
    CONFIRM-level actions, delegates to the real security boundary. When no
    profile is attached (e.g. hand-built contexts), it falls back to the
    deny-by-default ``allowed_tools`` whitelist.
    """

    task_id: str
    agent_id: str
    workspace: str = ""
    allowed_tools: list[str] = Field(default_factory=list)
    permissions: dict[str, Any] = Field(default_factory=dict)
    agent_permissions: AgentPermissions | None = None
    budget: ExecutionBudget = Field(default_factory=ExecutionBudget)
    current_plan_step: int | None = None
    relevant_context: dict[str, str] = Field(default_factory=dict)
    cancelled: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    def __setattr__(self, name: str, value) -> None:
        """Blocks rebinding the runtime permission profile.

        ``agent_permissions`` is the enforcement source :meth:`check_action`
        reads — letting an agent (or anything else) swap it would let it
        rewrite its own permissions. Once assigned by the runtime, the field
        is frozen for the lifetime of this context: any later assignment
        raises. (The frozen profile itself is already immutable; this closes
        the reference-swap hole.)
        """
        if (
            name == "agent_permissions"
            and "agent_permissions" in self.__dict__
            and self.__dict__["agent_permissions"] is not None
        ):
            raise ValueError(
                "agent_permissions is runtime-controlled and cannot be reassigned"
            )
        super().__setattr__(name, value)

    @property
    def is_cancelled(self) -> bool:
        return self.cancelled

    def request_cancel(self) -> None:
        """Requests cancellation; the agent should stop at the next safe point."""
        self.cancelled = True

    def allows_tool(self, tool_name: str) -> bool:
        """Deny-by-default: only explicitly listed tools are permitted."""
        return tool_name in self.allowed_tools

    def is_budget_exhausted(self) -> bool:
        return self.budget.is_exhausted()

    def check_action(
        self,
        action_type: str,
        target: str = "",
        content: str | None = None,
        boundary=None,
    ) -> Decision:
        """
        Permission verdict for one tool call within this context.

        With a section-7.2 ``agent_permissions`` profile attached, this is
        the runtime enforcement point: unknown/unlisted tools are denied,
        ALLOW/DENY levels decide directly, and CONFIRM levels delegate to
        the security boundary (guardrails + risk tier + mode) — the LLM can
        never bypass or change this.

        Without a profile (hand-built contexts) it falls back to the
        deny-by-default ``allowed_tools`` whitelist: listed tools are
        allowed, everything else is denied.
        """
        if self.agent_permissions is not None:
            return self.agent_permissions.check_action(
                action_type, target, content, boundary
            ).decision
        return Decision.ALLOW if self.allows_tool(action_type) else Decision.DENY

    def to_prompt_line(self, max_len: int = 200) -> str:
        tools = ", ".join(self.allowed_tools) if self.allowed_tools else "none"
        step = f"step {self.current_plan_step}" if self.current_plan_step else "no step"
        return (
            f"task={self.task_id} workspace={self.workspace or '(none)'} "
            f"tools=[{tools}] {step} budget({self.budget.summary()})"
        )[:max_len]


# ---------------------------------------------------------------------------
# Metadata / audit
# ---------------------------------------------------------------------------


class AgentStatusChange(BaseModel):
    """One lifecycle transition, recorded for the audit trail."""

    from_status: AgentStatus
    to_status: AgentStatus
    at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    reason: str = ""


class AgentMetadata(BaseModel):
    """Execution metadata for one agent run.

    Usage counters (steps / tool calls) live on the :class:`ExecutionBudget`
    (reachable via ``state.budget``) — they are not duplicated here.
    """

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    completed_at: datetime | None = None
    status_history: list[AgentStatusChange] = Field(default_factory=list)
    attempts: int = 0  # number of times the run was started (resumes count)

    @property
    def elapsed_seconds(self) -> float | None:
        """Wall-clock duration when the run has started (None otherwise)."""
        if self.started_at is None:
            return None
        end = self.completed_at or datetime.now(UTC)
        return (end - self.started_at).total_seconds()


# ---------------------------------------------------------------------------
# Agent state — the runtime record with lifecycle methods
# ---------------------------------------------------------------------------


class AgentState(BaseModel):
    """
    Runtime record of one orchestrated agent run.

    Binds everything the contract promises:

    - ``task_id`` — the task this run contributes to
    - ``identity`` — agent_id + agent_type (+ display name)
    - ``objective`` — the concrete goal handed to the agent
    - ``context`` — the scoped ExecutionContext (workspace, tools, budget)
    - ``status`` — lifecycle status (see lifecycle.py)
    - ``result`` — structured AgentResult once the run finishes
    - ``metadata`` — timing + status-history audit trail

    IMPORTANT: completing an agent run (:meth:`complete`) only ever changes
    THIS record. It never marks a TaskState complete — agent completion and
    task completion are separate concepts, by design (validated in tests).
    """

    task_id: str
    identity: AgentIdentity
    objective: str
    context: ExecutionContext | None = None
    status: AgentStatus = AgentStatus.PENDING
    result: AgentResult | None = None
    metadata: AgentMetadata = Field(default_factory=AgentMetadata)

    # -- convenience accessors -------------------------------------------------

    @property
    def agent_id(self) -> str:
        return self.identity.agent_id

    @property
    def agent_type(self) -> AgentType:
        return self.identity.agent_type

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    @property
    def is_active(self) -> bool:
        return self.status in ACTIVE_STATUSES

    @property
    def budget(self) -> ExecutionBudget | None:
        return self.context.budget if self.context is not None else None

    @property
    def is_cancelled(self) -> bool:
        return self.context is not None and self.context.is_cancelled

    # -- lifecycle -------------------------------------------------------------

    def _transition(self, target: AgentStatus, reason: str) -> None:
        """Validates and applies one transition, recording the audit trail."""
        assert_transition(self.status, target, context=self.identity.label)
        change = AgentStatusChange(
            from_status=self.status, to_status=target, reason=reason
        )
        self.status = target
        self.metadata.status_history.append(change)

    def assign(self, context: ExecutionContext, reason: str = "assigned") -> None:
        """PENDING -> ASSIGNED: bind the execution context.

        The context's task_id must agree with the state's — the context is
        the scoped view of THIS task, never another one.
        """
        if context.task_id != self.task_id:
            raise ValueError(
                f"Agent {self.identity.label} assigned a context for task "
                f"'{context.task_id}' but the state belongs to task "
                f"'{self.task_id}'"
            )
        self.context = context
        self._transition(AgentStatus.ASSIGNED, reason)

    def start(self, reason: str = "started") -> None:
        """ASSIGNED -> RUNNING: begin executing."""
        if self.metadata.started_at is None:
            self.metadata.started_at = datetime.now(UTC)
            if self.context is not None and self.context.budget.started_at is None:
                self.context.budget.started_at = self.metadata.started_at
        self.metadata.attempts += 1
        self._transition(AgentStatus.RUNNING, reason)

    def wait(self, result: AgentResult | None = None, reason: str = "awaiting input") -> None:
        """RUNNING -> WAITING: pause for confirmation/clarification.

        An explicitly-supplied result must be NEEDS_INPUT — a paused run can
        never carry a terminal result status (mirrors complete()/fail()).
        """
        if result is not None and result.status is not AgentResultStatus.NEEDS_INPUT:
            raise ValueError(
                f"Agent {self.identity.label} paused with a '"
                f"{result.status.value}' result — a WAITING lifecycle requires "
                "a NEEDS_INPUT result"
            )
        if result is not None:
            self.result = result
        self._transition(AgentStatus.WAITING, reason)

    def resume(self, reason: str = "resumed") -> None:
        """WAITING -> RUNNING: continue after the required input.

        A resume starts a new execution cycle, so it counts as another
        attempt (mirrors :meth:`start`).
        """
        self.metadata.attempts += 1
        self._transition(AgentStatus.RUNNING, reason)

    def complete(self, result: AgentResult, reason: str = "completed") -> None:
        """RUNNING -> COMPLETED: the run finished successfully.

        Only a SUCCESS result may complete a run — a completed lifecycle with
        a failed result would be a state corruption. This touches only the
        agent record; the TaskState is untouched (validated in tests).
        """
        if result.status is not AgentResultStatus.SUCCESS:
            raise ValueError(
                f"Agent {self.identity.label} completed with a "
                f"'{result.status.value}' result — only SUCCESS may complete a run"
            )
        self.result = result
        self.metadata.completed_at = datetime.now(UTC)
        self._transition(AgentStatus.COMPLETED, reason)

    def _validate_terminal_result(
        self, result: AgentResult, expected: AgentResultStatus
    ) -> AgentResult:
        """Rejects an explicitly-supplied result whose status contradicts the
        terminal lifecycle it accompanies (mirrors complete())."""
        if result.status is not expected:
            raise ValueError(
                f"Agent {self.identity.label} recorded a "
                f"'{result.status.value}' result for a terminal state that "
                f"requires '{expected.value}'"
            )
        return result

    def fail(
        self,
        result: AgentResult | None = None,
        reason: str = "failed",
    ) -> None:
        """Any active state -> FAILED: the run failed.

        An explicitly-supplied result must itself be FAILED (statuses must
        never contradict the lifecycle).
        """
        self.result = self._validate_terminal_result(
            result or AgentResult(status=AgentResultStatus.FAILED, summary=reason),
            AgentResultStatus.FAILED,
        )
        self.metadata.completed_at = datetime.now(UTC)
        self._transition(AgentStatus.FAILED, reason)

    def block(
        self,
        result: AgentResult | None = None,
        reason: str = "blocked",
    ) -> None:
        """Any active state -> BLOCKED: hard-stopped (security/policy)."""
        self.result = self._validate_terminal_result(
            result or AgentResult(status=AgentResultStatus.BLOCKED, summary=reason),
            AgentResultStatus.BLOCKED,
        )
        self.metadata.completed_at = datetime.now(UTC)
        self._transition(AgentStatus.BLOCKED, reason)

    def cancel(
        self,
        result: AgentResult | None = None,
        reason: str = "cancelled",
    ) -> None:
        """Any active state -> CANCELLED: cancelled by the controller.

        The stored result is forced to CANCELLED — a cancelled run can never
        record a success result, even if the agent ignored the cancellation.
        """
        provided = result or AgentResult(status=AgentResultStatus.CANCELLED, summary=reason)
        if provided.status is not AgentResultStatus.CANCELLED:
            provided = provided.model_copy(
                update={"status": AgentResultStatus.CANCELLED}
            )
        self.result = provided
        self.metadata.completed_at = datetime.now(UTC)
        self._transition(AgentStatus.CANCELLED, reason)

    # -- reporting -------------------------------------------------------------

    def summary(self) -> str:
        """Compact one-line description for logs and debugging."""
        outcome = self.result.status.value if self.result else "-"
        return (
            f"AgentState({self.identity.label}, task={self.task_id}, "
            f"status={self.status.value}, result={outcome}, "
            f"attempts={self.metadata.attempts})"
        )


# Resolve the AgentArtifact forward reference on AgentResult.
# Deferred to the bottom of this module so the dependency stays one-way
# (artifacts.py imports models.py; models.py never imports artifacts at the
# top of the module). The import is placed after every model definition so
# the partially-initialized-module hazard cannot occur.
from ultron.core.orchestration.artifacts import AgentArtifactUnion

AgentResult.model_rebuild(
    _types_namespace={"AgentArtifactUnion": AgentArtifactUnion}
)
