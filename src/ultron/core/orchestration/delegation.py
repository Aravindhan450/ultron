"""
ultron.core.orchestration.delegation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Supervisor + specialist delegation (Fix #7, section 7.4).

The minimal Supervisor for *sequential* delegation:

    TaskState -> Supervisor -> DelegationRequest -> specialist Agent
                            <- AgentResult (structured, with artifact) <-
              -> TaskState update -> decide whether to continue

:class:`DelegationRequest` is the unit of delegated work: it names the
task, the specialist type, the objective, the input artifacts (structured
:class:`~ultron.core.orchestration.artifacts.AgentArtifact` payloads — the
ONLY cross-agent channel, never raw trajectories), constraints, the frozen
permission profile, the expected output, and the budget/timeout.

The delegation lifecycle REUSES :class:`AgentStatus` (section 7.1) — the
states are exactly the ones the delegation needs (PENDING -> ASSIGNED ->
RUNNING -> COMPLETED, with FAILED / BLOCKED / CANCELLED as failure states,
plus WAITING for NEEDS_INPUT pauses) — so there is no second state machine
and no duplicated transition table.

:class:`Supervisor` responsibilities (minimal, sequential-only):

- receive a TaskState and select an agent type from the registry (invalid
  types fail safely with ``KeyError``),
- create the DelegationRequest (budget copied from the type's spec, frozen
  permission profile attached),
- dispatch the specialist inside a runtime-scoped :class:`ExecutionContext`
  (built by the registry — permissions can never be widened by the agent),
- enforce the timeout (``asyncio.wait_for`` + the budget) and cancellation,
- receive the structured AgentResult and its artifact,
- record the outcome on the TaskState (observation / error, never a
  completion claim — the section-7.1 invariant holds),
- decide whether the orchestration should continue.

No parallel execution, no dynamic routing, no workflows — those are later
sections. The supervisor is deliberately deterministic and has NO LLM: it
only routes, scopes, enforces and records.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from ultron.core.logging import get_logger
from ultron.core.orchestration.artifacts import (
    AgentArtifact,
    AgentArtifactUnion,
    ArtifactStore,
    task_key,
)
from ultron.core.orchestration.contract import Agent
from ultron.core.orchestration.lifecycle import AgentStatus, assert_transition
from ultron.core.orchestration.models import (
    AgentResult,
    AgentResultStatus,
    AgentState,
    AgentStatusChange,
    AgentType,
    ExecutionBudget,
)
from ultron.core.orchestration.permissions import AgentPermissions
from ultron.core.orchestration.registry import (
    DEFAULT_REGISTRY,
    AgentRegistry,
    AgentSpec,
)
from ultron.core.types import TaskError, TaskState

logger = get_logger("ultron.orchestration.delegation")

#: How the supervisor obtains a concrete Agent for a delegated run. The
#: factory receives the selected spec and the scoped state and returns an
#: Agent instance bound to that run.
AgentFactory = Callable[[AgentSpec, AgentState], Agent]


class SupervisorDecision(str, Enum):
    """What the orchestrator should do next after one delegation completes.

    - CONTINUE — the delegation succeeded; more work may follow
    - COMPLETE — nothing more to do (available for the workflow section)
    - NEEDS_INPUT — the specialist paused for confirmation/clarification
    - FAILED — the delegation failed / was blocked / was cancelled
    """

    CONTINUE = "continue"
    COMPLETE = "complete"
    NEEDS_INPUT = "needs_input"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Delegation request — the unit of delegated work
# ---------------------------------------------------------------------------


class DelegationRequest(BaseModel):
    """
    One unit of delegated work from the supervisor to a specialist.

    Fields:

    - ``delegation_id`` — unique id (defaults to ``task:type:hex``)
    - ``task_id`` / ``parent_task_id`` — ownership
    - ``agent_type`` — which specialist performs the work
    - ``objective`` — the concrete goal handed to the specialist
    - ``input_artifacts`` — structured artifacts (the ONLY cross-agent
      channel; never other agents' internal reasoning)
    - ``constraints`` — behavioral restrictions (e.g. "read-only")
    - ``permissions`` — the specialist type's frozen permission profile
      (attached by the supervisor at creation; runtime-controlled)
    - ``expected_output`` — description of the expected result/artifact
    - ``budget`` — per-run limits (copied from the type's spec)
    - ``timeout_seconds`` — wall-clock limit (folded into the budget)

    The lifecycle REUSES :class:`AgentStatus` — PENDING -> ASSIGNED ->
    RUNNING -> (WAITING) -> COMPLETED, with FAILED / BLOCKED / CANCELLED
    as failure states. Transitions are validated; terminal states accept
    nothing; result status is locked to the lifecycle status (a completed
    delegation always carries a SUCCESS result).
    """

    delegation_id: str = ""
    task_id: str
    parent_task_id: str | None = None
    agent_type: AgentType
    objective: str
    input_artifacts: list[AgentArtifactUnion] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    permissions: AgentPermissions | None = None
    expected_output: str = ""
    budget: ExecutionBudget = Field(default_factory=ExecutionBudget)
    timeout_seconds: int | None = None
    status: AgentStatus = AgentStatus.PENDING
    result: AgentResult | None = None
    agent_state_id: str | None = None
    error: str | None = None
    status_history: list[AgentStatusChange] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def model_post_init(self, context: Any) -> None:
        if not self.delegation_id:
            self.delegation_id = (
                f"{self.task_id}:{self.agent_type.value}:{uuid.uuid4().hex[:6]}"
            )

    @property
    def run_agent_id(self) -> str:
        """Agent id used for the specialist run spawned by this delegation."""
        return f"{self.agent_type.value}-{self.delegation_id}"

    # -- lifecycle (reuses the section-7.1 AgentStatus machine) ---------------

    def _transition(self, target: AgentStatus, reason: str) -> None:
        assert_transition(self.status, target, context=f"delegation {self.delegation_id}")
        self.status_history.append(
            AgentStatusChange(
                from_status=self.status, to_status=target, reason=reason
            )
        )
        self.status = target
        self.updated_at = datetime.now(UTC)

    def assign(self, reason: str = "assigned") -> None:
        """PENDING -> ASSIGNED: bound to a specialist run."""
        self._transition(AgentStatus.ASSIGNED, reason)

    def start(self, reason: str = "running") -> None:
        """ASSIGNED (or WAITING, on resume) -> RUNNING."""
        self._transition(AgentStatus.RUNNING, reason)

    def wait(self, result: AgentResult, reason: str = "awaiting input") -> None:
        """RUNNING -> WAITING: the specialist paused for input.

        Mirrors AgentState.wait: a paused delegation can only carry a
        NEEDS_INPUT result.
        """
        if result.status is not AgentResultStatus.NEEDS_INPUT:
            raise ValueError(
                f"Delegation {self.delegation_id} paused with a "
                f"'{result.status.value}' result — WAITING requires NEEDS_INPUT"
            )
        self.result = result
        self._transition(AgentStatus.WAITING, reason)

    def complete(self, result: AgentResult, reason: str = "completed") -> None:
        """RUNNING -> COMPLETED: only a SUCCESS result may complete a run."""
        if result.status is not AgentResultStatus.SUCCESS:
            raise ValueError(
                f"Delegation {self.delegation_id} completed with a "
                f"'{result.status.value}' result — only SUCCESS may complete"
            )
        self.result = result
        self._transition(AgentStatus.COMPLETED, reason)

    def _validate_terminal_result(
        self, result: AgentResult | None, expected: AgentResultStatus
    ) -> AgentResult:
        if result is None:
            return AgentResult(status=expected, summary="")
        if result.status is not expected:
            raise ValueError(
                f"Delegation {self.delegation_id} recorded a "
                f"'{result.status.value}' result for a terminal state that "
                f"requires '{expected.value}'"
            )
        return result

    def fail(
        self,
        result: AgentResult | None = None,
        reason: str = "failed",
    ) -> None:
        """Any active state -> FAILED."""
        result = result or AgentResult(status=AgentResultStatus.FAILED, summary=reason)
        self.result = self._validate_terminal_result(result, AgentResultStatus.FAILED)
        self.error = reason
        self._transition(AgentStatus.FAILED, reason)

    def block(
        self,
        result: AgentResult | None = None,
        reason: str = "blocked",
    ) -> None:
        """Any active state -> BLOCKED."""
        result = result or AgentResult(status=AgentResultStatus.BLOCKED, summary=reason)
        self.result = self._validate_terminal_result(result, AgentResultStatus.BLOCKED)
        self.error = reason
        self._transition(AgentStatus.BLOCKED, reason)

    def cancel(
        self,
        result: AgentResult | None = None,
        reason: str = "cancelled",
    ) -> None:
        """Any active state -> CANCELLED (result status forced to CANCELLED).

        Note (timeout semantics): a timed-out run ends with the DELEGATION
        record FAILED (carrying the timeout metadata — authoritative) while
        the underlying AgentState ends CANCELLED; the state's stored result
        is coerced to CANCELLED and does not carry timeout evidence.
        """
        provided = result or AgentResult(status=AgentResultStatus.CANCELLED, summary=reason)
        if provided.status is not AgentResultStatus.CANCELLED:
            provided = provided.model_copy(
                update={"status": AgentResultStatus.CANCELLED}
            )
        self.result = provided
        self.error = reason
        self._transition(AgentStatus.CANCELLED, reason)

    # -- reporting ------------------------------------------------------------

    @property
    def is_terminal(self) -> bool:
        return self.status.is_terminal

    def summary(self) -> str:
        outcome = self.result.status.value if self.result else "-"
        return (
            f"DelegationRequest({self.delegation_id}, type={self.agent_type.value}, "
            f"status={self.status.value}, result={outcome})"
        )


# ---------------------------------------------------------------------------
# Task brief — the relevant TaskState subset (context isolation)
# ---------------------------------------------------------------------------


def task_brief(task_state: TaskState | None) -> dict[str, Any]:
    """
    The relevant, string-only subset of a TaskState handed to a specialist.

    Includes the goal, task type, step tracking, requirement descriptions,
    and recent errors — NEVER the transcript, execution history, pending
    action, or full plan objects. The specialist does not need (and must
    not receive) the task's internal conversation.
    """
    if task_state is None:
        return {}
    step = task_state.current_plan_step()
    return {
        "goal": task_state.goal,
        "task_type": task_state.task_type.value if task_state.task_type else None,
        "current_step": task_state.current_step,
        "total_steps": task_state.total_steps,
        "plan_step": step.objective if step else None,
        "requirements": [r.description for r in task_state.requirements],
        "recent_errors": [e.message for e in task_state.errors[-3:]],
    }


def _artifact_brief(artifact: AgentArtifact) -> dict[str, Any]:
    """Compact, structured view of one artifact for a specialist's context."""
    return {
        "artifact_id": artifact.artifact_id,
        "type": artifact.artifact_type.value,
        "summary": artifact.summary,
        "related_files": list(artifact.related_files),
        "related_symbols": list(artifact.related_symbols),
        "source": artifact.source,
        "confidence": artifact.confidence.value if artifact.confidence else None,
    }


# ---------------------------------------------------------------------------
# Supervisor — minimal, sequential, deterministic
# ---------------------------------------------------------------------------


class Supervisor:
    """
    Minimal sequential supervisor for specialist delegation.

    - receives a TaskState (optional for tests),
    - selects an agent type from the registry (unknown -> KeyError),
    - creates a :class:`DelegationRequest` (budget + frozen permissions from
      the type's spec),
    - dispatches the specialist inside a scoped ExecutionContext,
    - enforces timeout (``asyncio.wait_for`` + budget) and cancellation,
    - records the structured result + artifact,
    - updates the TaskState (observation / error — never a completion claim),
    - decides whether orchestration should continue.

    ``agent_factory`` supplies concrete Agent implementations (the real
    specialists are later sections; until then callers provide fakes).
    Without a factory, dispatch raises ``ValueError``.
    """

    def __init__(
        self,
        registry: AgentRegistry | None = None,
        agent_factory: AgentFactory | None = None,
        store: ArtifactStore | None = None,
    ) -> None:
        self.registry = registry or DEFAULT_REGISTRY
        self.agent_factory = agent_factory
        self.store = store  # persists produced artifacts
        self._delegations: dict[str, DelegationRequest] = {}
        self._runs: dict[str, AgentState] = {}

    # -- delegation creation ---------------------------------------------------

    def create_delegation(
        self,
        task_state: TaskState | None,
        agent_type: AgentType | str,
        objective: str,
        *,
        parent_task_id: str | None = None,
        input_artifacts: list[AgentArtifact] | None = None,
        constraints: list[str] | None = None,
        expected_output: str = "",
        timeout_seconds: int | None = None,
    ) -> DelegationRequest:
        """
        Selects the specialist (invalid types raise KeyError) and builds a
        PENDING DelegationRequest with the type's budget + frozen permissions.
        """
        spec = self.registry.get(agent_type)  # KeyError → invalid agent type
        budget = spec.max_budget.model_copy(deep=True)
        if timeout_seconds is not None:
            budget.timeout_seconds = timeout_seconds
        request = DelegationRequest(
            task_id=task_key(task_state) if task_state is not None else "task",
            parent_task_id=parent_task_id,
            agent_type=spec.agent_type,
            objective=objective,
            input_artifacts=list(input_artifacts or []),
            constraints=list(constraints or []),
            permissions=spec.permissions,
            expected_output=expected_output,
            budget=budget,
            timeout_seconds=budget.timeout_seconds,
        )
        self._delegations[request.delegation_id] = request
        logger.info(
            "supervisor: created delegation %s (%s)", request.delegation_id, spec.name
        )
        return request

    # -- dispatch --------------------------------------------------------------

    async def dispatch(
        self,
        request: DelegationRequest,
        workspace: str = "",
        task_state: TaskState | None = None,
    ) -> DelegationRequest:
        """
        Runs one delegation to a terminal (or WAITING) state and returns it.

        Accepts a PENDING request (fresh run) or a WAITING request (resume
        after NEEDS_INPUT). Enforces the timeout with ``asyncio.wait_for``
        and the budget; persists the produced artifact; records the outcome
        on the TaskState.
        """
        if self.agent_factory is None:
            raise ValueError(
                "no agent_factory registered — cannot instantiate a specialist "
                f"for {request.agent_type.value}"
            )

        if request.status is AgentStatus.PENDING:
            state = self._spawn_run(request, workspace, task_state)
            request.assign(reason="dispatched")
            request.start(reason="running")
        elif request.status is AgentStatus.WAITING:
            state = self._runs.get(request.delegation_id)
            if state is None:
                raise ValueError(
                    f"cannot resume delegation {request.delegation_id}: no run record"
                )
            # Keep the specialist's view current (artifacts may have grown).
            state.context.relevant_context = self._build_relevant_context(
                request, task_state
            )
            request.start(reason="resumed")
        else:
            raise ValueError(
                f"cannot dispatch delegation {request.delegation_id}: "
                f"status is '{request.status.value}' (need PENDING or WAITING)"
            )

        agent = self.agent_factory(self.registry.get(request.agent_type), state)

        try:
            if request.timeout_seconds is not None:
                result = await asyncio.wait_for(
                    agent.run_with_state(state), timeout=request.timeout_seconds
                )
            else:
                result = await agent.run_with_state(state)
        except TimeoutError:
            return self._finalize_timeout(request, state, task_state)

        return self._finalize(request, state, result, task_state)

    def _spawn_run(
        self,
        request: DelegationRequest,
        workspace: str,
        task_state: TaskState | None = None,
    ) -> AgentState:
        """Builds the scoped AgentState via the registry (permissions applied)."""
        state = self.registry.instantiate(
            request.agent_type,
            request.run_agent_id,
            request.task_id,
            request.objective,
            workspace=workspace,
        )
        state.context.relevant_context = self._build_relevant_context(
            request, task_state
        )
        self._runs[request.delegation_id] = state
        request.agent_state_id = state.agent_id
        return state

    def _build_relevant_context(
        self, request: DelegationRequest, task_state: TaskState | None = None
    ) -> dict[str, Any]:
        """Context isolation: task brief + artifact summaries + constraints.

        Deliberately excludes the full TaskState, the conversation transcript,
        execution history, and any other agents' internal reasoning — the
        input artifacts are the only cross-agent channel.
        """
        return {
            "task": task_brief(task_state),
            "artifacts": [_artifact_brief(a) for a in request.input_artifacts],
            "constraints": list(request.constraints),
            "expected_output": request.expected_output,
        }

    def _finalize_timeout(
        self,
        request: DelegationRequest,
        state: AgentState,
        task_state: TaskState | None,
    ) -> DelegationRequest:
        state.context.request_cancel()
        result = AgentResult(
            status=AgentResultStatus.FAILED,
            summary=(
                f"specialist {request.agent_type.value} timed out after "
                f"{request.timeout_seconds}s"
            ),
            metadata={"timeout": True, "delegation_id": request.delegation_id},
        )
        state.cancel(result, reason="timeout")
        request.fail(result, reason=f"timeout after {request.timeout_seconds}s")
        self._persist_result_artifact(request)
        self._record_on_task(task_state, request)
        logger.warning("supervisor: delegation %s timed out", request.delegation_id)
        return request

    def _finalize(
        self,
        request: DelegationRequest,
        state: AgentState,
        result: AgentResult,
        task_state: TaskState | None,
    ) -> DelegationRequest:
        request.result = result
        if request.budget.timed_out():
            result = AgentResult(
                status=AgentResultStatus.FAILED,
                summary="specialist exceeded its execution budget",
                metadata={"timeout": True, "budget": request.budget.summary()},
            )
            state.cancel(result, reason="budget timed out")
            request.fail(result, reason="budget timed out")
        elif state.context.is_cancelled or result.status is AgentResultStatus.CANCELLED:
            request.cancel(result, reason="cancelled")
        elif result.status is AgentResultStatus.SUCCESS:
            request.complete(result)
        elif result.status is AgentResultStatus.NEEDS_INPUT:
            request.wait(result)
        elif result.status is AgentResultStatus.FAILED:
            request.fail(result)
        elif result.status is AgentResultStatus.BLOCKED:
            request.block(result)
        else:  # pragma: no cover — defensive; all statuses handled above
            request.cancel(result)
        self._persist_result_artifact(request)
        self._record_on_task(task_state, request)
        logger.info("supervisor: delegation %s -> %s", request.delegation_id, request.status.value)
        return request

    def _persist_result_artifact(self, request: DelegationRequest) -> None:
        """Persists the produced artifact (if any) to the store."""
        if self.store is None or request.result is None or request.result.artifact is None:
            return
        try:
            self.store.save(request.result.artifact)
        except ValueError as exc:  # duplicate artifact id — already stored
            logger.warning("supervisor: artifact not re-persisted: %s", exc)

    # -- cancellation ----------------------------------------------------------

    def cancel_delegation(
        self, request: DelegationRequest, reason: str = "cancelled by supervisor"
    ) -> DelegationRequest:
        """
        Cancels a delegation from any active state.

        Mid-run (RUNNING) cancellation only sets the context flag so the
        specialist can stop at its next safe checkpoint; the CANCELLED
        transition happens in :meth:`_finalize` when the run returns (this
        keeps the delegation record consistent with the actual run and
        avoids double-transitioning a terminal state). PENDING / ASSIGNED /
        WAITING delegations are transitioned to CANCELLED immediately.
        """
        if request.is_terminal:
            raise ValueError(
                f"cannot cancel delegation {request.delegation_id}: already "
                f"'{request.status.value}'"
            )
        state = self._runs.get(request.delegation_id)
        if state is not None and state.context is not None and request.status is AgentStatus.RUNNING:
            # Mid-run: ask the specialist to stop at its next safe checkpoint.
            # NOTE: if no timeout is set and the specialist never returns, the
            # delegation stays RUNNING until the run finishes — the run itself
            # is the only thing that can end it.
            state.context.request_cancel()
            request.error = reason
            logger.info(
                "supervisor: cancel requested for running delegation %s (%s)",
                request.delegation_id,
                reason,
            )
            return request
        # Not running (PENDING / ASSIGNED / WAITING): end both the delegation
        # and its parked run record, so no stale state is left behind.
        if state is not None and not state.is_terminal:
            state.cancel(reason=reason)
        request.cancel(reason=reason)
        logger.info("supervisor: cancelled delegation %s (%s)", request.delegation_id, reason)
        return request

    # -- TaskState update + continuation ---------------------------------------

    def _record_on_task(self, task_state: TaskState | None, request: DelegationRequest) -> None:
        """Records the delegation outcome on the TaskState.

        Updates the observation/error surface only — the supervisor NEVER
        marks a task complete (agent completion != task completion).
        """
        if task_state is None:
            return
        tag = f"[delegate:{request.agent_type.value}]"
        summary = (request.result.summary if request.result else "") or ""
        if request.status is AgentStatus.COMPLETED:
            task_state.last_observation = f"{tag} {summary}".strip()
        elif request.status is AgentStatus.WAITING:
            task_state.last_observation = f"{tag} awaiting input: {summary}".strip()
        else:  # FAILED / BLOCKED / CANCELLED
            note = f"{tag} {request.status.value}: {summary or request.error or ''}".strip()
            task_state.last_observation = note
            if request.status in (AgentStatus.FAILED, AgentStatus.BLOCKED):
                task_state.errors.append(
                    TaskError(message=note, step=task_state.current_step)
                )
        task_state.updated_at = datetime.now(UTC)

    def decide(self, request: DelegationRequest) -> SupervisorDecision:
        """Whether the orchestration should continue after this delegation.

        Sequential scope: a successful delegation means CONTINUE (more work
        may follow); NEEDS_INPUT pauses; anything else stops. COMPLETE is
        reserved for the workflow section, which verifies task requirements.
        """
        if request.result is None:
            return SupervisorDecision.CONTINUE
        status = request.result.status
        if status is AgentResultStatus.NEEDS_INPUT:
            return SupervisorDecision.NEEDS_INPUT
        if status is not AgentResultStatus.SUCCESS:
            return SupervisorDecision.FAILED
        return SupervisorDecision.CONTINUE

    def get_delegation(self, delegation_id: str) -> DelegationRequest | None:
        return self._delegations.get(delegation_id)

    def get_run(self, delegation_id: str) -> AgentState | None:
        """The specialist AgentState spawned for a delegation (validation uses
        it to inspect the run record). None when the delegation is unknown or
        has not been dispatched (or the run record is gone after a restart)."""
        return self._runs.get(delegation_id)
