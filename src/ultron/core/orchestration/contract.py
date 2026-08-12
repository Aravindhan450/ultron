"""
ultron.core.orchestration.contract
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The base Agent contract (Fix #7, section 7.1).

Every orchestrated agent (research, coding, test/QA, reviewer, security,
supervisor) implements this contract:

- carries an :class:`AgentIdentity` (agent_id + agent_type),
- executes a concrete ``objective`` within a scoped
  :class:`ExecutionContext`,
- returns a structured :class:`AgentResult` (never a bare string),
- cooperates with cancellation.

The contract deliberately does NOT know about TaskState, engines, tools, or
prompts. The supervisor (a later section) will drive agents through the
lifecycle via :class:`AgentState`; this section only fixes the interface
both sides speak. Section 7.1 intentionally implements no supervisor, no
delegation, no parallel execution and no workflows.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ultron.core.orchestration.lifecycle import AgentStatus
from ultron.core.orchestration.models import (
    AgentIdentity,
    AgentResult,
    AgentResultStatus,
    ExecutionContext,
)


class Agent(ABC):
    """
    The contract every orchestrated agent implements.

    Implementations are expected to:

    - respect the scoped ``context`` (only ``allowed_tools``, the budget,
      and the permission profile) and honor ``context.is_cancelled`` at the
      next safe checkpoint;
    - return a structured :class:`AgentResult` whose ``status`` matches the
      outcome (SUCCESS / FAILED / BLOCKED / CANCELLED / NEEDS_INPUT);
    - never bypass the security boundary and never claim task completion —
      an agent result is evidence for the controller, not a task verdict.
    """

    def __init__(self, identity: AgentIdentity) -> None:
        self.identity = identity

    @abstractmethod
    async def execute(
        self,
        objective: str,
        context: ExecutionContext,
    ) -> AgentResult:
        """
        Performs ``objective`` within ``context`` and returns the outcome.

        Raises are allowed for genuinely exceptional conditions (the caller
        converts them into a FAILED result), but the contract prefers a
        structured FAILED / BLOCKED / NEEDS_INPUT result over exceptions for
        expected outcomes.
        """

    async def cancel(self) -> None:
        """Best-effort cancellation hook; the default implementation is a
        no-op. Agents that run long operations should override this to
        interrupt them at the next checkpoint."""

    def describe(self) -> str:
        """One-line description of the agent for logs and prompts."""
        return self.identity.to_prompt_line()

    async def run_with_state(self, state) -> AgentResult:
        """
        Convenience: executes the agent against an :class:`AgentState` and
        drives it to a terminal lifecycle state.

        The state's status is validated at every step; the returned result
        is also stored on the state. This is the entry point a (future)
        supervisor will call; it can equally be used standalone in tests.

        Resume support: the state may be started from ANY active state — a
        PENDING state is assigned + started, an ASSIGNED state is started,
        a WAITING state (left after a NEEDS_INPUT result) is resumed, and a
        RUNNING state continues. The state must NOT already be terminal.
        """
        if state.context is None:
            raise ValueError(f"AgentState for {state.identity.label} has no context")
        if state.is_terminal:
            raise ValueError(
                f"AgentState for {state.identity.label} is already terminal "
                f"('{state.status.value}')"
            )
        if state.status is AgentStatus.PENDING:
            state.assign(state.context)
            state.start()
        elif state.status is AgentStatus.ASSIGNED:
            state.start()
        elif state.status is AgentStatus.WAITING:
            state.resume()
        # RUNNING: continue the run without re-starting.
        try:
            result = await self.execute(state.objective, state.context)
        except Exception as exc:  # noqa: BLE001 — contract-level safety net
            result = AgentResult(
                status=AgentResultStatus.FAILED,
                summary=f"agent raised an exception: {exc}",
                metadata={"exception": str(exc)},
            )
        if state.context.is_cancelled:
            # The controller asked to stop — even an agent that ignored the
            # flag must not be recorded as successful: coerce to CANCELLED.
            if result is not None and result.status is not AgentResultStatus.CANCELLED:
                result = result.model_copy(
                    update={"status": AgentResultStatus.CANCELLED}
                )
            state.cancel(result, reason="cancelled")
        elif result.status is AgentResultStatus.SUCCESS:
            state.complete(result)
        elif result.status is AgentResultStatus.NEEDS_INPUT:
            state.wait(result, reason="needs input")
        elif result.status is AgentResultStatus.FAILED:
            state.fail(result)
        elif result.status is AgentResultStatus.BLOCKED:
            state.block(result)
        else:
            state.cancel(result)
        return result

