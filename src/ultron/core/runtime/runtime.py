"""
ultron.core.runtime.runtime
~~~~~~~~~~~~~~~~~~~~~~~~~~~

AgentRuntime — Lifecycle and orchestration boundary for Ultron agents.

Owns:
- run ID and lifecycle state machine (CREATED -> RUNNING -> COMPLETED / FAILED / etc.)
- execution budget enforcement (iterations, tool calls, delegations, timeout)
- cooperative cancellation propagation
- event bus emissions across execution checkpoints
- coordination with BaseAgent (ReActAgent / SimpleAgent)
- structured RunResult generation
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

from ultron.core.agents.base import BaseAgent
from ultron.core.context.manager import RepositoryContextManager
from ultron.core.logging import get_logger
from ultron.core.runtime.budget import RuntimeBudget
from ultron.core.runtime.cancellation import CancellationToken
from ultron.core.runtime.events import EventBus, RuntimeEvent, RuntimeEventType
from ultron.core.runtime.result import RunResult
from ultron.core.runtime.state import RunState, RuntimeStatus
from ultron.core.types import ChatMessage, TaskState

logger = get_logger("ultron.runtime")


class AgentRuntime:
    """
    Orchestrates execution of an agent within a bounded, observable runtime.
    """

    def __init__(
        self,
        event_bus: EventBus | None = None,
        default_budget: RuntimeBudget | None = None,
        context_manager: RepositoryContextManager | None = None,
    ) -> None:
        self.event_bus = event_bus or EventBus()
        self.default_budget = default_budget or RuntimeBudget()
        self.context_manager = context_manager or RepositoryContextManager()

    async def execute(
        self,
        agent: BaseAgent,
        user_input: str,
        history: list[ChatMessage] | None = None,
        task: TaskState | None = None,
        budget: RuntimeBudget | None = None,
        cancellation_token: CancellationToken | None = None,
        session: Any | None = None,
    ) -> RunResult:
        """
        Executes an agent run through the runtime lifecycle.
        """
        run_id = f"run_{uuid.uuid4().hex[:8]}"
        task_id = getattr(task, "goal", None) if task else None
        active_budget = (budget.model_copy(deep=True) if budget else self.default_budget.model_copy(deep=True))
        token = cancellation_token or CancellationToken()

        run_state = RunState(
            run_id=run_id,
            task_id=task_id,
            budget=active_budget,
        )

        # Transition CREATED -> INITIALIZING -> RUNNING
        run_state.transition_to(RuntimeStatus.INITIALIZING)
        await self.event_bus.emit(
            RuntimeEvent(
                event_type=RuntimeEventType.RUN_STARTED,
                run_id=run_id,
                task_id=task_id,
                payload={"goal": user_input, "budget": active_budget.summary()},
            )
        )

        run_state.transition_to(RuntimeStatus.RUNNING)

        # Assemble context snapshot for this run using canonical RepositoryContextManager
        code_ctx = getattr(task, "code_context", None) if task else None
        context_snapshot = self.context_manager.assemble_snapshot(
            user_request=user_input,
            task=task,
            code_context=code_ctx,
            session=session,
        )

        # Check early cancellation
        if token.is_cancelled:
            run_state.request_cancellation(token.reason or "Cancelled before start")
            run_state.transition_to(RuntimeStatus.CANCELLED)
            await self.event_bus.emit(
                RuntimeEvent(
                    event_type=RuntimeEventType.RUN_CANCELLED,
                    run_id=run_id,
                    task_id=task_id,
                    payload={"reason": token.reason},
                )
            )
            return RunResult(
                run_id=run_id,
                status=RuntimeStatus.CANCELLED,
                run_state=run_state,
                context_snapshot=context_snapshot,
                termination_reason=token.reason or "Cancelled before start",
            )

        # Prepare execution coroutine with agent
        # We pass task/session if the agent's run() supports it (like ReActAgent)
        async def _run_agent() -> ChatMessage:
            kwargs: dict[str, Any] = {}
            import inspect
            sig = inspect.signature(agent.run)
            if "task" in sig.parameters:
                kwargs["task"] = task
            if "session" in sig.parameters:
                kwargs["session"] = session

            return await agent.run(user_input, history, **kwargs)

        agent_coro = _run_agent()
        timeout_seconds = active_budget.timeout_seconds

        try:
            if timeout_seconds is not None and timeout_seconds > 0:
                response_msg = await asyncio.wait_for(agent_coro, timeout=timeout_seconds)
            else:
                response_msg = await agent_coro

            # Successful completion or confirmation yield
            resolved_task = getattr(response_msg, "task_state", None) or task
            
            # Extract changed files and evidence from task if available
            changed_files: list[str] = []
            evidence: list[str] = []
            if resolved_task is not None:
                if resolved_task.code_context is not None:
                    changed_files = list(resolved_task.code_context.tracker.modified_files)
                for tool_ex in resolved_task.execution_history:
                    evidence.append(f"{tool_ex.tool_name}({tool_ex.target}) -> success={tool_ex.success}")
                    run_state.budget.record_tool_call(1)

            run_state.transition_to(RuntimeStatus.COMPLETED)
            await self.event_bus.emit(
                RuntimeEvent(
                    event_type=RuntimeEventType.RUN_COMPLETED,
                    run_id=run_id,
                    task_id=task_id,
                    payload={
                        "has_pending_action": response_msg.pending_action is not None,
                        "changed_files_count": len(changed_files),
                    },
                )
            )

            return RunResult(
                run_id=run_id,
                status=RuntimeStatus.COMPLETED,
                message=response_msg,
                task_state=resolved_task,
                context_snapshot=context_snapshot,
                run_state=run_state,
                changed_files=changed_files,
                evidence=evidence,
                termination_reason="Success",
            )

        except TimeoutError:
            run_state.transition_to(RuntimeStatus.TIMED_OUT, error=f"Run timed out after {timeout_seconds}s")
            await self.event_bus.emit(
                RuntimeEvent(
                    event_type=RuntimeEventType.RUN_FAILED,
                    run_id=run_id,
                    task_id=task_id,
                    payload={"error": run_state.error, "status": RuntimeStatus.TIMED_OUT.value},
                )
            )
            return RunResult(
                run_id=run_id,
                status=RuntimeStatus.TIMED_OUT,
                task_state=task,
                context_snapshot=context_snapshot,
                run_state=run_state,
                error=run_state.error,
                termination_reason="Execution timed out",
            )

        except asyncio.CancelledError:
            run_state.request_cancellation(token.reason or "Async task cancelled")
            run_state.transition_to(RuntimeStatus.CANCELLED)
            await self.event_bus.emit(
                RuntimeEvent(
                    event_type=RuntimeEventType.RUN_CANCELLED,
                    run_id=run_id,
                    task_id=task_id,
                    payload={"reason": run_state.cancellation_reason},
                )
            )
            return RunResult(
                run_id=run_id,
                status=RuntimeStatus.CANCELLED,
                task_state=task,
                context_snapshot=context_snapshot,
                run_state=run_state,
                termination_reason=run_state.cancellation_reason or "Run cancelled",
            )

        except Exception as exc:
            err_msg = str(exc)
            logger.exception(f"AgentRuntime run {run_id} failed: {err_msg}")
            run_state.transition_to(RuntimeStatus.FAILED, error=err_msg)
            await self.event_bus.emit(
                RuntimeEvent(
                    event_type=RuntimeEventType.RUN_FAILED,
                    run_id=run_id,
                    task_id=task_id,
                    payload={"error": err_msg},
                )
            )
            return RunResult(
                run_id=run_id,
                status=RuntimeStatus.FAILED,
                task_state=task,
                context_snapshot=context_snapshot,
                run_state=run_state,
                error=err_msg,
                termination_reason=f"Runtime error: {err_msg}",
            )
