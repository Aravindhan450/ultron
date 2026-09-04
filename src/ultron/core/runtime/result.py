"""
ultron.core.runtime.result
~~~~~~~~~~~~~~~~~~~~~~~~~~

Structured outcome returned by AgentRuntime executions.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ultron.core.runtime.state import RunState, RuntimeStatus
from ultron.core.types import ChatMessage, TaskState


class RunResult(BaseModel):
    """
    Structured outcome of an AgentRuntime run.
    """

    run_id: str
    status: RuntimeStatus
    message: ChatMessage | None = None
    task_state: TaskState | None = None
    run_state: RunState
    evidence: list[str] = Field(default_factory=list)
    changed_files: list[str] = Field(default_factory=list)
    error: str | None = None
    termination_reason: str = ""

    @property
    def is_success(self) -> bool:
        return self.status is RuntimeStatus.COMPLETED

    @property
    def is_terminal(self) -> bool:
        return self.status.is_terminal
