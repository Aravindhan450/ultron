"""
ultron.core.runtime.budget
~~~~~~~~~~~~~~~~~~~~~~~~~~

Execution budget model for AgentRuntime runs.

Tracks and enforces:
- max_iterations (reasoning loops)
- max_tool_calls (tool invocations)
- max_delegations (sub-agent / supervisor delegations)
- timeout_seconds (wall-clock execution timeout)
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel


class RuntimeBudget(BaseModel):
    """
    Execution constraints for an AgentRuntime execution.
    """

    max_iterations: int = 20
    max_tool_calls: int = 50
    max_delegations: int = 10
    timeout_seconds: float | None = None

    iterations_used: int = 0
    tool_calls_used: int = 0
    delegations_used: int = 0
    started_at: datetime | None = None

    def record_iteration(self, count: int = 1) -> None:
        """Records one reasoning/action iteration."""
        self.iterations_used += max(count, 0)

    def record_tool_call(self, count: int = 1) -> None:
        """Records one tool execution."""
        self.tool_calls_used += max(count, 0)

    def record_delegation(self, count: int = 1) -> None:
        """Records one sub-agent / supervisor delegation."""
        self.delegations_used += max(count, 0)

    def is_exhausted(self) -> bool:
        """Returns True if any execution limit has been reached or exceeded."""
        return (
            self.iterations_used >= self.max_iterations
            or self.tool_calls_used >= self.max_tool_calls
            or self.delegations_used >= self.max_delegations
        )

    def exhaustion_reason(self) -> str | None:
        """Returns the specific limit that was exhausted, if any."""
        if self.iterations_used >= self.max_iterations:
            return f"Exceeded max_iterations limit ({self.iterations_used}/{self.max_iterations})"
        if self.tool_calls_used >= self.max_tool_calls:
            return f"Exceeded max_tool_calls limit ({self.tool_calls_used}/{self.max_tool_calls})"
        if self.delegations_used >= self.max_delegations:
            return f"Exceeded max_delegations limit ({self.delegations_used}/{self.max_delegations})"
        return None

    def is_timed_out(self, now: datetime | None = None) -> bool:
        """Returns True if the wall-clock execution limit has expired."""
        if self.timeout_seconds is None or self.started_at is None:
            return False
        current_time = now or datetime.now(UTC)
        elapsed = (current_time - self.started_at).total_seconds()
        return elapsed > self.timeout_seconds

    def summary(self) -> str:
        """Human-readable budget consumption summary."""
        parts = [
            f"iterations: {self.iterations_used}/{self.max_iterations}",
            f"tools: {self.tool_calls_used}/{self.max_tool_calls}",
            f"delegations: {self.delegations_used}/{self.max_delegations}",
        ]
        if self.timeout_seconds is not None:
            parts.append(f"timeout: {self.timeout_seconds}s")
        return ", ".join(parts)
