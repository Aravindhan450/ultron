"""State-progress and thrashing analyzer for autonomous task execution.

Evaluates whether individual execution steps made meaningful progress, caused regressions,
or produced zero observable change (thrashing).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ProgressKind(str, Enum):
    PROGRESS = "PROGRESS"
    NO_PROGRESS = "NO_PROGRESS"
    REGRESSION = "REGRESSION"


@dataclass
class StateTransition:
    """One observed state transition before and after a model/tool action."""

    step_index: int
    tool: str
    target: str
    state_before: dict[str, Any]
    state_after: dict[str, Any]
    kind: ProgressKind
    reason: str
    delta: dict[str, Any] = field(default_factory=dict)


class StateProgressAnalyzer:
    """Deterministic analyzer of progress across action steps."""

    @staticmethod
    def analyze_step(
        step_index: int,
        tool: str,
        target: str,
        state_before: dict[str, Any],
        state_after: dict[str, Any],
        success: bool = True,
        error: str | None = None,
    ) -> StateTransition:
        """Determines progress kind by comparing before/after state snapshots."""
        files_before = set(state_before.get("files", []))
        files_after = set(state_after.get("files", []))
        
        new_files = files_after - files_before
        removed_files = files_before - files_after
        
        errors_before = state_before.get("last_error")

        if new_files:

            return StateTransition(
                step_index=step_index,
                tool=tool,
                target=target,
                state_before=state_before,
                state_after=state_after,
                kind=ProgressKind.PROGRESS,
                reason=f"Created new artifacts: {list(new_files)}",
                delta={"new_files": list(new_files)},
            )

        if success and tool in ("replace_in_file", "overwrite_file", "create_file"):
            return StateTransition(
                step_index=step_index,
                tool=tool,
                target=target,
                state_before=state_before,
                state_after=state_after,
                kind=ProgressKind.PROGRESS,
                reason=f"Modified existing file '{target}' successfully.",
                delta={"modified": target},
            )

        if not success:
            if errors_before and errors_before == error:
                return StateTransition(
                    step_index=step_index,
                    tool=tool,
                    target=target,
                    state_before=state_before,
                    state_after=state_after,
                    kind=ProgressKind.NO_PROGRESS,
                    reason=f"Action failed with identical error as previous step ('{error}').",
                    delta={"repeated_error": error},
                )
            return StateTransition(
                step_index=step_index,
                tool=tool,
                target=target,
                state_before=state_before,
                state_after=state_after,
                kind=ProgressKind.NO_PROGRESS,
                reason=f"Action failed: {error or 'Unknown error'}",
                delta={"error": error},
            )

        if removed_files:
            return StateTransition(
                step_index=step_index,
                tool=tool,
                target=target,
                state_before=state_before,
                state_after=state_after,
                kind=ProgressKind.REGRESSION,
                reason=f"Artifacts unexpectedly removed: {list(removed_files)}",
                delta={"removed": list(removed_files)},
            )

        return StateTransition(
            step_index=step_index,
            tool=tool,
            target=target,
            state_before=state_before,
            state_after=state_after,
            kind=ProgressKind.PROGRESS,
            reason="Step completed with observable state consistency.",
        )
