"""ultron.core.coding.context
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CodeContext — the structured execution context for a coding task.

It keeps the coding-relevant state SEPARATE from the raw conversation
history: the discovered workspace, the relevant files the agent identified,
structured observations (file content, command results, test/build results,
errors, diffs, repository state), and the modification tracker.

The context associates with the Fix #1/#2 task machinery WITHOUT duplicating
it: it references the task goal, task type and current plan step id (copied
at attach time), while the TaskState remains the runtime source of truth.
This context object survives confirmation because it lives on the TaskState
(the CLI hands the task back through continue_task_after_confirmation).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ultron.core.coding.edits import ModificationTracker
from ultron.core.coding.executor import CodingExecutor
from ultron.core.coding.intelligence_bridge import CodeIntelligenceBridge
from ultron.core.coding.observations import Observation, ObservationKind
from ultron.core.coding.workspace import CodingWorkspace


class CodeContext(BaseModel):
    """Structured coding execution context attached to a TaskState."""

    workspace: CodingWorkspace | None = None
    relevant_files: list[str] = Field(default_factory=list)
    observations: list[Observation] = Field(default_factory=list)
    tracker: ModificationTracker = Field(default_factory=ModificationTracker)
    executor: CodingExecutor = Field(default_factory=CodingExecutor)
    # Fix #4 integration: lazy bridge to the code-intelligence facade. It is
    # serialized with the context (root + observability log) and rebuilds the
    # live CodeIntelligence lazily after a confirmation round-trip.
    intelligence: CodeIntelligenceBridge = Field(
        default_factory=CodeIntelligenceBridge
    )

    # Task association (lightweight refs — TaskState stays the source of truth).
    task_goal: str | None = None
    task_type: str | None = None
    plan_step_id: int | None = None

    # ------------------------------------------------------------------
    # Task association
    # ------------------------------------------------------------------

    @staticmethod
    def _current_step_id(task: Any) -> int | None:
        """Reads the task's current plan step id (duck-typed: attribute or method)."""
        step = getattr(task, "current_plan_step", None)
        if callable(step):
            try:
                step = step()
            except (TypeError, ValueError):
                return None
        return step.id if step is not None else None

    def attach_task(self, task: Any) -> CodeContext:
        """Copies the task's goal / task type / current plan step id."""
        self.task_goal = getattr(task, "goal", None)
        task_type = getattr(task, "task_type", None)
        self.task_type = getattr(task_type, "value", None)
        self.plan_step_id = self._current_step_id(task)
        self.ensure_intelligence()
        return self

    def ensure_intelligence(self) -> CodeIntelligenceBridge:
        """
        Enables the code-intelligence bridge for the workspace root (if any).

        No-op when there is no workspace or the workspace is outside the
        allowed base directory / is Ultron's own repository (the bridge's
        ``enable`` decides). Safe to call repeatedly.
        """
        if not self.intelligence.enabled and self.workspace is not None:
            root = getattr(self.workspace, "project_root", None)
            if root:
                self.intelligence.enable(str(root))
        return self.intelligence

    def sync_plan_step(self, task: Any) -> CodeContext:
        """Refreshes the current plan step id from the task (called on resume)."""
        self.plan_step_id = self._current_step_id(task)
        return self

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------

    def add_relevant_file(self, path: str) -> CodeContext:
        if path and path not in self.relevant_files:
            self.relevant_files.append(path)
        return self

    def add_observation(self, observation: Observation) -> CodeContext:
        self.observations.append(observation)
        return self

    def record_observation(
        self,
        kind: ObservationKind,
        source: str,
        summary: str,
        detail: str = "",
        *,
        success: bool | None = None,
        exit_code: int | None = None,
    ) -> Observation:
        """Builds and records one observation in one call."""
        obs = Observation(
            kind=kind,
            source=source,
            summary=summary,
            detail=detail,
            success=success,
            exit_code=exit_code,
        )
        self.observations.append(obs)
        return obs

    def record_error(self, source: str, summary: str, detail: str = "") -> Observation:
        """Records a failed observation (kind=ERROR)."""
        obs = Observation.error(source, summary, detail)
        self.observations.append(obs)
        return obs

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def recent_observations(self, limit: int = 10) -> list[Observation]:
        """The most recent *limit* observations."""
        return self.observations[-limit:]

    def has_failures(self) -> bool:
        """True when any recorded observation failed."""
        return any(o.success is False for o in self.observations)

    def summary(self, max_observations: int = 8) -> str:
        """Compact context block for prompt injection (bounded, no raw repo dump)."""
        lines = ["CODING CONTEXT:"]
        if self.workspace is not None:
            lines.append(self.workspace.summary())
        if self.relevant_files:
            shown = ", ".join(self.relevant_files[:20])
            lines.append(f"Relevant files: {shown}")
        if self.observations:
            lines.append("Recent observations:")
            for obs in self.recent_observations(max_observations):
                lines.append("  " + obs.to_prompt_line())
        if self.tracker.modifications:
            lines.append("Modifications this task:")
            for mod in self.tracker.recent(5):
                lines.append("  " + mod.describe())
        return "\n".join(lines)
