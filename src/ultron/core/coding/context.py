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

FIX #6: the context also owns the workspace-scoped project memory store
(root + lazy, serializable — same pattern as the intelligence bridge). The
store is created on the FIRST ``attach_task`` for a workspace and rebuilt
lazily after a ``model_dump_json`` round-trip, so it survives confirmations
and (via the task snapshot store) process restarts without duplicating
TaskState.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, PrivateAttr

from ultron.core.coding.edits import ModificationTracker
from ultron.core.coding.executor import CodingExecutor
from ultron.core.coding.intelligence_bridge import CodeIntelligenceBridge
from ultron.core.coding.observations import Observation, ObservationKind
from ultron.core.coding.workspace import CodingWorkspace, _resolve_safe_path
from ultron.core.memory.project_memory import ProjectMemoryStore

# The Ultron repository root (src/ultron/core/coding/context.py -> parents:
# [0]=coding, [1]=core, [2]=ultron, [3]=src, [4]=repo). Project memory is
# never created inside Ultron's own repository — unit tests run from it and
# must stay side-effect free, exactly like the intelligence bridge.
_ULTRON_ROOT = Path(__file__).resolve().parents[4]


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

    # FIX #6: workspace-scoped project memory. Only the root is serialized;
    # the store itself is a lazy private handle rebuilt after round-trips.
    project_memory_root: str | None = None
    _project_memory: ProjectMemoryStore | None = PrivateAttr(default=None)

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
        if not self.project_memory_root:
            self.project_memory_root = self._memory_root()
        self.ensure_project_memory()
        return self

    # ------------------------------------------------------------------
    # Project memory (FIX #6)
    # ------------------------------------------------------------------

    def _memory_root(self) -> str | None:
        """The workspace root for project memory, when safe.

        Reuses the same gate as the intelligence bridge: the root must
        resolve inside ``ALLOWED_BASE_DIR`` and must not be Ultron's own
        repository. Returns None otherwise (silently — memory simply stays
        disabled for that workspace).
        """
        if self.workspace is None:
            return None
        root = getattr(self.workspace, "project_root", None)
        if not root:
            return None
        resolved = _resolve_safe_path(str(root))
        if resolved is None:
            return None
        resolved = resolved.resolve()
        if resolved == _ULTRON_ROOT or _ULTRON_ROOT in resolved.parents:
            return None
        return str(resolved)

    def ensure_project_memory(self):
        """
        Builds (lazily) and returns the workspace-scoped project memory
        store, recording the workspace's detected facts once on first build.

        No-op when the workspace has no safe root. The store is idempotent,
        so repeated calls (every reasoning turn) never churn the history.
        """
        if self._project_memory is None and self.project_memory_root:
            from ultron.core.memory.formation import remember_workspace_facts

            store = ProjectMemoryStore(self.project_memory_root)
            remember_workspace_facts(store, self.workspace)
            self._project_memory = store
        return self._project_memory

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
