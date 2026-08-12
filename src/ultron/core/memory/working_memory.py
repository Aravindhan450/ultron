"""ultron.core.memory.working_memory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Working memory: only what the CURRENT reasoning cycle needs.

It is a bounded, ephemeral view assembled from the authoritative sources —
TaskState (goal/step/failures), CodeContext (relevant files, observations),
and the current tool result. It is NOT a second copy of TaskState: it holds
short reference lines, and it is never persisted.

The ContextManager consumes this view plus the persistent stores to decide
what actually reaches the LLM.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


def _clip(text: str, limit: int = 200) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


class WorkingMemory(BaseModel):
    """Ephemeral context for one reasoning step."""

    goal: str = ""
    current_step: str = ""  # current plan step description
    current_step_id: int | None = None
    recent_observations: list[str] = Field(default_factory=list)
    active_failure: str | None = None
    relevant_files: list[str] = Field(default_factory=list)
    last_tool_result: str | None = None

    # Budgets.
    max_observations: int = 5
    max_files: int = 12
    clip_chars: int = 220

    # ------------------------------------------------------------------
    # Builders
    # ------------------------------------------------------------------

    @classmethod
    def from_task(cls, task, code_context=None) -> WorkingMemory:
        """Builds the working-memory view from TaskState + CodeContext.

        Duck-typed on purpose (no import of agent/task modules, mirroring
        the CodingExecutor pattern): TaskState stays the source of truth;
        this is only a bounded projection.
        """
        wm = cls()

        wm.goal = str(getattr(task, "goal", "") or "")

        # Current plan step.
        step = None
        if task is not None:
            current = getattr(task, "current_plan_step", None)
            if callable(current):
                try:
                    step = current()
                except (TypeError, ValueError):
                    step = None
        if step is not None:
            wm.current_step = str(
                getattr(step, "description", "") or ""
            )
            wm.current_step_id = getattr(step, "id", None)
            error = getattr(step, "error", None)
            if error:
                wm.active_failure = _clip(str(error), wm.clip_chars)

        # Recent observations from CodeContext (not raw transcript).
        if code_context is not None:
            observations = getattr(code_context, "recent_observations", None)
            if callable(observations):
                try:
                    recent = observations(limit=wm.max_observations)
                except TypeError:
                    recent = observations() if observations else []
            else:
                recent = []
            for obs in recent or []:
                line = getattr(obs, "to_prompt_line", None)
                if callable(line):
                    wm.recent_observations.append(_clip(line(), wm.clip_chars))
                else:
                    wm.recent_observations.append(_clip(str(obs), wm.clip_chars))
            files = list(getattr(code_context, "relevant_files", None) or [])
            wm.relevant_files = files[: wm.max_files]
            tracker = getattr(code_context, "tracker", None)
            if tracker is not None:
                modifications = getattr(tracker, "modifications", None) or []
                if modifications:
                    wm.recent_observations.append(
                        f"modifications this task: {len(modifications)}"
                    )

        # Executor failures (classified).
        executor = getattr(code_context, "executor", None) if code_context else None
        if executor is not None:
            failures = list(getattr(executor, "failures", None) or [])
            if failures:
                last = failures[-1]
                line = getattr(last, "to_prompt_line", None)
                wm.active_failure = _clip(
                    line() if callable(line) else str(last), wm.clip_chars
                )

        # Failed observations (e.g. a failing test run) surface as the active
        # failure when no step error / classified failure already did.
        if wm.active_failure is None:
            for obs in reversed(recent or []):
                if getattr(obs, "success", None) is False:
                    line = getattr(obs, "to_prompt_line", None)
                    wm.active_failure = _clip(
                        line() if callable(line) else str(obs), wm.clip_chars
                    )
                    break

        return wm

    def record_tool_result(self, result: str) -> WorkingMemory:
        self.last_tool_result = _clip(result, self.clip_chars)
        return self

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def to_context_lines(self) -> list[str]:
        lines: list[str] = []
        if self.goal:
            lines.append(f"Goal: {self.goal}")
        if self.current_step:
            step = f" (step {self.current_step_id})" if self.current_step_id else ""
            lines.append(f"Current plan step{step}: {self.current_step}")
        if self.active_failure:
            lines.append(f"Active failure: {self.active_failure}")
        if self.recent_observations:
            lines.append("Recent observations:")
            for obs in self.recent_observations:
                lines.append(f"  - {obs}")
        if self.relevant_files:
            lines.append(
                "Relevant files: " + ", ".join(self.relevant_files)
            )
        if self.last_tool_result:
            lines.append(f"Last tool result: {self.last_tool_result}")
        return lines
