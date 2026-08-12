"""ultron.core.memory.session_memory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Session memory: what matters about the CURRENT conversation/session.

Deliberately bounded and summary-shaped — raw tool output is never kept
indefinitely:

- ``recent_requests`` — the last few user asks
- ``decisions`` — important choices made this session
- ``notable_outputs`` — short summaries of important results (truncated)
- ``active_workspace`` — the project being worked on
- ``task_refs`` — references to tasks started this session

This is an in-memory model (not persisted by itself): the ContextManager
reads it to assemble each model turn. Long-term value is distilled into
project/long-term memory by higher layers — never the raw transcript.
"""

from __future__ import annotations

import time

from pydantic import BaseModel, Field


def _clip(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


class SessionMemory(BaseModel):
    """Bounded summary of one chat session."""

    session_id: str = ""
    started_at: float = Field(default_factory=time.time)
    active_workspace: str = ""
    recent_requests: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    notable_outputs: list[str] = Field(default_factory=list)
    task_refs: list[str] = Field(default_factory=list)

    # Budgets (configurable; see ContextManager for the total context budget).
    max_requests: int = 6
    max_decisions: int = 6
    max_outputs: int = 4
    output_clip_chars: int = 160

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def note_request(self, text: str) -> SessionMemory:
        if text and text.strip():
            self.recent_requests.append(text.strip())
            self.recent_requests = self.recent_requests[-self.max_requests :]
        return self

    def note_decision(self, text: str) -> SessionMemory:
        if text and text.strip():
            self.decisions.append(text.strip())
            self.decisions = self.decisions[-self.max_decisions :]
        return self

    def note_output(self, text: str) -> SessionMemory:
        """Records a truncated summary of an important output — never the raw dump."""
        clipped = _clip(text, self.output_clip_chars)
        if clipped:
            self.notable_outputs.append(clipped)
            self.notable_outputs = self.notable_outputs[-self.max_outputs :]
        return self

    def note_task(self, task_ref: str) -> SessionMemory:
        if task_ref and task_ref not in self.task_refs:
            self.task_refs.append(task_ref)
            self.task_refs = self.task_refs[-10:]
        return self

    def set_workspace(self, workspace: str) -> SessionMemory:
        self.active_workspace = workspace
        return self

    # ------------------------------------------------------------------
    # Assembly
    # ------------------------------------------------------------------

    def to_context_lines(self, max_requests: int = 4) -> list[str]:
        """Compact lines for ContextManager injection (bounded)."""
        lines: list[str] = []
        if self.active_workspace:
            lines.append(f"Active workspace: {self.active_workspace}")
        if self.recent_requests:
            lines.append("Recent requests:")
            for req in self.recent_requests[-max_requests:]:
                lines.append(f"  - {_clip(req, 120)}")
        if self.decisions:
            lines.append("Decisions this session:")
            for decision in self.decisions:
                lines.append(f"  - {_clip(decision, 120)}")
        if self.notable_outputs:
            lines.append("Notable outputs:")
            for out in self.notable_outputs:
                lines.append(f"  - {out}")
        if self.task_refs:
            lines.append(f"Tasks this session: {', '.join(self.task_refs)}")
        return lines

    def reset(self) -> SessionMemory:
        """Starts a fresh session: all content AND the workspace binding clear."""
        self.recent_requests = []
        self.decisions = []
        self.notable_outputs = []
        self.task_refs = []
        self.active_workspace = ""
        self.started_at = time.time()
        return self

    @property
    def is_empty(self) -> bool:
        return not (
            self.active_workspace
            or self.recent_requests
            or self.decisions
            or self.notable_outputs
            or self.task_refs
        )
