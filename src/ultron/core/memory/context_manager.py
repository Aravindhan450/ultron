"""ultron.core.memory.context_manager
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ContextManager — decides what actually reaches the LLM.

It is the single deterministic assembly point for model context (FIX #6).
Given the working-memory view, the task state, code context, project
memory, session memory and long-term memory, it produces a bounded,
prioritized block:

    Priority order (never overridden by older memory):
    1.  current user request
    2.  current task state (goal, status, step counts)
    3.  current plan step
    4.  latest tool observations
    5.  relevant code context (files / observations)
    6.  relevant test results
    7.  relevant project memory
    8.  session memory
    9.  older long-term memory

Rules:

- **Budget-first**: :class:`ContextBudget` caps each section AND the total
  character size. Once the budget is exhausted nothing more is appended —
  older memory is the first thing dropped, never current facts.
- **Source-of-truth priority**: stale/superseded memory records are
  excluded entirely; project memory (workspace-scoped) is preferred over
  global memory for project tasks.
- **Deterministic**: the same inputs produce the same block — no LLM call,
  no ordering ambiguity.
- **Reusable**: a pure model with ``assemble(...) -> str``, duck-typed
  against TaskState/CodeContext like the CodingExecutor. Agents call it;
  it never executes tools.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from ultron.core.memory.models import (
    MemoryConfidence,
    MemoryKind,
    MemoryRecord,
    MemoryValidity,
)
from ultron.core.memory.session_memory import SessionMemory
from ultron.core.memory.working_memory import WorkingMemory


class ContextBudget(BaseModel):
    """Configurable per-section and total caps for one assembled context."""

    max_user_chars: int = 400
    max_task_chars: int = 600
    max_step_chars: int = 400
    max_observations_chars: int = 700
    max_code_chars: int = 800
    max_test_chars: int = 500
    max_project_memory_chars: int = 600
    max_session_chars: int = 400
    max_long_term_chars: int = 400
    max_total_chars: int = 3500  # hard ceiling; enforced after sections


def _clip(text: str, limit: int) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    if len(text) <= limit:
        return text
    # A slice with a negative stop would GROW the string once the ellipsis
    # is added (limit-3 < 0) — refuse to emit a section that overflows the
    # budget instead.
    if limit < 4:
        return ""
    return text[: limit - 3].rstrip() + "..."


def _task_status(task) -> str:
    status = getattr(task, "status", None)
    return getattr(status, "value", None) or str(status or "")


def _records_for_kind(records: list[MemoryRecord] | None, kind: MemoryKind) -> list[MemoryRecord]:
    return [r for r in (records or []) if r.kind is kind]


class ContextManager(BaseModel):
    """Assembles bounded, prioritized model context from all memory types."""

    budget: ContextBudget = Field(default_factory=ContextBudget)

    # ------------------------------------------------------------------
    # Sections (each returns "" when there is nothing relevant)
    # ------------------------------------------------------------------

    def _section_user(self, user_request: str) -> str:
        return _clip(user_request, self.budget.max_user_chars)

    def _section_task(self, task) -> str:
        if task is None:
            return ""
        goal = str(getattr(task, "goal", "") or "")
        status = _task_status(task)
        completed = len(
            [r for r in getattr(task, "completed_requirements", []) or []]
        )
        total = len(getattr(task, "requirements", None) or [])
        bits = []
        if goal:
            bits.append(f"Goal: {goal}")
        if status:
            bits.append(f"Status: {status}")
        bits.append(f"Requirements: {completed}/{total} complete")
        if getattr(task, "remaining_requirements", None):
            remaining = task.remaining_requirements()
            if remaining:
                names = ", ".join(r.description for r in remaining)
                bits.append(f"Remaining: {names}")
        return _clip("\n".join(bits), self.budget.max_task_chars)

    def _section_step(self, working: WorkingMemory | None) -> str:
        if working is None or not working.current_step:
            return ""
        line = f"Current step: {working.current_step}"
        if working.current_step_id is not None:
            line = f"Current step {working.current_step_id}: {working.current_step}"
        if working.active_failure:
            line += f"\nActive failure: {working.active_failure}"
        return _clip(line, self.budget.max_step_chars)

    def _section_observations(self, working: WorkingMemory | None) -> str:
        if working is None or not working.recent_observations:
            return ""
        # Test/build lines belong to the dedicated TEST RESULTS section
        # (priority 6) — do not count them twice against the budget.
        lines = ["Recent observations:"] + [
            f"  - {o}"
            for o in working.recent_observations
            if not self._is_test_line(o)
        ]
        if working.last_tool_result:
            lines.append(f"Last tool result: {working.last_tool_result}")
        return _clip("\n".join(lines), self.budget.max_observations_chars)

    def _section_code(self, code_context) -> str:
        if code_context is None:
            return ""
        workspace = getattr(code_context, "workspace", None)
        bits = []
        if workspace is not None:
            summary = getattr(workspace, "summary", None)
            if callable(summary):
                bits.append(summary())
        relevant = list(getattr(code_context, "relevant_files", None) or [])
        if relevant:
            bits.append("Relevant files: " + ", ".join(relevant[:12]))
        # Code intelligence targeted context, if the bridge already built it.
        intelligence = getattr(code_context, "intelligence", None)
        usage = (
            intelligence.usage_summary()
            if intelligence is not None
            and getattr(intelligence, "enabled", False)
            and callable(getattr(intelligence, "usage_summary", None))
            else ""
        )
        if usage and "no code-intelligence queries" not in usage:
            bits.append(f"Code intelligence: {usage}")
        return _clip("\n".join(bits), self.budget.max_code_chars)

    def _section_tests(self, working: WorkingMemory | None) -> str:
        if working is None:
            return ""
        test_lines = [
            o for o in working.recent_observations if self._is_test_line(o)
        ]
        if not test_lines:
            return ""
        return _clip(
            "\n".join(["Test/build results:"] + [f"  - {t}" for t in test_lines]),
            self.budget.max_test_chars,
        )

    @staticmethod
    def _is_test_line(line: str) -> bool:
        lowered = line.lower()
        markers = (
            "test", "pytest", "assertion", "exit code", "failed", "passed",
            "build", "coverage", "test_assertion", "test_result",
        )
        return any(m in lowered for m in markers)

    def _section_project_memory(
        self,
        records: list[MemoryRecord] | None,
        workspace: str,
        task_terms: list[str],
    ) -> str:
        valid = [
            r
            for r in _records_for_kind(records, MemoryKind.PROJECT)
            if r.validity is MemoryValidity.VALID
        ]
        if not valid:
            return ""
        # Prefer workspace-scoped records for project tasks (exact match on
        # resolved paths — a substring test could conflate sibling dirs).
        try:
            resolved_workspace = str(Path(workspace).resolve()) if workspace else ""
        except (OSError, ValueError):
            resolved_workspace = workspace
        scoped = [
            r
            for r in valid
            if r.workspace
            and resolved_workspace
            and str(Path(r.workspace).resolve()) == resolved_workspace
        ]
        pool = scoped or valid
        # Cheap deterministic relevance: shared keyword between fact and task.
        if task_terms:
            keywords = [t.lower() for t in task_terms if len(t) >= 3]
            ranked = sorted(
                pool,
                key=lambda r: -sum(
                    1 for k in keywords if k in (r.name + " " + r.content).lower()
                ),
            )
            pool = ranked
        lines = [r.to_prompt_line() for r in pool[:6]]
        return _clip("\n".join(lines), self.budget.max_project_memory_chars)

    def _section_session(self, session: SessionMemory | None) -> str:
        if session is None or session.is_empty:
            return ""
        return _clip(
            "\n".join(session.to_context_lines()),
            self.budget.max_session_chars,
        )

    def _section_long_term(
        self, records: list[MemoryRecord] | None
    ) -> str:
        valid = [
            r
            for r in _records_for_kind(records, MemoryKind.LONG_TERM)
            if r.validity is MemoryValidity.VALID
            and r.confidence
            not in (MemoryConfidence.INFERRED, MemoryConfidence.UNKNOWN)
        ]
        if not valid:
            return ""
        return _clip(
            "\n".join(r.to_prompt_line() for r in valid[:6]),
            self.budget.max_long_term_chars,
        )

    # ------------------------------------------------------------------
    # Assembly
    # ------------------------------------------------------------------

    def _append(
        self, sections: list[str], used: int, title: str, body: str
    ) -> int:
        """Appends one bounded section under the hard total ceiling.

        Returns the updated ``used`` budget counter. Empty blocks and blocks
        that cannot fit in the remaining room are skipped entirely — the
        total budget is never exceeded, even by a section that fits alone.
        """
        if not body:
            return used
        room = self.budget.max_total_chars - used
        if room <= 0:
            return used
        block = _clip(f"{title}:\n{body}", room)
        if not block:  # nothing fit in the remaining room
            return used
        sections.append(block)
        return used + len(block) + 1

    def assemble(
        self,
        user_request: str = "",
        task=None,
        working: WorkingMemory | None = None,
        code_context=None,
        project_memory: list[MemoryRecord] | None = None,
        session: SessionMemory | None = None,
        long_term: list[MemoryRecord] | None = None,
        *,
        workspace: str = "",
        task_terms: list[str] | None = None,
    ) -> str:
        """
        Produces the final context block for one reasoning step.

        Sections are appended in strict priority order; the total budget is
        enforced as a hard ceiling, so older memory is dropped first and
        current task facts are never evicted by stale memory.
        """
        sections: list[str] = []
        used = 0

        used = self._append(
            sections, used, "USER REQUEST", self._section_user(user_request)
        )
        used = self._append(sections, used, "TASK STATE", self._section_task(task))
        used = self._append(sections, used, "PLAN STEP", self._section_step(working))
        used = self._append(
            sections, used, "OBSERVATIONS", self._section_observations(working)
        )
        used = self._append(sections, used, "CODE CONTEXT", self._section_code(code_context))
        used = self._append(sections, used, "TEST RESULTS", self._section_tests(working))
        used = self._append(
            sections,
            used,
            "PROJECT MEMORY",
            self._section_project_memory(
                project_memory, workspace, task_terms or []
            ),
        )
        used = self._append(sections, used, "SESSION MEMORY", self._section_session(session))
        self._append(sections, used, "LONG-TERM MEMORY", self._section_long_term(long_term))

        return "\n\n".join(sections)

    def memory_block(
        self,
        project_records: list[MemoryRecord] | None = None,
        session: SessionMemory | None = None,
        long_term: list[MemoryRecord] | None = None,
        *,
        workspace: str = "",
        task_terms: list[str] | None = None,
    ) -> str:
        """
        The memory-only sections (project > session > long-term), bounded.

        Used by the agent to inject RELEVANT MEMORY alongside its explicit
        task/plan/observation sections — those higher-priority sections are
        assembled by the agent itself (they must never be dropped by a
        memory budget), so this method intentionally starts below priority 7.
        """
        sections: list[str] = []
        used = 0
        used = self._append(
            sections,
            used,
            "PROJECT MEMORY",
            self._section_project_memory(
                project_records, workspace, task_terms or []
            ),
        )
        used = self._append(sections, used, "SESSION MEMORY", self._section_session(session))
        self._append(sections, used, "LONG-TERM MEMORY", self._section_long_term(long_term))
        return "\n\n".join(sections)
