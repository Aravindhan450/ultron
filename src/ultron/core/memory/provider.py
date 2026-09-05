"""
ultron.core.memory.provider
~~~~~~~~~~~~~~~~~~~~~~~~~~~

MemoryProvider — Memory evidence provider for the canonical ContextManager.

Extracts, filters, and formats project memory, session memory, and long-term
memory records without owning global context assembly.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from ultron.core.context.models import (
    ContextItem,
    ContextPriority,
    ContextSourceType,
    estimate_tokens,
)
from ultron.core.memory.models import (
    MemoryConfidence,
    MemoryKind,
    MemoryRecord,
    MemoryValidity,
)
from ultron.core.memory.session_memory import SessionMemory


def _records_for_kind(
    records: list[MemoryRecord] | None, kind: MemoryKind
) -> list[MemoryRecord]:
    return [r for r in (records or []) if r.kind is kind]


class MemoryProvider(BaseModel):
    """
    Evidence provider for memory subsystem records.
    Supplies structured ContextItems to RepositoryContextManager.
    """

    def provide_project_memory(
        self,
        records: list[MemoryRecord] | None,
        workspace: str = "",
        task_terms: list[str] | None = None,
        max_records: int = 6,
    ) -> list[ContextItem]:
        """
        Retrieves valid, workspace-scoped, relevance-ranked project memory records.
        """
        valid = [
            r
            for r in _records_for_kind(records, MemoryKind.PROJECT)
            if r.validity is MemoryValidity.VALID
        ]
        if not valid:
            return []

        try:
            resolved_workspace = (
                str(Path(workspace).resolve()) if workspace else ""
            )
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

        if task_terms:
            keywords = [t.lower() for t in task_terms if len(t) >= 3]
            ranked = sorted(
                pool,
                key=lambda r: -sum(
                    1
                    for k in keywords
                    if k in (r.name + " " + r.content).lower()
                ),
            )
            pool = ranked

        items: list[ContextItem] = []
        for r in pool[:max_records]:
            line = r.to_prompt_line()
            items.append(
                ContextItem(
                    source_type=ContextSourceType.PROJECT_MEMORY,
                    priority=ContextPriority.PROJECT_MEMORY,
                    title=f"Project Fact ({r.name})",
                    content=line,
                    target=r.name,
                    estimated_tokens=estimate_tokens(line),
                )
            )
        return items

    def provide_session_memory(
        self, session: SessionMemory | None
    ) -> list[ContextItem]:
        """
        Retrieves recent session continuity lines from SessionMemory.
        """
        if session is None or session.is_empty:
            return []
        lines = session.to_context_lines()
        if not lines:
            return []
        body = "\n".join(lines)
        return [
            ContextItem(
                source_type=ContextSourceType.SESSION_MEMORY,
                priority=ContextPriority.SESSION_MEMORY,
                title="Session Memory",
                content=body,
                target="session_memory",
                estimated_tokens=estimate_tokens(body),
            )
        ]

    def provide_long_term_memory(
        self, records: list[MemoryRecord] | None, max_records: int = 6
    ) -> list[ContextItem]:
        """
        Retrieves valid, high-confidence long-term memory records.
        """
        valid = [
            r
            for r in _records_for_kind(records, MemoryKind.LONG_TERM)
            if r.validity is MemoryValidity.VALID
            and r.confidence
            not in (MemoryConfidence.INFERRED, MemoryConfidence.UNKNOWN)
        ]
        if not valid:
            return []

        items: list[ContextItem] = []
        for r in valid[:max_records]:
            line = r.to_prompt_line()
            items.append(
                ContextItem(
                    source_type=ContextSourceType.PROJECT_MEMORY,
                    priority=ContextPriority.GENERAL_REPO,
                    title=f"Long-Term Memory ({r.name})",
                    content=line,
                    target=r.name,
                    estimated_tokens=estimate_tokens(line),
                )
            )
        return items
