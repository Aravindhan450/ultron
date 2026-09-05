"""
ultron.core.context.manager
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Repository-aware ContextManager coordinating retrieval, prioritization,
deduplication, token budgeting, and compaction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from ultron.core.context.models import (
    ContextItem,
    ContextPriority,
    ContextSnapshot,
    ContextSourceType,
)
from ultron.core.context.retrieval import RepositoryRetriever, estimate_tokens

if TYPE_CHECKING:
    from ultron.core.coding.context import CodeContext
    from ultron.core.coding.workspace import CodingWorkspace
    from ultron.core.memory.models import MemoryRecord
    from ultron.core.memory.provider import MemoryProvider
    from ultron.core.memory.session_memory import SessionMemory
    from ultron.core.types import TaskState


class ContextBudgetConfig(BaseModel):
    """Token budget limits for repository-aware context assembly."""

    max_total_tokens: int = 4000
    max_file_tokens: int = 1500
    max_search_tokens: int = 800
    max_symbol_tokens: int = 600
    max_git_tokens: int = 300
    max_task_tokens: int = 600
    max_observation_tokens: int = 800


class RepositoryContextManager:
    """
    Assembles evidence-based model context across repository, task, git,
    files, symbols, and observations.
    """

    def __init__(
        self,
        workspace: CodingWorkspace | None = None,
        budget: ContextBudgetConfig | None = None,
        memory_provider: MemoryProvider | None = None,
    ) -> None:
        self.retriever = RepositoryRetriever(workspace=workspace)
        self.budget = budget or ContextBudgetConfig()
        if memory_provider is None:
            from ultron.core.memory.provider import MemoryProvider as _MemoryProvider

            self.memory_provider = _MemoryProvider()
        else:
            self.memory_provider = memory_provider
        self._last_snapshot: ContextSnapshot | None = None

    @property
    def last_snapshot(self) -> ContextSnapshot | None:
        """Returns the most recent assembled context snapshot for observability."""
        return self._last_snapshot

    def build_context(
        self,
        user_request: str = "",
        task: TaskState | None = None,
        code_context: CodeContext | None = None,
        session: SessionMemory | None = None,
        requested_files: list[str] | None = None,
        candidate_symbols: list[str] | None = None,
        search_queries: list[str] | None = None,
        artifacts: list[Any] | None = None,
        project_memory: list[MemoryRecord] | None = None,
        long_term_memory: list[MemoryRecord] | None = None,
        task_terms: list[str] | None = None,
    ) -> str:
        """
        Assembles prioritized, deduplicated, and budgeted context.
        """
        raw_items: list[ContextItem] = []

        # 1. User task & goal
        if user_request:
            raw_items.append(
                ContextItem(
                    source_type=ContextSourceType.USER_TASK,
                    priority=ContextPriority.USER_TASK,
                    title="User Request",
                    content=user_request.strip(),
                    target="user_request",
                    estimated_tokens=estimate_tokens(user_request),
                )
            )

        # 2. TaskState goal / requirements / plan
        if task is not None:
            task_lines = [
                f"Goal: {task.goal}",
                f"Status: {task.status.value}",
                f"Step: {task.current_step}/{task.total_steps or '?'}",
            ]
            if task.requirements:
                task_lines.append("Requirements:")
                for r in task.requirements:
                    task_lines.append(f"  - [{'x' if r.completed else ' '}] {r.description}")
            if task.plan is not None:
                step = task.current_plan_step()
                if step:
                    task_lines.append(f"Active Plan Step: {step.id}. {step.description}")
            task_body = "\n".join(task_lines)
            raw_items.append(
                ContextItem(
                    source_type=ContextSourceType.USER_TASK,
                    priority=ContextPriority.USER_TASK,
                    title="Active Task State",
                    content=task_body,
                    target="task_state",
                    estimated_tokens=estimate_tokens(task_body),
                )
            )

        # 3. Workspace Summary & Config
        ws = self.retriever.workspace
        ws_summary = ws.summary()
        raw_items.append(
            ContextItem(
                source_type=ContextSourceType.PROJECT_CONFIG,
                priority=ContextPriority.PROJECT_CONFIG,
                title="Workspace Environment",
                content=ws_summary,
                target="workspace",
                estimated_tokens=estimate_tokens(ws_summary),
            )
        )

        # 4. Git State
        git_res = self.retriever.retrieve_git_context()
        if git_res.is_found:
            raw_items.extend(git_res.items)

        # 5. Explicitly requested files
        files_to_fetch = list(requested_files or [])
        if code_context and code_context.relevant_files:
            for f in code_context.relevant_files:
                if f not in files_to_fetch:
                    files_to_fetch.append(f)

        for file_path in files_to_fetch[:5]:
            file_res = self.retriever.retrieve_file(file_path)
            if file_res.is_found:
                raw_items.extend(file_res.items)

        # 6. Candidate Symbols
        for sym in (candidate_symbols or [])[:5]:
            sym_res = self.retriever.retrieve_symbol(sym)
            if sym_res.is_found:
                raw_items.extend(sym_res.items)

        # 7. Search Queries
        for q in (search_queries or [])[:3]:
            search_res = self.retriever.retrieve_search(q)
            if search_res.is_found:
                raw_items.extend(search_res.items)

        # 8. CodeContext observations and modifications
        if code_context is not None:
            if code_context.observations:
                obs_lines = [
                    f"- {obs.to_prompt_line()}"
                    for obs in code_context.recent_observations(5)
                ]
                obs_body = "\n".join(obs_lines)
                raw_items.append(
                    ContextItem(
                        source_type=ContextSourceType.OBSERVATION,
                        priority=ContextPriority.TESTS_AND_OBSERVATIONS,
                        title="Recent Observations",
                        content=obs_body,
                        target="observations",
                        estimated_tokens=estimate_tokens(obs_body),
                    )
                )
            if code_context.tracker.modifications:
                mod_lines = [
                    f"- {mod.describe()}" for mod in code_context.tracker.recent(5)
                ]
                mod_body = "\n".join(mod_lines)
                raw_items.append(
                    ContextItem(
                        source_type=ContextSourceType.CHANGES_AND_DIFF,
                        priority=ContextPriority.CHANGES_AND_DIFF,
                        title="Modifications This Task",
                        content=mod_body,
                        target="modifications",
                        estimated_tokens=estimate_tokens(mod_body),
                    )
                )

        # 9. Project & Session & Long-Term Memory Context via MemoryProvider
        if project_memory:
            ws_root = (
                str(getattr(code_context.workspace, "project_root", ""))
                if code_context and code_context.workspace
                else (
                    str(getattr(task.code_context.workspace, "project_root", ""))
                    if task and task.code_context and task.code_context.workspace
                    else str(getattr(self.retriever.workspace, "project_root", ""))
                )
            )
            terms = task_terms or ([task.goal] if task and task.goal else ([user_request] if user_request else None))
            raw_items.extend(
                self.memory_provider.provide_project_memory(
                    records=project_memory,
                    workspace=ws_root,
                    task_terms=terms,
                )
            )
        elif task is not None and task.code_context is not None:
            store = task.code_context.ensure_project_memory()
            if store is not None:
                records = store.recall(limit=40)
                ws_root = (
                    str(getattr(task.code_context.workspace, "project_root", ""))
                    if task.code_context.workspace
                    else str(getattr(self.retriever.workspace, "project_root", ""))
                )
                terms = task_terms or ([task.goal] if task.goal else ([user_request] if user_request else None))
                raw_items.extend(
                    self.memory_provider.provide_project_memory(
                        records=records,
                        workspace=ws_root,
                        task_terms=terms,
                    )
                )

        if session is not None and not session.is_empty:
            raw_items.extend(self.memory_provider.provide_session_memory(session))

        if long_term_memory:
            raw_items.extend(
                self.memory_provider.provide_long_term_memory(long_term_memory)
            )

        # 10. Structured Artifacts
        if artifacts:
            for i, art in enumerate(artifacts[:3]):
                art_summary = getattr(art, "summary", None) or str(art)
                raw_items.append(
                    ContextItem(
                        source_type=ContextSourceType.ARTIFACT,
                        priority=ContextPriority.ARTIFACTS,
                        title=f"Artifact {i + 1}",
                        content=art_summary,
                        target=f"artifact_{i}",
                        estimated_tokens=estimate_tokens(art_summary),
                    )
                )

        # Deduplicate and sort by priority
        deduped = self._deduplicate(raw_items)
        sorted_items = sorted(deduped, key=lambda x: x.priority.value)

        # Enforce budget & compaction
        accepted_items, snapshot = self._compact_and_budget(sorted_items)
        self._last_snapshot = snapshot

        # Render prompt text
        blocks = [item.prompt_block() for item in accepted_items]
        return "\n\n".join(blocks)

    def assemble_snapshot(
        self,
        user_request: str = "",
        task: TaskState | None = None,
        code_context: CodeContext | None = None,
        session: SessionMemory | None = None,
        requested_files: list[str] | None = None,
        candidate_symbols: list[str] | None = None,
        search_queries: list[str] | None = None,
        artifacts: list[Any] | None = None,
        project_memory: list[MemoryRecord] | None = None,
        long_term_memory: list[MemoryRecord] | None = None,
        task_terms: list[str] | None = None,
    ) -> ContextSnapshot:
        """
        Assembles context and returns the structured ContextSnapshot directly.
        """
        self.build_context(
            user_request=user_request,
            task=task,
            code_context=code_context,
            session=session,
            requested_files=requested_files,
            candidate_symbols=candidate_symbols,
            search_queries=search_queries,
            artifacts=artifacts,
            project_memory=project_memory,
            long_term_memory=long_term_memory,
            task_terms=task_terms,
        )
        return self._last_snapshot or ContextSnapshot()

    def _deduplicate(self, items: list[ContextItem]) -> list[ContextItem]:
        """
        Deduplicates items by signature while preserving distinct evidence.
        """
        seen_signatures: set[str] = set()
        result: list[ContextItem] = []

        for item in items:
            normalized_content = " ".join(item.content.split()[:30])
            sig = f"{item.source_type.value}:{item.target}:{normalized_content}"
            if sig not in seen_signatures:
                seen_signatures.add(sig)
                result.append(item)

        return result

    def _compact_and_budget(
        self, items: list[ContextItem]
    ) -> tuple[list[ContextItem], ContextSnapshot]:
        """
        Fits items into max_total_tokens by dropping lower-priority items first.
        """
        accepted: list[ContextItem] = []
        tokens_used = 0
        characters_used = 0
        dropped_count = 0
        contributions: dict[str, int] = {}

        any_truncated = False
        for item in items:
            item_tokens = item.estimated_tokens
            if tokens_used + item_tokens <= self.budget.max_total_tokens:
                accepted.append(item)
                tokens_used += item_tokens
                characters_used += len(item.content)
                source_key = item.source_type.value
                contributions[source_key] = contributions.get(source_key, 0) + item_tokens
            else:
                # If high-priority item (USER_TASK or DIRECT_FILE), try compacting
                if item.priority in (ContextPriority.USER_TASK, ContextPriority.DIRECT_FILE):
                    remaining_room = self.budget.max_total_tokens - tokens_used
                    if remaining_room >= 10:
                        max_chars = max(0, remaining_room * 4 - 20)
                        clipped_content = item.content[:max_chars].rstrip() + " ... [truncated]"
                        item_tok = estimate_tokens(clipped_content)
                        while item_tok > remaining_room and max_chars > 20:
                            max_chars -= 20
                            clipped_content = item.content[:max_chars].rstrip() + " ... [truncated]"
                            item_tok = estimate_tokens(clipped_content)
                        if item_tok <= remaining_room:
                            compacted_item = item.model_copy(
                                update={
                                    "content": clipped_content,
                                    "estimated_tokens": item_tok,
                                }
                            )
                            accepted.append(compacted_item)
                            tokens_used += compacted_item.estimated_tokens
                            characters_used += len(clipped_content)
                            any_truncated = True
                            break
                dropped_count += 1

        snapshot = ContextSnapshot(
            items=accepted,
            total_estimated_tokens=tokens_used,
            total_characters=characters_used,
            dropped_items_count=dropped_count,
            compacted=dropped_count > 0 or any_truncated,
            source_contributions=contributions,
        )

        return accepted, snapshot
