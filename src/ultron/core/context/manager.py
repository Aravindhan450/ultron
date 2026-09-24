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
from ultron.core.types import ChatMessage, Role

if TYPE_CHECKING:
    from ultron.core.coding.context import CodeContext
    from ultron.core.coding.workspace import CodingWorkspace
    from ultron.core.memory.models import MemoryRecord
    from ultron.core.memory.provider import MemoryProvider
    from ultron.core.memory.session_memory import SessionMemory
    from ultron.core.types import TaskState


class ContextBudgetConfig(BaseModel):
    """Token budget limits for repository-aware context assembly and model interaction."""

    model_context_limit: int = 16384
    reserved_output_tokens: int = 2048
    max_tool_output_tokens: int = 1000
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
    files, symbols, and observations, and enforces model context budgeting.
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

            # 2b. Granular Active Plan Step (Priority 2)
            if task.plan is not None:
                active_step = task.current_plan_step()
                if active_step:
                    step_desc = f"Step {active_step.id}: {active_step.description} (Status: {active_step.status.value})"
                    raw_items.append(
                        ContextItem(
                            source_type=ContextSourceType.ACTIVE_PLAN_STEP,
                            priority=ContextPriority.ACTIVE_PLAN_STEP,
                            title=f"Plan Step {active_step.id}",
                            content=step_desc,
                            target=f"plan_step_{active_step.id}",
                            estimated_tokens=estimate_tokens(step_desc),
                        )
                    )

            # 2c. Granular Latest Failure (Priority 3)
            latest_err = None
            if hasattr(task, "errors") and task.errors:
                latest_err = task.errors[-1]
            elif hasattr(task, "execution_history") and task.execution_history:
                for exec_entry in reversed(task.execution_history):
                    if not exec_entry.success:
                        latest_err = exec_entry
                        break
            if latest_err is not None:
                err_msg = getattr(latest_err, "message", None) or getattr(latest_err, "detail", None) or str(latest_err)
                raw_items.append(
                    ContextItem(
                        source_type=ContextSourceType.FAILURE,
                        priority=ContextPriority.LATEST_FAILURE,
                        title="Latest Failure Evidence",
                        content=f"Failure: {err_msg}",
                        target="latest_failure",
                        estimated_tokens=estimate_tokens(f"Failure: {err_msg}"),
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
                        priority=ContextPriority.RECENT_OBSERVATIONS,
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

    def budget_messages(
        self,
        messages: list[ChatMessage],
        model_context_limit: int | None = None,
        reserved_output_tokens: int | None = None,
    ) -> tuple[list[ChatMessage], dict[str, Any]]:
        """
        Enforces hard token budgeting on conversation messages before sending to model.

        Ensures:
          estimated_input_tokens + reserved_output_tokens <= model_context_limit

        Strategy:
        1. Truncate oversized single TOOL observations preserving head and tail.
        2. Protect critical anchors:
           - System prompt (messages[0] if Role.SYSTEM)
           - User original goal / prompt (first Role.USER message)
        3. If still over budget, condense/drop older intermediate dialogue & tool pairs
           starting from oldest history toward recent.
        4. If single system message or user prompt still exceeds available budget,
           compact system message or user prompt keeping critical instructions/goal.

        Returns:
            (budgeted_messages, metadata_dict)
        """
        limit = model_context_limit or self.budget.model_context_limit
        reserved = reserved_output_tokens or self.budget.reserved_output_tokens
        max_input_tokens = max(100, limit - reserved)

        if not messages:
            return [], {
                "input_tokens": 0,
                "model_limit": limit,
                "reserved_output_tokens": reserved,
                "dropped_messages": 0,
                "truncated_tool_messages": 0,
            }

        # Step 1: Pre-process and truncate oversized TOOL messages
        max_tool_tokens = self.budget.max_tool_output_tokens
        processed: list[ChatMessage] = []
        truncated_tools_count = 0

        for msg in messages:
            if msg.role == Role.TOOL:
                tok = estimate_tokens(msg.content)
                if tok > max_tool_tokens:
                    # Truncate retaining head & tail
                    half_chars = max(50, (max_tool_tokens * 4) // 2 - 40)
                    head = msg.content[:half_chars].rstrip()
                    tail = msg.content[-half_chars:].lstrip()
                    clipped = f"{head}\n... [truncated {len(msg.content)} chars, retained head & tail] ...\n{tail}"
                    processed.append(
                        ChatMessage(
                            role=msg.role,
                            name=msg.name,
                            content=clipped,
                            tool_call_id=msg.tool_call_id,
                        )
                    )
                    truncated_tools_count += 1
                else:
                    processed.append(msg)
            else:
                processed.append(msg)

        # Helper to compute total tokens
        def calc_total_tokens(msgs: list[ChatMessage]) -> int:
            return sum(estimate_tokens(m.content) for m in msgs)

        current_tokens = calc_total_tokens(processed)
        if current_tokens <= max_input_tokens:
            return processed, {
                "input_tokens": current_tokens,
                "model_limit": limit,
                "reserved_output_tokens": reserved,
                "dropped_messages": 0,
                "truncated_tool_messages": truncated_tools_count,
            }

        # Step 2: Separate anchor messages and intermediate dialogue
        # Anchor 1: System prompt (if role == SYSTEM at index 0)
        has_system = len(processed) > 0 and processed[0].role == Role.SYSTEM

        # Anchor 2: User initial task/goal message
        user_idx = -1
        for idx in range(1 if has_system else 0, len(processed)):
            if processed[idx].role == Role.USER:
                user_idx = idx
                break

        # Protected indices
        protected_indices = set()
        if has_system:
            protected_indices.add(0)
        if user_idx != -1:
            protected_indices.add(user_idx)

        # Intermediate messages that can be pruned (from oldest to newest)
        # We preserve recent messages by dropping oldest prunable messages first
        prunable_indices = [i for i in range(len(processed)) if i not in protected_indices]
        dropped_messages_count = 0

        # Create active copy of messages
        active_msgs = list(processed)
        # Drop oldest prunable until within budget or no prunable remain
        while calc_total_tokens(active_msgs) > max_input_tokens and prunable_indices:
            idx_to_drop = prunable_indices.pop(0)
            # Find and remove
            target_msg = processed[idx_to_drop]
            if target_msg in active_msgs:
                active_msgs.remove(target_msg)
                dropped_messages_count += 1

        # Step 3: If STILL over budget (e.g. huge system message or user prompt or remaining dialogue)
        current_tokens = calc_total_tokens(active_msgs)
        if current_tokens > max_input_tokens:
            for i, msg in reversed(list(enumerate(active_msgs))):
                if current_tokens <= max_input_tokens:
                    break
                excess = current_tokens - max_input_tokens
                msg_tok = estimate_tokens(msg.content)
                avail_tok = max(10, msg_tok - excess)
                avail_chars = max(20, avail_tok * 4)
                new_content = msg.content[:avail_chars].rstrip() + " ... [truncated]"
                # Keep shaving if estimation still exceeds
                while estimate_tokens(new_content) > avail_tok and avail_chars > 20:
                    avail_chars -= 10
                    new_content = msg.content[:avail_chars].rstrip() + " ... [truncated]"
                active_msgs[i] = ChatMessage(
                    role=msg.role,
                    name=msg.name,
                    content=new_content,
                    tool_call_id=msg.tool_call_id,
                )
                current_tokens = calc_total_tokens(active_msgs)

        return active_msgs, {
            "input_tokens": calc_total_tokens(active_msgs),
            "model_limit": limit,
            "reserved_output_tokens": reserved,
            "dropped_messages": dropped_messages_count,
            "truncated_tool_messages": truncated_tools_count,
        }

