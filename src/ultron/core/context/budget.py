"""
ultron.core.context.budget
~~~~~~~~~~~~~~~~~~~~~~~~~~

Single authoritative implementation of Ultron's model-context token budget
invariant::

    estimated_input_tokens + reserved_output_tokens <= model_context_limit

The algorithm is deliberately independent of how a caller represents messages.
Two thin adapters expose it:

- :func:`budget_messages` — for :class:`~ultron.core.types.ChatMessage` lists
  (used by :class:`~ultron.core.context.manager.RepositoryContextManager`).
- :func:`budget_openai_messages` — for OpenAI-style ``dict`` message lists
  (used by the engine-boundary :class:`~ultron.core.context.invocation.BudgetedEngine`).

There is intentionally ONE algorithm (:func:`_budget_core`); adapters must never
re-implement budgeting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from ultron.core.context.models import estimate_tokens
from ultron.core.types import ChatMessage, Role


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


# Minimum input budget the boundary will attempt to preserve. If the configured
# output reservation would reduce the input budget below this, the reservation
# is *reduced* (never the model limit exceeded) so the hard invariant holds.
MIN_INPUT_TOKENS = 64


@dataclass
class _BudgetMessage:
    """Neutral representation of one message for the budgeting algorithm."""

    role: str
    content: Any
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def is_text(self) -> bool:
        return isinstance(self.content, str)

    @property
    def text(self) -> str:
        # Non-text (e.g. multimodal content-part lists) is not token-counted by
        # this version of the boundary; see the module docstring/report.
        return self.content if isinstance(self.content, str) else ""


def _resolve_limits(
    config: ContextBudgetConfig,
    model_context_limit: int | None,
    reserved_output_tokens: int | None,
) -> tuple[int, int, int]:
    """
    Resolves the (model_limit, reserved_output, max_input) triple.

    The invariant ``max_input + reserved <= model_limit`` is guaranteed for any
    inputs: the previous ``max(100, limit - reserved)`` floor could exceed the
    model capacity, so it is removed. When the reservation would starve the
    prompt, the reservation is reduced instead.
    """
    limit = model_context_limit if model_context_limit is not None else config.model_context_limit
    reserved = (
        reserved_output_tokens
        if reserved_output_tokens is not None
        else config.reserved_output_tokens
    )
    limit = max(0, int(limit))
    reserved = max(0, int(reserved))

    max_input = limit - reserved
    if max_input < MIN_INPUT_TOKENS:
        reserved = max(0, limit - MIN_INPUT_TOKENS)
        max_input = max(0, limit - reserved)
    return limit, reserved, max_input


def _truncate_oversized_tool_messages(
    items: list[_BudgetMessage], max_tool_tokens: int
) -> tuple[list[_BudgetMessage], int]:
    """Truncates oversized TOOL observations, retaining head and tail."""
    processed: list[_BudgetMessage] = []
    truncated = 0
    for msg in items:
        if msg.role == "tool" and msg.is_text and estimate_tokens(msg.text) > max_tool_tokens:
            half_chars = max(50, (max_tool_tokens * 4) // 2 - 40)
            head = msg.text[:half_chars].rstrip()
            tail = msg.text[-half_chars:].lstrip()
            clipped = (
                f"{head}\n... [truncated {len(msg.text)} chars, retained head & tail] "
                f"...\n{tail}"
            )
            processed.append(_BudgetMessage(msg.role, clipped, dict(msg.extra)))
            truncated += 1
        else:
            processed.append(msg)
    return processed, truncated


def _total_tokens(items: list[_BudgetMessage]) -> int:
    return sum(estimate_tokens(m.text) for m in items)


def _anchor_indices(items: list[_BudgetMessage]) -> tuple[int | None, int | None, int | None]:
    """Returns (system_index, first_user_index, latest_user_index)."""
    system_idx = 0 if items and items[0].role == "system" else None
    first_user_idx: int | None = None
    latest_user_idx: int | None = None
    for idx, msg in enumerate(items):
        if msg.role == "user":
            if first_user_idx is None:
                first_user_idx = idx
            latest_user_idx = idx
    return system_idx, first_user_idx, latest_user_idx


def _shave_to_fit(
    items: list[_BudgetMessage],
    max_input: int,
    protected: set[int],
) -> None:
    """
    Last-resort compaction. Shaves non-protected material oldest-first, then the
    system prompt, then the original-goal anchor, and only finally the latest
    user request — so the information required to answer the current request is
    the last thing to be reduced.
    """
    system_idx, first_user_idx, latest_user_idx = _anchor_indices(items)
    order: list[int] = [i for i in range(len(items)) if i not in protected]
    for idx in (system_idx, first_user_idx, latest_user_idx):
        if idx is not None and idx not in order:
            order.append(idx)

    def _shave(index: int) -> None:
        msg = items[index]
        if not msg.is_text:
            return
        excess = _total_tokens(items) - max_input
        msg_tok = estimate_tokens(msg.text)
        avail_tok = max(10, msg_tok - excess)
        avail_chars = max(20, avail_tok * 4)
        new_content = msg.text[:avail_chars].rstrip() + " ... [truncated]"
        while estimate_tokens(new_content) > avail_tok and avail_chars > 20:
            avail_chars -= 10
            new_content = msg.text[:avail_chars].rstrip() + " ... [truncated]"
        items[index] = _BudgetMessage(msg.role, new_content, dict(msg.extra))

    for idx in order:
        if _total_tokens(items) <= max_input:
            return
        _shave(idx)

    # Absolute guarantee: blank any remaining text message (oldest first) so the
    # invariant can never be violated by text content.
    if _total_tokens(items) > max_input:
        for idx in range(len(items)):
            if _total_tokens(items) <= max_input:
                break
            if items[idx].is_text and items[idx].text:
                items[idx] = _BudgetMessage(items[idx].role, "", dict(items[idx].extra))


def _budget_core(
    items: list[_BudgetMessage],
    config: ContextBudgetConfig,
    model_context_limit: int | None,
    reserved_output_tokens: int | None,
) -> tuple[list[_BudgetMessage], dict[str, Any]]:
    """The single budgeting algorithm (role/content agnostic)."""
    limit, reserved, max_input = _resolve_limits(
        config, model_context_limit, reserved_output_tokens
    )

    if not items:
        return [], {
            "input_tokens": 0,
            "estimated_input_tokens": 0,
            "model_limit": limit,
            "reserved_output_tokens": reserved,
            "dropped_messages": 0,
            "truncated_tool_messages": 0,
        }

    processed, truncated_tools = _truncate_oversized_tool_messages(
        list(items), config.max_tool_output_tokens
    )

    if _total_tokens(processed) <= max_input:
        return processed, {
            "input_tokens": _total_tokens(processed),
            "estimated_input_tokens": _total_tokens(processed),
            "model_limit": limit,
            "reserved_output_tokens": reserved,
            "dropped_messages": 0,
            "truncated_tool_messages": truncated_tools,
        }

    system_idx, first_user_idx, latest_user_idx = _anchor_indices(processed)
    protected = {i for i in (system_idx, first_user_idx, latest_user_idx) if i is not None}

    active = list(processed)
    prunable = [i for i in range(len(active)) if i not in protected]
    dropped = 0
    while _total_tokens(active) > max_input and prunable:
        # Drop oldest prunable first; protected anchors + newest material last.
        drop_probable = prunable.pop(0)
        target = processed[drop_probable]
        if target in active:
            active.remove(target)
            dropped += 1

    if _total_tokens(active) > max_input:
        _shave_to_fit(active, max_input, protected)

    total = _total_tokens(active)
    return active, {
        "input_tokens": total,
        "estimated_input_tokens": total,
        "model_limit": limit,
        "reserved_output_tokens": reserved,
        "dropped_messages": dropped,
        "truncated_tool_messages": truncated_tools,
    }


def budget_messages(
    messages: list[ChatMessage],
    config: ContextBudgetConfig,
    model_context_limit: int | None = None,
    reserved_output_tokens: int | None = None,
) -> tuple[list[ChatMessage], dict[str, Any]]:
    """Enforces the token budget on a list of :class:`ChatMessage` objects."""
    items = [
        _BudgetMessage(
            role=msg.role.value,
            content=msg.content,
            extra={"name": msg.name, "tool_call_id": msg.tool_call_id},
        )
        for msg in messages
    ]
    budgeted, meta = _budget_core(items, config, model_context_limit, reserved_output_tokens)
    rebuilt = [
        ChatMessage(
            role=Role(m.role),
            content=m.content,
            name=m.extra.get("name"),
            tool_call_id=m.extra.get("tool_call_id"),
        )
        for m in budgeted
    ]
    return rebuilt, meta


def budget_openai_messages(
    messages: list[dict[str, Any]],
    config: ContextBudgetConfig,
    model_context_limit: int | None = None,
    reserved_output_tokens: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Enforces the token budget on OpenAI-style ``dict`` message lists."""
    items = []
    for raw in messages:
        extra = {k: v for k, v in raw.items() if k not in ("role", "content")}
        items.append(
            _BudgetMessage(
                role=str(raw.get("role", "user")),
                content=raw.get("content", ""),
                extra=extra,
            )
        )
    budgeted, meta = _budget_core(items, config, model_context_limit, reserved_output_tokens)
    rebuilt = [
        {**dict(m.extra), "role": m.role, "content": m.content} for m in budgeted
    ]
    return rebuilt, meta
