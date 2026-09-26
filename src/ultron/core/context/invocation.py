"""
ultron.core.context.invocation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The authoritative model-call boundary.

Every production model invocation in Ultron goes through ``BaseEngine.generate``
or ``BaseEngine.stream``. :class:`BudgetedEngine` is a transparent
:class:`~ultron.core.engine.base.BaseEngine` decorator installed at the single
production agent factory, so **no** caller — ReAct primary loop, verification,
planning, classification, synthesis, or SimpleAgent — can bypass the Phase 2
context-budget invariant::

    estimated_input_tokens + reserved_output_tokens <= model_context_limit

Budgeting itself lives in exactly one place
(:mod:`ultron.core.context.budget`); this module only wires it to the engine
boundary and emits sanitized ``CONTEXT_BUILT`` observability events.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ultron.core.context.budget import (
    ContextBudgetConfig,
    budget_openai_messages,
)
from ultron.core.engine.base import BaseEngine
from ultron.core.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = get_logger("ultron.context.invocation")


@dataclass
class ModelCallScope:
    """Ambient observability context for model calls in the current async task."""

    event_bus: Any | None = None
    task_id: str | None = None
    run_id: str | None = None


_scope_var: ContextVar[ModelCallScope | None] = ContextVar("ultron_model_call_scope", default=None)
_caller_var: ContextVar[str | None] = ContextVar("ultron_model_caller", default=None)


@contextmanager
def model_caller(name: str):
    """
    Tags model calls made inside the block with a caller label for
    observability. Uses a contextvar so no engine signature is affected (fake
    and real engines alike are called unchanged).
    """
    token = _caller_var.set(name)
    try:
        yield
    finally:
        _caller_var.reset(token)


@contextmanager
def model_call_scope(event_bus: Any | None, task_id: str | None, run_id: str | None = None):
    """
    Binds the ambient observability scope for model calls made inside the block.

    ``BudgetedEngine`` emits a ``CONTEXT_BUILT`` event per model call when a
    scope with an event bus and a task id is active; otherwise it stays silent
    (budgeting still happens, it is simply not persisted).
    """
    token = _scope_var.set(ModelCallScope(event_bus=event_bus, task_id=task_id, run_id=run_id))
    try:
        yield
    finally:
        _scope_var.reset(token)


class BudgetedEngine(BaseEngine):
    """
    ``BaseEngine`` decorator enforcing the context-budget invariant on every
    ``generate``/``stream`` call before delegating to the wrapped engine.

    The wrapper preserves the engine contract: attribute reads (``base_url``,
    ``model``, ``supports_images``, ``get_active_model``, ...) and writes
    (``base_url``, ``set_model``) are delegated to the inner engine.
    """

    def __init__(self, engine: BaseEngine, budget: ContextBudgetConfig | None = None) -> None:
        self._engine = engine
        self._budget = budget or ContextBudgetConfig()
        self._last_budget: dict[str, Any] | None = None

    # ------------------------------------------------------------------ utils
    @property
    def inner_engine(self) -> BaseEngine:
        return self._engine

    @property
    def budget(self) -> ContextBudgetConfig:
        return self._budget

    @property
    def last_budget(self) -> dict[str, Any] | None:
        """Budget metadata of the most recent model call (observability/testing)."""
        return self._last_budget

    def set_context_limit(self, limit: int | None) -> None:
        """Propagates the routed model's context limit into the budget boundary."""
        if limit and int(limit) > 0:
            self._budget.model_context_limit = int(limit)

    def set_model(self, model_id: str) -> None:
        setter = getattr(self._engine, "set_model", None)
        if callable(setter):
            setter(model_id)

    @property
    def base_url(self) -> Any:
        return getattr(self._engine, "base_url", None)

    @base_url.setter
    def base_url(self, value: Any) -> None:
        self._engine.base_url = value

    def __getattr__(self, name: str) -> Any:
        # Only called when normal attribute lookup fails; delegate reads to the
        # wrapped engine (keeps BaseEngine contract transparent for callers).
        engine = self.__dict__.get("_engine")
        if engine is not None:
            return getattr(engine, name)
        raise AttributeError(name)

    # --------------------------------------------------------------- boundary
    def _model_id(self) -> str | None:
        for attr in ("model", "default_model"):
            value = getattr(self._engine, attr, None)
            if isinstance(value, str):
                return value
        return None

    def _budget_call(
        self, messages: list[dict[str, Any]], caller: str | None
    ) -> list[dict[str, Any]]:
        budgeted, meta = budget_openai_messages(messages, self._budget)
        self._last_budget = meta
        self._emit(meta, caller or _caller_var.get())
        return budgeted

    def _emit(self, meta: dict[str, Any], caller: str | None) -> None:
        scope = _scope_var.get()
        if scope is None or scope.event_bus is None or not scope.task_id:
            return
        payload = {
            "kind": "model_call_budget",
            "caller": caller or "model",
            "model": self._model_id(),
            **meta,
        }
        try:
            from ultron.core.runtime.events import RuntimeEvent, RuntimeEventType

            scope.event_bus.emit_sync(
                RuntimeEvent(
                    event_type=RuntimeEventType.CONTEXT_BUILT,
                    run_id=scope.run_id or scope.task_id,
                    task_id=scope.task_id,
                    payload=payload,
                )
            )
        except Exception as exc:  # noqa: BLE001 — observability must never break a model call
            logger.warning("Failed to emit CONTEXT_BUILT for model call: %s", exc)

    async def generate(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        caller = kwargs.pop("caller", None)
        budgeted = self._budget_call(messages, caller)
        return await self._engine.generate(budgeted, **kwargs)

    async def stream(  # type: ignore[override]
        self, messages: list[dict[str, Any]], **kwargs: Any
    ) -> AsyncIterator[str]:
        caller = kwargs.pop("caller", None)
        budgeted = self._budget_call(messages, caller)
        async for chunk in self._engine.stream(budgeted, **kwargs):
            yield chunk


def ensure_budgeted_engine(
    engine: BaseEngine | None, budget: ContextBudgetConfig | None = None
) -> BaseEngine | None:
    """
    Returns ``engine`` wrapped in :class:`BudgetedEngine` (idempotently).

    Idempotent so ``/agent``, ``/reload`` and repair-agent paths that reuse an
    already-wrapped engine never double-wrap or reset the budget state.
    """
    if engine is None:
        return None
    if isinstance(engine, BudgetedEngine):
        if budget is not None:
            engine.budget.model_context_limit = budget.model_context_limit
        return engine
    return BudgetedEngine(engine, budget)
