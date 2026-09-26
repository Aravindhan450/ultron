"""
ultron.core.context
~~~~~~~~~~~~~~~~~~~

Phase 2 Repository-Aware ContextManager package.

Provides evidence-based repository discovery, file/symbol/search retrieval,
prioritization, deduplication, token budgeting, and compaction.
"""

from ultron.core.context.budget import (
    ContextBudgetConfig,
    budget_messages,
    budget_openai_messages,
)
from ultron.core.context.invocation import (
    BudgetedEngine,
    ModelCallScope,
    ensure_budgeted_engine,
    model_call_scope,
)
from ultron.core.context.manager import (
    RepositoryContextManager,
)
from ultron.core.context.models import (
    ContextItem,
    ContextPriority,
    ContextRetrievalResult,
    ContextRetrievalStatus,
    ContextSnapshot,
    ContextSourceType,
)
from ultron.core.context.retrieval import (
    RepositoryRetriever,
    estimate_tokens,
)

__all__ = [
    "BudgetedEngine",
    "ContextBudgetConfig",
    "ContextItem",
    "ContextPriority",
    "ContextRetrievalResult",
    "ContextRetrievalStatus",
    "ContextSnapshot",
    "ContextSourceType",
    "ModelCallScope",
    "RepositoryContextManager",
    "RepositoryRetriever",
    "budget_messages",
    "budget_openai_messages",
    "ensure_budgeted_engine",
    "estimate_tokens",
    "model_call_scope",
]
