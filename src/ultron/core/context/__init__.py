"""
ultron.core.context
~~~~~~~~~~~~~~~~~~~

Phase 2 Repository-Aware ContextManager package.

Provides evidence-based repository discovery, file/symbol/search retrieval,
prioritization, deduplication, token budgeting, and compaction.
"""

from ultron.core.context.manager import (
    ContextBudgetConfig,
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
    "ContextBudgetConfig",
    "ContextItem",
    "ContextPriority",
    "ContextRetrievalResult",
    "ContextRetrievalStatus",
    "ContextSnapshot",
    "ContextSourceType",
    "RepositoryContextManager",
    "RepositoryRetriever",
    "estimate_tokens",
]
