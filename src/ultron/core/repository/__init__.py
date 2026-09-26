"""
ultron.core.repository
~~~~~~~~~~~~~~~~~~~~~~

Repository intelligence, dependency graph, token-budgeted repo map, and task-aware retrieval.
"""

from __future__ import annotations

from ultron.core.repository.cache import (
    RepositoryCache,
    get_repository_cache,
    invalidate_file_cache,
)
from ultron.core.repository.graph import GraphEdge, GraphNode, RepositoryGraph
from ultron.core.repository.index import (
    FileClassification,
    FileInfo,
    IndexSummary,
    RepositoryIndex,
    classify_file,
    is_secret_file,
)
from ultron.core.repository.parser import (
    FileParseResult,
    ParsedImport,
    ParsedSymbol,
    parse_source_file,
)
from ultron.core.repository.repo_map import RepoMap, RepoMapGenerator
from ultron.core.repository.retriever import (
    RelevanceProvenance,
    TaskAwareRetriever,
    TaskRelevanceResult,
)

__all__ = [
    "FileClassification",
    "FileInfo",
    "FileParseResult",
    "GraphEdge",
    "GraphNode",
    "IndexSummary",
    "ParsedImport",
    "ParsedSymbol",
    "RelevanceProvenance",
    "RepoMap",
    "RepoMapGenerator",
    "RepositoryCache",
    "RepositoryGraph",
    "RepositoryIndex",
    "TaskAwareRetriever",
    "TaskRelevanceResult",
    "classify_file",
    "get_repository_cache",
    "invalidate_file_cache",
    "is_secret_file",
    "parse_source_file",
]
