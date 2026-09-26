"""
ultron.core.repository.retriever
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Task-aware multi-criteria repository retriever with explicit provenance tracking.
Combines lexical matching, symbol matching, dependency graph proximity, and
recent-change relevance to identify relevant repository context without dumping
the whole repository.
"""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, Field

from ultron.core.repository.cache import RepositoryCache, get_repository_cache
from ultron.core.repository.graph import RepositoryGraph
from ultron.core.repository.index import RepositoryIndex


class RelevanceProvenance(BaseModel):
    """Structured explanation of why a file or symbol was selected as relevant."""

    file_path: str
    symbol_name: str | None = None
    line: int | None = None
    end_line: int | None = None
    score: float = 0.0
    reason: str = ""

    def to_summary_line(self) -> str:
        loc = f":{self.line}" if self.line else ""
        sym = f" [{self.symbol_name}]" if self.symbol_name else ""
        return f"{self.file_path}{loc}{sym} (score {self.score:.2f}) - {self.reason}"


class TaskRelevanceResult(BaseModel):
    """Ranked results with provenance explanations."""

    query: str
    items: list[RelevanceProvenance] = Field(default_factory=list)
    top_files: list[str] = Field(default_factory=list)
    top_symbols: list[str] = Field(default_factory=list)

    def to_context_block(self) -> str:
        if not self.items:
            return f"No repository matches found for task: '{self.query}'"
        lines = [f"Relevant Repository Context for task: '{self.query}'"]
        for it in self.items:
            lines.append(f"  - {it.to_summary_line()}")
        return "\n".join(lines)


def _tokenize(text: str) -> list[str]:
    """Tokenizes natural language and code identifiers into lowercase words."""
    if not text:
        return []
    # Split on whitespace, underscores, hyphens, and camelCase
    words = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)|\d+", text)
    return [w.lower() for w in words if len(w) > 1]


class TaskAwareRetriever:
    """
    Retrieves prioritized files, symbols, and dependency contexts for a given task.
    """

    def __init__(
        self,
        root: str | Path,
        index: RepositoryIndex | None = None,
        graph: RepositoryGraph | None = None,
        cache: RepositoryCache | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.cache = cache or get_repository_cache()
        self.index = index or RepositoryIndex(self.root, cache=self.cache)
        self._graph = graph

    @property
    def graph(self) -> RepositoryGraph:
        if self._graph is None:
            root_data = self.cache.get_root_data(self.root)
            if root_data.get("graph") is None:
                root_data["graph"] = RepositoryGraph(self.index)
            self._graph = root_data["graph"]
        return self._graph

    def retrieve_relevant(
        self,
        task_query: str,
        max_files: int = 8,
        max_symbols: int = 12,
        recent_files: list[str] | None = None,
    ) -> TaskRelevanceResult:
        """
        Executes multi-criteria ranking and returns provenance-annotated items.
        """
        q = (task_query or "").strip()
        if not q:
            return TaskRelevanceResult(query=q)

        tokens = _tokenize(q)
        token_set = set(tokens)

        # 1. Direct symbol matching
        symbol_matches: dict[str, list[RelevanceProvenance]] = {}  # file -> list of items
        for finfo in self.index.files.values():
            for sym in finfo.symbols:
                sym_tokens = set(_tokenize(sym.name))
                overlap = token_set.intersection(sym_tokens)

                # Exact symbol name match (e.g. query has "TaskState" or "task_state")
                exact_name_match = (
                    sym.name.lower() in [t.lower() for t in q.split()]
                    or any(sym.name.lower() == t for t in tokens)
                )

                if exact_name_match:
                    item = RelevanceProvenance(
                        file_path=finfo.rel_path,
                        symbol_name=sym.name,
                        line=sym.line,
                        end_line=sym.end_line,
                        score=0.95,
                        reason=f"Exact symbol match for '{sym.name}'",
                    )
                    symbol_matches.setdefault(finfo.rel_path, []).append(item)
                elif len(overlap) >= 1 and sym.kind in ("class", "function"):
                    item = RelevanceProvenance(
                        file_path=finfo.rel_path,
                        symbol_name=sym.name,
                        line=sym.line,
                        end_line=sym.end_line,
                        score=0.60 + 0.1 * len(overlap),
                        reason=f"Partial symbol match for '{sym.name}' on {list(overlap)}",
                    )
                    symbol_matches.setdefault(finfo.rel_path, []).append(item)

        # 2. Lexical path & docstring matching
        path_matches: dict[str, RelevanceProvenance] = {}
        for rel_path, finfo in self.index.files.items():
            path_tokens = set(_tokenize(rel_path))
            path_overlap = token_set.intersection(path_tokens)
            if path_overlap:
                score = min(0.85, 0.40 + 0.15 * len(path_overlap))
                path_matches[rel_path] = RelevanceProvenance(
                    file_path=rel_path,
                    score=score,
                    reason=f"Path token match on {sorted(path_overlap)}",
                )

        # 3. Combine initial hits
        initial_file_scores: dict[str, float] = {}
        provenance_by_file: dict[str, list[RelevanceProvenance]] = {}

        for p, prov_list in symbol_matches.items():
            best_sc = max(it.score for it in prov_list)
            initial_file_scores[p] = max(initial_file_scores.get(p, 0.0), best_sc)
            provenance_by_file.setdefault(p, []).extend(prov_list)

        for p, prov in path_matches.items():
            initial_file_scores[p] = max(initial_file_scores.get(p, 0.0), prov.score)
            provenance_by_file.setdefault(p, []).append(prov)

        # 4. Dependency Proximity Expansion
        # Files that import or are imported by initial hits get a proximity boost
        seed_files = list(initial_file_scores.keys())
        for seed in seed_files:
            seed_score = initial_file_scores[seed]
            neighbors = self.graph.get_neighbors(seed, max_depth=1)
            for neighbor in neighbors:
                prox_score = seed_score * 0.55
                if prox_score > initial_file_scores.get(neighbor, 0.0):
                    initial_file_scores[neighbor] = prox_score
                    provenance_by_file.setdefault(neighbor, []).append(
                        RelevanceProvenance(
                            file_path=neighbor,
                            score=prox_score,
                            reason=f"Dependency proximity to seed '{seed}'",
                        )
                    )

        # 5. Recent changes boost
        if recent_files:
            for rf in recent_files:
                if rf in self.index.files:
                    initial_file_scores[rf] = initial_file_scores.get(rf, 0.0) + 0.35
                    provenance_by_file.setdefault(rf, []).append(
                        RelevanceProvenance(
                            file_path=rf,
                            score=0.75,
                            reason="Recently modified in workspace",
                        )
                    )

        # 6. Rank files by score
        ranked_files = sorted(
            initial_file_scores.keys(),
            key=lambda p: (initial_file_scores[p], self.graph.get_pagerank(p)),
            reverse=True,
        )

        selected_files = ranked_files[:max_files]

        # Consolidate top provenance items
        all_items: list[RelevanceProvenance] = []
        top_symbols: list[str] = []

        for p in selected_files:
            items = provenance_by_file.get(p, [])
            # Pick highest score item for this file
            if items:
                best_item = max(items, key=lambda it: it.score)
                all_items.append(best_item)
                if best_item.symbol_name and best_item.symbol_name not in top_symbols:
                    top_symbols.append(best_item.symbol_name)
            else:
                all_items.append(
                    RelevanceProvenance(
                        file_path=p,
                        score=initial_file_scores.get(p, 0.1),
                        reason="Topological PageRank relevance",
                    )
                )

        return TaskRelevanceResult(
            query=q,
            items=all_items,
            top_files=selected_files,
            top_symbols=top_symbols[:max_symbols],
        )
