"""
ultron.core.repository.repo_map
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Token-budgeted Aider-class Repository Map generator.
Produces a compact, structured representation of the repository's directory layout,
high-value files, and key symbols/signatures ranked by PageRank centrality and
task relevance, strictly honoring a hard token budget.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from ultron.core.context.models import estimate_tokens
from ultron.core.repository.cache import RepositoryCache, get_repository_cache
from ultron.core.repository.graph import RepositoryGraph
from ultron.core.repository.index import FileInfo, RepositoryIndex


class RepoMap(BaseModel):
    """Rendered repository map with token usage metrics."""

    text: str
    token_count: int
    included_files: list[str] = Field(default_factory=list)
    elided_files_count: int = 0
    top_symbols: list[str] = Field(default_factory=list)
    max_tokens_budget: int = 1000


class RepoMapGenerator:
    """
    Constructs a budgeted, prioritized structural overview of a repository.
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

    def generate(
        self,
        max_tokens: int = 1000,
        focus_files: list[str] | None = None,
        focus_query: str | None = None,
    ) -> RepoMap:
        """
        Generates a compact repo map strictly within max_tokens.
        """
        # Check cache if no focus parameters provided
        root_data = self.cache.get_root_data(self.root)
        cache_key = f"{max_tokens}:{focus_files}:{focus_query}"
        if "repo_maps" in root_data and cache_key in root_data["repo_maps"]:
            return root_data["repo_maps"][cache_key]

        all_files = self.index.files
        if not all_files:
            empty_map = RepoMap(
                text="[Repository is empty or unindexed]",
                token_count=estimate_tokens("[Repository is empty or unindexed]"),
                max_tokens_budget=max_tokens,
            )
            return empty_map

        # Calculate file relevance scores
        file_scores = self._score_files(focus_files=focus_files, focus_query=focus_query)

        # Sort files by descending relevance score
        sorted_files = sorted(
            all_files.keys(),
            key=lambda p: (
                file_scores.get(p, 0.0),
                1 if all_files[p].is_source else 0,
                -len(all_files[p].symbols),
            ),
            reverse=True,
        )

        # Multi-tier fit into max_tokens
        repo_map = self._fit_to_budget(sorted_files, max_tokens=max_tokens)

        # Save to cache
        root_data["repo_maps"][cache_key] = repo_map
        return repo_map

    def _score_files(
        self, focus_files: list[str] | None = None, focus_query: str | None = None
    ) -> dict[str, float]:
        """Calculates composite file scores based on PageRank and focus terms."""
        all_files = self.index.files
        base_scores = {p: self.graph.get_pagerank(p) for p in all_files}

        # Apply personalized PageRank if focus_files provided
        if focus_files:
            seed_scores = self.graph.compute_personalized_pagerank(focus_files)
            for p, sc in seed_scores.items():
                base_scores[p] = 0.5 * base_scores.get(p, 0.0) + 0.5 * sc

        # Boost matching files if focus_query provided
        if focus_query:
            q_terms = [t.lower() for t in focus_query.split() if len(t) > 2]
            for p, finfo in all_files.items():
                p_lower = p.lower()
                # Path match boost
                for term in q_terms:
                    if term in p_lower:
                        base_scores[p] = base_scores.get(p, 0.0) + 0.4
                    # Symbol name match boost
                    for sym in finfo.symbols:
                        if term in sym.name.lower():
                            base_scores[p] = base_scores.get(p, 0.0) + 0.6
                            break

        return base_scores

    def _format_file_tier1(self, finfo: FileInfo) -> list[str]:
        """Tier 1: Full detail with classes, methods, signatures."""
        lines = [f"{finfo.rel_path}:"]
        # Top-level classes
        classes = [s for s in finfo.symbols if s.kind == "class"]
        funcs = [s for s in finfo.symbols if s.kind in ("function", "method") and not s.parent]
        methods_by_parent: dict[str, list[str]] = {}
        for s in finfo.symbols:
            if s.parent and s.kind == "method":
                methods_by_parent.setdefault(s.parent, []).append(s.name)

        for c in classes[:4]:
            bases_str = f"({', '.join(c.bases)})" if c.bases else ""
            lines.append(f"  class {c.name}{bases_str}:")
            m_list = methods_by_parent.get(c.name, [])
            for m in m_list[:4]:
                lines.append(f"    def {m}(...)")
            if len(m_list) > 4:
                lines.append(f"    [+{len(m_list) - 4} methods...]")

        for f in funcs[:5]:
            sig = f.signature or f"def {f.name}(...)"
            lines.append(f"  {sig}")

        if not classes and not funcs and finfo.symbols:
            for s in finfo.symbols[:4]:
                lines.append(f"  {s.kind} {s.name}")

        return lines

    def _format_file_tier2(self, finfo: FileInfo) -> list[str]:
        """Tier 2: Compact symbols (names only)."""
        lines = [f"{finfo.rel_path}:"]
        classes = [s.name for s in finfo.symbols if s.kind == "class"]
        funcs = [s.name for s in finfo.symbols if s.kind in ("function", "method") and not s.parent]
        if classes:
            lines.append(f"  classes: {', '.join(classes[:4])}")
        if funcs:
            lines.append(f"  funcs: {', '.join(funcs[:5])}")
        if not classes and not funcs and finfo.symbols:
            names = [s.name for s in finfo.symbols[:5]]
            lines.append(f"  symbols: {', '.join(names)}")
        return lines

    def _format_file_tier3(self, finfo: FileInfo) -> list[str]:
        """Tier 3: File path with symbol count."""
        count = len(finfo.symbols)
        info = f" ({count} symbols)" if count > 0 else ""
        return [f"{finfo.rel_path}{info}"]

    def _render_candidate(
        self,
        sorted_files: list[str],
        tier1_count: int,
        tier2_count: int,
        tier3_count: int,
    ) -> tuple[str, list[str], int, list[str]]:
        """Renders repo map text with specified tier counts."""
        lines: list[str] = []
        included_files: list[str] = []
        top_symbols: list[str] = []

        idx = 0
        total_files = len(sorted_files)

        # Tier 1 files
        end_t1 = min(idx + tier1_count, total_files)
        for i in range(idx, end_t1):
            p = sorted_files[i]
            finfo = self.index.files[p]
            lines.extend(self._format_file_tier1(finfo))
            included_files.append(p)
            for s in finfo.symbols[:3]:
                top_symbols.append(s.name)
        idx = end_t1

        # Tier 2 files
        end_t2 = min(idx + tier2_count, total_files)
        for i in range(idx, end_t2):
            p = sorted_files[i]
            finfo = self.index.files[p]
            lines.extend(self._format_file_tier2(finfo))
            included_files.append(p)
            for s in finfo.symbols[:2]:
                top_symbols.append(s.name)
        idx = end_t2

        # Tier 3 files
        end_t3 = min(idx + tier3_count, total_files)
        for i in range(idx, end_t3):
            p = sorted_files[i]
            finfo = self.index.files[p]
            lines.extend(self._format_file_tier3(finfo))
            included_files.append(p)
        idx = end_t3

        elided = max(0, total_files - idx)
        if elided > 0:
            lines.append(f"[+{elided} other files elided...]")

        rendered = "\n".join(lines)
        return rendered, included_files, elided, top_symbols

    def _fit_to_budget(self, sorted_files: list[str], max_tokens: int) -> RepoMap:
        """
        Dynamically adjusts tier allocations to strictly satisfy the max_tokens budget.
        """
        total = len(sorted_files)
        if total == 0:
            return RepoMap(text="[No files indexed]", token_count=4, max_tokens_budget=max_tokens)

        # Start with ambitious counts based on total files
        t1 = min(15, total)
        t2 = min(20, max(0, total - t1))
        t3 = min(40, max(0, total - t1 - t2))

        # Iteratively shrink until budget is satisfied
        while True:
            text, inc_files, elided, top_syms = self._render_candidate(sorted_files, t1, t2, t3)
            tokens = estimate_tokens(text)

            if tokens <= max_tokens:
                return RepoMap(
                    text=text,
                    token_count=tokens,
                    included_files=inc_files,
                    elided_files_count=elided,
                    top_symbols=top_syms,
                    max_tokens_budget=max_tokens,
                )

            # Reduce tiers in order of detail
            if t1 > 2:
                t1 -= 1
                t2 += 1
            elif t2 > 2:
                t2 -= 1
                t3 += 1
            elif t3 > 2:
                t3 -= 2
            elif t1 > 0:
                t1 -= 1
            elif t2 > 0:
                t2 -= 1
            elif t3 > 0:
                t3 -= 1
            else:
                # Emergency fallback if budget is extremely tight
                minimal_text = f"Repository map: {total} files indexed (+elided to fit {max_tokens} tokens)"
                return RepoMap(
                    text=minimal_text,
                    token_count=estimate_tokens(minimal_text),
                    included_files=[],
                    elided_files_count=total,
                    max_tokens_budget=max_tokens,
                )
