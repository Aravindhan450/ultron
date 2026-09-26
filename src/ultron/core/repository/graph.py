"""
ultron.core.repository.graph
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dependency graph and PageRank centrality engine for repository intelligence.
Builds exact and inferred relationships between files and symbols, calculating
structural importance scores to rank files for RepoMap generation and task-aware retrieval.
"""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel

from ultron.core.repository.index import RepositoryIndex


class GraphEdge(BaseModel):
    """A directed dependency relationship from source to target."""

    source: str  # rel_path of dependent
    target: str  # rel_path of dependency
    kind: str = "import"  # "import", "reference"
    weight: float = 1.0


class GraphNode(BaseModel):
    """A node representing a file in the repository graph."""

    rel_path: str
    pagerank: float = 0.0
    in_degree: int = 0
    out_degree: int = 0
    symbols_count: int = 0


class RepositoryGraph:
    """
    Constructs and analyzes the dependency graph across indexed repository files.
    """

    def __init__(self, index: RepositoryIndex) -> None:
        self.index = index
        self._adj_out: dict[str, set[str]] = defaultdict(set)  # u -> {v} (u imports v)
        self._adj_in: dict[str, set[str]] = defaultdict(set)   # v -> {u} (u imported by v)
        self._pagerank_scores: dict[str, float] = {}
        self._build_graph()

    def _resolve_import_to_file(self, source_file: str, imported_module: str, is_relative: bool) -> str | None:
        """
        Attempts to resolve an imported module string to a concrete repository file path.
        """
        if not imported_module:
            return None

        source_path = Path(source_file)
        source_dir = source_path.parent

        # 1. Relative import resolution
        if is_relative or imported_module.startswith("."):
            clean_mod = imported_module.lstrip(".")
            candidate_parts = clean_mod.split(".") if clean_mod else []
            # Calculate parent steps
            dot_count = len(imported_module) - len(imported_module.lstrip("."))
            target_dir = source_dir
            for _ in range(max(0, dot_count - 1)):
                target_dir = target_dir.parent

            rel_candidate_base = target_dir.joinpath(*candidate_parts)
            # Try .py or /__init__.py
            cand1 = f"{rel_candidate_base}.py"
            cand2 = f"{rel_candidate_base}/__init__.py"
            cand1_norm = os.path.normpath(cand1).replace("\\", "/")
            cand2_norm = os.path.normpath(cand2).replace("\\", "/")
            if cand1_norm in self.index.files:
                return cand1_norm
            if cand2_norm in self.index.files:
                return cand2_norm

        # 2. Absolute package import resolution
        mod_parts = imported_module.split(".")
        # Try matching against known files in the repository
        # e.g., "ultron.core.runtime" -> matches "src/ultron/core/runtime.py" or "ultron/core/runtime.py"
        for rel_file in self.index.files:
            file_no_ext = os.path.splitext(rel_file)[0].replace("\\", "/")
            file_parts = file_no_ext.split("/")
            # Check if mod_parts is a suffix or subpath of file_parts
            if len(file_parts) >= len(mod_parts) and file_parts[-len(mod_parts):] == mod_parts:
                return rel_file
            # Check for __init__.py package match
            if (
                file_parts[-1] == "__init__"
                and len(file_parts) - 1 >= len(mod_parts)
                and file_parts[-1 - len(mod_parts): -1] == mod_parts
            ):
                return rel_file

        return None

    def _build_graph(self) -> None:
        """Populates edges from indexed file imports and symbol definitions."""
        files = self.index.files
        # Map of symbol name -> list of files defining it
        symbol_def_files: dict[str, list[str]] = defaultdict(list)
        for rel_path, finfo in files.items():
            for sym in finfo.symbols:
                if sym.kind in ("class", "function", "interface"):
                    symbol_def_files[sym.name].append(rel_path)

        for rel_path, finfo in files.items():
            for imp in finfo.imports:
                target_file = self._resolve_import_to_file(rel_path, imp.imported, imp.is_relative)
                if target_file and target_file != rel_path:
                    self._adj_out[rel_path].add(target_file)
                    self._adj_in[target_file].add(rel_path)
                elif imp.member and imp.member in symbol_def_files:
                    # Import member resolves directly to a defined symbol
                    for def_file in symbol_def_files[imp.member]:
                        if def_file != rel_path:
                            self._adj_out[rel_path].add(def_file)
                            self._adj_in[def_file].add(rel_path)

        self._compute_pagerank()

    def _compute_pagerank(self, damping: float = 0.85, max_iter: int = 50, tol: float = 1e-4) -> None:
        """
        Computes standard PageRank scores over the repository graph.
        Files with high incoming dependencies receive higher centrality.
        """
        all_nodes = list(self.index.files.keys())
        N = len(all_nodes)
        if N == 0:
            self._pagerank_scores = {}
            return

        # Initialize uniform distribution
        scores = {node: 1.0 / N for node in all_nodes}

        for _ in range(max_iter):
            new_scores = {}
            # Base teleportation
            teleport = (1.0 - damping) / N
            for node in all_nodes:
                incoming_sum = 0.0
                for inbound in self._adj_in.get(node, set()):
                    out_deg = len(self._adj_out.get(inbound, set()))
                    if out_deg > 0:
                        incoming_sum += scores[inbound] / out_deg
                new_scores[node] = teleport + damping * incoming_sum

            # Check convergence
            diff = sum(abs(new_scores[n] - scores[n]) for n in all_nodes)
            scores = new_scores
            if diff < tol:
                break

        # Normalize so maximum score is 1.0
        max_score = max(scores.values()) if scores and max(scores.values()) > 0 else 1.0
        self._pagerank_scores = {k: v / max_score for k, v in scores.items()}

    def get_pagerank(self, rel_path: str) -> float:
        """Returns the calculated PageRank importance score (0.0 to 1.0)."""
        return self._pagerank_scores.get(rel_path, 0.0)

    def get_node(self, rel_path: str) -> GraphNode:
        """Returns structural metrics for a specific file node."""
        finfo = self.index.get_file(rel_path)
        sym_count = len(finfo.symbols) if finfo else 0
        return GraphNode(
            rel_path=rel_path,
            pagerank=self.get_pagerank(rel_path),
            in_degree=len(self._adj_in.get(rel_path, set())),
            out_degree=len(self._adj_out.get(rel_path, set())),
            symbols_count=sym_count,
        )

    def get_neighbors(self, rel_path: str, max_depth: int = 1) -> set[str]:
        """
        Returns connected dependencies (upstream imports + downstream dependents)
        up to max_depth hops.
        """
        visited: set[str] = set()
        current_layer = {rel_path}

        for _ in range(max_depth):
            next_layer: set[str] = set()
            for node in current_layer:
                # Add upstream imports
                next_layer.update(self._adj_out.get(node, set()))
                # Add downstream dependents
                next_layer.update(self._adj_in.get(node, set()))
            next_layer.discard(rel_path)
            visited.update(next_layer)
            current_layer = next_layer
            if not current_layer:
                break

        return visited

    def compute_personalized_pagerank(
        self, seed_files: list[str], damping: float = 0.85, max_iter: int = 30
    ) -> dict[str, float]:
        """
        Computes personalized PageRank biased towards the seed_files.
        Useful for ranking files close to a specific task target.
        """
        all_nodes = list(self.index.files.keys())
        N = len(all_nodes)
        if N == 0:
            return {}

        valid_seeds = [s for s in seed_files if s in self.index.files]
        if not valid_seeds:
            return self._pagerank_scores.copy()

        seed_set = set(valid_seeds)
        teleport_weight = 1.0 / len(valid_seeds)

        scores = {node: (teleport_weight if node in seed_set else 0.0) for node in all_nodes}

        for _ in range(max_iter):
            new_scores = {}
            for node in all_nodes:
                teleport = (1.0 - damping) * (teleport_weight if node in seed_set else 0.0)
                incoming_sum = 0.0
                for inbound in self._adj_in.get(node, set()):
                    out_deg = len(self._adj_out.get(inbound, set()))
                    if out_deg > 0:
                        incoming_sum += scores[inbound] / out_deg
                new_scores[node] = teleport + damping * incoming_sum

            scores = new_scores

        max_score = max(scores.values()) if scores and max(scores.values()) > 0 else 1.0
        return {k: v / max_score for k, v in scores.items()}
