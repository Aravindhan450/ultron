"""
ultron.core.repository.cache
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Cache management and incremental invalidation for repository intelligence.
Ensures that file modifications, additions, or deletions immediately invalidate
cached ASTs, dependency graphs, and repo maps for affected roots.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any


class RepositoryCache:
    """
    Per-workspace in-memory cache for parsed symbols, dependency graphs,
    and rendered repo maps.

    Architecture Note:
    In accordance with the Ultron Autonomous Coding Harness Gap Closure Plan
    (Phase 3 & Section 2.1 Invariants), repository index caching is intentionally
    process-local and thread-safe. Bounded in-memory caching ensures low-latency
    retrieval while eliminating stale cache hazards across git branch switches or
    external workspace mutations. Thread synchronization via RLock guarantees
    integrity across concurrent tool execution and background subagent tasks.
    """

    _instance: RepositoryCache | None = None
    _class_lock: threading.RLock = threading.RLock()

    def __init__(self) -> None:
        # root -> { "files": {rel_path: FileInfo}, "graph": RepositoryGraph, "repo_maps": {key: RepoMap} }
        self._roots: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

    @classmethod
    def get_instance(cls) -> RepositoryCache:
        with cls._class_lock:
            if cls._instance is None:
                cls._instance = RepositoryCache()
            return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Resets the singleton instance (useful for testing)."""
        with cls._class_lock:
            cls._instance = RepositoryCache()

    def get_root_data(self, root: str | Path) -> dict[str, Any]:
        key = str(Path(root).resolve())
        with self._lock:
            if key not in self._roots:
                self._roots[key] = {
                    "files": {},
                    "graph": None,
                    "repo_maps": {},
                }
            return self._roots[key]

    def invalidate_file(self, file_path: str | Path, root: str | Path | None = None) -> None:
        """
        Invalidates cached data for a specific file across all relevant roots.
        """
        target_path = Path(file_path).resolve()
        with self._lock:
            for root_str, data in list(self._roots.items()):
                root_path = Path(root_str)
                if root is not None and str(Path(root).resolve()) != root_str:
                    continue

                try:
                    rel = str(target_path.relative_to(root_path))
                except ValueError:
                    # File is not under this root
                    continue

                # Remove file from file cache
                if "files" in data and rel in data["files"]:
                    del data["files"][rel]

                # Invalidate graph and repo maps for this root
                data["graph"] = None
                data["repo_maps"] = {}

    def invalidate_root(self, root: str | Path) -> None:
        """
        Invalidates all cached data for a repository root.
        """
        key = str(Path(root).resolve())
        with self._lock:
            if key in self._roots:
                del self._roots[key]

    def clear(self) -> None:
        """
        Clears all cached data across all repositories.
        """
        with self._lock:
            self._roots.clear()



def invalidate_file_cache(file_path: str | Path, root: str | Path | None = None) -> None:
    """Convenience helper to invalidate cache for a file."""
    RepositoryCache.get_instance().invalidate_file(file_path, root=root)


def get_repository_cache() -> RepositoryCache:
    """Returns the singleton RepositoryCache instance."""
    return RepositoryCache.get_instance()
