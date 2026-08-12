"""ultron.core.coding.test_selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Deterministic test discovery and affected-test selection (Fix #5).

The coding agent must know WHICH tests exist and WHICH tests are likely
affected by a change — without asking the LLM and without running anything.

:func:`discover_test_files` builds a deterministic inventory of test files
under a project's test directories (or the whole tree when no conventional
test directory exists). Test-file detection is name-based and multi-language
(pytest ``test_*`` / ``*_test``, jest/vitest ``*.test.*`` / ``*.spec.*``,
go ``*_test.go``, rust ``*_test.rs``, java ``*Test.java``).

:func:`select_affected_tests` maps changed source files to candidate test
files using deterministic conventions:

- mirror:  ``src/auth/service.py`` -> ``tests/auth/test_service.py``
- sibling: ``src/auth/service.py`` -> ``tests/auth/service_test.py``
- module:  ``src/auth/service.py`` -> ``tests/test_auth.py``
- node:    ``src/auth/service.ts`` -> ``tests/auth/service.test.ts``

Candidates are filtered against the on-disk test inventory, so only tests
that actually exist are returned. When a code-intelligence facade is
provided, dependents that are test files (modules that import the changed
module) are added — the FIX #4 dependency graph feeds test selection.

Security: pure read-only filesystem inspection inside *root*; never
executes anything and never leaves the workspace.
"""

from __future__ import annotations

import os
from pathlib import Path

# Directories that are never part of a project's source/test surface.
# Mirrors workspace._IGNORED_DIRS (kept local to avoid a private cross-module
# import; keep the two sets in sync when adding new vendored dirs).
_IGNORED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "dist",
    "build",
    ".next",
    ".nuxt",
    ".cache",
    ".idea",
    ".vscode",
    "coverage",
    "htmlcov",
}

# Conventional test directories, tried in order.
DEFAULT_TEST_DIRS = ("tests", "test", "__tests__", "spec", "specs")

# Source prefixes stripped before mirroring a path into a test directory.
_SOURCE_PREFIXES = ("src/", "lib/", "app/", "ultron/")


def _is_test_file(rel: str) -> bool:
    """True when *rel* (a relative path) names a test file by convention."""
    name = rel.replace("\\", "/").rsplit("/", 1)[-1]
    stem = name.rsplit(".", 1)[0] if "." in name else name
    return stem.startswith("test_") or stem.endswith(
        ("_test", ".test", ".spec", "Test", "Tests")
    )


def discover_test_files(
    root: str | Path,
    test_dirs: list[str] | None = None,
    max_files: int = 200,
) -> list[str]:
    """
    Discovers test files under *root* (deterministic, bounded, read-only).

    Walks each existing conventional test directory (or the whole tree when
    none exists), skipping ignored/vendored directories, and returns sorted
    relative paths whose names match test conventions. Capped at
    *max_files* to keep large repositories bounded.
    """
    base = Path(root).resolve()
    if not base.is_dir():
        return []

    dirs = [d for d in (test_dirs or DEFAULT_TEST_DIRS) if (base / d).is_dir()]
    if not dirs:
        # No conventional test dir — scan the whole tree (pruned).
        dirs = ["."]

    found: list[str] = []

    def _walk(start: Path) -> None:
        for dirpath, dirnames, filenames in os.walk(start):
            dirnames[:] = sorted(
                n for n in dirnames if n not in _IGNORED_DIRS and not n.startswith(".")
            )
            for filename in sorted(filenames):
                rel = str(Path(dirpath).relative_to(base) / filename)
                if _is_test_file(rel):
                    found.append(rel)
                    if len(found) >= max_files:
                        return
            if len(found) >= max_files:
                return

    for d in dirs:
        _walk(base / d)
        if len(found) >= max_files:
            break
    return sorted(found)


def _test_candidates(rel: str, stem: str, ext: str) -> list[str]:
    """Deterministic candidate test paths for one changed source file."""
    parts = stem.split("/")
    leaf = parts[-1]
    parent = parts[:-1]
    candidates: list[str] = []

    # Strip source prefixes and mirror the remainder (src/auth/service.py ->
    # tests/auth/test_service.py). The first loop mirrors the RAW path; the
    # srcp loop mirrors the path minus the source prefix. Loop for prefix
    # priority; at most one applies.
    sub_stem, sub_ext = stem, ext
    for srcp in _SOURCE_PREFIXES:
        if rel.startswith(srcp):
            sub = rel[len(srcp) :]
            sub_stem, _, sub_ext = sub.rpartition(".")
            break
    sparts = sub_stem.split("/")
    s_leaf = sparts[-1]
    s_parent = sparts[:-1]
    module = sparts[0] if len(sparts) >= 2 else None

    for td in DEFAULT_TEST_DIRS:
        prefix = f"{td}/"
        # Mirror + sibling variants (leaf-stem based).
        candidates.append(f"{prefix}{'/'.join(s_parent + [f'test_{s_leaf}.{sub_ext}'])}")
        candidates.append(f"{prefix}{'/'.join(s_parent + [f'{s_leaf}_test.{sub_ext}'])}")
        if sub_ext in ("ts", "js", "tsx", "jsx"):
            candidates.append(
                f"{prefix}{'/'.join(s_parent + [f'{s_leaf}.test.{sub_ext}'])}"
            )
            candidates.append(
                f"{prefix}{'/'.join(s_parent + [f'{s_leaf}.spec.{sub_ext}'])}"
            )
        # Module-level variant: src/auth/service.py -> tests/test_auth.py.
        if module is not None:
            candidates.append(f"{prefix}test_{module}.{sub_ext}")
        # Legacy raw-path mirror (kept for layouts without a source prefix).
        if parts != sparts:
            candidates.append(f"{prefix}{'/'.join(parent + [f'test_{leaf}.{ext}'])}")
            candidates.append(f"{prefix}{'/'.join(parent + [f'{leaf}_test.{ext}'])}")

    return list(dict.fromkeys(candidates))


def _test_target_stem(name: str) -> str:
    """The tested module name implied by a test filename (best-effort)."""
    # Work on the stem only — markers never span the extension
    # (``service_test.go`` -> ``service_test`` -> ``service``).
    stem = name.rsplit(".", 1)[0] if "." in name else name
    for marker in ("test_", "_test", ".test", ".spec", "Test", "Tests"):
        if marker == "test_" and stem.startswith(marker):
            return stem[len(marker) :]
        if stem.endswith(marker):
            return stem[: -len(marker)]
    return stem


def select_affected_tests(
    changed_files: list[str],
    root: str | Path,
    test_dirs: list[str] | None = None,
    intelligence=None,
    max_results: int = 12,
) -> list[str]:
    """
    Selects the test files most likely affected by *changed_files*.

    Deterministic convention mapping (mirror / sibling / module / node
    variants) filtered against the on-disk test inventory, plus:

    - a changed test file selects itself;
    - when *intelligence* (a code-intelligence facade with
      ``get_dependents(rel_path) -> list[str]``) is provided, dependents
      that are test files are added — the import graph feeds selection.

    Only existing tests are returned; results are sorted and capped at
    *max_results*.
    """
    base = Path(root).resolve()
    test_files = discover_test_files(base, test_dirs=test_dirs)
    test_set = set(test_files)

    # Index test files by the module they target, for stem matching.
    by_target: dict[str, list[str]] = {}
    for tf in test_files:
        core = _test_target_stem(tf.rsplit("/", 1)[-1])
        if core:
            by_target.setdefault(core, []).append(tf)

    selected: set[str] = set()

    def _exists(candidate: str) -> bool:
        # Accept either a discovered test file or a real path on disk.
        return candidate in test_set or (base / candidate).is_file()

    for rel in changed_files:
        rel = rel.replace("\\", "/").lstrip("./")
        if not rel:
            continue
        if _is_test_file(rel):
            selected.add(rel)  # a changed test affects itself
            continue
        stem, _, ext = rel.rpartition(".")
        if not stem:
            continue
        for candidate in _test_candidates(rel, stem, ext):
            if _exists(candidate):
                selected.add(candidate)
        # Stem fallback: tests whose target matches the changed module.
        leaf = stem.split("/")[-1]
        for hit in by_target.get(leaf, []):
            if _exists(hit):
                selected.add(hit)

    # Intelligence hook (Fix #4): dependents that are test files. Only
    # on-disk test files are added (stale index entries are ignored).
    if intelligence is not None and hasattr(intelligence, "get_dependents"):
        for rel in changed_files:
            rel = rel.replace("\\", "/").lstrip("./")
            if not rel:
                continue
            try:
                dependents = intelligence.get_dependents(rel)
            except Exception:  # noqa: BLE001 — intelligence must never crash selection
                dependents = []
            for dep in dependents or []:
                dep = dep.replace("\\", "/")
                if _is_test_file(dep) and (base / dep).is_file():
                    selected.add(dep)

    return sorted(selected)[:max_results]
