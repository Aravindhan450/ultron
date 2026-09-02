"""Dynamic repository subject discovery (Phase 4 of STEP 3).

The framework must discover arbitrary test subjects from the CURRENT
repository — classes, functions, modules, files — never from a hardcoded
list.  This module reuses the existing Code Intelligence index (via the
facade's public API) so no second repository index is created.

Subjects are consumed by the task generator, which renders them into varied
natural-language requests.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ultron.core.coding.intelligence.facade import CodeIntelligence
from ultron.core.coding.intelligence.symbols import SymbolKind
from ultron.core.coding.workspace import _IGNORED_DIRS

# Kinds that make good "where is X defined/used" subjects.
_SUBJECT_KINDS = {
    SymbolKind.CLASS,
    SymbolKind.INTERFACE,
    SymbolKind.ENUM,
    SymbolKind.STRUCT,
    SymbolKind.TRAIT,
    SymbolKind.FUNCTION,
    SymbolKind.TYPE_ALIAS,
}

_SOURCE_EXTENSIONS = {".py", ".md", ".toml", ".yaml", ".yml", ".json", ".txt", ".cfg", ".ini"}


@dataclass(frozen=True)
class Subject:
    """One repository entity usable as a task subject."""

    name: str
    kind: str  # class / function / enum / file / module
    rel_path: str  # repository-relative path
    line: int = 0

    @property
    def display(self) -> str:
        return f"{self.name} ({self.kind}, {self.rel_path}:{self.line})"


def _walk_source_py(root: Path) -> list[str]:
    """Deterministic list of repository-relative Python files (src-first)."""
    rels: list[str] = []
    for path in sorted(root.rglob("*.py")):
        parts = path.relative_to(root).parts
        if any(p in _IGNORED_DIRS or p.startswith(".") for p in parts[:-1]):
            continue
        rels.append(str(path.relative_to(root)))
    # src/ implementation files first — they are the best task subjects.
    src = [r for r in rels if r.startswith("src/")]
    rest = [r for r in rels if not r.startswith("src/")]
    return src + rest


def _symbols_in_file(ci: CodeIntelligence, rel: str) -> list:
    """Symbols for one indexed file; never raises on unparsed files."""
    try:
        return ci.index.find_symbols_in_file(rel)
    except Exception:  # noqa: BLE001 — index lookups are best-effort
        return []


def _symbol_subjects(ci: CodeIntelligence, root: Path, max_count: int) -> list[Subject]:
    """Classes/functions/enums from the existing index (deterministic sample).

    Prefers implementation symbols under ``src/`` — implementation questions
    must not be dominated by test/doc subjects — then fills remaining slots
    from the broad name index when the src pool is small.
    """
    subjects: list[Subject] = []
    for rel in _walk_source_py(root):
        if len(subjects) >= max_count:
            break
        symbols = _symbols_in_file(ci, rel)
        # Take up to two *definition-kind* symbols per file (raw rows include
        # imports/references that are not valid subjects).
        taken = 0
        for symbol in symbols:
            if taken >= 2 or len(subjects) >= max_count:
                break
            if symbol.kind not in _SUBJECT_KINDS:
                continue
            if symbol.name.startswith("_") or len(symbol.name) < 2:
                continue
            subjects.append(
                Subject(
                    name=symbol.name,
                    kind=symbol.kind.value,
                    rel_path=rel,
                    line=symbol.location.line,
                )
            )
            taken += 1
    if len(subjects) < max_count:
        # Fill with a deterministic spread over the full name index.
        names = ci.index.all_symbol_names()
        stride = max(1, len(names) // (max_count * 2)) if names else 1
        seen = {s.name for s in subjects}
        for name in names[::stride]:
            if len(subjects) >= max_count:
                break
            if not name or name.startswith("_") or name in seen:
                continue
            for symbol in ci.find_symbol(name)[:1]:
                if symbol.kind not in _SUBJECT_KINDS:
                    continue
                if not (root / symbol.location.file).is_file():
                    continue
                subjects.append(
                    Subject(
                        name=symbol.name,
                        kind=symbol.kind.value,
                        rel_path=symbol.location.file,
                        line=symbol.location.line,
                    )
                )
                seen.add(symbol.name)
                break
    return subjects


def _file_subjects(root: Path, max_count: int) -> list[Subject]:
    """Repository source files (deterministic walk, ignore-dir aware)."""
    subjects: list[Subject] = []
    for path in sorted(root.rglob("*")):
        if len(subjects) >= max_count:
            break
        if not path.is_file():
            continue
        parts = path.relative_to(root).parts
        if any(p in _IGNORED_DIRS or p.startswith(".") for p in parts[:-1]):
            continue
        if path.suffix.lower() not in _SOURCE_EXTENSIONS:
            continue
        subjects.append(
            Subject(name=path.name, kind="file", rel_path=str(path.relative_to(root)))
        )
    return subjects


def _directory_subjects(root: Path) -> list[Subject]:
    """Workspace root plus top-level directories — directory-listing subjects."""
    subjects = [Subject(name=root.name or "workspace", kind="directory", rel_path=".")]
    for child in sorted(root.iterdir()):
        if child.is_dir() and not child.name.startswith(".") and child.name not in _IGNORED_DIRS:
            subjects.append(Subject(name=child.name, kind="directory", rel_path=child.name))
    return subjects


def discover_subjects(
    root: str | Path, *, max_symbols: int = 30, max_files: int = 10, with_directories: bool = True
) -> list[Subject]:
    """Discovers a diverse subject pool from the current repository.

    Uses the existing Code Intelligence index for symbols (refreshing it
    first, like the agent's own tools do) plus a gitignore/ignore-dir-aware
    file walk for file subjects and the workspace's top-level directories.
    Deterministic: same repository yields the same pool.
    """
    root = Path(root).resolve()
    with CodeIntelligence(root=root) as ci:
        ci.refresh()
        symbols = _symbol_subjects(ci, root, max_symbols)
    files = _file_subjects(root, max_files)
    if with_directories:
        dirs = _directory_subjects(root)
    else:
        dirs = []
    # Symbols first (they exercise code intelligence), files, then directories.
    return symbols + files + dirs


def subjects_by_kind(subjects: list[Subject], kind: str) -> list[Subject]:
    """Filters a subject pool to one kind (class / function / file / ...)."""
    return [s for s in subjects if s.kind == kind]
