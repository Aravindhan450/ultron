"""ultron.core.coding.intelligence.dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dependency graph over the repository index (Fix #4).

Relationships are explicitly classified so uncertain edges are never
presented as facts:

- ``EXACT`` — import edges parsed from the source (``file A imports file B``).
- ``INFERRED`` — structural guesses (class A references class B, function A
  calls function B) derived from reference hits in the index. These are
  best-effort and labelled as such.

Callers/callees at this layer are inferred from reference occurrences: a
function name appearing in another file's references is treated as a
candidate caller/callee, not a fact. Exact call graphs are the LSP layer's
job (see ``lsp.py``).
"""

from __future__ import annotations

import ast
import re
import sys
from enum import Enum

from pydantic import BaseModel

from ultron.core.coding.intelligence.index import RepositoryIndex

# Standard library module names in Python (using sys.stdlib_module_names with rich fallback)
STDLIB_MODULE_NAMES: frozenset[str] = getattr(
    sys,
    "stdlib_module_names",
    frozenset(
        {
            "tkinter",
            "sqlite3",
            "sys",
            "os",
            "re",
            "json",
            "math",
            "asyncio",
            "typing",
            "pathlib",
            "subprocess",
            "time",
            "datetime",
            "collections",
            "itertools",
            "functools",
            "shutil",
            "tempfile",
            "unittest",
            "urllib",
            "http",
            "logging",
            "io",
            "csv",
            "random",
            "hashlib",
            "copy",
            "dataclasses",
            "enum",
            "abc",
            "inspect",
            "struct",
            "fcntl",
            "termios",
            "select",
            "threading",
            "multiprocessing",
            "queue",
            "socket",
            "ssl",
            "email",
            "html",
            "xml",
            "xmlrpc",
            "zipfile",
            "tarfile",
            "gzip",
            "bz2",
            "lzma",
            "zlib",
            "contextlib",
            "warnings",
            "traceback",
            "builtins",
        }
    ),
)


def is_stdlib_module(name: str) -> bool:
    """Returns True if the given module name is part of Python's standard library."""
    clean = name.strip().lower()
    root_pkg = clean.split(".")[0]
    return root_pkg in STDLIB_MODULE_NAMES


def filter_python_dependencies(packages: list[str]) -> list[str]:
    """Filters out standard library modules from a list of package names / specifiers."""
    valid: list[str] = []
    for pkg in packages:
        clean = pkg.strip()
        if not clean or clean.startswith(("#", "-")):
            continue
        pkg_name = re.split(r"[=<>!~@]", clean)[0].strip().lower()
        if not is_stdlib_module(pkg_name):
            valid.append(clean)
    return valid


def sanitize_requirements_txt(content: str) -> str:
    """
    Sanitizes requirements.txt content by removing standard library modules
    while preserving comments, options, and valid external dependencies.
    """
    lines = content.splitlines()
    clean_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "-")):
            clean_lines.append(line)
            continue
        # Extract package name before comments or version specifiers
        pkg_part = stripped.split("#")[0].strip()
        pkg_name = re.split(r"[=<>!~@]", pkg_part)[0].strip().lower()
        if not is_stdlib_module(pkg_name):
            clean_lines.append(line)
    return "\n".join(clean_lines)


def extract_external_imports(source_code: str) -> set[str]:
    """
    Parses Python source code AST and returns the set of top-level imported
    packages that are external (non-stdlib).
    """
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return set()

    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                imported_roots.add(root)
        elif isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".")[0]
            imported_roots.add(root)

    return {pkg for pkg in imported_roots if not is_stdlib_module(pkg)}



class EdgeConfidence(str, Enum):
    """How certain a dependency edge is."""

    EXACT = "exact"
    INFERRED = "inferred"


class DependencyEdge(BaseModel):
    """One dependency relationship between two source locations."""

    source: str  # file (relative) or qualified symbol
    target: str  # file (relative) or qualified symbol
    kind: str = "import"  # "import" | "references" | "inherits" | "calls"
    confidence: EdgeConfidence = EdgeConfidence.INFERRED
    detail: str = ""

    def to_prompt_line(self) -> str:
        return f"{self.source} -> {self.target} [{self.kind}/{self.confidence.value}]"


class DependencyGraph:
    """
    Dependency relationships derived from a :class:`RepositoryIndex`.

    Imports are EXACT; cross-file symbol references are INFERRED. The graph
    never fabricates certainty.
    """

    def __init__(self, index: RepositoryIndex) -> None:
        self.index = index

    # ------------------------------------------------------------------
    # Exact edges (from parsed imports)
    # ------------------------------------------------------------------

    def imports(self, rel_path: str) -> list[DependencyEdge]:
        """EXACT import edges of one file."""
        edges = self.index.get_imports(rel_path)
        return [
            DependencyEdge(
                source=rel_path,
                target=edge.target,
                kind="import",
                confidence=EdgeConfidence.EXACT,
                detail=edge.to_prompt_line(),
            )
            for edge in edges
        ]

    def dependents(self, rel_path: str) -> list[DependencyEdge]:
        """EXACT reverse edges: files that import *rel_path*."""
        files = self.index.get_dependents(rel_path)
        return [
            DependencyEdge(
                source=other,
                target=rel_path,
                kind="import",
                confidence=EdgeConfidence.EXACT,
            )
            for other in files
        ]

    # ------------------------------------------------------------------
    # Inferred edges (from reference occurrences)
    # ------------------------------------------------------------------

    def references_to(self, symbol_name: str) -> list[DependencyEdge]:
        """INFERRED: files/locations that reference *symbol_name*."""
        refs = self.index.find_references(symbol_name)
        return [
            DependencyEdge(
                source=ref.location.file,
                target=symbol_name,
                kind="references",
                confidence=EdgeConfidence.INFERRED,
                detail=f"{ref.location.line}: {ref.context[:120]}",
            )
            for ref in refs
        ]

    def callers_of(self, symbol_name: str) -> list[DependencyEdge]:
        """
        INFERRED callers of a function: reference sites in other files whose
        context actually contains a call-looking pattern (``name(``). Never
        claims exactness.
        """
        refs = self.index.find_references(symbol_name)
        edges: list[DependencyEdge] = []
        for ref in refs:
            if f"{symbol_name}(" not in ref.context:
                continue
            edges.append(
                DependencyEdge(
                    source=ref.location.file,
                    target=symbol_name,
                    kind="calls",
                    confidence=EdgeConfidence.INFERRED,
                    detail=f"{ref.location.line}: {ref.context[:120]}",
                )
            )
        return edges

    def inheritors_of(self, symbol_name: str) -> list[DependencyEdge]:
        """
        INFERRED subclasses: symbols whose parsed ``bases`` list contains
        *symbol_name* (exact base names from the parser, but the resolution
        of "who implements what" is only as good as the parser's base data).
        """
        edges: list[DependencyEdge] = []
        for symbol in self.index.find_symbol(symbol_name):
            if symbol_name in symbol.bases:
                edges.append(
                    DependencyEdge(
                        source=symbol.qualified_name,
                        target=symbol_name,
                        kind="inherits",
                        confidence=EdgeConfidence.INFERRED,
                        detail=symbol.location.to_prompt_line(),
                    )
                )
        # Walk every symbol with a matching base via a full scan (bounded).
        for name in self.index.all_symbol_names():
            for symbol in self.index.find_symbol(name):
                if symbol.bases and symbol_name in symbol.bases:
                    edges.append(
                        DependencyEdge(
                            source=symbol.qualified_name,
                            target=symbol_name,
                            kind="inherits",
                            confidence=EdgeConfidence.INFERRED,
                            detail=symbol.location.to_prompt_line(),
                        )
                    )
        # Deduplicate by (source, target, kind).
        seen: set[tuple[str, str, str]] = set()
        unique: list[DependencyEdge] = []
        for edge in edges:
            key = (edge.source, edge.target, edge.kind)
            if key not in seen:
                seen.add(key)
                unique.append(edge)
        return unique
