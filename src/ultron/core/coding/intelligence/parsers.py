"""ultron.core.coding.intelligence.parsers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Modular source parsing for the code intelligence layer (Fix #4).

The architecture is deliberately parser-agnostic: a :class:`SourceParser`
protocol (``parse`` -> :class:`ParseResult`) with an extensible registry
(:data:`PARSERS`, :func:`get_parser`). Language-specific implementations
slot in without touching the index or the facade.

Two implementations ship:

- :class:`PythonAstParser` — real AST parsing via the stdlib ``ast`` module:
  exact symbols (class / function / method / variable / constant), imports,
  and base-class relationships. Nothing here is regex-based.
- :class:`RegexParser` — heuristic extraction for languages without a
  bundled parser (javascript, typescript, go, rust, java, ...). Symbols are
  marked ``inferred=True`` and must never be presented as exact facts.

Malformed sources never raise: ``parse_source`` catches per-file errors and
returns a partial ``ParseResult`` (best-effort, deterministic).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Protocol

from ultron.core.coding.intelligence.symbols import (
    ImportEdge,
    ParseResult,
    Symbol,
    SymbolKind,
    SymbolLocation,
    SymbolRelationship,
)

# Source-extension -> language map (shared with the indexer). Files whose
# extension is unknown are not parsed for symbols (still searchable).
EXTENSION_LANGUAGES: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".rb": "ruby",
    ".php": "php",
    ".cs": "csharp",
    ".swift": "swift",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".hpp": "cpp",
    ".sh": "shell",
}


def language_for_path(path: str) -> str:
    """Language for a source path via its extension ('' when unknown)."""
    return EXTENSION_LANGUAGES.get(Path(path).suffix.lower(), "")


class SourceParser(Protocol):
    """The parser contract: one source file -> structured symbols/imports."""

    language: str

    def parse(self, source: str, file_path: str) -> ParseResult: ...


class PythonAstParser:
    """Exact symbol extraction for Python using the stdlib ``ast`` module."""

    language = "python"

    def parse(self, source: str, file_path: str) -> ParseResult:
        tree = ast.parse(source)  # raises SyntaxError on malformed input
        visitor = _PythonVisitor(file_path, source)
        visitor.visit(tree)
        return ParseResult(
            symbols=visitor.symbols,
            imports=visitor.imports,
            relationships=visitor.relationships,
        )


class _PythonVisitor(ast.NodeVisitor):
    """Walks a Python AST collecting symbols, imports and relationships.

    Scopes: a class name becomes the ``parent``/``scope`` of its methods;
    module-level functions are top-level. Bases of classes become
    ``inherits`` relationships (kind='inherits').
    """

    def __init__(self, file_path: str, source: str) -> None:
        self.file_path = file_path
        self.source = source
        self.symbols: list[Symbol] = []
        self.imports: list[ImportEdge] = []
        self.relationships: list[SymbolRelationship] = []
        self._scopes: list[str] = []

    # -- helpers ----------------------------------------------------------

    def _loc(self, node: ast.AST) -> SymbolLocation:
        return SymbolLocation(
            file=self.file_path,
            line=getattr(node, "lineno", 1),
            column=getattr(node, "col_offset", 0),
            end_line=getattr(node, "end_lineno", None),
            end_column=getattr(node, "end_col_offset", None),
        )

    def _signature(self, node: ast.AST) -> str:
        """Reconstructs the definition line(s) from the source, bounded."""
        try:
            text = ast.get_source_segment(self.source, node)
        except (TypeError, ValueError):
            text = None
        if not text:
            return ""
        first = text.splitlines()[0].strip()
        return (first + " ...") if "\n" in text else first

    def _doc(self, node: ast.AST) -> str:
        body = getattr(node, "body", [])
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
            return body[0].value.value.splitlines()[0][:200] if body[0].value.value else ""
        return ""

    # -- visitors ---------------------------------------------------------

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        base_names: list[str] = []
        for base in node.bases:
            name = _name_of(base)
            if name:
                base_names.append(name)
                self.relationships.append(
                    SymbolRelationship(
                        source_symbol=node.name,
                        source_kind=SymbolKind.CLASS,
                        target_symbol=name,
                        target_kind=SymbolKind.CLASS,
                        kind="inherits",
                        inferred=False,
                    )
                )
        self.symbols.append(
            Symbol(
                name=node.name,
                kind=SymbolKind.CLASS,
                language="python",
                location=self._loc(node),
                signature=self._signature(node),
                doc=self._doc(node),
                bases=base_names,
                inferred=False,
            )
        )
        self._scopes.append(node.name)
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.visit_FunctionDef(item)
            elif isinstance(item, ast.ClassDef):
                self.visit_ClassDef(item)
        self._scopes.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node, is_async=False)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node, is_async=True)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef, *, is_async: bool) -> None:
        parent = self._scopes[-1] if self._scopes else None
        kind = SymbolKind.METHOD if parent else SymbolKind.FUNCTION
        sig = self._signature(node)
        if is_async and sig and not sig.startswith("async "):
            sig = f"async {sig}"
        self.symbols.append(
            Symbol(
                name=node.name,
                kind=kind,
                language="python",
                location=self._loc(node),
                scope=parent or "",
                parent=parent,
                signature=sig,
                doc=self._doc(node),
                inferred=False,
            )
        )

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id.isidentifier():
                self._add_variable(target.id, node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if isinstance(node.target, ast.Name) and node.target.id.isidentifier():
            self._add_variable(node.target.id, node)

    def _add_variable(self, name: str, node: ast.AST) -> None:
        kind = SymbolKind.CONSTANT if name.isupper() else SymbolKind.VARIABLE
        self.symbols.append(
            Symbol(
                name=name,
                kind=kind,
                language="python",
                location=self._loc(node),
                signature=self._signature(node),
                inferred=False,
            )
        )

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports.append(
                ImportEdge(
                    source=self.file_path,
                    imported=alias.name,
                    alias=alias.asname,
                    language="python",
                )
            )
            self.symbols.append(
                Symbol(
                    name=alias.asname or alias.name,
                    kind=SymbolKind.IMPORT,
                    language="python",
                    location=self._loc(node),
                    signature=f"import {alias.name}",
                    inferred=False,
                )
            )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        for alias in node.names:
            if alias.name == "*":
                continue
            self.imports.append(
                ImportEdge(
                    source=self.file_path,
                    imported=module,
                    member=alias.name,
                    alias=alias.asname,
                    is_relative=bool(node.level),
                    language="python",
                )
            )
            self.symbols.append(
                Symbol(
                    name=alias.asname or alias.name,
                    kind=SymbolKind.IMPORT,
                    language="python",
                    location=self._loc(node),
                    signature=f"from {'.' * node.level}{module} import {alias.name}",
                    inferred=False,
                )
            )


def _name_of(node: ast.AST) -> str | None:
    """Best-effort dotted name of an AST expression (base classes, callables)."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _name_of(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    if isinstance(node, ast.Subscript):
        return _name_of(node.value)
    return None


# ---------------------------------------------------------------------------
# Heuristic (regex) parsers for languages without a bundled parser
# ---------------------------------------------------------------------------

# Per-language definition patterns: (regex, kind). Order matters —
# more specific patterns must come first within a language. Patterns are
# plain strings; the RegexParser compiles them once per parse.
_REGEX_DEFINITIONS: dict[str, list[tuple[str, SymbolKind]]] = {
    "javascript": [
        (r"\bclass\s+([A-Za-z_$][\w$]*)", SymbolKind.CLASS),
        (r"\b(?:async\s+)?function\s+\*?\s*([A-Za-z_$][\w$]*)", SymbolKind.FUNCTION),
        (r"\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s+)?(?:function|\([^)]*\)\s*=>)", SymbolKind.FUNCTION),
        (r"\binterface\s+([A-Za-z_$][\w$]*)", SymbolKind.INTERFACE),
        (r"\b(?:const|let|var)\s+([A-Z][A-Z0-9_]*)\s*=", SymbolKind.CONSTANT),
        (r"\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=", SymbolKind.VARIABLE),
    ],
    "typescript": [
        (r"\b(?:abstract\s+)?class\s+([A-Za-z_$][\w$]*)", SymbolKind.CLASS),
        (r"\binterface\s+([A-Za-z_$][\w$]*)", SymbolKind.INTERFACE),
        (r"\btype\s+([A-Za-z_$][\w$]*)\s*=", SymbolKind.TYPE_ALIAS),
        (r"\benum\s+([A-Za-z_$][\w$]*)", SymbolKind.ENUM),
        (r"\b(?:async\s+)?function\s+\*?\s*([A-Za-z_$][\w$]*)", SymbolKind.FUNCTION),
        (r"\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s+)?(?:function|\([^)]*\)\s*=>)", SymbolKind.FUNCTION),
        (r"\b(?:const|let|var)\s+([A-Z][A-Z0-9_]*)\s*=", SymbolKind.CONSTANT),
    ],
    "go": [
        (r"\btype\s+([A-Za-z_]\w*)\s+struct\b", SymbolKind.STRUCT),
        (r"\btype\s+([A-Za-z_]\w*)\s+interface\b", SymbolKind.INTERFACE),
        (r"\bfunc\s+\([^)]*\)\s+([A-Za-z_]\w*)\s*\(", SymbolKind.METHOD),
        (r"\bfunc\s+([A-Za-z_]\w*)\s*\(", SymbolKind.FUNCTION),
        (r"\bconst\s*\(\s*([A-Z][A-Z0-9_]*)\b", SymbolKind.CONSTANT),
        (r"\bvar\s+([A-Za-z_]\w*)\s*=", SymbolKind.VARIABLE),
    ],
    "rust": [
        (r"\bstruct\s+([A-Z]\w*)", SymbolKind.STRUCT),
        (r"\benum\s+([A-Z]\w*)", SymbolKind.ENUM),
        (r"\btrait\s+([A-Z]\w*)", SymbolKind.TRAIT),
        (r"\b(?:pub\s+)?fn\s+([a-z_]\w*)\s*\(", SymbolKind.FUNCTION),
        (r"\b(?:pub\s+)?const\s+([A-Z][A-Z0-9_]*)\s*:", SymbolKind.CONSTANT),
    ],
    "java": [
        (r"\b(?:public|protected|private)?\s*(?:abstract\s+|final\s+|static\s+)*class\s+([A-Z]\w*)", SymbolKind.CLASS),
        (r"\binterface\s+([A-Z]\w*)", SymbolKind.INTERFACE),
        (r"\benum\s+([A-Z]\w*)", SymbolKind.ENUM),
        (r"\b(?:public|protected|private)\s+[\w<>\[\],\s.]+?\s+([a-z]\w*)\s*\(", SymbolKind.METHOD),
        (r"\b(?:public|protected|private)\s+static\s+final\s+[\w<>]+?\s+([A-Z][A-Z0-9_]*)\s*=", SymbolKind.CONSTANT),
    ],
    "kotlin": [
        (r"\b(?:data\s+)?class\s+([A-Z]\w*)", SymbolKind.CLASS),
        (r"\binterface\s+([A-Z]\w*)", SymbolKind.INTERFACE),
        (r"\benum\s+class\s+([A-Z]\w*)", SymbolKind.ENUM),
        (r"\bfun\s+([a-z]\w*)\s*\(", SymbolKind.FUNCTION),
        (r"\bval\s+([A-Z][A-Z0-9_]*)\s*=\s*", SymbolKind.CONSTANT),
    ],
    "ruby": [
        (r"\bclass\s+([A-Z]\w*)", SymbolKind.CLASS),
        (r"\bmodule\s+([A-Z]\w*)", SymbolKind.MODULE),
        (r"\bdef\s+self\.([a-z_]\w*)", SymbolKind.METHOD),
        (r"\bdef\s+([a-z_]\w*)", SymbolKind.FUNCTION),
    ],
    "php": [
        (r"\bclass\s+([A-Za-z_]\w*)", SymbolKind.CLASS),
        (r"\binterface\s+([A-Za-z_]\w*)", SymbolKind.INTERFACE),
        (r"\b(?:public|protected|private)?\s*function\s+([A-Za-z_]\w*)\s*\(", SymbolKind.FUNCTION),
    ],
    "csharp": [
        (r"\bclass\s+([A-Z]\w*)", SymbolKind.CLASS),
        (r"\binterface\s+([A-Z]\w*)", SymbolKind.INTERFACE),
        (r"\benum\s+([A-Z]\w*)", SymbolKind.ENUM),
        (r"\b(?:public|protected|private|internal)?\s*(?:static\s+|async\s+|virtual\s+|override\s+)*[\w<>\[\],?]+?\s+([a-z]\w*)\s*\(", SymbolKind.METHOD),
        (r"\bconst\s+[\w<>?]+\s+([A-Z][A-Z0-9_]*)\s*=", SymbolKind.CONSTANT),
    ],
    "swift": [
        (r"\bclass\s+([A-Z]\w*)", SymbolKind.CLASS),
        (r"\bprotocol\s+([A-Z]\w*)", SymbolKind.INTERFACE),
        (r"\bstruct\s+([A-Z]\w*)", SymbolKind.STRUCT),
        (r"\benum\s+([A-Z]\w*)", SymbolKind.ENUM),
        (r"\bfunc\s+([a-z]\w*)\s*\(", SymbolKind.FUNCTION),
    ],
    "shell": [
        (r"^([a-zA-Z_]\w*)\s*\(\)\s*\{", SymbolKind.FUNCTION),
    ],
}

# Per-language import patterns (name, is_relative): match -> capture imported.
_REGEX_IMPORTS: dict[str, list[tuple[re.Pattern[str], bool]]] = {
    "javascript": [
        (re.compile(r"\bimport\s+[\w\s,*{}$-]*?\s+from\s+['\"]([^'\"]+)['\"]"), False),
        (re.compile(r"\bimport\s+['\"]([^'\"]+)['\"]"), False),
        (re.compile(r"\brequire\s*\(\s*['\"]([^'\"]+)['\"]\s*\)"), False),
    ],
    "typescript": [
        (re.compile(r"\bimport\s+[\w\s,*{}$-]*?\s+from\s+['\"]([^'\"]+)['\"]"), False),
        (re.compile(r"\bimport\s+['\"]([^'\"]+)['\"]"), False),
        (re.compile(r"\brequire\s*\(\s*['\"]([^'\"]+)['\"]\s*\)"), False),
    ],
    "go": [
        (re.compile(r'\bimport\s+"([^"]+)"'), False),
        (re.compile(r'\bimport\s+alias\s+"([^"]+)"'), False),
        (re.compile(r'\bimport\s*\(\s*([^)]*)\)'), False),  # block form, expanded below
    ],
    "rust": [
        (re.compile(r"\buse\s+([\w:]+)(?:\s+as\s+\w+)?\s*;"), False),
        (re.compile(r"\buse\s+[\w:]+::\{([^}]*)\}\s*;"), False),
    ],
    "java": [
        (re.compile(r"\bimport\s+(?:static\s+)?([\w.]+);"), False),
    ],
    "kotlin": [
        (re.compile(r"\bimport\s+([\w.]+)"), False),
    ],
    "ruby": [
        (re.compile(r"\brequire\s+['\"]([^'\"]+)['\"]"), False),
        (re.compile(r"\brequire_relative\s+['\"]([^'\"]+)['\"]"), True),
    ],
    "php": [
        (re.compile(r"\buse\s+([\w\\]+)(?:\s+as\s+\w+)?\s*;"), False),
    ],
    "csharp": [
        (re.compile(r"\busing\s+([\w.]+);"), False),
    ],
    "swift": [
        (re.compile(r"\bimport\s+([\w.]+)"), False),
    ],
}


class RegexParser:
    """Heuristic symbol/import extraction for languages without a bundled parser.

    All symbols produced are marked ``inferred=True`` — they are best-effort
    and must never be presented to the user as exact facts.
    """

    def __init__(self, language: str) -> None:
        self.language = language

    def parse(self, source: str, file_path: str) -> ParseResult:
        symbols: list[Symbol] = []
        imports: list[ImportEdge] = []
        relationships: list[SymbolRelationship] = []
        lines = source.splitlines()

        def location_for(match: re.Match[str]) -> SymbolLocation:
            line = source.count("\n", 0, match.start()) + 1
            col = match.start() - (source.rfind("\n", 0, match.start()) + 1)
            return SymbolLocation(file=file_path, line=line, column=col)

        # Higher-priority patterns run first; keep the FIRST symbol for each
        # (name, line) so e.g. a CONSTANT is never also reported as a generic
        # VARIABLE by a later, broader pattern.
        seen_defs: set[tuple[str, int]] = set()
        for pattern_text, kind in _REGEX_DEFINITIONS.get(self.language, []):
            pattern = re.compile(pattern_text)
            for match in pattern.finditer(source):
                name = match.group(1)
                if not name:
                    continue
                loc = location_for(match)
                key = (name, loc.line)
                if key in seen_defs:
                    continue
                seen_defs.add(key)
                line_text = lines[loc.line - 1].strip() if lines and loc.line <= len(lines) else ""
                symbols.append(
                    Symbol(
                        name=name,
                        kind=kind,
                        language=self.language,
                        location=loc,
                        signature=line_text[:200],
                        inferred=True,
                    )
                )

        for pattern, is_relative in _REGEX_IMPORTS.get(self.language, []):
            for match in re.finditer(pattern, source):
                captured = match.group(1)
                # Block-form imports (`import ( "a" "b" )`) carry several
                # quoted tokens in one capture — expand each token as its own
                # edge and skip the raw block string itself.
                tokens = re.findall(r'["\']([^"\']+)["\']', captured)
                if tokens:
                    for token in tokens:
                        imports.append(
                            ImportEdge(
                                source=file_path,
                                imported=token,
                                is_relative=is_relative,
                                language=self.language,
                            )
                        )
                    continue
                if not captured.strip():
                    continue
                imports.append(
                    ImportEdge(
                        source=file_path,
                        imported=captured.strip(),
                        is_relative=is_relative,
                        language=self.language,
                    )
                )

        # Deduplicate import edges (name, member) while preserving order.
        seen: set[tuple[str, str, str]] = set()
        unique: list[ImportEdge] = []
        for edge in imports:
            key = (edge.source, edge.imported, edge.member)
            if key not in seen:
                seen.add(key)
                unique.append(edge)

        return ParseResult(
            symbols=symbols,
            imports=unique,
            relationships=relationships,
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Extensible registry: language -> parser instance. Add a new language by
# appending a parser here (either a RegexParser entry or a bespoke class).
PARSERS: dict[str, SourceParser] = {
    "python": PythonAstParser(),
    **{lang: RegexParser(lang) for lang in _REGEX_DEFINITIONS},
}


def get_parser(language: str) -> SourceParser | None:
    """Returns the parser for a language, or None when unsupported."""
    return PARSERS.get(language or "")


def parse_source(source: str, file_path: str) -> ParseResult:
    """
    Parses source text for a file, dispatching on its extension's language.

    Never raises: malformed sources return a partial ``ParseResult`` (the
    Python AST parser may raise SyntaxError, which is caught and reported as
    an empty result). Unsupported languages return an empty result.
    """
    language = language_for_path(file_path)
    parser = get_parser(language)
    if parser is None:
        return ParseResult(file_path=file_path, language=language)
    try:
        result = parser.parse(source, file_path)
    except (SyntaxError, ValueError, TypeError, RecursionError):
        result = ParseResult()
    result.file_path = file_path
    result.language = language
    return result
