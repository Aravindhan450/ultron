"""
ultron.core.repository.parser
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Source code parsing for repository intelligence.
Extracts symbols, definitions, signatures, docstrings, and import relationships
across supported languages with robust AST parsing and graceful regex fallbacks.
"""

from __future__ import annotations

import ast
from pathlib import Path

from pydantic import BaseModel, Field

from ultron.core.coding.intelligence.parsers import (
    language_for_path,
)
from ultron.core.coding.intelligence.parsers import (
    parse_source as _intelligence_parse_source,
)


class ParsedSymbol(BaseModel):
    """A symbol extracted from a source file."""

    name: str
    kind: str  # "class", "function", "method", "variable", "constant", "interface", etc.
    file_path: str  # relative path
    line: int = 1
    end_line: int | None = None
    column: int = 0
    signature: str = ""
    doc: str = ""
    parent: str | None = None
    scope: str = ""
    bases: list[str] = Field(default_factory=list)
    inferred: bool = False

    @property
    def qualified_name(self) -> str:
        if self.parent:
            return f"{self.parent}.{self.name}"
        return self.name

    def to_summary_line(self) -> str:
        sig = f": {self.signature}" if self.signature else ""
        return f"{self.kind} {self.qualified_name} (line {self.line}){sig}"


class ParsedImport(BaseModel):
    """An import statement extracted from a source file."""

    source_file: str
    imported: str
    member: str = ""
    alias: str | None = None
    is_relative: bool = False

    def to_summary_line(self) -> str:
        if self.member:
            return f"from {self.imported} import {self.member}"
        return f"import {self.imported}"


class FileParseResult(BaseModel):
    """Result of parsing a single source file."""

    file_path: str
    language: str = ""
    symbols: list[ParsedSymbol] = Field(default_factory=list)
    imports: list[ParsedImport] = Field(default_factory=list)
    error: str | None = None


class _PythonASTCollector(ast.NodeVisitor):
    """Collects symbols and imports from Python AST."""

    def __init__(self, rel_path: str, source_lines: list[str]) -> None:
        self.rel_path = rel_path
        self.source_lines = source_lines
        self.symbols: list[ParsedSymbol] = []
        self.imports: list[ParsedImport] = []
        self._scope_stack: list[str] = []

    def _get_docstring(self, node: ast.AST) -> str:
        doc = ast.get_docstring(node)
        if doc:
            first_line = doc.strip().split("\n")[0].strip()
            return first_line[:160]
        return ""

    def _get_signature(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
        try:
            line_idx = node.lineno - 1
            if 0 <= line_idx < len(self.source_lines):
                line = self.source_lines[line_idx].strip()
                line = line.removesuffix(":")
                return line[:160]
        except Exception:  # noqa: BLE001, S110
            pass
        return f"def {node.name}(...)"

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        bases = []
        for b in node.bases:
            if isinstance(b, ast.Name):
                bases.append(b.id)
            elif isinstance(b, ast.Attribute):
                bases.append(f"{getattr(b.value, 'id', '')}.{b.attr}")

        parent = self._scope_stack[-1] if self._scope_stack else None
        sym = ParsedSymbol(
            name=node.name,
            kind="class",
            file_path=self.rel_path,
            line=node.lineno,
            end_line=getattr(node, "end_lineno", node.lineno),
            column=node.col_offset,
            doc=self._get_docstring(node),
            parent=parent,
            scope=".".join(self._scope_stack),
            bases=bases,
            inferred=False,
        )
        self.symbols.append(sym)

        self._scope_stack.append(node.name)
        self.generic_visit(node)
        self._scope_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._handle_function(node, is_async=False)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._handle_function(node, is_async=True)

    def _handle_function(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef, is_async: bool
    ) -> None:
        parent = self._scope_stack[-1] if self._scope_stack else None
        kind = "method" if parent else "function"
        sym = ParsedSymbol(
            name=node.name,
            kind=kind,
            file_path=self.rel_path,
            line=node.lineno,
            end_line=getattr(node, "end_lineno", node.lineno),
            column=node.col_offset,
            signature=self._get_signature(node),
            doc=self._get_docstring(node),
            parent=parent,
            scope=".".join(self._scope_stack),
            inferred=False,
        )
        self.symbols.append(sym)

        # Do not recurse into nested function bodies to avoid symbol explosion
        if not parent:
            self._scope_stack.append(node.name)
            self.generic_visit(node)
            self._scope_stack.pop()

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports.append(
                ParsedImport(
                    source_file=self.rel_path,
                    imported=alias.name,
                    member="",
                    alias=alias.asname,
                    is_relative=False,
                )
            )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        mod = node.module or ""
        is_rel = node.level > 0
        for alias in node.names:
            self.imports.append(
                ParsedImport(
                    source_file=self.rel_path,
                    imported=mod,
                    member=alias.name,
                    alias=alias.asname,
                    is_relative=is_rel,
                )
            )


def parse_source_file(file_path: str | Path, rel_path: str, content: str) -> FileParseResult:
    """
    Parses source code into structured symbols and imports.
    Never throws an exception; gracefully captures errors.
    """
    path_obj = Path(file_path)
    lang = language_for_path(str(path_obj))

    if not content.strip():
        return FileParseResult(file_path=rel_path, language=lang)

    # 1. Exact AST parse for Python
    if lang == "python":
        try:
            tree = ast.parse(content, filename=str(file_path))
            collector = _PythonASTCollector(rel_path, content.splitlines())
            collector.visit(tree)
            return FileParseResult(
                file_path=rel_path,
                language=lang,
                symbols=collector.symbols,
                imports=collector.imports,
            )
        except SyntaxError as e:
            return FileParseResult(
                file_path=rel_path,
                language=lang,
                error=f"SyntaxError: {e.msg} (line {e.lineno})",
            )
        except Exception as e:  # noqa: BLE001
            return FileParseResult(
                file_path=rel_path,
                language=lang,
                error=f"ParseError: {e}",
            )

    # 2. Heuristic fallback for other languages (JS/TS, Go, Rust, etc.)
    try:
        raw_result = _intelligence_parse_source(content, rel_path)
        symbols: list[ParsedSymbol] = []
        for s in raw_result.symbols:
            symbols.append(
                ParsedSymbol(
                    name=s.name,
                    kind=s.kind.value,
                    file_path=rel_path,
                    line=s.location.line,
                    end_line=s.location.end_line,
                    column=s.location.column,
                    signature=s.signature,
                    doc=s.doc,
                    parent=s.parent,
                    scope=s.scope,
                    bases=s.bases,
                    inferred=True,
                )
            )

        imports: list[ParsedImport] = []
        for imp in raw_result.imports:
            imports.append(
                ParsedImport(
                    source_file=rel_path,
                    imported=imp.imported,
                    member=imp.member,
                    alias=imp.alias,
                    is_relative=imp.is_relative,
                )
            )

        return FileParseResult(
            file_path=rel_path,
            language=lang,
            symbols=symbols,
            imports=imports,
        )
    except Exception as e:  # noqa: BLE001
        return FileParseResult(
            file_path=rel_path,
            language=lang,
            error=f"FallbackParseError: {e}",
        )
