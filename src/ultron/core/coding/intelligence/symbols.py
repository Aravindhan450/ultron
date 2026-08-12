"""ultron.core.coding.intelligence.symbols
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Symbol representation for the code intelligence layer (Fix #4).

A :class:`Symbol` describes one named entity in a source file — a module,
class, function, method, interface, variable, constant or import — with its
location, scope, parent, signature and optional docstring. Locations are
1-based line / 0-based column (the Python ``ast`` convention), stored per
file with paths RELATIVE to the repository root.

Symbols are plain pydantic models so they serialize into the repository
index and into prompts without any parser coupling. ``inferred`` marks
symbols recovered by heuristic (regex) parsers — AST-derived symbols are
exact, heuristic ones are best-effort and must not be presented as facts.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class SymbolKind(str, Enum):
    """Kinds of named entities a parser can extract."""

    MODULE = "module"
    CLASS = "class"
    INTERFACE = "interface"
    FUNCTION = "function"
    METHOD = "method"
    VARIABLE = "variable"
    CONSTANT = "constant"
    IMPORT = "import"
    TYPE_ALIAS = "type_alias"
    STRUCT = "struct"
    TRAIT = "trait"
    ENUM = "enum"
    UNKNOWN = "unknown"


class SymbolLocation(BaseModel):
    """A location inside one source file (relative to the repository root)."""

    file: str
    line: int = 1
    column: int = 0
    end_line: int | None = None
    end_column: int | None = None

    def to_prompt_line(self) -> str:
        base = f"{self.file}:{self.line}"
        if self.end_line and self.end_line != self.line:
            base += f"-{self.end_line}"
        return base


class Symbol(BaseModel):
    """One named entity extracted from a source file."""

    name: str
    kind: SymbolKind = SymbolKind.UNKNOWN
    language: str = ""
    location: SymbolLocation
    scope: str = ""  # enclosing scope name (e.g. the class for a method)
    parent: str | None = None  # enclosing symbol name (e.g. the class)
    signature: str = ""  # reconstructed definition line(s), bounded
    doc: str = ""  # first docstring / leading comment line, bounded
    bases: list[str] = Field(default_factory=list)  # base classes / interfaces
    inferred: bool = False  # True when recovered by a heuristic (regex) parser

    @property
    def qualified_name(self) -> str:
        """Dotted name: parent.name when nested (e.g. ``Service.login``)."""
        if self.parent:
            return f"{self.parent}.{self.name}"
        return self.name

    def to_prompt_line(self, max_len: int = 200) -> str:
        head = f"{self.qualified_name} ({self.kind.value}) {self.location.to_prompt_line()}"
        if self.signature:
            return f"{head}: {self.signature[: max_len - len(head) - 2]}"
        return head


class SymbolReference(BaseModel):
    """One usage of a symbol outside its definition site."""

    name: str
    location: SymbolLocation
    context: str = ""  # the source line, bounded

    def to_prompt_line(self) -> str:
        head = f"{self.name} @ {self.location.to_prompt_line()}"
        if self.context:
            return f"{head}: {self.context[:150]}"
        return head


class ImportEdge(BaseModel):
    """One import statement found in a source file."""

    source: str  # importing file, relative to the repository root
    imported: str  # module / package path
    member: str = ""  # for ``from x import y`` — the imported member
    alias: str | None = None  # ``import x as y`` / ``from x import y as z``
    is_relative: bool = False  # ``from .x import ...``
    language: str = ""

    @property
    def target(self) -> str:
        """The dotted name that resolves to the imported symbol/module."""
        return f"{self.imported}.{self.member}" if self.member else self.imported

    def to_prompt_line(self) -> str:
        return f"{self.source} -> {self.target}"


class SymbolRelationship(BaseModel):
    """A structural relationship between two symbols (kind-qualified)."""

    source_symbol: str  # qualified name of the source
    source_kind: SymbolKind = SymbolKind.UNKNOWN
    target_symbol: str  # qualified name of the target
    target_kind: SymbolKind = SymbolKind.UNKNOWN
    kind: str = "references"  # e.g. "references", "calls", "inherits", "implements"
    inferred: bool = True  # structural relationships are heuristic unless LSP


class ParseResult(BaseModel):
    """The structured output of one parsed source file."""

    file_path: str = ""
    language: str = ""
    symbols: list[Symbol] = Field(default_factory=list)
    imports: list[ImportEdge] = Field(default_factory=list)
    relationships: list[SymbolRelationship] = Field(default_factory=list)

    def symbol_count(self) -> int:
        return len(self.symbols)

    def import_count(self) -> int:
        return len(self.imports)
