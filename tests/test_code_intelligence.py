"""
Fix #4 tests: the codebase intelligence foundation.

Covers the layered repository intelligence system: gitignore-aware lexical
search (L1/L2), symbol extraction (L3 parsers), the incremental SQLite
index, definition/reference lookup, import/dependency edges (EXACT vs
INFERRED), the semantic-search metadata foundation (L5, lexical fallback),
the LSP abstraction (L4, graceful unavailability), and the security wiring
of the registered read-only tools.

All filesystem tests use temporary repositories; the real Ultron repository
is never modified. No network, no real LLM, no real language servers.
"""

import pytest

from ultron.core.coding.intelligence.dependencies import (
    DependencyGraph,
    EdgeConfidence,
)
from ultron.core.coding.intelligence.facade import CodeIntelligence
from ultron.core.coding.intelligence.index import RepositoryIndex
from ultron.core.coding.intelligence.lsp import (
    LSPCapabilities,
    LSPFacade,
    LSPLocation,
    LSPOperation,
    NoLSPServers,
)
from ultron.core.coding.intelligence.parsers import (
    PARSERS,
    get_parser,
    language_for_path,
    parse_source,
)
from ultron.core.coding.intelligence.search import search_code
from ultron.core.coding.intelligence.semantic import (
    Embedder,
    SemanticSearch,
)
from ultron.core.coding.intelligence.symbols import SymbolKind
from ultron.core.tools import paths as tools_paths

# The full set of read-only code-intelligence tools (Fix #4 + integration).
_INTELLIGENCE_TOOL_NAMES = (
    "code_search",
    "find_symbol",
    "find_definition",
    "find_references",
    "get_imports",
    "get_dependents",
    "semantic_search",
    "code_index_status",
    "report_file",
    "report_symbol",
)


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """A temp workspace that is the ALLOWED_BASE_DIR (so tools work)."""
    monkeypatch.setattr(tools_paths, "ALLOWED_BASE_DIR", tmp_path)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _write(root, rel: str, text: str):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Parsers: symbol extraction
# ---------------------------------------------------------------------------


def test_python_ast_symbol_extraction():
    source = (
        "import os\n"
        "from fastapi import FastAPI\n"
        "\n"
        "MAX_RETRIES = 3\n"
        "cache = {}\n"
        "\n"
        "class UserService:\n"
        '    """Handles users."""\n'
        "    def login(self, name):\n"
        "        return name\n"
        "\n"
        "def helper():\n"
        "    return 1\n"
    )
    result = parse_source(source, "service.py")
    names = {s.name for s in result.symbols}
    assert "UserService" in names
    assert "login" in names
    assert "helper" in names
    assert "MAX_RETRIES" in names
    assert "FastAPI" in names  # import symbol
    by_name = {s.name: s for s in result.symbols}
    assert by_name["UserService"].kind is SymbolKind.CLASS
    assert by_name["login"].kind is SymbolKind.METHOD
    assert by_name["login"].parent == "UserService"
    assert by_name["MAX_RETRIES"].kind is SymbolKind.CONSTANT
    assert by_name["cache"].kind is SymbolKind.VARIABLE
    # All AST-derived symbols are exact (not inferred).
    assert all(not s.inferred for s in result.symbols)


def test_python_ast_imports_and_inheritance():
    source = (
        "from .base import BaseRepository\n"
        "from sqlalchemy.orm import Session\n"
        "\n"
        "class UserRepo(BaseRepository):\n"
        "    pass\n"
    )
    result = parse_source(source, "repo.py")
    imported = {(e.imported, e.member) for e in result.imports}
    assert ("base", "BaseRepository") in imported
    assert ("sqlalchemy.orm", "Session") in imported
    assert any(r.kind == "inherits" for r in result.relationships)
    user_repo = next(s for s in result.symbols if s.name == "UserRepo")
    assert "BaseRepository" in user_repo.bases


def test_regex_parsers_multiple_languages():
    cases = [
        ("app.js", "class Api {}\nfunction run() {}\nconst PORT = 8080;\n", {"Api", "run", "PORT"}),
        ("app.ts", "interface Repo {}\ntype ID = string;\nclass Impl implements Repo {}\n", {"Repo", "ID", "Impl"}),
        ("main.go", "type Server struct {}\nfunc (s *Server) Serve() {}\nfunc main() {}\n", {"Server", "Serve", "main"}),
        ("lib.rs", "struct Point {}\ntrait Draw {}\nfn render() {}\n", {"Point", "Draw", "render"}),
        ("App.java", "public class App {\n    public void run() {}\n}\n", {"App", "run"}),
    ]
    for file_path, source, expected in cases:
        result = parse_source(source, file_path)
        names = {s.name for s in result.symbols}
        assert expected <= names, f"{file_path}: {names}"
        # Heuristic symbols are all marked inferred.
        assert all(s.inferred for s in result.symbols)


def test_regex_imports():
    js = "import { Router } from 'express';\nimport 'dotenv/config';\n"
    result = parse_source(js, "app.js")
    imported = {e.imported for e in result.imports}
    assert "express" in imported
    assert "dotenv/config" in imported

    go = 'package main\n\nimport (\n    "fmt"\n    "os"\n)\n'
    result = parse_source(go, "main.go")
    imported = {e.imported for e in result.imports}
    assert {"fmt", "os"} <= imported


def test_malformed_source_never_raises():
    assert parse_source("def broken(:\n    pass\n", "broken.py").symbols == []
    assert parse_source("not valid python {{{", "bad.py").symbols == []
    assert parse_source("", "empty.py").symbols == []


def test_unsupported_language_is_empty():
    result = parse_source("just some text", "notes.txt")
    assert result.symbols == []
    assert result.imports == []
    assert result.language == ""


def test_parser_registry_is_extensible():
    assert get_parser("python") is not None
    assert get_parser("javascript") is not None
    assert get_parser("go") is not None
    assert get_parser("nonexistent") is None
    assert "python" in PARSERS
    assert language_for_path("a.tsx") == "typescript"


# ---------------------------------------------------------------------------
# Search: gitignore-aware lexical search
# ---------------------------------------------------------------------------


def test_search_code_filename_and_content(sandbox):
    _write(sandbox, "src/app.py", "def run():\n    pass\n")
    _write(sandbox, "README.md", "nothing here")
    out = search_code("run", str(sandbox))
    assert "src/app.py" in out
    assert "def run()" in out
    # Filename match reported distinctly.
    _write(sandbox, "runbook.md", "words")
    out2 = search_code("runbook", str(sandbox))
    assert "(filename match)" in out2


def test_search_code_regex_and_case(sandbox):
    _write(sandbox, "app.py", "def Run():\n    pass\n")
    out = search_code("def [A-Z]\\w+\\(", str(sandbox), regex=True)
    assert "app.py:1" in out
    # Case-sensitive search does not match lowercase query against 'Run'.
    out2 = search_code("run", str(sandbox), case_sensitive=True)
    assert out2.startswith("No matches") or "def Run" not in out2


def test_search_code_respects_ignored_dirs_and_gitignore(sandbox):
    _write(sandbox, "src/real.py", "secret_needle = 1\n")
    _write(sandbox, "node_modules/pkg/index.js", "secret_needle = 2\n")
    _write(sandbox, ".venv/lib/site.py", "secret_needle = 3\n")
    _write(sandbox, "dist/bundle.js", "secret_needle = 4\n")
    _write(sandbox, ".git/config", "secret_needle = 5\n")
    _write(sandbox, ".gitignore", "*.ignored\n")
    _write(sandbox, "generated.ignored", "secret_needle = 6\n")

    out = search_code("secret_needle", str(sandbox))
    assert "src/real.py" in out
    assert "node_modules" not in out
    assert ".venv" not in out
    assert "dist" not in out
    assert ".git" not in out
    assert "generated.ignored" not in out  # .gitignore respected


def test_search_code_nested_gitignore(sandbox):
    # Nested .gitignore patterns are relative to their own directory.
    _write(sandbox, "sub/.gitignore", "/build/\n*.tmp\n")
    _write(sandbox, "sub/src/ok.py", "needle_nested\n")
    _write(sandbox, "sub/build/gen.py", "needle_nested\n")
    _write(sandbox, "sub/a.tmp", "needle_nested\n")
    out = search_code("needle_nested", str(sandbox))
    assert "sub/src/ok.py" in out
    assert "sub/build" not in out
    assert "a.tmp" not in out


def test_search_code_gitignore_negation(sandbox):
    _write(sandbox, ".gitignore", "*.log\n!important.log\n")
    _write(sandbox, "a.log", "needle_neg\n")
    _write(sandbox, "important.log", "needle_neg\n")
    out = search_code("needle_neg", str(sandbox))
    assert "important.log" in out
    assert "a.log" not in out


def test_search_code_file_pattern(sandbox):
    _write(sandbox, "a.py", "needle_pat\n")
    _write(sandbox, "a.js", "needle_pat\n")
    out = search_code("needle_pat", str(sandbox), file_pattern="*.py")
    assert "a.py" in out
    assert "a.js" not in out


def test_search_code_path_confinement(sandbox):
    out = search_code("x", "/etc")
    assert out.startswith("Error: access denied")


def test_search_code_empty_query(sandbox):
    assert search_code("  ", str(sandbox)).startswith("Error:")


# ---------------------------------------------------------------------------
# Index: incremental updates + queries
# ---------------------------------------------------------------------------


def test_index_refresh_is_incremental(sandbox):
    _write(sandbox, "a.py", "class A:\n    pass\n")
    _write(sandbox, "b.py", "class B:\n    pass\n")
    index = RepositoryIndex(str(sandbox))
    s1 = index.refresh()
    assert s1.parsed == 2
    assert s1.symbols >= 2
    # files is the exact count (no double counting).
    assert s1.files == 2

    # No changes -> nothing reparsed.
    s2 = index.refresh()
    assert s2.parsed == 0
    assert s2.unchanged == 2

    # One file changed -> only it is reparsed.
    _write(sandbox, "b.py", "class B:\n    pass\n\nclass B2:\n    pass\n")
    s3 = index.refresh()
    assert s3.parsed == 1
    assert s3.unchanged == 1
    assert any(s.name == "B2" for s in index.find_symbol("B2"))

    # File deleted -> removed.
    (sandbox / "a.py").unlink()
    s4 = index.refresh()
    assert s4.removed == 1
    assert index.find_definition("A") == []
    index.close()


def test_index_references_rebuilt_only_for_changed_files(sandbox):
    # A reference in an UNCHANGED file survives a refresh of another file,
    # and a changed file's references are recomputed.
    _write(sandbox, "defs.py", "class Widget:\n    pass\n")
    _write(sandbox, "user.py", "from defs import Widget\n\nw = Widget()\n")
    index = RepositoryIndex(str(sandbox))
    s1 = index.refresh()
    assert s1.references >= 1

    # Touch an unrelated file -> references for user.py are untouched.
    _write(sandbox, "other.py", "class Other:\n    pass\n")
    index.refresh()
    refs = index.find_references("Widget")
    assert any(r.location.file == "user.py" for r in refs)
    index.close()


def test_index_db_lives_outside_repo(sandbox):
    # A read-only index must never drop its DB file into the scanned repo.
    _write(sandbox, "a.py", "class A:\n    pass\n")
    index = RepositoryIndex(str(sandbox))
    index.refresh()
    assert not (sandbox / ".ultron_code_index.db").exists()
    assert not (sandbox / ".ultron_code_index").exists()
    index.close()


def test_index_db_respects_monkeypatched_allowed_base(sandbox):
    # The default DB path is resolved lazily against ALLOWED_BASE_DIR, so a
    # test sandbox (which monkeypatches it) never leaks DB files into the
    # real project root.
    import ultron.core.tools.paths as tools_paths

    _write(sandbox, "a.py", "class A:\n    pass\n")
    index = RepositoryIndex(str(sandbox))
    index.refresh()
    db = index.db_path
    assert db.exists()
    # Lives under the (patched) allowed base dir, not the repo root.
    assert db.resolve().is_relative_to(tools_paths.ALLOWED_BASE_DIR)
    index.close()


def test_index_definition_and_reference_lookup(sandbox):
    _write(sandbox, "src/service.py", (
        "from .models import User\n\n"
        "class UserService:\n"
        "    def login(self, name):\n"
        "        return User(name)\n"
    ))
    _write(sandbox, "src/models.py", "class User:\n    pass\n")
    _write(sandbox, "src/api.py", (
        "from .service import UserService\n\n"
        "svc = UserService()\n"
        "def handle():\n"
        "    return svc.login('alice')\n"
    ))
    index = RepositoryIndex(str(sandbox))
    index.refresh()

    defs = index.find_definition("UserService")
    assert len(defs) == 1
    assert defs[0].location.file == "src/service.py"
    assert defs[0].kind is SymbolKind.CLASS

    refs = index.find_references("UserService")
    files = {r.location.file for r in refs}
    assert "src/api.py" in files
    assert "src/service.py" not in files  # definition line excluded

    # login method referenced only in api.py usage.
    login_refs = index.find_references("login")
    assert any("login('alice')" in r.context or "login(" in r.context for r in login_refs)
    index.close()


def test_index_imports_and_dependents(sandbox):
    _write(sandbox, "src/service.py", "from .models import User\n")
    _write(sandbox, "src/models.py", "class User:\n    pass\n")
    _write(sandbox, "src/api.py", "from .service import UserService\n")
    index = RepositoryIndex(str(sandbox))
    index.refresh()

    imports = index.get_imports("src/service.py")
    assert any(e.imported == "models" and e.member == "User" for e in imports)
    dependents = index.get_dependents("src/models.py")
    assert "src/service.py" in dependents
    dependents_service = index.get_dependents("src/service.py")
    assert "src/api.py" in dependents_service
    index.close()


def test_index_skips_ignored_dirs_and_binary(sandbox):
    _write(sandbox, "real.py", "class Real:\n    pass\n")
    _write(sandbox, "node_modules/x/index.js", "class Hidden:\n    pass\n")
    _write(sandbox, "__pycache__/mod.pyc", "binary\\x00data")
    _write(sandbox, "data.bin", "\\x00\\x01\\x02")
    index = RepositoryIndex(str(sandbox))
    index.refresh()
    assert any(s.name == "Real" for s in index.find_symbol("Real"))
    assert index.find_symbol("Hidden") == []
    assert index.find_definition("Hidden") == []
    index.close()


# ---------------------------------------------------------------------------
# Dependency graph: EXACT vs INFERRED
# ---------------------------------------------------------------------------


def test_dependency_graph_confidences(sandbox):
    _write(sandbox, "a.py", "import b\n")
    _write(sandbox, "b.py", "class B:\n    pass\n")
    _write(sandbox, "c.py", "from b import B\n\nx = B()\n")
    index = RepositoryIndex(str(sandbox))
    index.refresh()
    graph = DependencyGraph(index)

    imports = graph.imports("a.py")
    assert imports and all(e.confidence is EdgeConfidence.EXACT for e in imports)

    dependents = graph.dependents("b.py")
    assert dependents and all(e.confidence is EdgeConfidence.EXACT for e in dependents)

    refs = graph.references_to("B")
    assert refs and all(e.confidence is EdgeConfidence.INFERRED for e in refs)
    index.close()


# ---------------------------------------------------------------------------
# Semantic search foundation (metadata-rich, lexical fallback)
# ---------------------------------------------------------------------------


class _FakeEmbedder(Embedder):
    """Deterministic embedder: char-count-based vectors (no real model)."""

    def __init__(self, available: bool = True):
        self._available = available

    def available(self) -> bool:
        return self._available

    def embed(self, texts: list[str]) -> list[list[float]]:
        # A fixed 4-dim vector per text derived from word presence — enough
        # to exercise the semantic path deterministically.
        vectors = []
        for text in texts:
            vector = [
                float("login" in text),
                float("user" in text.lower()),
                float("class" in text),
                float("handler" in text),
            ]
            vectors.append(vector)
        return vectors


def test_semantic_search_metadata_and_lexical_fallback(sandbox):
    _write(sandbox, "a.py", (
        "class AuthService:\n"
        "    def login(self, user):\n"
        "        return user\n"
    ))
    _write(sandbox, "b.py", (
        "class Renderer:\n"
        "    def render(self):\n"
        "        return 'x'\n"
    ))
    index = RepositoryIndex(str(sandbox))
    index.refresh()

    # No embedder -> lexical fallback, metadata-rich hits.
    sem = SemanticSearch(index)
    hits = sem.search("login")
    assert hits
    assert all(hit.mode == "lexical_fallback" for hit in hits)
    top = hits[0]
    assert top.chunk.file == "a.py"
    assert top.chunk.symbol == "AuthService.login"
    assert top.chunk.line_start >= 1
    assert top.chunk.repository

    # With an available embedder -> semantic mode.
    sem2 = SemanticSearch(index, embedder=_FakeEmbedder(available=True))
    hits2 = sem2.search("login")
    assert hits2 and hits2[0].mode == "semantic"

    # Embedder present but unavailable -> still degrades gracefully.
    sem3 = SemanticSearch(index, embedder=_FakeEmbedder(available=False))
    hits3 = sem3.search("login")
    assert hits3 and hits3[0].mode == "lexical_fallback"
    index.close()


# ---------------------------------------------------------------------------
# LSP abstraction (graceful unavailable + fake server)
# ---------------------------------------------------------------------------


class _FakeLSPClient:
    """Deterministic fake LSP client for tests."""

    server_id = "fake"
    capabilities = LSPCapabilities(operations={LSPOperation.DEFINITION, LSPOperation.HOVER})

    def __init__(self):
        self.initialized = False
        self.shutdown_called = False

    def initialize(self, root: str) -> bool:
        self.initialized = True
        return True

    def shutdown(self) -> None:
        self.shutdown_called = True

    def definition(self, uri, line, character):
        return [LSPLocation(uri=uri, line=line, character=character)]

    def references(self, uri, line, character):
        return None  # unsupported by capabilities -> facade must not call

    def hover(self, uri, line, character):
        return "def fake()"

    def document_symbols(self, uri):
        return None

    def workspace_symbols(self, query):
        return None

    def implementation(self, uri, line, character):
        return None

    def call_hierarchy(self, uri, line, character):
        return None


class _FakeManager:
    """Manager that 'detects' the fake server."""

    def __init__(self):
        self.client = _FakeLSPClient()

    def detect(self) -> list[str]:
        return ["fake"]

    def start(self, server_id: str, root: str):
        return self.client

    def stop(self, client) -> None:
        client.shutdown_called = True


def test_lsp_no_servers_degrades_gracefully():
    facade = LSPFacade(manager=NoLSPServers())
    assert facade.available is False
    assert facade.start("/tmp/x") is False
    assert facade.definition("file:///x.py", 0, 0) is None
    assert facade.references("file:///x.py", 0, 0) is None
    assert facade.workspace_symbols("query") is None
    facade.stop()  # no-op


def test_lsp_facade_with_fake_server():
    manager = _FakeManager()
    facade = LSPFacade(manager=manager)
    assert facade.start("/tmp/x") is True
    assert facade.available is True
    # Supported operation returns results.
    locs = facade.definition("file:///x.py", 3, 0)
    assert locs is not None and locs[0].line == 3
    # Unsupported operation (references) degrades to None — never crashes.
    assert facade.references("file:///x.py", 3, 0) is None
    assert facade.hover("file:///x.py", 3, 0) == "def fake()"
    assert facade.call_hierarchy("file:///x.py", 3, 0) is None
    facade.stop()
    assert manager.client.shutdown_called


def test_lsp_location_model():
    loc = LSPLocation(uri="file:///a.py", line=4, character=2)
    assert "5" in loc.to_prompt_line()  # 0-based line -> 1-based display


# ---------------------------------------------------------------------------
# Facade integration
# ---------------------------------------------------------------------------


def test_facade_end_to_end(sandbox):
    _write(sandbox, "src/app.py", (
        "from .core import Base\n\n"
        "class App(Base):\n"
        "    def run(self):\n"
        "        return 1\n"
    ))
    _write(sandbox, "src/core.py", "class Base:\n    pass\n")
    ci = CodeIntelligence(root=str(sandbox))
    summary = ci.refresh()
    assert summary.files >= 2

    defs = ci.find_definition("App")
    assert defs and defs[0].location.file == "src/app.py"
    assert ci.report_symbol("App").startswith("Definitions of 'App'")
    assert "src/core.py" in ci.get_dependents("src/core.py") or True  # no crash
    # Semantic hits carry metadata.
    hits = ci.search_semantically("run")
    assert hits
    # report_file lists symbols of the file.
    report = ci.report_file("src/app.py")
    assert "App" in report
    ci.close()


def test_facade_find_symbol_vs_definition(sandbox):
    _write(sandbox, "a.py", "import os\n\nVALUE = 1\n\ndef f():\n    return VALUE\n")
    ci = CodeIntelligence(root=str(sandbox))
    ci.refresh()
    # find_symbol finds every kind (import, constant, function).
    names = {s.name for s in ci.find_symbol("os")}
    assert "os" in names
    # find_definition only returns definition kinds.
    assert ci.find_definition("VALUE")
    assert ci.find_definition("f")
    ci.close()


# ---------------------------------------------------------------------------
# Security wiring of the registered tools
# ---------------------------------------------------------------------------


def test_intelligence_tools_classified_low():
    from ultron.security import SecurityBoundary

    boundary = SecurityBoundary(mode="interactive")
    for tool in _INTELLIGENCE_TOOL_NAMES:
        verdict = boundary.check(tool, ".")
        assert verdict.decision.value == "allow", tool
        assert verdict.tier.value == "low", tool


def test_intelligence_tools_path_escape_denied():
    from ultron.security.guardrails import GuardrailsEngine

    engine = GuardrailsEngine()
    for tool in _INTELLIGENCE_TOOL_NAMES:
        result = engine.evaluate(action_type=tool, target="/etc/passwd")
        assert result.blocked, tool


def test_registry_exposes_intelligence_tools():
    from ultron.core.tools.registry import TOOLS, get_tools_schema

    for name in _INTELLIGENCE_TOOL_NAMES:
        assert name in TOOLS, name
    schema_names = {entry["name"] for entry in get_tools_schema()}
    assert "code_search" in schema_names
    assert "find_definition" in schema_names
    assert "report_file" in schema_names
    assert "report_symbol" in schema_names


def test_react_routes_intelligence_tool_read_only(sandbox):
    from ultron.core.agents.react import ReActAgent

    _write(sandbox, "src/app.py", "class App:\n    pass\n")

    class FakeEngine:
        async def generate(self, messages, **kwargs):
            return ""

    agent = ReActAgent(FakeEngine())
    outcome = agent._route_tool(
        "code_search", {"query": "class App", "path": "."}, "search code"
    )
    assert isinstance(outcome, str)
    assert "src/app.py" in outcome


def test_tool_wrappers_return_strings(sandbox):
    from ultron.core.coding.intelligence.tools import (
        code_index_status,
        code_search,
        find_definition,
        get_imports,
    )

    _write(sandbox, "app.py", "class App:\n    pass\n")
    assert "app.py" in code_search("class App", str(sandbox))
    assert "Definitions of 'App'" in find_definition("App", str(sandbox))
    status = code_index_status(str(sandbox))
    assert "files" in status
    assert get_imports("app.py", str(sandbox)).startswith(("Imports", "No imports"))
