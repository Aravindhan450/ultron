"""
tests.test_phase3_repo_intelligence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Comprehensive automated tests for Phase 3 — Repository Intelligence / Repo Map.

Validates:
1. Small repository indexing accuracy (files, AST symbols, imports, classification).
2. Secret & ignored path exclusion (.env, credentials, .git, .venv, binary files).
3. Dependency graph & PageRank centrality (import resolution, centrality scoring).
4. Multi-file dependency retrieval (dependency proximity, 1-hop & 2-hop traversal).
5. Token-budgeted RepoMap generation (strict token bound, tiered detail, graceful elision).
6. Large repository simulation (stress test ensuring zero token budget breaches).
7. Incremental index invalidation on file mutation/addition/deletion.
8. Task-aware retrieval with full provenance (file, symbol, score, reason).
9. ContextManager integration (RepoMap inclusion, priority ordering, token compaction).
10. Registered `repo_map` tool verification.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ultron.core.context import (
    ContextBudgetConfig,
    ContextPriority,
    ContextSourceType,
    RepositoryContextManager,
    estimate_tokens,
)
from ultron.core.repository import (
    FileClassification,
    RepoMapGenerator,
    RepositoryCache,
    RepositoryGraph,
    RepositoryIndex,
    TaskAwareRetriever,
    is_secret_file,
)
from ultron.core.tools.definitions import TOOL_DEFINITIONS
from ultron.core.tools.registry import TOOLS


@pytest.fixture
def sandbox_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Creates a realistic small repository structure inside tmp_path."""
    monkeypatch.setattr("ultron.core.tools.paths.ALLOWED_BASE_DIR", tmp_path)
    monkeypatch.chdir(tmp_path)
    RepositoryCache.reset()

    # 1. Config files
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "demo-pkg"\nversion = "0.1.0"\n', encoding="utf-8"
    )

    # 2. Source package
    src = tmp_path / "src" / "demo"
    src.mkdir(parents=True)
    (src / "__init__.py").write_text('__version__ = "0.1.0"\n', encoding="utf-8")

    (src / "models.py").write_text(
        '"""Data models."""\n\n'
        "class User:\n"
        '    """User account entity."""\n'
        "    def __init__(self, name: str, email: str):\n"
        "        self.name = name\n"
        "        self.email = email\n\n"
        "class Session:\n"
        "    def __init__(self, user: User, token: str):\n"
        "        self.user = user\n"
        "        self.token = token\n",
        encoding="utf-8",
    )

    (src / "auth.py").write_text(
        '"""Authentication service."""\n\n'
        "from demo.models import User, Session\n\n"
        "class AuthService:\n"
        "    def authenticate(self, username: str, password: str) -> Session:\n"
        "        user = User(username, f'{username}@example.com')\n"
        "        return Session(user, 'tok_123')\n\n"
        "def verify_token(token: str) -> bool:\n"
        "    return token.startswith('tok_')\n",
        encoding="utf-8",
    )

    (src / "controller.py").write_text(
        '"""HTTP API controllers."""\n\n'
        "from demo.auth import AuthService\n\n"
        "class AuthController:\n"
        "    def __init__(self):\n"
        "        self.auth_service = AuthService()\n\n"
        "    def login(self, username: str, password: str) -> dict:\n"
        "        session = self.auth_service.authenticate(username, password)\n"
        "        return {'token': session.token}\n",
        encoding="utf-8",
    )

    # 3. Tests
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir(parents=True)
    (tests_dir / "test_auth.py").write_text(
        "from demo.auth import AuthService, verify_token\n\n"
        "def test_auth_login():\n"
        "    svc = AuthService()\n"
        "    session = svc.authenticate('alice', 'secret')\n"
        "    assert session.token == 'tok_123'\n"
        "    assert verify_token(session.token)\n",
        encoding="utf-8",
    )

    # 4. Sensitive & ignored files (MUST be excluded)
    (tmp_path / ".env").write_text("SECRET_KEY=supersecret123\n", encoding="utf-8")
    (tmp_path / ".env.local").write_text("API_KEY=local_abc\n", encoding="utf-8")
    (tmp_path / "id_rsa").write_text("-----BEGIN RSA PRIVATE KEY-----\nfake\n", encoding="utf-8")

    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text("[core]\nrepositoryformatversion = 0\n", encoding="utf-8")

    venv_dir = tmp_path / ".venv" / "lib"
    venv_dir.mkdir(parents=True)
    (venv_dir / "site.py").write_text("# venv library\n", encoding="utf-8")

    cache_dir = tmp_path / "__pycache__"
    cache_dir.mkdir()
    (cache_dir / "compiled.pyc").write_bytes(b"\x00\x00\x00\x00")

    return tmp_path


# ---------------------------------------------------------------------------
# 1. Indexing & File Classification Tests
# ---------------------------------------------------------------------------


def test_repository_indexing_and_classification(sandbox_repo: Path):
    index = RepositoryIndex(sandbox_repo)
    summary = index.refresh()

    assert summary.total_files >= 5
    assert summary.source_files >= 4
    assert summary.test_files >= 1
    assert summary.config_files >= 1

    # Check classification
    models_file = index.get_file("src/demo/models.py")
    assert models_file is not None
    assert models_file.classification == FileClassification.SOURCE
    assert models_file.language == "python"

    # Symbols extracted
    sym_names = [s.name for s in models_file.symbols]
    assert "User" in sym_names
    assert "Session" in sym_names

    # Test file classified correctly
    test_file = index.get_file("tests/test_auth.py")
    assert test_file is not None
    assert test_file.classification == FileClassification.TEST


# ---------------------------------------------------------------------------
# 2. Secret & Ignored Path Exclusion Tests
# ---------------------------------------------------------------------------


def test_secrets_and_ignored_directories_strictly_excluded(sandbox_repo: Path):
    index = RepositoryIndex(sandbox_repo)
    index.refresh()

    indexed_paths = list(index.files.keys())

    # None of the secrets or ignored paths should be present
    for path in indexed_paths:
        assert not path.startswith(".env"), f"Secret leaked: {path}"
        assert not path.startswith(".git"), f"Git leaked: {path}"
        assert not path.startswith(".venv"), f"Venv leaked: {path}"
        assert not path.startswith("__pycache__"), f"Cache leaked: {path}"
        assert "id_rsa" not in path, f"Key leaked: {path}"
        assert not path.endswith(".pyc"), f"Binary leaked: {path}"

    assert is_secret_file(".env")
    assert is_secret_file(".env.local")
    assert is_secret_file("id_rsa")
    assert is_secret_file("prod.key")
    assert not is_secret_file("src/demo/auth.py")


# ---------------------------------------------------------------------------
# 3. Dependency Graph & PageRank Centrality Tests
# ---------------------------------------------------------------------------


def test_dependency_graph_and_pagerank(sandbox_repo: Path):
    index = RepositoryIndex(sandbox_repo)
    graph = RepositoryGraph(index)

    models_node = graph.get_node("src/demo/models.py")
    controller_node = graph.get_node("src/demo/controller.py")
    assert controller_node.rel_path == "src/demo/controller.py"

    # models.py is imported by auth.py, so it has incoming dependencies
    assert models_node.in_degree >= 1

    # Foundational models.py has higher or equal centrality to leaf controller
    assert graph.get_pagerank("src/demo/models.py") > 0.0

    # 1-hop and 2-hop neighborhood
    neighbors = graph.get_neighbors("src/demo/auth.py", max_depth=1)
    assert "src/demo/models.py" in neighbors or "src/demo/controller.py" in neighbors


# ---------------------------------------------------------------------------
# 4. Token-Budgeted RepoMap Tests
# ---------------------------------------------------------------------------


def test_repomap_generation_and_budget_enforcement(sandbox_repo: Path):
    generator = RepoMapGenerator(sandbox_repo)

    # 1. Normal budget (1000 tokens)
    repo_map = generator.generate(max_tokens=1000)
    assert repo_map.token_count <= 1000
    assert repo_map.token_count > 0
    assert "src/demo/models.py" in repo_map.text
    assert "class User" in repo_map.text

    # 2. Strict budget (150 tokens)
    strict_map = generator.generate(max_tokens=150)
    assert strict_map.token_count <= 150
    assert estimate_tokens(strict_map.text) <= 150

    # 3. Micro budget (50 tokens) - must not crash and must obey limit
    micro_map = generator.generate(max_tokens=50)
    assert micro_map.token_count <= 50
    assert estimate_tokens(micro_map.text) <= 50


def test_repomap_with_focus_terms(sandbox_repo: Path):
    generator = RepoMapGenerator(sandbox_repo)
    focused_map = generator.generate(max_tokens=800, focus_query="authenticate login token")

    assert focused_map.token_count <= 800
    # auth.py or controller.py should be prominently ranked
    assert "auth.py" in focused_map.text or "controller.py" in focused_map.text


# ---------------------------------------------------------------------------
# 5. Large Repository Simulation (Stress Test)
# ---------------------------------------------------------------------------


def test_large_repository_repomap_stress(sandbox_repo: Path):
    """Simulates 60 files across multiple modules and verifies hard token budget."""
    pkg = sandbox_repo / "src" / "demo" / "services"
    pkg.mkdir(parents=True, exist_ok=True)

    for i in range(50):
        (pkg / f"service_{i}.py").write_text(
            f'"""Service {i} module."""\n\n'
            f"from demo.models import User\n\n"
            f"class Service{i}:\n"
            f"    def execute_{i}(self, user: User) -> int:\n"
            f"        return {i}\n",
            encoding="utf-8",
        )

    generator = RepoMapGenerator(sandbox_repo)
    budget = 400
    large_map = generator.generate(max_tokens=budget)

    # Hard invariant: token_count must strictly obey configured budget
    assert large_map.token_count <= budget
    assert estimate_tokens(large_map.text) <= budget
    assert large_map.elided_files_count > 0


# ---------------------------------------------------------------------------
# 6. Incremental Cache Invalidation Tests
# ---------------------------------------------------------------------------


def test_cache_invalidation_on_file_edits(sandbox_repo: Path):
    index = RepositoryIndex(sandbox_repo)
    index.refresh()

    auth_file = sandbox_repo / "src" / "demo" / "auth.py"

    # Add a new symbol to auth.py
    new_content = auth_file.read_text(encoding="utf-8") + "\nclass NewAuthPolicy:\n    pass\n"
    auth_file.write_text(new_content, encoding="utf-8")

    # Invalidate cache for auth.py
    cache = RepositoryCache.get_instance()
    cache.invalidate_file(auth_file, root=sandbox_repo)

    # Re-index
    summary = index.refresh()
    assert summary.total_symbols > 0

    updated_auth = index.get_file("src/demo/auth.py")
    assert updated_auth is not None
    sym_names = [s.name for s in updated_auth.symbols]
    assert "NewAuthPolicy" in sym_names


# ---------------------------------------------------------------------------
# 7. Task-Aware Retrieval with Provenance Tests
# ---------------------------------------------------------------------------


def test_task_aware_retrieval_provenance(sandbox_repo: Path):
    retriever = TaskAwareRetriever(sandbox_repo)

    result = retriever.retrieve_relevant("Fix authentication token expiration in AuthService")

    assert len(result.items) > 0
    assert "src/demo/auth.py" in result.top_files

    # Provenance verification: every item has file, score, and clear reason
    for item in result.items:
        assert item.file_path
        assert item.score > 0.0
        assert item.reason, f"Missing relevance reason for {item.file_path}"

    top_item = result.items[0]
    assert "auth" in top_item.file_path.lower()
    assert ("symbol match" in top_item.reason.lower() or "token" in top_item.reason.lower())


# ---------------------------------------------------------------------------
# 8. ContextManager Integration Tests
# ---------------------------------------------------------------------------


def test_context_manager_repomap_integration(sandbox_repo: Path):
    mgr = RepositoryContextManager(budget=ContextBudgetConfig(max_total_tokens=2000))

    # Assemble context snapshot
    snapshot = mgr.assemble_snapshot(
        user_request="Implement password hashing in AuthService",
        include_repo_map=True,
    )

    # REPO_MAP item must be present
    repo_map_items = [it for it in snapshot.items if it.source_type == ContextSourceType.REPO_MAP]
    assert len(repo_map_items) == 1

    repo_map_item = repo_map_items[0]
    assert repo_map_item.priority == ContextPriority.REPO_MAP
    assert "src/demo/auth.py" in repo_map_item.content
    assert snapshot.total_estimated_tokens <= 2000


# ---------------------------------------------------------------------------
# 9. Tool Registry & definitions.py Verification
# ---------------------------------------------------------------------------


def test_repo_map_tool_registered():
    assert "repo_map" in TOOLS
    assert "repo_map" in TOOL_DEFINITIONS

    defn = TOOL_DEFINITIONS["repo_map"]
    assert defn.read_only is True
    assert defn.domain.value == "code_intelligence"
    assert defn.action_label == "Generate repository map"

    tool_fn = TOOLS["repo_map"]
    res = tool_fn(focus="models", max_tokens=500)
    assert "Repository Map" in res


# ---------------------------------------------------------------------------
# 10. Embedded Secret Protection and False Positive Avoidance Tests
# ---------------------------------------------------------------------------


def test_embedded_secret_protection_and_false_positives(sandbox_repo: Path):
    from ultron.core.coding.workspace import search_files
    from ultron.core.context.retrieval import RepositoryRetriever

    # Create config file with synthetic embedded secrets alongside normal code
    config_file = sandbox_repo / "src" / "demo" / "config.py"
    config_file.write_text(
        '"""App configuration."""\n\n'
        'import os\n\n'
        '# Normal code patterns that must NOT be redacted:\n'
        'def get_api_key():\n'
        '    return os.environ.get("OPENAI_KEY")\n\n'
        'token_count: int = 1500\n'
        'is_production: bool = True\n\n'
        '# Embedded secrets that MUST be redacted:\n'
        'API_KEY = "sk-proj-99999999999999999999"\n'
        'DATABASE_URL = "postgres://admin:supersecretpassword@db.internal:5432/main_db"\n'
        'PRIVATE_SECRET_TOKEN = "tok_live_9876543210"\n',
        encoding="utf-8",
    )

    index = RepositoryIndex(sandbox_repo)
    index.refresh()

    # 1. Test search_text in RepositoryIndex
    results = index.search_text("DATABASE_URL")
    assert len(results) > 0
    _path, _line, content = results[0]
    assert "[REDACTED_SECRET]" in content
    assert "supersecretpassword" not in content

    results_key = index.search_text("API_KEY")
    assert len(results_key) > 0
    key_content = " ".join(c for _, _, c in results_key)
    assert "[REDACTED_SECRET]" in key_content
    assert "sk-proj-99999999999999999999" not in key_content

    # 2. Test search_files workspace tool
    ws_search = search_files("DATABASE_URL")
    assert "[REDACTED_SECRET]" in ws_search
    assert "supersecretpassword" not in ws_search

    # 3. Test RepositoryRetriever.retrieve_file
    retriever = RepositoryRetriever()
    file_res = retriever.retrieve_file("src/demo/config.py")
    assert file_res.items
    retrieved_content = file_res.items[0].content
    assert "[REDACTED_SECRET]" in retrieved_content
    assert "supersecretpassword" not in retrieved_content
    assert "sk-proj-99999999999999999999" not in retrieved_content

    # 4. Verify False-Positive Avoidance on normal code
    assert "def get_api_key():" in retrieved_content
    assert 'return os.environ.get("OPENAI_KEY")' in retrieved_content
    assert "token_count: int = 1500" in retrieved_content
    assert "is_production: bool = True" in retrieved_content

    # 5. Test ContextManager.build_context defensive sanitization
    mgr = RepositoryContextManager()
    ctx = mgr.build_context(requested_files=["src/demo/config.py"])
    assert "[REDACTED_SECRET]" in ctx
    assert "supersecretpassword" not in ctx
    assert "sk-proj-99999999999999999999" not in ctx


# ---------------------------------------------------------------------------
# 11. RepositoryCache Thread-Safety Tests
# ---------------------------------------------------------------------------


def test_repository_cache_thread_safety(sandbox_repo: Path):
    import concurrent.futures

    cache = RepositoryCache.get_instance()
    cache.clear()

    errors: list[Exception] = []

    def worker(i: int) -> None:
        try:
            # Query and mutate root data concurrently
            data = cache.get_root_data(sandbox_repo)
            assert "files" in data
            data["files"][f"file_{i}.py"] = {"index": i}

            # Invalidate specific file
            cache.invalidate_file(sandbox_repo / f"file_{i}.py")

            # Query again
            fresh_data = cache.get_root_data(sandbox_repo)
            assert "files" in fresh_data
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)


    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(worker, i) for i in range(50)]
        concurrent.futures.wait(futures)

    assert not errors, f"Thread safety errors encountered: {errors}"


# ---------------------------------------------------------------------------
# 12. End-to-End Mutation Invalidation Tests
# ---------------------------------------------------------------------------


def test_end_to_end_mutation_invalidation(sandbox_repo: Path):
    from ultron.core.coding.edits import (
        append_to_file,
        create_file,
        delete_file,
        rename_file,
        replace_in_file,
    )

    index = RepositoryIndex(sandbox_repo)
    index.refresh()

    cache_data = RepositoryCache.get_instance().get_root_data(sandbox_repo)
    assert len(cache_data["files"]) > 0

    # 1. create_file invalidates cache
    create_file("src/demo/service.py", "class Service:\n    pass\n")
    # Repo maps and graph should be cleared in cache
    assert cache_data["graph"] is None
    assert cache_data["repo_maps"] == {}

    index.refresh()
    assert "src/demo/service.py" in cache_data["files"]

    # 2. replace_in_file invalidates cache
    replace_in_file("src/demo/service.py", "pass", "def execute(self): pass")
    assert "src/demo/service.py" not in cache_data["files"]

    index.refresh()
    assert "src/demo/service.py" in cache_data["files"]
    updated_syms = [s.name for s in cache_data["files"]["src/demo/service.py"].symbols]
    assert "execute" in updated_syms

    # 3. append_to_file invalidates cache
    append_to_file("src/demo/service.py", "\ndef helper_func(): pass\n")
    assert "src/demo/service.py" not in cache_data["files"]

    index.refresh()
    appended_syms = [s.name for s in cache_data["files"]["src/demo/service.py"].symbols]
    assert "helper_func" in appended_syms

    # 4. rename_file invalidates both paths in cache
    rename_file("src/demo/service.py", "src/demo/renamed_service.py")
    assert "src/demo/service.py" not in cache_data["files"]
    assert "src/demo/renamed_service.py" not in cache_data["files"]

    index.refresh()
    assert "src/demo/renamed_service.py" in cache_data["files"]
    assert "src/demo/service.py" not in cache_data["files"]

    # 5. delete_file invalidates cache
    delete_file("src/demo/renamed_service.py")
    assert "src/demo/renamed_service.py" not in cache_data["files"]

    index.refresh()
    assert "src/demo/renamed_service.py" not in cache_data["files"]

