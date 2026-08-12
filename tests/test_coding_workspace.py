"""
Fix #3 stage-1 tests: coding workspace discovery + repository inspection.

Covers workspace discovery, extensible project detection (python / node /
go / rust / java), the read-only list_directory and search_files tools,
and best-effort git status/diff integration. All tests use temporary
directories — the real Ultron repository is never modified.
"""

import subprocess

import pytest

from ultron.core.coding.workspace import (
    CodingWorkspace,
    discover_workspace,
    discover_workspace_summary,
    git_diff,
    git_status,
    list_directory,
    search_files,
)
from ultron.core.tools import paths as tools_paths


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Points the file-policy allowlist + cwd at a temp dir."""
    monkeypatch.setattr(tools_paths, "ALLOWED_BASE_DIR", tmp_path)
    monkeypatch.chdir(tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# Workspace discovery
# ---------------------------------------------------------------------------


def test_discover_workspace_empty_dir(sandbox):
    ws = discover_workspace(str(sandbox))
    assert ws.project_root == str(sandbox)
    assert ws.project_type in ("unknown", "generic")
    assert ws.is_git_repo is False


def test_discover_python_project(sandbox):
    (sandbox / "pyproject.toml").write_text(
        '[build-system]\nrequires = ["setuptools"]\n\n[tool.pytest.ini_options]\n',
        encoding="utf-8",
    )
    (sandbox / "src").mkdir()
    (sandbox / "tests").mkdir()
    ws = discover_workspace(str(sandbox))
    assert ws.project_type == "python"
    assert "python" in ws.languages
    assert ws.test_framework == "pytest"
    assert ws.build_system == "setuptools"
    assert "src" in ws.source_dirs
    assert "tests" in ws.test_dirs
    assert "pyproject.toml" in ws.config_files


def test_discover_node_project(sandbox):
    (sandbox / "package.json").write_text(
        '{"scripts": {"build": "vite build"}, "devDependencies": {"vitest": "^1.0"}}',
        encoding="utf-8",
    )
    (sandbox / "pnpm-lock.yaml").write_text("lockfileVersion: '9.0'", encoding="utf-8")
    (sandbox / "tsconfig.json").write_text("{}", encoding="utf-8")
    (sandbox / "src").mkdir()
    ws = discover_workspace(str(sandbox))
    assert ws.project_type == "node"
    assert "typescript" in ws.languages
    assert ws.package_manager == "pnpm"
    assert ws.build_system == "vite"
    assert ws.test_framework == "vitest"
    assert "src" in ws.source_dirs


def test_discover_go_and_rust(sandbox):
    (sandbox / "go.mod").write_text("module example.com/demo\n\ngo 1.21\n", encoding="utf-8")
    ws = discover_workspace(str(sandbox))
    assert ws.project_type == "go"
    assert ws.package_manager == "go modules"
    assert ws.test_framework == "go test"

    rust_dir = sandbox / "rustapp"
    rust_dir.mkdir()
    (rust_dir / "Cargo.toml").write_text("[package]\nname = 'demo'\n", encoding="utf-8")
    ws_rust = discover_workspace(str(rust_dir))
    assert ws_rust.project_type == "rust"
    assert ws_rust.package_manager == "cargo"
    assert ws_rust.test_framework == "cargo test"


def test_discover_java_maven(sandbox):
    (sandbox / "pom.xml").write_text("<project/>", encoding="utf-8")
    (sandbox / "src").mkdir()
    ws = discover_workspace(str(sandbox))
    assert ws.project_type == "java"
    assert ws.package_manager == "maven"
    assert ws.build_system == "maven"


def test_project_root_walks_up(sandbox):
    nested = sandbox / "a" / "b"
    nested.mkdir(parents=True)
    (sandbox / "package.json").write_text("{}", encoding="utf-8")
    ws = discover_workspace(str(nested))
    assert ws.project_root == str(sandbox)  # found by walking up


def test_workspace_summary_compact(sandbox):
    (sandbox / "pyproject.toml").write_text("[tool.pytest.ini_options]\n", encoding="utf-8")
    summary = discover_workspace_summary(str(sandbox))
    assert summary.startswith("WORKSPACE")
    assert "python" in summary


def test_workspace_model_serializes(sandbox):
    ws = discover_workspace(str(sandbox))
    restored = CodingWorkspace.model_validate_json(ws.model_dump_json())
    assert restored.project_root == ws.project_root


# ---------------------------------------------------------------------------
# list_directory
# ---------------------------------------------------------------------------


def test_list_directory_tree(sandbox):
    (sandbox / "main.py").write_text("print(1)", encoding="utf-8")
    (sandbox / "pkg").mkdir()
    (sandbox / "pkg" / "mod.py").write_text("x = 1", encoding="utf-8")
    out = list_directory(".", max_entries=50, max_depth=3)
    assert "main.py" in out
    assert "pkg/" in out
    assert "mod.py" in out


def test_list_directory_skips_vcs_and_artifacts(sandbox):
    (sandbox / "node_modules").mkdir()
    (sandbox / "node_modules" / "lib").mkdir()
    (sandbox / ".git").mkdir()
    (sandbox / ".venv").mkdir()
    (sandbox / "app.py").write_text("x = 1", encoding="utf-8")
    out = list_directory(".", max_entries=200, max_depth=3)
    assert "app.py" in out
    assert "node_modules" not in out
    assert ".git" not in out
    assert ".venv" not in out


def test_list_directory_bounded(sandbox):
    for i in range(10):
        (sandbox / f"f{i}.txt").write_text("x", encoding="utf-8")
    out = list_directory(".", max_entries=3, max_depth=1)
    lines = [ln for ln in out.splitlines() if ln and not ln.startswith("(")]
    assert len(lines) == 3


def test_list_directory_unsafe_path_rejected(sandbox):
    # /etc exists outside the sandbox allowlist.
    out = list_directory("/etc", max_entries=5, max_depth=1)
    assert out.startswith("Error: access denied")


# ---------------------------------------------------------------------------
# search_files
# ---------------------------------------------------------------------------


def test_search_files_matches_content(sandbox):
    (sandbox / "auth.py").write_text("def login(): pass", encoding="utf-8")
    (sandbox / "main.py").write_text("from auth import login\n", encoding="utf-8")
    out = search_files("login", ".", max_results=30)
    assert "auth.py" in out
    assert "main.py" in out


def test_search_files_matches_filename(sandbox):
    (sandbox / "login_module.py").write_text("x = 1", encoding="utf-8")
    out = search_files("login_module", ".", max_results=30)
    assert "login_module.py" in out


def test_search_files_no_match(sandbox):
    (sandbox / "auth.py").write_text("def login(): pass", encoding="utf-8")
    out = search_files("zzz_nonexistent", ".", max_results=30)
    assert "No matches" in out


def test_search_files_unsafe_path_rejected(sandbox):
    out = search_files("login", "/etc", max_results=5)
    assert out.startswith("Error: access denied")


# ---------------------------------------------------------------------------
# git integration (best-effort; skipped when git is unavailable)
# ---------------------------------------------------------------------------


def _git_available() -> bool:
    try:
        subprocess.run(["git", "--version"], capture_output=True, timeout=3, check=False)
        return True
    except (OSError, subprocess.SubprocessError):
        return False


@pytest.mark.skipif(not _git_available(), reason="git not installed")
def test_git_status_and_diff_integration(sandbox):
    subprocess.run(["git", "init", "-q"], cwd=sandbox, check=True, timeout=10)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=sandbox, check=True, timeout=5)
    subprocess.run(["git", "config", "user.name", "t"], cwd=sandbox, check=True, timeout=5)
    (sandbox / "a.txt").write_text("hello", encoding="utf-8")
    subprocess.run(["git", "add", "a.txt"], cwd=sandbox, check=True, timeout=5)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=sandbox, check=True, timeout=5)

    ws = discover_workspace(str(sandbox))
    assert ws.is_git_repo
    assert ws.git_clean is True

    (sandbox / "a.txt").write_text("hello world", encoding="utf-8")
    assert git_status(str(sandbox)) != ""
    assert git_diff(str(sandbox)) != ""
    ws_dirty = discover_workspace(str(sandbox))
    assert ws_dirty.git_clean is False


def test_git_helpers_without_repo_return_empty(sandbox):
    assert git_status(str(sandbox)) == ""
    assert git_diff(str(sandbox)) == ""
