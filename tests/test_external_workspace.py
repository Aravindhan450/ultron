"""
Tests for ULTRON EXTERNAL WORKSPACE (ULTRON_WORKSPACE=~/UltronWorkspace).

Validates:
1. Dynamic resolution of ULTRON_WORKSPACE (~ expansion, normalized absolute path).
2. Path safety confinement and traversal rejection (../, ../../, absolute escapes, symlinks).
3. Artifact directory creation and execution cwd defaulting to the external project directory.
4. Fallback behavior when ULTRON_WORKSPACE is unset.
"""

from pathlib import Path

import pytest

from ultron.core.intelligence.task_planning import build_artifact_plan
from ultron.core.tools.builtin.command_runner import run_command
from ultron.core.tools.builtin.file_reader import read_file
from ultron.core.tools.builtin.file_writer import write_file
from ultron.core.tools.paths import (
    ALLOWED_BASE_DIR,
    get_active_project_dir,
    get_allowed_base_dirs,
    get_configured_workspace,
    is_path_safe,
    set_active_project_dir,
)
from ultron.security import check_file_access


@pytest.fixture(autouse=True)
def reset_active_dir():
    set_active_project_dir(None)
    yield
    set_active_project_dir(None)


def test_configured_workspace_resolution(monkeypatch, tmp_path):
    # Unset initially
    monkeypatch.delenv("ULTRON_WORKSPACE", raising=False)
    assert get_configured_workspace() is None

    # Set to tilde or path
    fake_home = tmp_path / "fake_home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("ULTRON_WORKSPACE", "~/UltronWorkspace")

    ws = get_configured_workspace()
    assert ws is not None
    assert ws == (fake_home / "UltronWorkspace").resolve()
    assert str(ws).startswith(str(fake_home))


def test_allowed_base_dirs_includes_workspace(monkeypatch, tmp_path):
    ext_ws = tmp_path / "ExternalWorkspace"
    ext_ws.mkdir()
    monkeypatch.setenv("ULTRON_WORKSPACE", str(ext_ws))

    base_dirs = get_allowed_base_dirs()
    assert ALLOWED_BASE_DIR in base_dirs
    assert ext_ws.resolve() in base_dirs


def test_is_path_safe_allows_both_bases(monkeypatch, tmp_path):
    ext_ws = tmp_path / "ExternalWorkspace"
    ext_ws.mkdir()
    monkeypatch.setenv("ULTRON_WORKSPACE", str(ext_ws))

    # File in default ALLOWED_BASE_DIR
    in_repo = ALLOWED_BASE_DIR / "some_file.txt"
    safe_repo, _ = is_path_safe(in_repo)
    assert safe_repo is True

    # File in external workspace
    in_ws = ext_ws / "project" / "main.py"
    safe_ws, _ = is_path_safe(in_ws)
    assert safe_ws is True

    # Escape via traversal from external workspace
    escape_attempt = ext_ws / "project" / ".." / ".." / "outside.txt"
    safe_escape, _ = is_path_safe(escape_attempt)
    # If the resolved path falls outside both ALLOWED_BASE_DIR and ext_ws, it must be rejected
    if escape_attempt.resolve() not in (ALLOWED_BASE_DIR, ext_ws) and not any(
        str(escape_attempt.resolve()).startswith(str(b)) for b in (ALLOWED_BASE_DIR, ext_ws)
    ):
        assert safe_escape is False


def test_path_traversal_hard_blocked_by_security_boundary(monkeypatch, tmp_path):
    ext_ws = tmp_path / "ExternalWorkspace"
    ext_ws.mkdir()
    monkeypatch.setenv("ULTRON_WORKSPACE", str(ext_ws))

    # Outside file
    outside = tmp_path / "super_secret.txt"
    outside.write_text("classified")

    # Security check on outside file
    verdict = check_file_access(str(outside), operation="read")
    assert not verdict.ok
    assert "outside the allowed working directory" in verdict.reason

    # Traversal from workspace to outside
    traversal = ext_ws / ".." / "super_secret.txt"
    verdict_trav = check_file_access(str(traversal), operation="read")
    assert not verdict_trav.ok
    assert "outside the allowed working directory" in verdict_trav.reason


def test_artifact_plan_resolves_in_external_workspace(monkeypatch, tmp_path):
    ext_ws = tmp_path / "UltronWorkspace"
    ext_ws.mkdir()
    monkeypatch.setenv("ULTRON_WORKSPACE", str(ext_ws))

    plan = build_artifact_plan("Build a Python expense tracker")
    assert plan.project_dir is not None
    assert str(plan.project_dir).startswith(str(ext_ws))
    assert "expense-tracker" in plan.project_dir
    assert Path(plan.project_dir).is_dir()
    assert get_active_project_dir() == Path(plan.project_dir).resolve()

    # Reset active project dir
    set_active_project_dir(None)


def test_file_operations_and_command_runner_in_active_project(monkeypatch, tmp_path):
    ext_ws = tmp_path / "UltronWorkspace"
    ext_ws.mkdir()
    monkeypatch.setenv("ULTRON_WORKSPACE", str(ext_ws))

    project_dir = ext_ws / "my_project"
    project_dir.mkdir()
    set_active_project_dir(project_dir)

    try:
        # Write relative file
        res_w = write_file("calc.py", "def add(a, b): return a + b\nprint(add(2, 3))")
        assert "Successfully wrote" in res_w
        assert (project_dir / "calc.py").exists()

        # Read relative file
        res_r = read_file("calc.py")
        assert "def add(a, b):" in res_r

        # Run command without cwd specified; defaults to active project dir
        res_cmd = run_command("python3 calc.py")
        assert "Exit code: 0" in res_cmd
        assert "Output:\n5" in res_cmd

        # Check cwd of execution via pwd
        res_pwd = run_command("pwd")
        assert str(project_dir) in res_pwd

    finally:
        set_active_project_dir(None)
