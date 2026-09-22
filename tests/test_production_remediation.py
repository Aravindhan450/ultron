from __future__ import annotations

import tempfile
from pathlib import Path

from ultron.core.agents.react import (
    ReActAgent,
    _check_action_preflight,
)
from ultron.core.coding.context import CodeContext
from ultron.core.coding.edits import create_file
from ultron.core.coding.intelligence.dependencies import (
    extract_external_imports,
    filter_python_dependencies,
    is_stdlib_module,
    sanitize_requirements_txt,
)
from ultron.core.intelligence.task_classification import TaskType
from ultron.core.intelligence.task_planning import TaskState
from ultron.core.tools.builtin.file_writer import write_file
from ultron.core.types import ToolExecution


def test_dependency_intelligence_stdlib_detection():
    """Verify is_stdlib_module correctly classifies Python standard library vs 3rd-party."""
    # Stdlib modules
    assert is_stdlib_module("tkinter")
    assert is_stdlib_module("sqlite3")
    assert is_stdlib_module("sys")
    assert is_stdlib_module("os")
    assert is_stdlib_module("json")
    assert is_stdlib_module("math")
    assert is_stdlib_module("pathlib")
    assert is_stdlib_module("urllib.parse")
    assert is_stdlib_module("collections.abc")

    # 3rd-party / external packages
    assert not is_stdlib_module("pandas")
    assert not is_stdlib_module("numpy")
    assert not is_stdlib_module("requests")
    assert not is_stdlib_module("fastapi")
    assert not is_stdlib_module("pydantic")
    assert not is_stdlib_module("rich")
    assert not is_stdlib_module("pytest")


def test_filter_python_dependencies():
    """Verify filter_python_dependencies removes stdlib modules from package lists."""
    pkgs = [
        "tkinter",
        "requests>=2.28.0",
        "sqlite3",
        "pandas==2.0.0",
        "json",
        "pytest",
        "# comment line",
        "-r other.txt",
    ]
    filtered = filter_python_dependencies(pkgs)
    assert filtered == ["requests>=2.28.0", "pandas==2.0.0", "pytest"]


def test_sanitize_requirements_txt():
    """Verify sanitize_requirements_txt strips stdlib modules while preserving valid entries."""
    raw = (
        "# Core dependencies\n"
        "tkinter\n"
        "sqlite3\n"
        "requests>=2.28.0\n"
        "pandas==2.0.0\n"
        "\n"
        "# Dev tools\n"
        "pytest>=7.0.0\n"
        "math\n"
    )
    sanitized = sanitize_requirements_txt(raw)
    expected = (
        "# Core dependencies\n"
        "requests>=2.28.0\n"
        "pandas==2.0.0\n"
        "\n"
        "# Dev tools\n"
        "pytest>=7.0.0"
    )
    assert sanitized == expected


def test_extract_external_imports():
    """Verify AST extraction of external top-level imports."""
    code = """
import os
import sys
import tkinter as tk
import sqlite3
import pandas as pd
from requests.auth import HTTPBasicAuth
from my_local_module import helper
"""
    external = extract_external_imports(code)
    assert "pandas" in external
    assert "requests" in external
    assert "my_local_module" in external
    assert "os" not in external
    assert "sys" not in external
    assert "tkinter" not in external
    assert "sqlite3" not in external


def test_file_writer_and_edits_sanitize_requirements(monkeypatch):
    """Verify write_file and create_file sanitize requirements.txt files on creation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        monkeypatch.setenv("ULTRON_WORKSPACE_DIR", tmpdir)
        import ultron.core.tools.paths as pth

        monkeypatch.setattr(pth, "ALLOWED_BASE_DIR", Path(tmpdir).resolve())
        req_path = Path(tmpdir) / "requirements.txt"

        # Test write_file
        content = "tkinter\nsqlite3\nrich>=13.0.0\n"
        res = write_file(str(req_path), content, overwrite=True)
        assert "Successfully wrote" in res
        saved_content = req_path.read_text()
        assert "rich>=13.0.0" in saved_content
        assert "tkinter" not in saved_content
        assert "sqlite3" not in saved_content

        # Test create_file via edits
        req2_path = Path(tmpdir) / "dev-requirements.txt"
        create_file(str(req2_path), "pytest\njson\nos\n")
        saved2 = req2_path.read_text()
        assert "pytest" in saved2
        assert "json" not in saved2
        assert "os" not in saved2


def test_action_preflight_pip_install():
    """Verify _check_action_preflight blocks pip install when requirements file does not exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        missing_file = Path(tmpdir) / "missing_reqs.txt"
        msg = _check_action_preflight(
            "run_command",
            {"command": f"pip install -r {missing_file}", "cwd": tmpdir},
        )
        assert msg is not None
        assert "Preflight Error" in msg
        assert "missing_reqs.txt" in msg

        # When file exists, preflight check succeeds
        existing_file = Path(tmpdir) / "exists_reqs.txt"
        existing_file.write_text("rich\n")
        msg = _check_action_preflight(
            "run_command",
            {"command": f"pip install -r {existing_file}", "cwd": tmpdir},
        )
        assert msg is None


def test_action_preflight_python_execution():
    """Verify _check_action_preflight blocks python execution when script does not exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        missing_script = Path(tmpdir) / "app.py"
        msg = _check_action_preflight(
            "run_command",
            {"command": f"python {missing_script}", "cwd": tmpdir},
        )
        assert msg is not None
        assert "Preflight Error" in msg
        assert "app.py" in msg

        # When script exists, preflight check succeeds
        missing_script.write_text("print('hello')\n")
        msg = _check_action_preflight(
            "run_command",
            {"command": f"python {missing_script}", "cwd": tmpdir},
        )
        assert msg is None


def test_failure_thrashing_circuit_breaker():
    """Verify coding gate prevents repeating failing actions after repeated failures."""
    from unittest.mock import MagicMock

    with tempfile.TemporaryDirectory() as tmpdir:
        script = Path(tmpdir) / "run.py"
        script.write_text("print('test')")

        agent = ReActAgent(MagicMock())
        task = TaskState(
            goal="Build something",
            task_type=TaskType.SOFTWARE_ENGINEERING,
        )
        task.code_context = CodeContext()

        # 1 failure: allowed to retry
        exec_record1 = ToolExecution(
            tool_name="run_command",
            target=f"python {script}",
            detail="Error: command failed",
            success=False,
        )
        task.execution_history.append(exec_record1)
        gate_err1 = agent._coding_gate(
            task, "run_command", {"command": f"python {script}", "cwd": tmpdir}
        )
        assert gate_err1 is None

        # 2nd failure: circuit breaker triggers and blocks 3rd attempt
        exec_record2 = ToolExecution(
            tool_name="run_command",
            target=f"python {script}",
            detail="Error: command failed",
            success=False,
        )
        task.execution_history.append(exec_record2)
        gate_err2 = agent._coding_gate(
            task, "run_command", {"command": f"python {script}", "cwd": tmpdir}
        )
        assert gate_err2 is not None
        assert "has already failed 2 times" in gate_err2


import pytest


@pytest.mark.anyio
async def test_verification_requires_execution_evidence():
    """Verify task verification rejects completion on software engineering tasks without execution."""
    from unittest.mock import AsyncMock

    mock_engine = AsyncMock()
    mock_engine.generate.return_value = '[{"description": "App is complete", "satisfied": true}]'

    agent = ReActAgent(mock_engine)
    task = TaskState(
        goal="Build app",
        task_type=TaskType.SOFTWARE_ENGINEERING,
    )

    # Case 1: Initial state with no executions
    verified, _ = await agent._verify_task(task, "Build app", "I am done", [])
    assert not verified

    # Case 2: Add only edit execution (no run/test execution)
    task.execution_history.append(
        ToolExecution(
            tool_name="create_file",
            target="app.py",
            detail="Created",
            success=True,
        )
    )
    verified, _ = await agent._verify_task(task, "Build app", "I am done", [])
    assert not verified

    # Case 3: Add successful test/run execution
    task.execution_history.append(
        ToolExecution(
            tool_name="run_command",
            target="pytest tests/",
            detail="Passed 5 tests",
            success=True,
        )
    )
    verified, _ = await agent._verify_task(task, "Build app", "I am done", [])
    assert verified
    assert task.status.value == "task_completed"
