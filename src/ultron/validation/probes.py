"""Deterministic state probes for environment and workspace validation.

Probes independently inspect the real filesystem, workspace confinement, and
process/command state without asking an LLM or relying on model self-assertions.
"""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ultron.core.tools.paths import ALLOWED_BASE_DIR


class WorkspaceProbe:
    """Deterministic probe for workspace confinement and path boundaries."""

    def __init__(self, workspace_root: str | Path | None = None) -> None:
        if workspace_root is not None:
            self.workspace_root = Path(workspace_root).resolve()
        else:
            env_ws = os.environ.get("ULTRON_WORKSPACE")
            if env_ws:
                self.workspace_root = Path(env_ws).resolve()
            elif ALLOWED_BASE_DIR:
                self.workspace_root = Path(ALLOWED_BASE_DIR).resolve()
            else:
                self.workspace_root = Path.cwd().resolve()

    def is_inside(self, path: str | Path) -> bool:
        """Determines if a target path is strictly inside the workspace root."""
        try:
            resolved = Path(path).resolve()
            # If path does not exist yet, resolve its parent
            if not resolved.exists():
                parent_resolved = resolved.parent.resolve()
                try:
                    parent_resolved.relative_to(self.workspace_root)
                    return True
                except ValueError:
                    return False
            resolved.relative_to(self.workspace_root)
            return True
        except (ValueError, OSError):
            return False

    def check_confinement(self, paths: list[str | Path]) -> tuple[bool, list[str]]:
        """Checks a list of paths and returns (is_confined, list_of_escaped_paths)."""
        escaped: list[str] = []
        for p in paths:
            if not self.is_inside(p):
                escaped.append(str(p))
        return len(escaped) == 0, escaped


class FilesystemProbe:
    """Deterministic probe for filesystem existence, types, and content."""

    @staticmethod
    def exists(path: str | Path) -> bool:
        try:
            return Path(path).resolve().exists()
        except OSError:
            return False

    @staticmethod
    def not_exists(path: str | Path) -> bool:
        return not FilesystemProbe.exists(path)

    @staticmethod
    def is_file(path: str | Path) -> bool:
        try:
            return Path(path).resolve().is_file()
        except OSError:
            return False

    @staticmethod
    def is_dir(path: str | Path) -> bool:
        try:
            return Path(path).resolve().is_dir()
        except OSError:
            return False

    @staticmethod
    def read_text(path: str | Path) -> str | None:
        try:
            p = Path(path).resolve()
            if p.is_file():
                return p.read_text(encoding="utf-8", errors="replace")
            return None
        except OSError:
            return None

    @staticmethod
    def content_contains(path: str | Path, pattern: str | re.Pattern[str]) -> bool:
        content = FilesystemProbe.read_text(path)
        if content is None:
            return False
        if isinstance(pattern, re.Pattern):
            return pattern.search(content) is not None
        return pattern in content

    @staticmethod
    def content_matches_predicate(path: str | Path, predicate: Callable[[str], bool]) -> bool:
        content = FilesystemProbe.read_text(path)
        if content is None:
            return False
        try:
            return predicate(content)
        except (ValueError, TypeError, AttributeError):
            return False


class CommandProcessProbe:
    """Deterministic probe for command execution history and process status."""

    @staticmethod
    def execute(
        cmd: list[str] | str,
        cwd: str | Path | None = None,
        timeout: float = 10.0,
        env: dict[str, str] | None = None,
    ) -> tuple[int, str, str]:
        """Runs a probe command returning (exit_code, stdout, stderr)."""
        try:
            if isinstance(cmd, str):
                res = subprocess.run(
                    cmd,
                    shell=True,
                    cwd=str(cwd) if cwd else None,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    env=env,
                    check=False,
                )
            else:
                res = subprocess.run(
                    cmd,
                    cwd=str(cwd) if cwd else None,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    env=env,
                    check=False,
                )
            return res.returncode, res.stdout, res.stderr
        except subprocess.TimeoutExpired:
            return -1, "", f"Timed out after {timeout}s"
        except (OSError, subprocess.SubprocessError) as exc:
            return -1, "", str(exc)

    @staticmethod
    def check_command_history(
        history: list[dict[str, Any]],
        cmd_substring: str,
        expected_code: int = 0,
    ) -> bool:
        """Checks if a command containing cmd_substring exited with expected_code."""
        for entry in history:
            cmd = entry.get("command", "")
            code = entry.get("exit_code")
            if cmd_substring in cmd and code == expected_code:
                return True
        return False

