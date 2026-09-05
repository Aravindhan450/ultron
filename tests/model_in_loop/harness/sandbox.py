"""
MITLSandbox: Isolated temporary repository sandbox for model-in-the-loop tests.
Guarantees that the real Ultron codebase is never modified.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from ultron.core.coding.workspace import CodingWorkspace, discover_workspace
from ultron.core.tools import paths as tools_paths


class MITLSandbox:
    """
    Manages an isolated temporary repository environment for a model-in-the-loop scenario.
    """

    def __init__(self, root_dir: Path, files: dict[str, str] | None = None) -> None:
        self.path = root_dir.resolve()
        self.path.mkdir(parents=True, exist_ok=True)
        self._initial_files = files or {}
        self.initial_commit: str | None = None
        self._setup_files()
        self._init_git()
        self.workspace: CodingWorkspace = discover_workspace(str(self.path))

    def _setup_files(self) -> None:
        """Populates the initial scenario files in the sandbox."""
        for rel_path, content in self._initial_files.items():
            dest = self.path / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")

    def _init_git(self) -> None:
        """Initializes a clean git repository in the sandbox with an initial commit."""
        try:
            subprocess.run(
                ["git", "init", "-b", "main"],
                cwd=self.path,
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Ultron-MITL-Test"],
                cwd=self.path,
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                ["git", "config", "user.email", "mitl@ultron.test"],
                cwd=self.path,
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                ["git", "add", "."],
                cwd=self.path,
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                ["git", "commit", "-m", "Initial scenario commit"],
                cwd=self.path,
                check=True,
                capture_output=True,
                text=True,
            )
            proc = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.path,
                check=True,
                capture_output=True,
                text=True,
            )
            self.initial_commit = proc.stdout.strip()
        except (subprocess.SubprocessError, OSError):
            # If git fails for any reason, proceed in non-git sandbox mode
            self.initial_commit = None

    def apply_security_boundary(self, monkeypatch) -> None:
        """Restricts Ultron tools and path checking strictly to this sandbox."""
        monkeypatch.setattr(tools_paths, "ALLOWED_BASE_DIR", self.path)
        monkeypatch.chdir(self.path)

    def read_file(self, relative_path: str) -> str:
        """Reads a file from the sandbox directory."""
        target = self.path / relative_path
        if not target.exists():
            raise FileNotFoundError(f"File not found in sandbox: {relative_path}")
        return target.read_text(encoding="utf-8")

    def write_file(self, relative_path: str, content: str) -> None:
        """Writes content to a file in the sandbox directory."""
        target = self.path / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    def file_exists(self, relative_path: str) -> bool:
        """Checks if a file exists in the sandbox."""
        return (self.path / relative_path).exists()

    def get_diff(self) -> str:
        """Returns the git diff of changes made inside the sandbox."""
        if self.initial_commit:
            try:
                proc = subprocess.run(
                    ["git", "diff", self.initial_commit],
                    cwd=self.path,
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
                diff = proc.stdout.strip()
                if diff:
                    return diff
            except (subprocess.SubprocessError, OSError):
                pass

        # Fallback diff generation via difflib
        import difflib

        diff_lines = []
        for rel_path, initial_content in self._initial_files.items():
            p = self.path / rel_path
            if p.exists():
                curr_content = p.read_text(encoding="utf-8")
                if curr_content != initial_content:
                    diff_lines.extend(
                        difflib.unified_diff(
                            initial_content.splitlines(keepends=True),
                            curr_content.splitlines(keepends=True),
                            fromfile=f"a/{rel_path}",
                            tofile=f"b/{rel_path}",
                        )
                    )
        return "".join(diff_lines)

    def get_modified_files(self) -> list[str]:
        """Returns list of modified or newly created file paths relative to sandbox root."""
        modified: list[str] = []
        if self.initial_commit:
            try:
                proc = subprocess.run(
                    ["git", "diff", "--name-only", self.initial_commit],
                    cwd=self.path,
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
                for line in proc.stdout.splitlines():
                    f = line.strip()
                    if f and f not in modified:
                        modified.append(f)

                status_proc = subprocess.run(
                    ["git", "status", "--porcelain"],
                    cwd=self.path,
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
                for line in status_proc.stdout.splitlines():
                    parts = line.strip().split(maxsplit=1)
                    if len(parts) == 2 and parts[1] not in modified:
                        modified.append(parts[1])
            except (subprocess.SubprocessError, OSError):
                pass

        # Also check file content diffs directly
        for rel_path, initial_content in self._initial_files.items():
            p = self.path / rel_path
            if p.exists() and p.read_text(encoding="utf-8") != initial_content and rel_path not in modified:
                modified.append(rel_path)

        return modified

    def run_pytest(self, timeout: float = 30.0) -> tuple[int, str, str]:
        """
        Executes pytest inside the sandbox to independently evaluate test status.
        Returns (exit_code, stdout, stderr).
        """
        try:
            proc = subprocess.run(
                [sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider"],
                cwd=self.path,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            return proc.returncode, proc.stdout, proc.stderr
        except subprocess.TimeoutExpired as exc:
            return 124, "", f"Pytest execution timed out after {timeout}s: {exc}"
        except OSError as exc:
            return 1, "", f"Failed to run pytest: {exc}"
