"""ultron.core.coding.workspace
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Coding workspace awareness.

A :class:`CodingWorkspace` describes the repository/environment the coding
agent is operating on: working directory, project root, git state, detected
project type, languages, package manager, build system, test framework,
relevant config files, and source/test directories.

Project detection is an EXTENSIBLE registry of detector functions
(``PROJECT_DETECTORS``) — add a detector for a new ecosystem instead of
hardcoding it. Detection is pure filesystem inspection (no tools executed,
no shell spawned), deterministic, and bounded.

This module also exposes the read-only ``list_directory`` and
``search_files`` tools registered for the ReAct agent, plus best-effort
``git_status`` / ``git_diff`` helpers.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, Field

from ultron.core.tools.paths import is_path_safe

# ---------------------------------------------------------------------------
# Project detection
# ---------------------------------------------------------------------------

# Markers that identify an existing software project. Used to find the
# project root by walking up from the working directory.
PROJECT_MARKERS = (
    "pyproject.toml",
    "package.json",
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "Cargo.toml",
    "go.mod",
    "requirements.txt",
    "setup.py",
    "setup.cfg",
    "Pipfile",
    "poetry.lock",
    "Gemfile",
    "composer.json",
    "pom.xml",
    "build.gradle",
    "build.gradle.kts",
    "mix.exs",
    "CMakeLists.txt",
    "Makefile",
    "Dockerfile",
    ".git",
)

# Directories that are never part of a project's source surface.
_IGNORED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "dist",
    "build",
    ".next",
    ".nuxt",
    ".cache",
    ".idea",
    ".vscode",
}

# Deterministic package-manager markers per ecosystem.
_PACKAGE_MANAGER_MARKERS = (
    ("poetry.lock", "poetry"),
    ("Pipfile.lock", "pipenv"),
    ("Pipfile", "pipenv"),
    ("uv.lock", "uv"),
    ("pnpm-lock.yaml", "pnpm"),
    ("yarn.lock", "yarn"),
    ("package-lock.json", "npm"),
    ("bun.lockb", "bun"),
    ("Cargo.lock", "cargo"),
    ("go.sum", "go modules"),
    ("Gemfile.lock", "bundler"),
)


class ProjectProfile(BaseModel):
    """Detected characteristics of one project ecosystem."""

    project_type: str = "unknown"
    languages: list[str] = Field(default_factory=list)
    package_manager: str | None = None
    build_system: str | None = None
    test_framework: str | None = None
    config_files: list[str] = Field(default_factory=list)
    source_dirs: list[str] = Field(default_factory=list)
    test_dirs: list[str] = Field(default_factory=list)


def _existing_dirs(root: Path, names: list[str]) -> list[str]:
    """Names that exist as directories under root, in the given order."""
    return [n for n in names if (root / n).is_dir()]


def _detect_python(root: Path) -> ProjectProfile | None:
    markers = [m for m in ("pyproject.toml", "requirements.txt", "setup.py", "setup.cfg", "Pipfile", "tox.ini") if (root / m).exists()]
    if not markers and not (root / "setup.py").exists():
        return None
    config = [m for m in markers if (root / m).is_file()]

    package_manager = "pip"
    for marker, manager in _PACKAGE_MANAGER_MARKERS:
        if (root / marker).exists():
            package_manager = manager
            break
    if (root / "pyproject.toml").is_file() and "tool.poetry" in _read_small(root / "pyproject.toml"):
        package_manager = "poetry"

    build_system = None
    if (root / "pyproject.toml").is_file():
        text = _read_small(root / "pyproject.toml")
        for system in ("hatchling", "setuptools", "flit_core", "poetry-core", "mesonpy"):
            if f'"{system}"' in text or f"'{system}'" in text:
                build_system = system
                break

    test_framework = None
    if (root / "pyproject.toml").is_file() and "[tool.pytest" in _read_small(root / "pyproject.toml") or any((root / m).exists() for m in ("pytest.ini", "tox.ini")):
        test_framework = "pytest"

    return ProjectProfile(
        project_type="python",
        languages=["python"],
        package_manager=package_manager,
        build_system=build_system,
        test_framework=test_framework,
        config_files=config,
        source_dirs=_existing_dirs(root, ["src", "lib", "app", "ultron"]),
        test_dirs=_existing_dirs(root, ["tests", "test"]),
    )


def _detect_node(root: Path) -> ProjectProfile | None:
    if not (root / "package.json").exists() and not (root / "tsconfig.json").exists():
        return None
    config = [m for m in ("package.json", "package-lock.json", "pnpm-lock.yaml", "yarn.lock", "bun.lockb", "tsconfig.json") if (root / m).exists()]

    languages = ["javascript"]
    if (root / "tsconfig.json").exists():
        languages.append("typescript")

    package_manager = None
    for marker, manager in _PACKAGE_MANAGER_MARKERS:
        if (root / marker).exists():
            package_manager = manager
            break

    build_system = None
    test_framework = None
    if (root / "package.json").is_file():
        text = _read_small(root / "package.json")
        for system in ("webpack", "vite", "next", "rollup", "esbuild", "tsc"):
            if system in text:
                build_system = system
                break
        for framework in ("jest", "vitest", "mocha", "playwright", "cypress", "ava"):
            if framework in text:
                test_framework = framework
                break

    return ProjectProfile(
        project_type="node",
        languages=languages,
        package_manager=package_manager,
        build_system=build_system,
        test_framework=test_framework,
        config_files=config,
        source_dirs=_existing_dirs(root, ["src", "lib", "app"]),
        test_dirs=_existing_dirs(root, ["__tests__", "test", "tests", "spec"]),
    )


def _detect_rust(root: Path) -> ProjectProfile | None:
    if not (root / "Cargo.toml").exists():
        return None
    return ProjectProfile(
        project_type="rust",
        languages=["rust"],
        package_manager="cargo",
        build_system="cargo build",
        test_framework="cargo test",
        config_files=["Cargo.toml"] if (root / "Cargo.toml").is_file() else [],
        source_dirs=_existing_dirs(root, ["src"]),
        test_dirs=_existing_dirs(root, ["tests"]),
    )


def _detect_go(root: Path) -> ProjectProfile | None:
    if not (root / "go.mod").exists():
        return None
    return ProjectProfile(
        project_type="go",
        languages=["go"],
        package_manager="go modules",
        build_system="go build",
        test_framework="go test",
        config_files=["go.mod"] if (root / "go.mod").is_file() else [],
        source_dirs=_existing_dirs(root, ["cmd", "internal", "pkg"]),
        test_dirs=[],
    )


def _detect_java(root: Path) -> ProjectProfile | None:
    gradle = any((root / m).exists() for m in ("build.gradle", "build.gradle.kts", "settings.gradle"))
    if not (root / "pom.xml").exists() and not gradle:
        return None
    config = [m for m in ("pom.xml", "build.gradle", "build.gradle.kts", "settings.gradle") if (root / m).exists()]
    return ProjectProfile(
        project_type="java",
        languages=["java"],
        package_manager="gradle" if gradle else "maven",
        build_system="gradle" if gradle else "maven",
        test_framework="junit",
        config_files=config,
        source_dirs=_existing_dirs(root, ["src/main/java"]),
        test_dirs=_existing_dirs(root, ["src/test/java"]),
    )


# Extensible registry: name -> detector(root: Path) -> ProjectProfile | None.
# Add a new ecosystem by appending a detector here.
PROJECT_DETECTORS: list[Callable[[Path], ProjectProfile | None]] = [
    _detect_java,  # java before generic; most specific markers first
    _detect_python,
    _detect_node,
    _detect_rust,
    _detect_go,
]


def detect_project(root: Path) -> ProjectProfile:
    """Runs the detector registry against *root*; returns a generic profile if none match."""
    for detector in PROJECT_DETECTORS:
        profile = detector(root)
        if profile is not None:
            return profile
    return ProjectProfile(
        project_type="generic",
        languages=[],
        source_dirs=_existing_dirs(root, ["src", "lib", "app", "include"]),
        test_dirs=_existing_dirs(root, ["tests", "test"]),
    )


def _read_small(path: Path, limit: int = 200_000) -> str:
    """Reads a text file (bounded) without raising."""
    try:
        if path.stat().st_size > limit:
            return ""
        return path.read_text(encoding="utf-8", errors="replace")
    except (OSError, ValueError):
        return ""


# ---------------------------------------------------------------------------
# CodingWorkspace
# ---------------------------------------------------------------------------


class CodingWorkspace(BaseModel):
    """Deterministic description of the active coding workspace."""

    cwd: str
    project_root: str
    project_type: str = "unknown"
    languages: list[str] = Field(default_factory=list)
    package_manager: str | None = None
    build_system: str | None = None
    test_framework: str | None = None
    config_files: list[str] = Field(default_factory=list)
    source_dirs: list[str] = Field(default_factory=list)
    test_dirs: list[str] = Field(default_factory=list)
    is_git_repo: bool = False
    git_branch: str | None = None
    git_clean: bool | None = None
    git_status_short: str = ""

    def summary(self) -> str:
        """Compact one-line description for prompt/context injection."""
        parts = [self.project_type]
        if self.languages:
            parts.append(",".join(self.languages))
        bits = []
        if self.package_manager:
            bits.append(f"pkg={self.package_manager}")
        if self.build_system:
            bits.append(f"build={self.build_system}")
        if self.test_framework:
            bits.append(f"tests={self.test_framework}")
        if self.is_git_repo:
            state = "clean" if self.git_clean else "dirty"
            bits.append(f"git={self.git_branch or '?'}:{state}")
        suffix = " (" + ", ".join(bits) + ")" if bits else ""
        return f"WORKSPACE {self.project_root}: {' '.join(parts)}{suffix}"


def _find_project_root(cwd: Path) -> Path:
    """Walks up from *cwd* to the nearest directory containing a project marker."""
    current = cwd
    for _ in range(6):  # bounded walk
        if any((current / m).exists() for m in PROJECT_MARKERS):
            return current
        if current.parent == current:
            break
        current = current.parent
    return cwd


def _git_info(root: Path) -> tuple[bool, str | None, bool | None, str]:
    """Best-effort git state. Never raises; returns (is_repo, branch, clean, short_status)."""
    if not (root / ".git").exists():
        return False, None, None, ""
    try:
        branch_proc = subprocess.run(
            ["git", "-C", str(root), "branch", "--show-current"],
            capture_output=True, text=True, timeout=3, check=False,
        )
        branch = branch_proc.stdout.strip() or None if branch_proc.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        branch = None
    try:
        status_proc = subprocess.run(
            ["git", "-C", str(root), "status", "--short"],
            capture_output=True, text=True, timeout=3, check=False,
        )
        if status_proc.returncode == 0:
            short = status_proc.stdout.strip()
            return True, branch, (short == ""), short[:500]
        return True, branch, None, ""
    except (OSError, subprocess.SubprocessError):
        return True, branch, None, ""


def discover_workspace(cwd: str | None = None) -> CodingWorkspace:
    """
    Discovers the coding workspace for *cwd* (defaults to the process CWD).

    Pure filesystem inspection + read-only git queries — no tools executed,
    no files modified.
    """
    base = Path(cwd).resolve() if cwd else Path.cwd().resolve()
    root = _find_project_root(base)
    profile = detect_project(root)
    is_git, branch, clean, short = _git_info(root)
    return CodingWorkspace(
        cwd=str(base),
        project_root=str(root),
        project_type=profile.project_type,
        languages=profile.languages,
        package_manager=profile.package_manager,
        build_system=profile.build_system,
        test_framework=profile.test_framework,
        config_files=profile.config_files,
        source_dirs=profile.source_dirs,
        test_dirs=profile.test_dirs,
        is_git_repo=is_git,
        git_branch=branch,
        git_clean=clean,
        git_status_short=short,
    )


# ---------------------------------------------------------------------------
# Read-only repository inspection tools (registered for the ReAct agent)
# ---------------------------------------------------------------------------


def _resolve_safe_path(path: str) -> Path | None:
    """Resolves *path* inside ALLOWED_BASE_DIR; returns None when unsafe."""
    try:
        ok, resolved = is_path_safe(path)
    except (OSError, ValueError):
        return None
    return resolved if ok else None


def list_directory(path: str = ".", max_entries: int = 200, max_depth: int = 3) -> str:
    """
    Lists the directory tree at *path* (bounded, deterministic).

    Skips VCS dirs, virtualenvs, node_modules, build artifacts, and hidden
    entries. Returns a formatted tree; errors return an "Error: ..." string
    exactly like the other file tools.
    """
    resolved = _resolve_safe_path(path)
    if resolved is None:
        return "Error: access denied, that directory is outside the allowed project folder."
    if not resolved.exists():
        return f"Error: directory not found at {path}"
    if not resolved.is_dir():
        return f"Error: {path} is not a directory"

    lines: list[str] = []
    count = 0

    def walk(directory: Path, depth: int) -> None:
        nonlocal count
        if depth > max_depth or count >= max_entries:
            return
        try:
            entries = sorted(directory.iterdir(), key=lambda e: e.name.lower())
        except (OSError, ValueError):
            return
        for entry in entries:
            if count >= max_entries:
                return
            name = entry.name
            if entry.is_dir():
                if name in _IGNORED_DIRS or name.startswith("."):
                    continue
                lines.append("  " * depth + f"{name}/")
                count += 1
                walk(entry, depth + 1)
            elif not name.startswith("."):
                lines.append("  " * depth + name)
                count += 1

    walk(resolved, 0)
    if not lines:
        return "(empty directory)"
    return "\n".join(lines)


def search_files(query: str, path: str = ".", max_results: int = 30) -> str:
    """
    Searches *path* for files whose name or content matches *query*.

    Bounded and deterministic: only text files up to a size limit are
    scanned, matches are capped at *max_results*, and results are reported
    as ``relative/path:line: content`` lines. Errors return an "Error: ..."
    string.
    """
    if not query or not query.strip():
        return "Error: search_files requires a non-empty 'query'."
    resolved = _resolve_safe_path(path)
    if resolved is None:
        return "Error: access denied, that directory is outside the allowed project folder."
    if not resolved.is_dir():
        return f"Error: {path} is not a directory"

    needle = query.strip().lower()
    results: list[str] = []
    max_file_size = 512 * 1024  # 512 KB

    def walk(directory: Path) -> None:
        if len(results) >= max_results:
            return
        try:
            entries = sorted(directory.iterdir(), key=lambda e: e.name.lower())
        except (OSError, ValueError):
            return
        for entry in entries:
            if len(results) >= max_results:
                return
            name = entry.name
            if entry.is_dir():
                if name not in _IGNORED_DIRS and not name.startswith("."):
                    walk(entry)
                continue
            if name.startswith("."):
                continue
            rel = str(entry.relative_to(resolved))
            if needle in name.lower():
                results.append(f"{rel}: (filename match)")
                continue
            try:
                if entry.stat().st_size > max_file_size:
                    continue
                content = entry.read_text(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                continue
            for line_no, line in enumerate(content.splitlines(), start=1):
                if needle in line.lower():
                    results.append(f"{rel}:{line_no}: {line.strip()[:200]}")
                    break  # one match line per file

    walk(resolved)
    if not results:
        return f"No matches for '{query}' in {path}."
    return "\n".join(results)


def git_status(cwd: str | None = None) -> str:
    """Returns ``git status --short`` for *cwd* (best-effort, never raises)."""
    root = Path(cwd).resolve() if cwd else Path.cwd().resolve()
    if not (root / ".git").exists():
        return ""
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "status", "--short"],
            capture_output=True, text=True, timeout=3, check=False,
        )
        if proc.returncode != 0:
            return ""
        return proc.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def git_diff(cwd: str | None = None, max_chars: int = 8000) -> str:
    """Returns ``git diff`` for *cwd* (best-effort, bounded, never raises)."""
    root = Path(cwd).resolve() if cwd else Path.cwd().resolve()
    if not (root / ".git").exists():
        return ""
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "diff"],
            capture_output=True, text=True, timeout=3, check=False,
        )
        if proc.returncode != 0:
            return ""
        return proc.stdout[:max_chars]
    except (OSError, subprocess.SubprocessError):
        return ""


def discover_workspace_summary(cwd: str | None = None) -> str:
    """Tool wrapper: discovers the workspace and returns its one-line summary."""
    return discover_workspace(cwd).summary()
