"""ultron.core.nlp.project
~~~~~~~~~~~~~~~~~~~~~~~~~~

Natural-language → project-specific command discovery.

``discover_project_command()`` maps a semantic request ("run the tests",
"build the project", "start the backend") to the *actual* command supported
by the repository, discovered from its configuration — never invented.

``resolve_test_command()`` is the project-aware test-command resolver: it
returns structured data (framework / environment / executable / arguments /
working directory / source) so callers can run *targeted* tests inside the
project's configured environment (virtualenv, poetry, uv, …).

Rules:

- Reuses ``ultron.core.coding.workspace.detect_project`` so there is a single
  source of truth for project detection.
- Returns None when the project does not provide evidence for the requested
  action (e.g. a Python project with no test framework → no test command).
- The caller decides what to do with None (fall back to a default, or ask).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Test-command resolution (structured)
# ---------------------------------------------------------------------------


@dataclass
class ResolvedTestCommand:
    """A project-aware test command, resolved from repository evidence."""

    framework: str
    environment: str | None  # "venv" | "poetry" | "uv" | None
    executable: str
    arguments: list[str]
    working_directory: str
    source: str  # where the evidence came from (pyproject.toml / package.json …)

    def command(self, targets: list[str] | None = None) -> str:
        """Joins executable + arguments + optional targeted test paths."""
        parts = [self.executable, *self.arguments]
        if targets:
            parts.extend(targets)
        return " ".join(parts)


def _find_venv_python(root: Path) -> str | None:
    """Returns the venv's python when the project uses a virtualenv."""
    for name in (".venv", "venv"):
        candidate = root / name / "bin" / "python"
        if candidate.exists():
            return str(candidate)
    return None


def _python_test_framework(profile, root: Path) -> str | None:
    """pytest / unittest / None for a python project (best-effort)."""
    if profile.test_framework in {"pytest", "unittest"}:
        return profile.test_framework
    if (root / "pyproject.toml").exists() or (root / "pytest.ini").exists():
        return "pytest"
    return None


def resolve_test_command(cwd: str | None = None) -> ResolvedTestCommand | None:
    """
    Resolves the project's test command with its environment, or None when the
    repository gives no evidence for a test framework.

    Python projects get their virtualenv (``.venv`` / ``venv``) or tool-runner
    (poetry / uv) resolved so bare ``pytest`` is never assumed on PATH.
    """
    root = Path(cwd or ".").resolve()
    from ultron.core.coding.workspace import detect_project

    profile = detect_project(root)

    if profile.project_type == "python":
        framework = _python_test_framework(profile, root)
        if framework is None:
            return None
        env_python = _find_venv_python(root)
        if env_python:
            return ResolvedTestCommand(
                framework=framework,
                environment="venv",
                executable=env_python,
                arguments=["-m", framework],
                working_directory=str(root),
                source="detected virtualenv",
            )
        if (root / "poetry.lock").exists():
            return ResolvedTestCommand(
                framework=framework, environment="poetry", executable="poetry",
                arguments=["run", framework], working_directory=str(root),
                source="poetry.lock",
            )
        if (root / "uv.lock").exists():
            return ResolvedTestCommand(
                framework=framework, environment="uv", executable="uv",
                arguments=["run", framework], working_directory=str(root),
                source="uv.lock",
            )
        return ResolvedTestCommand(
            framework=framework, environment=None, executable="python",
            arguments=["-m", framework], working_directory=str(root),
            source="pyproject.toml / detected framework",
        )

    if profile.project_type == "node":
        package = _read_package_json(root)
        scripts = package.get("scripts", {}) if package else {}
        if "test" in scripts or package is not None:
            return ResolvedTestCommand(
                framework="npm test", environment=None, executable="npm",
                arguments=["test"], working_directory=str(root),
                source="package.json",
            )
        return None

    if profile.project_type == "go":
        return ResolvedTestCommand(
            framework="go test", environment=None, executable="go",
            arguments=["test", "./..."], working_directory=str(root),
            source="go.mod",
        )

    if profile.project_type == "rust":
        return ResolvedTestCommand(
            framework="cargo test", environment=None, executable="cargo",
            arguments=["test"], working_directory=str(root),
            source="Cargo.toml",
        )

    return None


# ---------------------------------------------------------------------------
# Per-ecosystem command tables (keyed by ProjectProfile.project_type)
# ---------------------------------------------------------------------------

_PYTHON_COMMANDS: dict[str, str] = {
    "test": "pytest -v",
    "build": "python -m build",
    "lint": "ruff check . --output-format=concise",
    "typecheck": "mypy .",
    "format": "black --check .",
    "start": "uvicorn app.main:app --reload",
    "stop": "",
}

_NODE_COMMANDS: dict[str, str] = {
    "test": "npm test",
    "build": "npm run build",
    "lint": "npm run lint",
    "typecheck": "npx tsc --noEmit",
    "format": "npm run format",
    "start": "npm start",
    "stop": "",
}

_GO_COMMANDS: dict[str, str] = {
    "test": "go test ./...",
    "build": "go build ./...",
    "lint": "gofmt -l .",
    "typecheck": "go vet ./...",
    "format": "gofmt -w .",
    "start": "",
    "stop": "",
}

_RUST_COMMANDS: dict[str, str] = {
    "test": "cargo test",
    "build": "cargo build",
    "lint": "cargo clippy -- -D warnings",
    "typecheck": "cargo check",
    "format": "cargo fmt -- --check",
    "start": "cargo run",
    "stop": "",
}

_ECOSYSTEM_COMMANDS: dict[str, dict[str, str]] = {
    "python": _PYTHON_COMMANDS,
    "node": _NODE_COMMANDS,
    "go": _GO_COMMANDS,
    "rust": _RUST_COMMANDS,
}

# A profile supports an action only when its markers exist — the command
# tables above are *candidates*, and these gate them.
_TEST_FRAMEWORKS = {"pytest", "jest", "mocha", "vitest", "cargo test", "go test"}
_NODE_LINT_SCRIPTS = {"lint", "eslint", "tslint"}
_NODE_FORMAT_SCRIPTS = {"format", "prettier"}
_NODE_BUILD_SCRIPTS = {"build", "compile", "tsc", "vite build", "next build"}
_NODE_START_SCRIPTS = {"start", "dev", "serve"}


def resolve_explicit_test_command(command: str, cwd: str | None = None) -> str:
    """
    Rewrites an explicit test command to use the project's environment.

    "Run pytest tests/test_api.py" extracted ``pytest tests/test_api.py`` —
    when the project has a virtualenv, this becomes
    ``.venv/bin/python -m pytest tests/test_api.py`` so a bare ``pytest`` on
    PATH is never assumed.  Commands that are not pytest-shaped or projects
    without a venv are returned unchanged.
    """
    if not command or not command.strip():
        return command
    cmd = command.strip()
    root = Path(cwd or ".").resolve()
    env_python = _find_venv_python(root)
    if not env_python:
        return command
    m = re.match(r"^(?:python(?:3)?\s+-m\s+pytest|pytest)(\s+.*)?$", cmd)
    if not m:
        return command
    rest = (m.group(1) or "").strip()
    return f"{env_python} -m pytest {rest}".strip()


def discover_project_command(what: str, cwd: str | None = None) -> str | None:
    """
    Returns the project-specific command for a semantic request, or None when
    the repository gives no evidence for it.

    ``what`` is one of: ``test``, ``build``, ``lint``, ``typecheck``,
    ``format``, ``start``, ``stop``.
    """
    root = Path(cwd or ".").resolve()
    from ultron.core.coding.workspace import detect_project

    profile = detect_project(root)
    table = _ECOSYSTEM_COMMANDS.get(profile.project_type)
    if not table:
        return None
    candidate = table.get(what)
    if not candidate:
        return None

    # Gate each candidate on actual project evidence.
    if profile.project_type == "python":
        if what == "test" and profile.test_framework not in {"pytest", "unittest"}:
            return None
        if what == "start" and not any(
            (root / f).exists() for f in ("app/main.py", "main.py", "app.py", "manage.py")
        ):
            return None
        if what == "stop":
            return None
        return candidate

    if profile.project_type == "node":
        package = _read_package_json(root)
        scripts = package.get("scripts", {}) if package else {}
        if what in ("lint", "typecheck", "format", "build", "start") and what in scripts:
            # Prefer the named npm script when it exists.
            return f"npm run {what}" if what != "start" else "npm start"
        if what == "test":
            if "test" in scripts:
                return "npm test"
            return "npm test" if "test" in scripts else None
        # Fall back to framework defaults only with supporting deps.
        deps = {
            *(package.get("dependencies") or {}),
            *(package.get("devDependencies") or {}),
        }
        if what == "lint" and any(d in deps for d in ("eslint", "tslint", "biome")):
            return "npx eslint ."
        if what == "typecheck" and "typescript" in deps:
            return "npx tsc --noEmit"
        if what == "build" and any(d in deps for d in ("typescript", "vite", "webpack", "next")):
            return "npm run build"
        if what == "format" and any(d in deps for d in ("prettier", "biome")):
            return "npx prettier --check ."
        return candidate

    # Go / Rust: table commands are already framework-accurate.
    return candidate


def _read_package_json(root: Path) -> dict | None:
    try:
        import json

        with (root / "package.json").open(encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None
