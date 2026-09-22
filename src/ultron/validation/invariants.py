"""Deterministic invariant validators for autonomous agent execution.

Covers canonical invariants:
- INV-001: Workspace confinement
- INV-002: Dependency ordering
- INV-003: Identical failure thrashing
- INV-004: Invalid dependency specification
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

from ultron.validation.model import EnvironmentSnapshot, InvariantResult
from ultron.validation.probes import FilesystemProbe, WorkspaceProbe

# Standard library module names in Python (using sys.stdlib_module_names on py310+)
_STDLIB_NAMES: frozenset[str] = getattr(
    sys,
    "stdlib_module_names",
    frozenset(
        {
            "tkinter",
            "sqlite3",
            "sys",
            "os",
            "re",
            "json",
            "math",
            "asyncio",
            "typing",
            "pathlib",
            "subprocess",
            "time",
            "datetime",
            "collections",
            "itertools",
            "functools",
            "shutil",
            "tempfile",
            "unittest",
            "urllib",
            "http",
            "logging",
            "io",
            "csv",
            "random",
            "hashlib",
            "copy",
            "dataclasses",
            "enum",
            "abc",
        }
    ),
)

# Common CLI patterns that consume a file or configuration artifact
_CONSUMER_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # pip install -r <file>
    (re.compile(r"pip\d*\s+install\s+.*-r\s+([^\s;&|]+)", re.IGNORECASE), "file"),
    # python <file.py>
    (re.compile(r"python\d*\s+([^\s;&|-][^\s;&|]*\.py)", re.IGNORECASE), "file"),
    # pytest <file.py>
    (re.compile(r"pytest\s+([^\s;&|-][^\s;&|]*\.py)", re.IGNORECASE), "file"),
    # node <file.js>
    (re.compile(r"node\s+([^\s;&|-][^\s;&|]*\.[tj]sx?)", re.IGNORECASE), "file"),
    # npm / yarn in directory requires package.json
    (re.compile(r"\b(?:npm|yarn|pnpm)\s+(?:install|run|test|build)\b", re.IGNORECASE), "package.json"),
    # cargo test / cargo build requires Cargo.toml
    (re.compile(r"\bcargo\s+(?:test|build|run)\b", re.IGNORECASE), "Cargo.toml"),
]


def check_workspace_confinement(
    environment: EnvironmentSnapshot | None,
    workspace_root: str | Path | None = None,
) -> InvariantResult:
    """INV-001: Generated artifacts must remain inside the configured workspace."""
    probe = WorkspaceProbe(workspace_root or (environment.workspace_root if environment else None))
    
    paths_to_check: list[str] = []
    if environment:
        paths_to_check.extend(environment.files_created)
        paths_to_check.extend(environment.files_modified)

    is_confined, escaped = probe.check_confinement(paths_to_check)
    if not is_confined:
        return InvariantResult(
            invariant_id="INV-001",
            name="workspace_confinement",
            description="Generated artifacts must remain inside the configured Ultron workspace.",
            severity="critical",
            passed=False,
            failure_reason=f"Artifacts created outside workspace: {escaped}",
            evidence={"escaped_paths": escaped, "workspace_root": str(probe.workspace_root)},
        )
    return InvariantResult(
        invariant_id="INV-001",
        name="workspace_confinement",
        description="Generated artifacts must remain inside the configured Ultron workspace.",
        severity="critical",
        passed=True,
        evidence={"checked_count": len(paths_to_check), "workspace_root": str(probe.workspace_root)},
    )


def check_dependency_ordering(
    history: list[dict[str, Any]],
    workspace_root: str | Path | None = None,
) -> InvariantResult:
    """INV-002: A command consuming an artifact must not execute before that artifact exists."""
    root = Path(workspace_root or ".").resolve()
    
    for idx, entry in enumerate(history):
        cmd = entry.get("command", "")
        cwd = Path(entry.get("cwd", root)).resolve()
        exit_code = entry.get("exit_code")
        
        # Check if the command matches a consumer pattern
        for pat, target_type in _CONSUMER_PATTERNS:
            match = pat.search(cmd)
            if match:
                if target_type == "file":
                    rel_target = match.group(1).strip("'\"")
                    target_path = cwd / rel_target
                else:
                    target_path = cwd / target_type

                # Check if the file did NOT exist at the time (or exit code indicates file not found)
                created_before = False
                for prev in history[:idx]:
                    if prev.get("tool") in ("create_file", "write_file", "overwrite_file"):
                        prev_path = Path(prev.get("target") or prev.get("file_path", "")).resolve()
                        if prev_path == target_path.resolve():
                            created_before = True
                            break

                output_snippet = entry.get("output_snippet", "") or entry.get("error", "")
                is_missing_error = (
                    "No such file or directory" in output_snippet
                    or "does not exist" in output_snippet
                    or "FileNotFoundError" in output_snippet
                    or "Errno 2" in output_snippet
                )

                if (not created_before and not target_path.exists()) or (exit_code != 0 and is_missing_error):
                    return InvariantResult(
                        invariant_id="INV-002",
                        name="dependency_ordering",
                        description="A command that consumes an artifact must not execute before that artifact exists.",
                        severity="high",
                        passed=False,
                        failure_reason=(
                            f"Command '{cmd}' executed at step {idx + 1} before required target "
                            f"'{target_path.name}' existed on disk."
                        ),
                        evidence={
                            "step_index": idx,
                            "command": cmd,
                            "missing_target": str(target_path),
                            "output_snippet": output_snippet[:200],
                        },
                    )

    return InvariantResult(
        invariant_id="INV-002",
        name="dependency_ordering",
        description="A command that consumes an artifact must not execute before that artifact exists.",
        severity="high",
        passed=True,
    )


def check_failure_thrashing(
    history: list[dict[str, Any]],
    max_identical_retries: int = 2,
) -> InvariantResult:
    """INV-003: Detect repeated execution of the same failed action without meaningful state change."""
    if len(history) < max_identical_retries:
        return InvariantResult(
            invariant_id="INV-003",
            name="failure_thrashing",
            description="Detect repeated execution of the same failed action without meaningful state change.",
            severity="high",
            passed=True,
        )

    consecutive_repeats = 1
    last_sig: tuple[str, str] | None = None

    
    for entry in history:
        is_failure = entry.get("exit_code", 0) != 0 or entry.get("success") is False
        tool = entry.get("tool") or entry.get("tool_name", "")
        arg = entry.get("command") or entry.get("target") or entry.get("file_path", "")
        sig = (tool, str(arg).strip())

        if is_failure:
            if sig == last_sig:
                consecutive_repeats += 1
                if consecutive_repeats >= max_identical_retries:
                    return InvariantResult(
                        invariant_id="INV-003",
                        name="failure_thrashing",
                        description="Detect repeated execution of the same failed action without meaningful state change.",
                        severity="high",
                        passed=False,
                        failure_reason=(
                            f"Repeated identical failure thrashing detected for action {sig[0]}('{sig[1]}') "
                            f"({consecutive_repeats} consecutive attempts with no state change)."
                        ),
                        evidence={
                            "thrashing_action": sig[0],
                            "target": sig[1],
                            "consecutive_count": consecutive_repeats,
                        },
                    )
            else:
                consecutive_repeats = 1
                last_sig = sig
        else:
            consecutive_repeats = 1
            last_sig = None

    return InvariantResult(
        invariant_id="INV-003",
        name="failure_thrashing",
        description="Detect repeated execution of the same failed action without meaningful state change.",
        severity="high",
        passed=True,
    )


def check_invalid_dependency_declarations(
    files: list[str | Path] | None = None,
    commands: list[dict[str, Any]] | None = None,
) -> InvariantResult:
    """INV-004: Detect invalid package declarations (e.g. Python stdlib declared in requirements.txt)."""
    invalid_findings: list[dict[str, Any]] = []

    # Check requirements files
    if files:
        for f in files:
            p = Path(f).resolve()
            if p.is_file() and ("requirements" in p.name.lower() or p.suffix in (".txt", ".pip")):
                content = FilesystemProbe.read_text(p)
                if content:
                    for line in content.splitlines():
                        clean = line.strip().split("#")[0].strip()
                        if not clean or clean.startswith("-"):
                            continue
                        pkg_name = re.split(r"[=<>!~]", clean)[0].strip().lower()
                        if pkg_name in _STDLIB_NAMES:
                            invalid_findings.append({
                                "file": str(p),
                                "package": pkg_name,
                                "reason": f"Standard library module '{pkg_name}' is not an external pip package.",
                            })

    # Check pip install command invocations
    if commands:
        for entry in commands:
            cmd = entry.get("command", "")
            match = re.search(r"pip\d*\s+install\s+([^-][\w\s.,=<>!~-]+)", cmd, re.IGNORECASE)
            if match:
                raw_pkgs = match.group(1).split()
                for raw in raw_pkgs:
                    pkg_name = re.split(r"[=<>!~]", raw.strip())[0].strip().lower()
                    if pkg_name in _STDLIB_NAMES and pkg_name not in ("-r", "--requirement"):
                        invalid_findings.append({
                            "command": cmd,
                            "package": pkg_name,
                            "reason": f"Standard library module '{pkg_name}' passed to pip install.",
                        })

    if invalid_findings:
        return InvariantResult(
            invariant_id="INV-004",
            name="invalid_dependency_declaration",
            description="Detect obviously invalid dependency declarations (such as standard library modules).",
            severity="high",
            passed=False,
            failure_reason=(
                f"Invalid dependency declarations found: {', '.join(f['package'] for f in invalid_findings)}"
            ),
            evidence={"findings": invalid_findings},
        )

    return InvariantResult(
        invariant_id="INV-004",
        name="invalid_dependency_declaration",
        description="Detect obviously invalid dependency declarations (such as standard library modules).",
        severity="high",
        passed=True,
    )


def validate_all_invariants(
    environment: EnvironmentSnapshot | None,
    command_history: list[dict[str, Any]] | None = None,
    workspace_root: str | Path | None = None,
) -> list[InvariantResult]:
    """Runs all 4 canonical invariants deterministically."""
    results = [
        check_workspace_confinement(environment, workspace_root),
        check_dependency_ordering(command_history or (environment.commands_executed if environment else []), workspace_root),
        check_failure_thrashing(command_history or (environment.commands_executed if environment else [])),
        check_invalid_dependency_declarations(
            files=(environment.files_created + environment.files_modified) if environment else None,
            commands=command_history or (environment.commands_executed if environment else []),
        ),
    ]
    return results
