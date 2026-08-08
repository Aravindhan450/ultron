"""ultron.core.intelligence.debug_context
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Environmental-state debugging.

When a user asks Ultron to debug failing code, this module produces a report
that couples a **deterministic failure diagnosis** with the **exact
environmental state** the failure happened in: OS version, Python runtime,
installed library versions, and the declared-versus-installed dependency
picture. The goal is to turn "there is a traceback" into "this fails because
``pandas`` is declared in pyproject.toml as ``pandas>=2.0`` but is not
installed" — the kind of answer that makes fixes fast.

Everything here is read-only and in-process (or short-timeout ``--version``
probes); nothing is fabricated and nothing is guessed from memory.

- :func:`capture_environment` — OS / Python / tool versions / packages /
  declared requirements snapshot.
- :func:`format_environment` — human-readable rendering of the snapshot.
- :func:`diagnose_failure` — classifies a command result (exit code, stderr
  patterns, pytest summaries) into a cause + suggested fix.
- :func:`check_dependency` — is a package installed, at what version, and does
  it match its declared requirement.
- :func:`format_debug_report` — the full debug report shown to the user.
"""

from __future__ import annotations

import importlib.metadata
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Environment snapshot
# ---------------------------------------------------------------------------

# Curated packages always shown in the report — the ones most often implicated
# in "works on my machine" failures. Everything else is summarized by count.
_INTERESTING_PACKAGES = (
    "numpy", "pandas", "requests", "httpx", "flask", "fastapi", "django",
    "pydantic", "sqlalchemy", "pytest", "ruff", "black", "mypy", "boto3",
    "torch", "tensorflow", "scikit-learn", "matplotlib", "beautifulsoup4",
    "questionary", "prompt_toolkit", "rich", "ddgs",
)

# Read-only version probes; a failure just omits the tool from the report.
_TOOL_PROBES = (
    ("git", ("git", "--version")),
    ("ruff", ("ruff", "--version")),
    ("pytest", ("pytest", "--version")),
    ("node", ("node", "--version")),
    ("npm", ("npm", "--version")),
    ("gcc", ("gcc", "--version")),
)

_REQUIREMENT_FILES = ("requirements.txt", "requirements-dev.txt")


def _probe_version(argv: tuple[str, ...]) -> str | None:
    """Runs a short ``--version`` probe; returns the first line or None."""
    try:
        proc = subprocess.run(
            list(argv),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=3,
            check=False,
        )
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    first = (proc.stdout or "").strip().splitlines()
    return first[0].strip() if first else None


def _installed_distributions() -> dict[str, str]:
    """Maps normalized package names to installed versions (best-effort)."""
    result: dict[str, str] = {}
    try:
        for dist in importlib.metadata.distributions():
            name = (dist.metadata.get("Name") or "").strip().lower()
            version = (dist.metadata.get("Version") or "").strip()
            if name:
                result[name] = version
    except Exception:  # noqa: BLE001 — metadata is best-effort
        return result
    return result


def _declared_requirements() -> list[tuple[str, str, str]]:
    """
    Reads declared requirements from ``pyproject.toml`` and requirement files.

    Returns a list of ``(name, spec, source)`` tuples — e.g.
    ``("pandas", ">=2.0", "pyproject.toml")``. Names are normalized to
    lowercase; specifiers are the raw strings after the name.
    """
    declared: list[tuple[str, str, str]] = []

    def _split_req(line: str) -> tuple[str, str] | None:
        """Splits a requirement line into (name, specifier).

        Handles the standard PEP 508 space-less style (``pandas>=2.0``) as
        well as spaced style, strips trailing ``# comment``s and environment
        markers (``; python_version<...``), and rejects optionals/extras
        gracefully (``requests[extra]>=2`` keeps name ``requests``).
        """
        line = line.split("#", 1)[0].strip()
        match = re.match(r"^([a-zA-Z0-9_.-]+)\s*(.*)$", line)
        if not match:
            return None
        name = match.group(1).lower()
        spec = re.split(r"[;]", match.group(2), maxsplit=1)[0].strip()
        return name, spec

    pyproject = Path.cwd() / "pyproject.toml"
    if pyproject.is_file():
        try:
            import tomllib

            data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
            deps = data.get("project", {}).get("dependencies", []) or []
        except (OSError, ValueError, ImportError):
            deps = []
        for dep in deps:
            if isinstance(dep, str) and dep.strip():
                parsed = _split_req(dep.strip())
                if parsed:
                    declared.append((parsed[0], parsed[1], "pyproject.toml"))

    for reqfile in _REQUIREMENT_FILES:
        path = Path.cwd() / reqfile
        if not path.is_file():
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            line = line.strip()
            if not line or line.startswith(("#", "-")):
                continue
            parsed = _split_req(line)
            if parsed:
                declared.append((parsed[0], parsed[1], reqfile))

    return declared


def _parse_version(value: str) -> tuple[int, ...]:
    """Splits a version string into a comparable tuple of ints."""
    parts: list[int] = []
    for token in re.split(r"[^\d]+", value):
        if token.isdigit():
            parts.append(int(token))
        if len(parts) >= 4:
            break
    return tuple(parts) or (0,)


def _spec_satisfied(version: str, spec: str) -> bool | None:
    """
    Checks ``version`` against a simple PEP-440-ish specifier.

    Supports the common operators (>= <= > < == != ~=) on numeric versions.
    Returns None when the spec cannot be parsed — the caller then skips the
    mismatch flag instead of guessing.
    """
    spec = spec.strip()
    if not spec:
        return True

    def _single(part: str) -> bool | None:
        match = re.match(
            r"^(>=|<=|>|<|==|!=|~=)?\s*(v?\d[\w.]*)$", part, re.IGNORECASE
        )
        if not match:
            return None
        op = match.group(1) or "=="
        target = match.group(2).lstrip("v")
        try:
            got, want = _parse_version(version), _parse_version(target)
        except (TypeError, ValueError):
            return None
        if op == ">=":
            return got >= want
        if op == "<=":
            return got <= want
        if op == ">":
            return got > want
        if op == "<":
            return got < want
        if op == "!=":
            return got != want
        if op == "~=":
            return got[: len(want)] == want if want else None
        return got == want

    # Comma-separated specifiers (``>=2.0,<3.0``): every part must hold;
    # an unparseable part makes the whole check unknown (None).
    results = [_single(part.strip()) for part in spec.split(",")]
    if any(result is None for result in results):
        return None
    return all(results)


def capture_environment() -> dict:
    """
    Captures the full environmental state relevant to debugging.

    Returns a dict with keys: ``os`` (dict), ``python`` (dict), ``cwd``,
    ``tools`` (dict), ``packages`` (dict of interesting packages),
    ``package_count``, and ``requirements`` (list of declared-requirement
    dicts annotated with ``installed`` and ``ok``). Every value is read from
    the live process — nothing is fabricated.
    """
    installed = _installed_distributions()

    os_info = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "platform": platform.platform(),
    }

    python_info = {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "executable": sys.executable,
    }

    tools: dict[str, str] = {}
    for label, argv in _TOOL_PROBES:
        version = _probe_version(argv)
        if version:
            tools[label] = version

    interesting = {name: installed[name] for name in _INTERESTING_PACKAGES if name in installed}

    requirements: list[dict] = []
    for name, spec, source in _declared_requirements():
        entry: dict = {
            "name": name,
            "spec": spec,
            "source": source,
            "installed": installed.get(name),
        }
        if spec:
            ok = _spec_satisfied(entry["installed"] or "", spec)
            entry["ok"] = ok if ok is not None else True
        else:
            entry["ok"] = bool(entry["installed"])
        requirements.append(entry)

    return {
        "os": os_info,
        "python": python_info,
        "cwd": os.getcwd(),
        "tools": tools,
        "packages": interesting,
        "package_count": len(installed),
        "requirements": requirements,
    }


def get_debug_context() -> str:
    """
    Tool entry point: captures and renders the full environment snapshot.

    Returns the formatted environment block (OS, Python, tool versions,
    installed packages, declared-but-missing requirements).
    """
    return format_environment(capture_environment())


def format_environment(env: dict | None = None) -> str:
    """Renders :func:`capture_environment` output as a readable block."""
    env = env or capture_environment()
    os_info, python_info = env["os"], env["python"]
    lines = [
        "🌍 Environment",
        f"  OS       {os_info['system']} {os_info['release']} ({os_info['machine']})",
        f"  Python   {python_info['version']} · {python_info['implementation']}",
        f"  CWD      {env['cwd']}",
    ]

    tools = env["tools"]
    if tools:
        lines.append("  Tools    " + ", ".join(f"{k} {v}" for k, v in sorted(tools.items())))

    packages = env["packages"]
    pkg_line = ", ".join(f"{k} {v}" for k, v in sorted(packages.items()))
    lines.append(f"  Packages {pkg_line or '(none of the common ones installed)'} ({env['package_count']} total)")

    missing = [r for r in env["requirements"] if not r["installed"]]
    mismatched = [r for r in env["requirements"] if r["installed"] and not r["ok"]]
    for req in missing:
        spec_note = f" ({req['source']}: {req['name']}{req['spec']})"
        lines.append(f"  ⚠ Declared-but-missing: {req['name']}{spec_note}")
    for req in mismatched:
        lines.append(
            f"  ⚠ Version mismatch: {req['name']} declared {req['spec']}, "
            f"installed {req['installed']}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Failure diagnosis
# ---------------------------------------------------------------------------

_CAUSE_MATRIX: list[tuple[re.Pattern[str], str, str]] = [
    (re.compile(r"ModuleNotFoundError:\s*No module named ['\"]([^'\"]+)['\"]", re.IGNORECASE),
     "missing_dependency", "a module the code imports is not installed"),
    (re.compile(r"ImportError:", re.IGNORECASE),
     "missing_dependency", "an import failed (missing or broken module)"),
    (re.compile(r"SyntaxError", re.IGNORECASE),
     "syntax_error", "the code does not parse — check the flagged line and column"),
    (re.compile(r"NameError:\s*name ['\"]([^'\"]+)['\"]", re.IGNORECASE),
     "name_error", "a variable or function is used before it exists (typo?)"),
    (re.compile(r"FileNotFoundError", re.IGNORECASE),
     "missing_file", "the code tried to open a file that does not exist"),
    (re.compile(r"No such file or directory", re.IGNORECASE),
     "missing_file", "a path referenced by the command does not exist"),
    (re.compile(r"Permission denied", re.IGNORECASE),
     "permission", "the process lacks permission for a file or directory"),
    (re.compile(r"command not found", re.IGNORECASE),
     "command_not_found", "the shell cannot find the executable (install it or fix PATH)"),
    (re.compile(r"ConnectionRefusedError|Connection refused|network is unreachable|ConnectionError", re.IGNORECASE),
     "network", "a network connection failed — check the endpoint/URL and connectivity"),
    (re.compile(r"timed out after \d+ seconds", re.IGNORECASE),
     "timeout", "the command ran past its time limit — split it up or raise the limit"),
]


def _extract_traceback_modules(text: str) -> list[str]:
    """Collects top-level package names referenced in traceback paths."""
    modules: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(r"site-packages[/\\]([a-zA-Z0-9_.]+)", text):
        name = match.group(1).split(".")[0]
        if name not in seen:
            seen.add(name)
            modules.append(name)
    return modules


def diagnose_failure(text: str, command: str | None = None) -> dict:
    """
    Classifies a command result into a structured diagnosis.

    Accepts the same string ``run_command`` returns (with "Exit code: N",
    "Output:", "Error Output:" sections) or a bare error/traceback. Returns a
    dict with ``exit_code``, ``stdout_tail``, ``stderr_tail``, ``cause``,
    ``probable_cause``, ``suggested_fix``, ``modules``, and (for pytest output)
    ``passed``/``failed`` counts. ``cause`` is always one of the matrix keys
    above or ``"unknown"``.
    """
    text = str(text or "")
    lower = text.lower()

    # --- pytest summary beats generic classification when present, but ONLY
    # for actual failures — a passing run prints "0 failed" in its summary
    # line, which must never be diagnosed as a test failure.
    summary = re.search(r"(\d+)\s+failed", lower)
    if summary and "====" in lower and int(summary.group(1)) > 0:
        passed = re.search(r"(\d+)\s+passed", lower)
        return {
            "exit_code": 1,
            "stdout_tail": text[-800:],
            "stderr_tail": "",
            "cause": "tests_failed",
            "probable_cause": "a test assertion failed",
            "suggested_fix": "open the failing test and the asserted value",
            "modules": _extract_traceback_modules(text),
            "passed": int(passed.group(1)) if passed else 0,
            "failed": int(summary.group(1)),
            "full_output": text,
        }

    # --- timeout is produced by our own runner ---
    if "timed out after" in lower:
        return {
            "exit_code": None,
            "stdout_tail": text[-800:],
            "stderr_tail": "",
            "cause": "timeout",
            "probable_cause": "the command exceeded its time limit",
            "suggested_fix": "split the work into smaller commands or raise the timeout",
            "modules": [],
            "full_output": text,
        }

    exit_code: int | None = None
    code_match = re.search(r"Exit code:\s*(\d+)", text)
    if code_match:
        exit_code = int(code_match.group(1))

    stdout_tail = stderr_tail = ""
    if "Error Output:" in text:
        stdout_tail, _, stderr_tail = text.partition("Error Output:")
        stdout_tail = stdout_tail.strip()[-800:]
        stderr_tail = stderr_tail.strip()[-800:]
    else:
        stdout_tail = text.strip()[-800:]

    cause = "unknown"
    probable_cause = "the output does not match a known failure pattern"
    suggested_fix = "read the output below — or paste the traceback for a precise diagnosis"
    module_name: str | None = None

    for pattern, matched_cause, description in _CAUSE_MATRIX:
        match = pattern.search(text)
        if match:
            cause = matched_cause
            probable_cause = description
            module_name = match.group(1) if match.groups() else None
            break

    if cause == "missing_dependency" and module_name:
        suggested_fix = f"pip install {module_name}"
        # Prefer the declared version when we can see one.
        declared = [r for r in _declared_requirements() if r[0] == module_name.lower()]
        if declared:
            suggested_fix += f"   (declared: {module_name}{declared[0][1]})"
    elif cause == "syntax_error":
        suggested_fix = "fix the syntax at the flagged line:column and re-run"
    elif cause == "name_error" and module_name:
        suggested_fix = f"define or correctly spell '{module_name}' before it is used"
    elif cause == "missing_file":
        suggested_fix = "verify the path exists and is spelled correctly"
    elif cause == "permission":
        suggested_fix = "check file/directory permissions or run with the right user"
    elif cause == "command_not_found":
        suggested_fix = "install the tool (e.g. pip install) or add it to PATH"
    elif cause == "network":
        suggested_fix = "verify the URL/endpoint and your connectivity"

    return {
        "exit_code": exit_code,
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
        "cause": cause,
        "probable_cause": probable_cause,
        "suggested_fix": suggested_fix,
        "modules": _extract_traceback_modules(text),
        "full_output": text,
    }


# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------

def check_dependency(name: str) -> str:
    """
    Reports whether a package is installed, at what version, and whether that
    satisfies any declared requirement (pyproject.toml / requirements files).
    """
    name = (name or "").strip().lower()
    if not name:
        return "Error: check_dependency expects a package name."

    installed = _installed_distributions()
    version = installed.get(name)

    declared = [r for r in _declared_requirements() if r[0] == name]
    declared_note = ""
    if declared:
        spec = declared[0][1]
        source = declared[0][2]
        declared_note = f" — declared {name}{spec} in {source}"

    if not version:
        return (
            f"'{name}' is NOT installed.{declared_note} "
            f"Install it with: pip install {name}"
        )
    if declared and declared[0][1]:
        ok = _spec_satisfied(version, declared[0][1])
        if ok is False:
            return (
                f"'{name}' {version} is installed but does NOT satisfy the "
                f"declared requirement {name}{declared[0][1]} "
                f"(from {declared[0][2]}). Upgrade with: pip install -U {name}"
            )
    return f"'{name}' {version} is installed.{declared_note}"


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------

def format_debug_report(
    command: str | None = None,
    diagnosis: dict | None = None,
    expected: str | None = None,
    env: dict | None = None,
    note: str | None = None,
) -> str:
    """
    Assembles the full debug report: diagnosis + expected-vs-actual + the
    environmental snapshot. Every section is built from real data.
    """
    env = env or capture_environment()
    lines: list[str] = ["🔍 Debug report"]
    if command:
        lines.append(f"Command: {command}")
    lines.append("─" * 40)

    if diagnosis:
        exit_code = diagnosis.get("exit_code")
        code_str = f"Exit code: {exit_code}" if exit_code is not None else "Exit code: —"
        lines.append(f"🏷 {code_str}")
        cause = diagnosis.get("cause", "unknown")
        lines.append(f"📋 Diagnosis: {cause} — {diagnosis.get('probable_cause', '')}")
        lines.append(f"💡 Suggested fix: {diagnosis.get('suggested_fix', '')}")

        if expected is not None and expected.strip():
            # Match against the FULL output when available so a satisfied
            # expectation outside the truncated tail is not mis-reported.
            full = str(diagnosis.get("full_output") or "")
            haystack = full or "\n".join(
                [
                    str(diagnosis.get("stdout_tail", "")),
                    str(diagnosis.get("stderr_tail", "")),
                ]
            )
            satisfied = expected.strip().lower() in haystack.lower()
            marker = "satisfied" if satisfied else "not satisfied"
            lines.append(f"\nExpected: \"{expected.strip()}\" → {marker}")
    elif note:
        lines.append(f"📋 {note}")

    lines.append("")
    lines.append(format_environment(env))

    if diagnosis:
        tail = diagnosis.get("stderr_tail") or diagnosis.get("stdout_tail")
        if tail:
            lines.append("\nError output (tail):")
            lines.append("  " + tail.replace("\n", "\n  "))

    return "\n".join(lines)
