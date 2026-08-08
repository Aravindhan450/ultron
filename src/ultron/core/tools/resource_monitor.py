"""ultron.core.tools.resource_monitor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Resource constraint awareness for shell commands.

Three capabilities, all stdlib-first with optional psutil acceleration:

1. **System snapshot** — ``check_resources()`` reports CPU cores + load,
   memory used/total, and (when psutil is present) live CPU percent.
2. **Run measurement** — helpers for ``command_runner`` to measure a child
   process's wall time, CPU time (``getrusage`` deltas) and peak RSS (live
   sampling of the child pid via psutil / ``/proc`` / ``ps``).
3. **Forecast** — ``forecast_command()`` predicts a command's resource
   profile from static command-family patterns plus the measured history of
   previous runs (SQLite). ``resource_forecast()`` is the registered tool.

Everything degrades gracefully: a missing psutil, kernel, or DB only makes
the report smaller — it never breaks a command. See docs/resources.md.
"""

from __future__ import annotations

import os
import re
import sqlite3
import subprocess
import sys
import threading

from ultron.core.tools.paths import ALLOWED_BASE_DIR

# POSIX-only; on Windows these signals simply report 0 / are omitted.
try:
    import resource
except ImportError:  # pragma: no cover — Windows
    resource = None  # type: ignore[assignment]

# Optional psutil gives richer live data; stdlib fallbacks cover the rest.
try:
    import psutil  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover — psutil is optional
    psutil = None

# Path to the run-history database. Tests may repoint this at a temp file.
RESOURCES_DB_PATH = ALLOWED_BASE_DIR / ".ultron_resources.db"

# Severity ladder, from quiet to dangerous-for-the-machine.
LIGHT = "light"
MODERATE = "moderate"
HEAVY = "heavy"
CRITICAL = "critical"

_SEVERITY_RANK = {LIGHT: 0, MODERATE: 1, HEAVY: 2, CRITICAL: 3}

# Thresholds used to classify measured/forecast runs.
_HEAVY_DURATION_S = 60.0
_CRITICAL_DURATION_S = 300.0
_HEAVY_PEAK_MB = 1536.0  # 1.5 GB
_CRITICAL_PEAK_MB = 4096.0  # 4 GB

# ---------------------------------------------------------------------------
# Static command-family profiles: regex -> (duration_s, peak_mb, severity)
# ---------------------------------------------------------------------------
_HEAVY_COMMAND_PATTERNS: list[tuple[str, str, float, float, str]] = [
    # (name, regex, expected_duration_s, expected_peak_mb, severity)
    ("pip_install", r"\b(?:pip|pip3|pipx|uv\s+pip|poetry)\s+install\b", 120.0, 600.0, HEAVY),
    ("node_install", r"\b(?:npm|yarn|pnpm|bun)\s+(?:install|ci|add|\bi\b)\b", 90.0, 800.0, HEAVY),
    ("docker_build", r"\bdocker\s+(?:build|compose\s+up|pull|push)\b", 180.0, 1500.0, HEAVY),
    ("full_fs_scan", r"\bfind\s+/", 300.0, 300.0, CRITICAL),
    ("recursive_grep_root", r"\bgrep\s+-r\s+/", 240.0, 600.0, HEAVY),
    ("compile", r"\b(?:make|cmake\s+--build|gcc|clang|clang\+\+|rustc|cargo\s+build|go\s+build)\b", 120.0, 900.0, HEAVY),
    ("db_dump", r"\b(?:pg_dump|mysqldump|sqlite3\b.*\.dump)\b", 60.0, 800.0, MODERATE),
    ("git_clone", r"\bgit\s+clone\b", 45.0, 400.0, MODERATE),
    ("big_download", r"\b(?:curl|wget)\b.*(?:-o\s+\S+|>|--output)", 45.0, 300.0, MODERATE),
    ("dd_write", r"\bdd\b", 120.0, 100.0, HEAVY),
    ("test_suite", r"\bpytest\b", 60.0, 500.0, MODERATE),
    # The interpreter must be the FIRST token (optionally via path/sudo/env)
    # — a bare mention of the word is not a script run.
    (
        "script_run",
        r"^(?:\S*/)?(?:sudo\s+)?(?:env\s+\S+\s+)*(?:python3?|node|ruby|perl)(?:\s|$)",
        20.0,
        250.0,
        MODERATE,
    ),
]


def _command_family(command: str) -> str:
    """
    Reduces a shell command to a coarse family name used to share history.

    Strips `sudo` / env-prefix sugar and maps common launchers onto the
    real tool (``python3 -m pytest`` -> ``pytest``).
    """
    cmd = (command or "").strip()
    cmd = re.sub(r"^\s*(?:sudo\s+)?(?:env\s+[A-Za-z_][A-Za-z0-9_]*=\S+\s+)*", "", cmd)
    first = cmd.split(None, 1)[0].split("/")[-1] if cmd.split(None, 1) else ""
    lowered = cmd.lower()
    if first in {"python", "python3", "python3.11", "python3.12"}:
        if " -m pytest" in lowered:
            return "pytest"
        if " -m " in lowered:
            module_match = re.search(r"-m\s+(\S+)", lowered)
            if module_match:
                return f"{first} -m {module_match.group(1)}"
            return first
        return first
    # Package managers get finer families ("pip install" vs "pip list") so
    # one heavy install never taints unrelated subcommands of the same tool.
    if first in {"pip", "pip3", "pipx", "npm", "yarn", "pnpm", "bun"}:
        parts = cmd.split(None, 2)
        verb = parts[1] if len(parts) > 1 else ""
        return f"{first} {verb}".strip()
    return first


# ---------------------------------------------------------------------------
# History store
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    family TEXT NOT NULL,
    command TEXT NOT NULL,
    duration REAL NOT NULL,
    peak_mb REAL NOT NULL DEFAULT 0,
    cpu_seconds REAL NOT NULL DEFAULT 0,
    exit_ok INTEGER NOT NULL DEFAULT 1,
    ran_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_runs_family ON runs (family, id DESC);
"""


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(RESOURCES_DB_PATH))
    conn.executescript(_SCHEMA_SQL)
    return conn


def record_run(
    command: str,
    duration: float,
    peak_mb: float = 0.0,
    cpu_seconds: float = 0.0,
    exit_ok: bool = True,
) -> None:
    """Stores one measured run for future forecasts. Best-effort."""
    try:
        with _connect() as conn:
            conn.execute(
                "INSERT INTO runs (family, command, duration, peak_mb, cpu_seconds, exit_ok) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (_command_family(command), command, duration, peak_mb, cpu_seconds, int(exit_ok)),
            )
            # Bound the store: keep the newest 50 runs per family so a single
            # anomalous run ages out and the table never grows unboundedly.
            conn.execute(
                "DELETE FROM runs WHERE id NOT IN "
                "(SELECT id FROM runs WHERE family=? ORDER BY id DESC LIMIT 50)",
                (_command_family(command),),
            )
            conn.commit()
    except (sqlite3.Error, OSError):
        pass


def _recent_runs(family: str, limit: int = 3) -> list[tuple[float, float]]:
    """Returns recent (duration, peak_mb) pairs for a command family."""
    try:
        with _connect() as conn:
            rows = conn.execute(
                "SELECT duration, peak_mb FROM runs WHERE family=? ORDER BY id DESC LIMIT ?",
                (family, limit),
            ).fetchall()
        return [(r[0], r[1]) for r in rows]
    except (sqlite3.Error, OSError):
        return []


# ---------------------------------------------------------------------------
# System snapshot
# ---------------------------------------------------------------------------

def _system_memory() -> tuple[float | None, float | None]:
    """Returns (used_gb, total_gb) or (None, None) when unavailable."""
    if psutil is not None:
        try:
            vm = psutil.virtual_memory()
            return vm.used / 1e9, vm.total / 1e9
        except Exception:  # noqa: BLE001 — psutil failures degrade to fallbacks
            return None, None
    # Linux: /proc/meminfo
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            total = avail = None
            for line in fh:
                if line.startswith("MemTotal:"):
                    total = int(line.split()[1]) / 1024 / 1024
                elif line.startswith("MemAvailable:"):
                    avail = int(line.split()[1]) / 1024 / 1024
            if total and avail is not None:
                return total - avail, total
    except (OSError, ValueError):
        pass
    # macOS: sysctl + vm_stat
    try:
        total = (
            float(
                subprocess.run(
                    ["sysctl", "-n", "hw.memsize"],
                    capture_output=True,
                    text=True,
                    timeout=2,
                    check=False,
                ).stdout.strip()
            )
            / 1e9
        )
        out = subprocess.run(
            ["vm_stat"], capture_output=True, text=True, timeout=2, check=False
        ).stdout
        pages: dict[str, float] = {}
        for line in out.splitlines():
            if ":" in line:
                key, _, val = line.partition(":")
                # vm_stat values may carry units ("page size of 4096 bytes").
                num_match = re.search(r"[\d.]+\b", val)
                if num_match:
                    pages[key.strip()] = float(num_match.group(0))
        page_size = pages.get("page size of", 4096.0)
        free = (pages.get("Pages free", 0.0) + pages.get("Pages speculative", 0.0)) * page_size / 1e9
        return max(0.0, total - free), total
    except Exception:  # noqa: BLE001 — memory stats are best-effort
        return None, None


def system_snapshot() -> dict:
    """Collects a best-effort snapshot of current system resources."""
    used, total = _system_memory()
    load: tuple[float, float, float] | None = None
    try:
        load = os.getloadavg()
    except (OSError, AttributeError):
        load = None

    cpu_percent: float | None = None
    if psutil is not None:
        try:
            cpu_percent = psutil.cpu_percent(interval=0.3)
        except Exception:  # noqa: BLE001 — psutil failures degrade to fallbacks
            cpu_percent = None

    return {
        "cpu_count": os.cpu_count(),
        "load": load,
        "cpu_percent": cpu_percent,
        "memory_used_gb": used,
        "memory_total_gb": total,
        "psutil": psutil is not None,
    }


def check_resources() -> str:
    """
    Registered tool: human-readable system resource snapshot.
    """
    snap = system_snapshot()
    lines = ["System resources:"]
    cpu = snap["cpu_count"]
    if cpu:
        lines.append(f"- CPU: {cpu} core{'s' if cpu != 1 else ''}")
    if snap["load"]:
        lines.append(f"- Load average: {' / '.join(f'{x:.2f}' for x in snap['load'])}")
    if snap["cpu_percent"] is not None:
        lines.append(f"- CPU usage: {snap['cpu_percent']:.0f}%")
    if snap["memory_total_gb"]:
        used = snap["memory_used_gb"] or 0.0
        lines.append(
            f"- Memory: {used:.1f} GB used / {snap['memory_total_gb']:.1f} GB total "
            f"({100.0 * used / snap['memory_total_gb']:.0f}%)"
        )
    else:
        lines.append("- Memory: unavailable on this platform")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Run measurement (used by command_runner)
# ---------------------------------------------------------------------------

def _rss_of_pid(pid: int) -> float | None:
    """Current RSS of *pid* in MB, or None when it cannot be read."""
    if psutil is not None:
        try:
            return psutil.Process(pid).memory_info().rss / (1024 * 1024)
        except Exception:  # noqa: BLE001 — process may have exited
            return None
    # Linux: /proc/<pid>/status
    try:
        with open(f"/proc/{pid}/status", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except (OSError, ValueError):
        pass
    # macOS / other POSIX: ps -o rss= (KB)
    if sys.platform != "win32":
        try:
            out = subprocess.run(
                ["ps", "-o", "rss=", "-p", str(pid)],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            ).stdout.strip()
            if out:
                return float(out) / 1024.0
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    return None


def sample_peak_rss(pid: int, stop: threading.Event, result: dict) -> None:
    """Thread target: track the peak RSS of *pid* until *stop* is set."""
    peak = 0.0
    while not stop.is_set():
        rss = _rss_of_pid(pid)
        if rss is not None:
            peak = max(peak, rss)
        stop.wait(0.1)
    result["peak_mb"] = peak


def child_usage() -> tuple[float, float]:
    """
    Returns cumulative (cpu_seconds, peak_rss_mb) across all child processes
    so far. Callers record the value before and after a run and diff them:
    the CPU delta is accurate, and the peak-RSS delta is a fallback when
    live pid sampling is unavailable. Returns (0, 0) when the platform
    lacks the signal.
    """
    if resource is None:
        return 0.0, 0.0
    try:
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        # ru_maxrss: KB on Linux, bytes on macOS — normalize to MB.
        divisor = 1024.0 if sys.platform != "darwin" else 1024.0 * 1024.0
        return usage.ru_utime + usage.ru_stime, usage.ru_maxrss / divisor
    except (ValueError, OSError):
        return 0.0, 0.0


def format_metrics(
    duration: float,
    peak_mb: float = 0.0,
    cpu_seconds: float = 0.0,
    prefix: str = "[resources]",
) -> str:
    """Formats measured run metrics into the compact report line."""
    parts = [f"{prefix} finished in {fmt_duration(duration)}"]
    if cpu_seconds > 0:
        parts.append(f"CPU {fmt_duration(cpu_seconds)}")
    if peak_mb > 0:
        parts.append(f"peak ~{peak_mb:.0f} MB")
    return " · ".join(parts)


def fmt_duration(seconds: float) -> str:
    if seconds >= 60:
        return f"{seconds / 60:.1f} min"
    return f"{seconds:.2f} s"


# ---------------------------------------------------------------------------
# Forecast engine
# ---------------------------------------------------------------------------

def _severity_for(duration: float | None, peak_mb: float | None) -> str:
    if duration is not None and duration >= _CRITICAL_DURATION_S:
        return CRITICAL
    if peak_mb is not None and peak_mb >= _CRITICAL_PEAK_MB:
        return CRITICAL
    if duration is not None and duration >= _HEAVY_DURATION_S:
        return HEAVY
    if peak_mb is not None and peak_mb >= _HEAVY_PEAK_MB:
        return HEAVY
    return LIGHT


def forecast_command(command: str) -> dict:
    """
    Predicts the resource profile of a command.

    Returns a dict: {severity, duration_s, peak_mb, reasons}.
    Static command-family patterns provide a baseline; measured history of
    the same family overrides it with reality when available.
    """
    family = _command_family(command)
    duration: float | None = None
    peak_mb: float | None = None
    severity = LIGHT
    reasons: list[str] = []

    # 1. Static pattern baseline.
    for _name, pattern, exp_duration, exp_peak, sev in _HEAVY_COMMAND_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            duration, peak_mb, severity = exp_duration, exp_peak, sev
            reasons.append(f"pattern: {_name}")
            break

    # 2. Historical evidence overrides the baseline with measured numbers.
    history = _recent_runs(family)
    if history:
        hist_duration = max(d for d, _p in history)
        hist_peak = max(p for _d, p in history)
        duration = hist_duration
        peak_mb = hist_peak
        severity = _severity_for(hist_duration, hist_peak)
        if severity == LIGHT and (hist_duration > 5 or hist_peak > 200):
            severity = MODERATE
        reasons.append(f"last run took {fmt_duration(hist_duration)}")

    return {
        "family": family,
        "severity": severity,
        "duration_s": duration,
        "peak_mb": peak_mb,
        "reasons": reasons,
    }


def forecast_severity(command: str) -> str:
    """Returns the forecast severity (light/moderate/heavy/critical) for a command."""
    return forecast_command(command)["severity"]


def forecast_warning(command: str) -> str | None:
    """
    Returns the human warning text for *command* when its forecast is
    moderate or worse; None when the command looks light.

    Callers prefix it with "[resources] ⚠" — the message itself carries no
    label so it can be embedded in confirmation prompts and notes without
    doubled prefixes.
    """
    fc = forecast_command(command)
    if fc["severity"] == LIGHT:
        return None
    bits = [f"{fc['severity']} resource forecast"]
    if fc["duration_s"]:
        bits.append(f"~{fmt_duration(fc['duration_s'])}")
    if fc["peak_mb"]:
        bits.append(f"peak ~{fc['peak_mb']:.0f} MB")
    if fc["reasons"]:
        bits.append(f"({', '.join(fc['reasons'])})")
    return " · ".join(bits)


def resource_forecast(command: str) -> str:
    """
    Registered tool: human-readable resource forecast for one command.
    """
    if not command or not command.strip():
        return "Error: resource_forecast needs a command to inspect."
    fc = forecast_command(command)
    sev = fc["severity"]
    line = f"Resource forecast for '{command}': {sev}"
    if fc["duration_s"]:
        line += f", ~{fmt_duration(fc['duration_s'])}"
    if fc["peak_mb"]:
        line += f", peak ~{fc['peak_mb']:.0f} MB"
    if fc["reasons"]:
        line += f" — evidence: {', '.join(fc['reasons'])}"
    else:
        line += " — no pattern match or history yet"
    return line
