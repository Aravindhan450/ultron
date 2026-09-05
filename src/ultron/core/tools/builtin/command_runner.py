import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from ultron.core.tools import resource_monitor as rm


def _run_one(command: str, timeout: int) -> str:
    """
    Executes a single shell command and returns its formatted result.

    Shared by run_command and run_parallel so both report failures the same
    way: non-zero exit codes are surfaced (not raised), timeouts and process
    errors become "Error: ..." strings.

    Every run is measured — wall time, CPU time (getrusage child delta), and
    peak RSS (live pid sampling, with the getrusage high-water delta as the
    shell-descendant fallback) — reported as a ``[resources]`` line and
    recorded for future forecasts. Measurement never changes the command's
    output format or error contract.
    """
    import os
    import sys
    from pathlib import Path

    env = dict(os.environ)
    venv_bin = str(Path(sys.executable).parent)
    if venv_bin and venv_bin not in env.get("PATH", "").split(os.pathsep):
        env["PATH"] = f"{venv_bin}{os.pathsep}{env.get('PATH', '')}"

    start = time.monotonic()
    cpu_before, rss_before = rm.child_usage()

    try:
        proc = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return f"Error: {exc!s}"

    stop = threading.Event()
    sampled: dict = {"peak_mb": 0.0}
    sampler = threading.Thread(
        target=rm.sample_peak_rss, args=(proc.pid, stop, sampled), daemon=True
    )
    sampler.start()

    try:
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            stop.set()
            sampler.join(timeout=1.0)
            proc.kill()
            proc.wait()
            elapsed = time.monotonic() - start
            rm.record_run(command, elapsed, exit_ok=False)
            return (
                f"Error: command timed out after {timeout} seconds.\n"
                f"[resources] timed out after {rm.fmt_duration(elapsed)}"
            )
        stop.set()
        sampler.join(timeout=1.0)

        elapsed = time.monotonic() - start
        cpu_after, rss_after = rm.child_usage()
        cpu_seconds = max(0.0, cpu_after - cpu_before)
        rss_delta = max(0.0, rss_after - rss_before)
        peak_mb = max(sampled["peak_mb"], rss_delta)
        rm.record_run(command, elapsed, peak_mb=peak_mb, cpu_seconds=cpu_seconds, exit_ok=proc.returncode == 0)

        output_parts = [f"Exit code: {proc.returncode}"]
        if stdout:
            output_parts.append(f"Output:\n{stdout.strip()}")
        if stderr:
            output_parts.append(f"Error Output:\n{stderr.strip()}")
        output_parts.append(rm.format_metrics(elapsed, peak_mb, cpu_seconds))
        return "\n".join(output_parts)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        stop.set()
        elapsed = time.monotonic() - start
        rm.record_run(command, elapsed, exit_ok=False)
        return f"Error: {exc!s}"


def run_command(command: str) -> str:
    """
    Executes a shell command using Python's subprocess module.
    
    - Times out after 15 seconds to prevent hanging processes.
    - Captures stdout and stderr.
    - Reports measured resources ([resources] line) and records them for
      future forecasts.
    - Safety and user confirmation are handled at the agent level before calling this tool.
    """
    return _run_one(command, timeout=15)


def run_parallel(commands: list[str], timeout: int = 15) -> str:
    """
    Executes multiple shell commands concurrently and returns a combined report.

    - Commands run simultaneously (one worker per command) instead of in a
      queue, so the batch wall-clock time is ~the slowest command, not the
      sum of all commands.
    - Each command keeps its own timeout: a hung process times out on its own
      without delaying or cancelling the others.
    - A failure in one command does not stop the rest — every result is
      reported, plus an overall summary line with measured elapsed time.
    - Each command's [resources] line is embedded in its own result; the
      batch gets a summary line too.
    - Blank/whitespace-only entries are dropped so they can't be reported as
      successful no-op commands.
    - Safety and user confirmation are handled at the agent level before
      calling this tool: every command in the batch is gated individually
      (see simple.py handle_parallel).
    """
    if isinstance(commands, str):
        commands = [commands]
    if not isinstance(commands, list):
        return "Error: run_parallel expects a list of commands."
    # Blank/whitespace-only entries are dropped so they can't be reported as
    # successful no-op commands (a stray "" would otherwise count as a win).
    cleaned = [str(c).strip() for c in commands]
    commands = [c for c in cleaned if c]
    if not commands:
        return "Error: run_parallel expects a non-empty list of commands."
    # Tolerate a numeric string for timeout (small models sometimes emit
    # "5" instead of 5); a non-numeric value is an error, never a crash.
    try:
        timeout = int(timeout)
    except (TypeError, ValueError):
        return "Error: run_parallel expects timeout as a number of seconds."

    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=len(commands)) as pool:
        futures = {
            pool.submit(_run_one, str(cmd), timeout): i
            for i, cmd in enumerate(commands)
        }
        results: dict[int, str] = {}
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    elapsed = time.monotonic() - start

    lines: list[str] = []
    succeeded = 0
    for i, cmd in enumerate(commands):
        body = results[i]
        failed = body.startswith("Error") or bool(
            re.search(r"Exit code:\s*([1-9])", body)
        )
        if not failed:
            succeeded += 1
        status = "FAIL" if failed else "OK"
        lines.append(f"[{i + 1}] {status} — {cmd}\n{body}")

    summary = f"{succeeded}/{len(commands)} commands succeeded in {elapsed:.2f}s"
    return (
        summary
        + "\n\n"
        + "\n\n".join(lines)
        + f"\n[resources] batch: {len(commands)} commands in {elapsed:.2f}s"
    )
