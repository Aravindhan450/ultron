"""ultron.core.nlp.interpret
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Deterministic interpretation of command/tool results.

The terminal tool reports structured output (``Exit code: N`` plus captured
stdout/stderr and a ``[resources]`` line).  :func:`interpret_command_result`
turns that into a short human-readable verdict so the agent never claims
success from raw text alone:

- exit 0  -> "succeeded"
- exit 1+ -> "failed with exit code N" + the first meaningful error line
- "Error:"-prefixed tool output (timeouts, subprocess failures) -> kept as-is
"""

from __future__ import annotations

import re

_EXIT_RE = re.compile(r"^Exit code:\s*(-?\d+)\s*$", re.MULTILINE)
_ERROR_OUTPUT_RE = re.compile(r"^Error Output:\s*\n(.*?)(?:\n\[resources\]|\Z)", re.DOTALL | re.MULTILINE)
_OUTPUT_RE = re.compile(r"^Output:\s*\n(.*?)(?:\nError Output:|\n\[resources\]|\Z)", re.DOTALL | re.MULTILINE)
_TIMEOUT_RE = re.compile(r"command timed out after (\d+) seconds")


def _first_useful_line(text: str, limit: int = 3) -> str:
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return ""
    return " | ".join(lines[:limit])


def interpret_command_result(output: str, command: str | None = None) -> str:
    """Returns a concise, accurate verdict for a command's raw output.

    Never fabricates success: non-zero exits and tool-level errors are
    reported explicitly.
    """
    if not output or not output.strip():
        return "No output returned."

    stripped = output.strip()

    # Tool-level error (timeout / subprocess failure) — never a crash, and
    # never claimed as success.
    if stripped.startswith("Error:"):
        timeout = _TIMEOUT_RE.search(stripped)
        if timeout:
            return f"The command timed out after {timeout.group(1)} seconds."
        return f"The command could not be run: {_first_useful_line(stripped)}"

    m = _EXIT_RE.search(stripped)
    if not m:
        # Not terminal output (e.g. a dedicated tool result) — pass through
        # with a light summary, never a fabricated exit verdict.
        return stripped

    exit_code = int(m.group(1))
    label = f"{command!r}" if command else "The command"

    if exit_code == 0:
        detail = _first_useful_line(_OUTPUT_RE.search(stripped).group(1) if _OUTPUT_RE.search(stripped) else "")
        return f"{label} succeeded." + (f" Output: {detail}" if detail else "")

    # Non-zero exit: surface the error section when present.
    err = ""
    err_m = _ERROR_OUTPUT_RE.search(stripped)
    if err_m and err_m.group(1).strip():
        err = f" Error: {_first_useful_line(err_m.group(1))}"
    elif exit_code == 127:
        err = " (command not found)"
    elif exit_code == 126:
        err = " (not executable)"
    return f"{label} failed with exit code {exit_code}.{err}"
