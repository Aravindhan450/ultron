"""ultron.core.coding.command
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Structured command execution for the coding agent.

``run_command`` (the existing tool) returns a formatted string with exit
code, output, error output and resource metrics. This module parses that
string into a structured :class:`CommandResult` (command, exit code, stdout,
stderr, duration, timeout status) so the executor can act on the pieces
without losing any compiler/test output.

:func:`capture_command` runs the REAL ``run_command`` tool (so security and
resource measurement are untouched) and returns the structured result.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

# run_command output sections we parse. The format is stable:
#   Exit code: N
#   Output:\n...
#   Error Output:\n...
#   [resources] wall ... cpu ... peak ...
_TIMEOUT_RE = re.compile(r"timed out after", re.IGNORECASE)
_EXIT_CODE_RE = re.compile(r"Exit code:\s*(-?\d+)", re.IGNORECASE)


class CommandResult(BaseModel):
    """Structured result of one shell command execution."""

    command: str
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    duration_ms: float | None = None
    timed_out: bool = False
    output: str = ""  # the full original formatted output (never hidden)
    success: bool = Field(default=False)

    def __init__(self, **data) -> None:
        super().__init__(**data)
        # success is derived from the exit code; callers may override after
        # construction if they have better knowledge.
        if self.exit_code == 0:
            self.success = True


def _extract_section(text: str, header: str) -> str:
    """Extracts the text block that follows ``header:`` until the next known header."""
    marker = f"{header}:"
    idx = text.find(marker)
    if idx == -1:
        return ""
    start = idx + len(marker)
    rest = text[start:]
    # Section ends at the next header or the [resources] footer.
    for stop_header in ("\nError Output:", "\n[resources]"):
        stop = rest.find(stop_header)
        if stop != -1:
            rest = rest[:stop]
    return rest.strip()


def parse_command_output(text: str) -> CommandResult:
    """
    Parses a run_command formatted result string into a CommandResult.

    Never raises: any malformed input degrades gracefully to an empty
    result with the raw output preserved.
    """
    text = (text or "").strip()
    result = CommandResult(command="", output=text)

    # Timeout is reported as an Error line by run_command.
    if _TIMEOUT_RE.search(text):
        result.timed_out = True
        return result

    match = _EXIT_CODE_RE.search(text)
    if match:
        result.exit_code = int(match.group(1))
        result.success = result.exit_code == 0

    result.stdout = _extract_section(text, "Output")
    result.stderr = _extract_section(text, "Error Output")

    # Duration: "wall X.XXs" or "in X.XXs" in the resources footer.
    duration = re.search(r"(?:wall|in)\s+([\d.]+)s", text)
    if duration:
        try:
            result.duration_ms = round(float(duration.group(1)) * 1000, 1)
        except ValueError:
            pass
    return result


def capture_command(command: str) -> CommandResult:
    """
    Executes *command* through the existing run_command tool and returns a
    structured CommandResult.

    Security is unchanged: run_command itself is what the agent layer gates
    through the security boundary; this helper only parses its output.
    """
    from ultron.core.tools.registry import get_tool

    func = get_tool("run_command")
    if func is None:
        return CommandResult(command=command, success=False, output="Error: run_command tool not found.")
    raw = func(command)
    result = parse_command_output(raw)
    result.command = command
    return result
