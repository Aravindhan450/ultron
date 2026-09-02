"""Real model execution harness + trace capture (Phases 8-9 of STEP 3).

Tasks are executed through the ACTUAL Ultron CLI (``python -m ultron.main
chat``) in a pty, exactly like a human typing at the prompt.  The framework
never calls internal functions (``select_capability``, ``find_definition``,
...) for the primary benchmark; those are used only for evaluation/ground
truth.

The executor is injectable so deterministic framework self-tests can feed
canned transcripts without a live model.
"""

from __future__ import annotations

import fcntl
import os
import pty
import re
import select
import struct
import subprocess
import sys
import termios
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from ultron.core.tools.definitions import TOOL_DEFINITIONS
from ultron.validation.model import CapabilityTestCase, TaskTrace


def tool_capabilities(tool: str) -> tuple[str, ...]:
    """Canonical capabilities of one registered tool (execution-truth ground)."""
    definition = TOOL_DEFINITIONS.get(tool)
    if definition is None:
        return ()
    return tuple(c.value for c in definition.capabilities)

_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")

# Security-decision markers observable in a transcript.
_SECURITY_MARKERS: tuple[tuple[str, str], ...] = (
    (r"\bdenied\b", "denied"),
    (r"\bdeny\b", "deny"),
    (r"confirmation\s+required", "confirmation_required"),
    (r"requires?\s+confirmation", "confirmation_required"),
    (r"\bapproved\b", "approved"),
    (r"\bconfirmed\b", "confirmed"),
    (r"\ballowed\b", "allowed"),
    (r"🔒", "security_gate"),
)

_FAILURE_MARKERS: tuple[tuple[str, str], ...] = (
    (r"couldn['’]t\s+generate\s+a\s+response", "empty_response"),
    (r"could not generate a response", "empty_response"),
    (r"command\s+not\s+found", "command_not_found"),
    (r"no\s+(?:verified\s+)?definition\s+found", "no_definition"),
    (r"no\s+(?:verified\s+)?references?\s+found", "no_references"),
    (r"traceback\s*\(most\s+recent\s+call\s+last\)", "traceback"),
    (r"error\s+executing\s+tool", "tool_error"),
    (r"\bis\s+likely\s+the\s+(?:definition|main\s+location)\b", "speculative"),
    (r"likely\s+the\s+definition", "speculative"),
    # External-search routing on a repository question (Part 13: this is a
    # routing/capability-selection failure, not an evidence failure).
    (r"search\s+the\s+web", "web_search_routing"),
    # Deterministic-router / model clarification fallback instead of an answer
    # ("It sounds like you want to run a command — which command?").
    (r"it\s+sounds\s+like\s+you\s+want\s+to", "clarification_prompt"),
    (r"which\s+(?:command|file|one)\?", "clarification_prompt"),
)

# Quoted phrases the CLI tends to use for the resolved entity:
#   References to 'taskstate'  /  Find definition of 'X'  /  Symbol: X
_QUOTED_ENTITY = re.compile(r"['\"]([A-Za-z_][\w]*(?:\s+[\w.]+)*)['\"]")


# ---------------------------------------------------------------------------
# Pty harness (same pattern the STEP 2C sanity check uses).
# ---------------------------------------------------------------------------


def _set_size(fd: int, h: int, w: int) -> None:
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", h, w, 0, 0))


class _PtyLog:
    def __init__(self, master: int) -> None:
        self._master = master
        self._log = bytearray()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self) -> None:
        while not self._stop.is_set():
            r, _, _ = select.select([self._master], [], [], 0.1)
            if r:
                try:
                    chunk = os.read(self._master, 65536)
                except OSError:
                    break
                if not chunk:
                    break
                with self._lock:
                    self._log.extend(chunk)

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)

    def snapshot(self) -> str:
        with self._lock:
            return _ANSI.sub("", self._log.decode(errors="replace"))


def _pty_run(args: list[str], prompt: str, wait_s: float, settle_s: float = 1.5) -> tuple[str, float]:
    """Runs one prompt in a pty; returns (transcript, elapsed).

    The reply is considered complete when (a) the prompt was echoed, (b) the
    output has grown past the echo, and (c) the terminal has been idle for
    ``settle_s`` seconds.  This is a general idle detector — it does not rely
    on any particular theme marker — with ``wait_s`` as the hard deadline.
    """
    start = time.monotonic()
    master, slave = pty.openpty()
    _set_size(slave, 40, 120)
    log = _PtyLog(master)
    proc = subprocess.Popen(
        args,
        stdin=slave,
        stdout=slave,
        stderr=slave,
        close_fds=True,
        start_new_session=True,
    )
    os.close(slave)
    time.sleep(2.5)  # startup
    os.write(master, (prompt + "\n").encode())
    deadline = time.monotonic() + wait_s
    echo_len = 0
    last_len = 0
    idle_since = time.monotonic()
    while time.monotonic() < deadline:
        snap = log.snapshot()
        n = len(snap)
        if n != last_len:
            last_len = n
            idle_since = time.monotonic()
            if echo_len == 0 and prompt in snap[-len(prompt) - 100 :]:
                echo_len = n  # the prompt was echoed; reply follows
        elif (
            time.monotonic() - idle_since >= settle_s
            and last_len > 300  # past the startup banner
            and ("❯" in snap[-160:] or (echo_len and last_len > echo_len))
        ):
            # The terminal returned to the input prompt and is idle: reply done.
            break
        time.sleep(0.25)
    time.sleep(0.3)
    transcript = log.snapshot()
    log.stop()
    try:
        proc.terminate()
    except OSError:
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    os.close(master)
    return transcript, time.monotonic() - start


def default_cli_args(agent: str = "simple") -> list[str]:
    base = [sys.executable, "-m", "ultron.main", "chat"]
    if agent and agent != "simple":
        base += ["--agent", agent]
    return base


# ---------------------------------------------------------------------------
# Trace signal parsing (heuristic, deterministic).
# ---------------------------------------------------------------------------


@dataclass
class ParsedTrace:
    """Signals extracted from a raw transcript window."""

    tool_names: list[str] = field(default_factory=list)
    tool_capabilities: list[str] = field(default_factory=list)
    security_decision: str | None = None
    failure_markers: list[str] = field(default_factory=list)
    quoted_entities: list[str] = field(default_factory=list)
    file_lines: list[str] = field(default_factory=list)
    has_answer: bool = False
    empty_response: bool = False
    speculative: bool = False

    @property
    def observed_tool(self) -> str | None:
        return self.tool_names[0] if self.tool_names else None

    @property
    def observed_capabilities(self) -> tuple[str, ...]:
        """Capabilities of every observed tool (canonical ground truth)."""
        out: list[str] = []
        for tool in self.tool_names:
            out.extend(tool_capabilities(tool))
        return tuple(dict.fromkeys(out))


_FILE_LINE_RE = re.compile(r"[\w./-]+\.(?:py|ts|js|rs|go|java|md|toml|yaml|yml|json):\d+(?:[-,]\d+)*")


# Markers are matched inside the *reply window* — the tail of the
# transcript — not the whole transcript.  A task may legitimately display
# file contents that contain the words "traceback" or a tool name; only
# markers in the reply region signal the model's actual behavior.
_REPLY_WINDOW = 800
_EMPTY_WINDOW = 300


def parse_trace(transcript: str) -> ParsedTrace:
    """Extracts observable signals from a transcript (no model, no LLM).

    Tool names and failure/security markers are matched only within the reply
    window (the transcript tail), so file contents displayed by the agent do
    not produce false signals.
    """
    parsed = ParsedTrace()
    lowered = transcript.lower()
    reply = lowered[-_REPLY_WINDOW:]
    for pat, label in _FAILURE_MARKERS:
        window = lowered[-_EMPTY_WINDOW:] if label == "empty_response" else reply
        if re.search(pat, window):
            parsed.failure_markers.append(label)
            if label == "empty_response":
                parsed.empty_response = True
            if label == "speculative":
                parsed.speculative = True
    for pat, label in _SECURITY_MARKERS:
        if re.search(pat, reply):
            parsed.security_decision = label
            break
    for name in TOOL_DEFINITIONS:
        if re.search(rf"\b{re.escape(name)}\b", reply):
            parsed.tool_names.append(name)
    parsed.tool_capabilities = list(parsed.observed_capabilities)
    for m in _QUOTED_ENTITY.finditer(transcript[-_REPLY_WINDOW:]):
        ent = m.group(1)
        if ent not in parsed.quoted_entities:
            parsed.quoted_entities.append(ent)
    parsed.file_lines = _FILE_LINE_RE.findall(transcript)[:20]
    # An answer exists when the transcript has more than the startup banner
    # and does not end in an empty-response marker.
    body = transcript.strip()
    parsed.has_answer = bool(body) and not parsed.empty_response and "traceback" not in reply
    return parsed


# ---------------------------------------------------------------------------
# The runner.
# ---------------------------------------------------------------------------

Executor = Callable[[list[str], str, float], tuple[str, float]]


class ValidationRunner:
    """Executes capability tasks through the real CLI and records traces."""

    def __init__(
        self,
        *,
        agent: str = "simple",
        wait_s: float = 60.0,
        executor: Executor | None = None,
    ) -> None:
        self.agent = agent
        self.wait_s = wait_s
        # Injecting an executor keeps framework self-tests deterministic.
        self.executor = executor if executor is not None else _pty_run

    def run_one(self, case: CapabilityTestCase) -> TaskTrace:
        args = default_cli_args(self.agent)
        transcript, latency = self.executor(args, case.task, self.wait_s)
        return TaskTrace(case=case, transcript=transcript, latency_s=latency)

    def run_many(self, cases: list[CapabilityTestCase]) -> list[TaskTrace]:
        return [self.run_one(case) for case in cases]
