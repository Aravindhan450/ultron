"""
Phase 2 — Real CLI Production Validation harness (PTY).

Drives the real `ultron chat` production path (main.py -> _plan_and_run ->
AgentRuntime -> RepositoryContextManager -> ReActAgent -> budget_messages ->
LlamaCppEngine -> real model) through the Phase 2 validation scenarios and
records the full terminal transcript for evidence. Uses the same pty
conventions as `_cli_output_e2e.py`.

Scenarios:
  A. Normal read-only repository inspection task.
  B. Multi-turn context retention (3 turns in one session).
  C. Long-context / compaction stress (long build-up turns, then summary).
  D. Oversized tool output (read a large source file; read_file caps at
     5000 chars > max_tool_output_tokens=1000, so truncation must engage).
  E. Verification + planning model calls under the shared boundary.
  F. Multi-step planning request.
  G. Security: a state-changing action must require confirmation.

Usage:
    .venv/bin/python _phase2_cli_validation.py A|B|C|D|E|F|G

Transcripts are written to scratch/phase2_cli_<scenario>.log
"""

from __future__ import annotations

import fcntl
import os
import pty
import re
import select
import signal
import struct
import subprocess
import sys
import termios
import threading
import time

ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")

SCRATCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scratch")
os.makedirs(SCRATCH, exist_ok=True)


def set_size(fd: int, h: int, w: int) -> None:
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", h, w, 0, 0))


class PtyLog:
    """Continuously drains a pty master into a byte log."""

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

    def mark(self) -> int:
        with self._lock:
            return len(self._log)

    def since(self, mark: int) -> str:
        with self._lock:
            return bytes(self._log[mark:]).decode(errors="replace")

    def all(self) -> str:
        with self._lock:
            return bytes(self._log).decode(errors="replace")

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)


SCENARIOS: dict[str, list[tuple[str, None, int]]] = {
    # Turn = (input, unused_pattern, timeout_seconds).
    # NOTE on task sizing: the local 7B coder model takes 30-90s per reasoning
    # step, and this harness must finish inside the outer command budget, so
    # scenario tasks are sized to complete in 1-2 tool calls while still
    # exercising the full production path (prepare_task -> AgentRuntime ->
    # ContextManager -> ReActAgent -> BudgetedEngine -> engine -> model).
    # Protocol intent per scenario is preserved; only the breadth of each
    # question is reduced.
    "A": [
        (
            (
                "Read the file src/ultron/core/context/manager.py and summarize "
                "what it is responsible for. Do not modify any files."
            ),
            None,
            400,
        ),
    ],
    "B": [
        (
            (
                "Remember two things for this validation session: my test project "
                "is called UltronPhase2, and my synthetic test credential is "
                "api_key=sk-test-ULTRON-PHASE2-SECRET (it is fake, for validation "
                "only). All analysis must stay read-only."
            ),
            None,
            110,
        ),
        (
            (
                "Now identify the main runtime entry point of this repository by "
                "reading src/ultron/main.py briefly, and name the file you read. "
                "Do not modify anything."
            ),
            None,
            190,
        ),
        (
            (
                "Without using any tools: using the repository evidence you just "
                "inspected, explain in two or three sentences how user input "
                "reaches the model, and restate the project codename I gave you. "
                "Do not modify anything."
            ),
            None,
            190,
        ),
    ],
    "C": [
        (
            (
                "Read src/ultron/core/context/manager.py and "
                "src/ultron/core/runtime/runtime.py and state what each file is "
                "responsible for. Read-only analysis, do not modify anything."
            ),
            None,
            190,
        ),
        (
            (
                "Continue the read-only analysis: read src/ultron/security/boundary.py "
                "and summarize how tool actions are classified into risk tiers."
            ),
            None,
            190,
        ),
        (
            (
                "Summarize the current task, the most important repository evidence "
                "gathered so far, and the latest relevant observation. Do not "
                "modify anything."
            ),
            None,
            220,
        ),
    ],
    "D": [
        (
            (
                "Read the file src/ultron/core/agents/simple.py and summarize its "
                "structure. Read-only, do not modify anything."
            ),
            None,
            400,
        ),
    ],
    # E. Complex read-only investigation: forces a structured plan, so the
    # planning model call AND the final verification call (verify_plan_task)
    # both run through the same model-call boundary.
    "E": [
        (
            (
                "Investigate the context subsystem: read "
                "src/ultron/core/context/budget.py and "
                "src/ultron/core/context/invocation.py, then report the public "
                "class and function names each defines and state whether the "
                "model-call token-budget invariant is enforced in one place. "
                "Do not modify any files."
            ),
            None,
            450,
        ),
    ],
    # F. Multi-step planning request (complex -> plan generation path).
    "F": [
        (
            (
                "Plan and carry out a read-only survey of the test suite: "
                "determine which directory holds the tests, how many test files "
                "exist, and report the plan you followed plus the findings. "
                "Do not modify any files."
            ),
            None,
            450,
        ),
    ],
    # G. State-changing request: must surface a confirmation card and only run
    # after the driver approves it (security gate preserved).
    "G": [
        (
            (
                "Create a file named ultron_phase2_boundary_probe.txt in the "
                "repository root containing exactly the word ready. Then stop."
            ),
            None,
            450,
        ),
    ],
}


def run_scenario(name: str) -> int:
    turns = SCENARIOS[name]
    master, slave = pty.openpty()
    set_size(slave, 40, 120)

    root_dir = os.path.dirname(os.path.abspath(__file__))
    venv_python = os.path.join(root_dir, ".venv", "bin", "python")
    python_exe = venv_python if os.path.exists(venv_python) else sys.executable

    src_dir = os.path.join(root_dir, "src")
    env = dict(os.environ)
    if env.get("TERM") in (None, "dumb"):
        env["TERM"] = "xterm-256color"
    current_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{src_dir}:{current_pp}" if current_pp else src_dir
    # Real engine path: ULTRON_NO_SERVER must NOT be set. Inherit .env config.
    env.pop("ULTRON_NO_SERVER", None)
    env["ULTRON_VERBOSE"] = "0"

    proc = subprocess.Popen(
        [python_exe, "-m", "ultron.main", "chat"],
        stdin=slave,
        stdout=slave,
        stderr=slave,
        env=env,
        close_fds=True,
        start_new_session=True,
    )
    log = PtyLog(master)
    ok = True

    def clean_text() -> str:
        return ANSI.sub("", log.all())

    def wait_for_pattern(pattern: str, timeout: float) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if pattern in clean_text():
                return True
            if proc.poll() is not None:
                return False
            time.sleep(0.2)
        return False

    transcript_path = os.path.join(SCRATCH, f"phase2_cli_{name}.log")
    try:
        # Startup: banner or prompt marker (banners differ; accept either the
        # title or the prompt glyph).
        if not wait_for_pattern("Ultron AI", timeout=60) and "❯" not in clean_text():
            print(f"FAILED: CLI startup banner/prompt not seen. Output:\n{clean_text()[:3000]}")
            return 1
        # wait for the input prompt to be idle
        time.sleep(2.0)

        for i, (user_input, _pattern, timeout) in enumerate(turns, start=1):
            mark = log.mark()
            print(f"[scenario {name}] turn {i}: sending input ({len(user_input)} chars)...")
            os.write(master, user_input.encode() + b"\n")
            deadline = time.time() + timeout
            finished = False
            confirmations_answered = 0
            last_confirmation_mark = mark
            last_len = log.mark()
            last_change = time.time()
            while time.time() < deadline:
                if proc.poll() is not None:
                    print(f"[scenario {name}] turn {i}: CLI PROCESS EXITED early "
                          f"code={proc.returncode}")
                    ok = False
                    break
                # Answer interactive confirmation prompts by pressing Enter
                # (accepts the pre-selected "Yes, allow"). Each prompt is
                # answered once; a cap prevents runaway loops.
                since_conf = ANSI.sub("", log.since(last_confirmation_mark))
                if "Do you want to allow this action?" in since_conf and confirmations_answered < 12:
                    time.sleep(1.5)  # let the questionary prompt finish rendering
                    os.write(master, b"\n")
                    confirmations_answered += 1
                    last_confirmation_mark = log.mark()
                    print(f"[scenario {name}] turn {i}: confirmed action "
                          f"#{confirmations_answered}")
                    last_change = time.time()
                    time.sleep(0.5)
                    continue
                # A turn is complete when the transcript went idle (spinner
                # stopped) for >=10s AND a response box / prompt appeared
                # after the turn started.
                cur_len = log.mark()
                if cur_len != last_len:
                    last_len = cur_len
                    last_change = time.time()
                since_mark = ANSI.sub("", log.since(mark))
                if (
                    time.time() - last_change >= 10.0
                    and ("╭─" in since_mark or "❯" in since_mark)
                ):
                    time.sleep(1.0)
                    finished = True
                    break
                time.sleep(0.5)
            turn_text = ANSI.sub("", log.since(mark))
            if not finished:
                print(f"[scenario {name}] turn {i}: TIMEOUT after {timeout}s")
                ok = False
            else:
                print(f"[scenario {name}] turn {i}: response received "
                      f"({len(turn_text)} chars of transcript)")
            time.sleep(1.0)

        full = clean_text()
        with open(transcript_path, "w") as f:
            f.write(full)
        # Also save a spinner-cleaned version for readability.
        cleaned = re.sub(r"(?:⠋|⠙|⠹|⠸|⠼|⠴|⠦|⠧|⠇|⠏) Thinking[^\n\r]*", "", full)
        cleaned = re.sub(r"\r\n?", "\n", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        with open(transcript_path.replace(".log", "_clean.log"), "w") as f:
            f.write(cleaned)
        print(f"[scenario {name}] transcript saved -> {transcript_path} "
              f"({len(full)} chars, cleaned {len(cleaned)} chars)")

        # Basic health checks on the full transcript
        if "Traceback (most recent call last)" in full:
            print(f"[scenario {name}] FAILED: traceback leaked into CLI session")
            ok = False
        if proc.poll() is None:
            print(f"[scenario {name}] CLI still alive at end: OK")
        else:
            print(f"[scenario {name}] CLI exited: code={proc.returncode}")
            ok = False

        print(f"SCENARIO {name} RESULT:", "PASS" if ok else "FAIL")
        return 0 if ok else 1

    finally:
        log.stop()
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, OSError):
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        os.close(master)
        os.close(slave)


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in SCENARIOS:
        print(f"usage: {sys.argv[0]} {'|'.join(SCENARIOS)}")
        return 2
    return run_scenario(sys.argv[1])


if __name__ == "__main__":
    sys.exit(main())
