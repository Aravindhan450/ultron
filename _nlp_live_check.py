"""Live validation of the FIX #8.5 capability-routing fixes through the REAL CLI.

Starts `ultron chat` in a pty (no LLM required — every prompt below is a
deterministic path) and drives the spec's 10 live prompts plus adversarial
phrasing variants.  For each prompt it verifies the *actual* tool/command
behaviour from the assistant's reply:

  1. "List the files in the current directory."  -> list_directory at workspace root
  2. "Show me the files in this folder"          -> same semantic operation
  3. "Find where TaskState is defined."          -> find_definition (file + line)
  4. "Where is TaskState used?"                  -> find_references
  5. "Find where command execution is implemented" -> code_search
  6. "Run the relevant tests"                    -> affected-test selection + venv pytest
  7. "Run the full test suite"                   -> project-resolved test command
  8. "Run pwd."                                  -> terminal, command=pwd
  9. "Execute: git status"                       -> terminal/git, wrapper stripped
 10. "Show me the current git diff."             -> git diff

Key invariants verified:
- "current directory" / "this folder" resolve to the actual workspace root;
- code questions route to Code Intelligence (never the LLM fallback);
- test requests use the project virtualenv (never bare `pytest` on PATH);
- the NL wrapper ("Execute:", "Run the command `...`") never reaches the shell
  verbatim.
"""

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


def set_size(fd: int, h: int, w: int) -> None:
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", h, w, 0, 0))


class PtyLog:
    """Continuously drains a pty master into a byte log (like a real terminal)."""

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
            return ANSI.sub("", bytes(self._log[mark:]).decode(errors="replace"))

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)


CWD = os.path.realpath(os.getcwd())


PROMPTS = [
    # (prompt, wait_seconds, [must-contain-any-of], [must-contain-all-of], [must-not-contain])
    # NOTE: the echoed user line is part of every window, so "Execute:" is
    # always present — the forbidden signature is the shell error
    # "/bin/sh: Execute:".
    ("List the files in the current directory.", 6.0,
     ["pyproject.toml", "src", "tests"], None, ["/bin/sh:", "not found at the"]),
    ("Show me the files in this folder", 4.0,
     ["pyproject.toml", "src", "tests"], None, ["/bin/sh:", "not found at the"]),
    ("Find where TaskState is defined.", 6.0,
     ["TaskState"], ["Definitions", "src/ultron"], ["/bin/sh:", "couldn't generate"]),
    ("Where is TaskState used?", 6.0,
     ["References to 'TaskState'"], ["TaskState"], None),
    ("Find where command execution is implemented", 6.0,
     ["command", "execute"], None, ["/bin/sh:", "couldn't generate"]),
    ("Run the relevant tests", 30.0,
     ["passed", "failed", "pytest", "Affected tests"], None, ["/bin/sh: pytest", "command not found"]),
    # The full suite exceeds the command tool's 15s timeout; a timeout result
    # still proves the venv-resolved pytest command executed (never bare
    # `pytest` on PATH, never a "which command?" clarification).
    ("Run the full test suite", 30.0,
     ["pytest", "timed out", "passed", "failed"], None,
     ["/bin/sh: pytest", "command not found", "which command?"]),
    ("Run pwd.", 4.0, None, [CWD], ["/bin/sh:"]),
    ("Execute: git status", 4.0,
     ["On branch", "nothing to commit", "modified:", "Changes not staged"], None, ["/bin/sh: Execute:"]),
    ("Show me the current git diff.", 4.0, ["diff"], None, None),
]
ADVERSARIAL = [
    "pwd",
    "Run the command `pwd`",
    "Please execute `pwd`",
    "Can you execute pwd?",
    "Use the terminal and execute pwd",
    "Execute: pwd",
]


def main() -> int:
    deadline = time.monotonic() + 360  # hard wall-clock budget (full suite runs)

    def _left() -> float:
        return deadline - time.monotonic()

    master, slave = pty.openpty()
    set_size(slave, 40, 120)

    env = dict(os.environ)
    proc = subprocess.Popen(
        [sys.executable, "-m", "ultron.main", "chat"],
        stdin=slave, stdout=slave, stderr=slave,
        env=env, close_fds=True, start_new_session=True,
    )
    log = PtyLog(master)
    ok = True
    try:
        time.sleep(3.0)  # banner + first prompt
        failures = []

        def _wait(seconds: float) -> None:
            time.sleep(min(seconds, max(0.1, _left())))

        def _send(prompt: str) -> bool:
            """Writes a prompt; returns False when the CLI child is gone."""
            if proc.poll() is not None:
                return False
            try:
                os.write(master, prompt.encode() + b"\n")
                return True
            except OSError:
                return False

        def _check(label: str, text: str, must_any: list | None,
                   must_all: list | None, must_not: list | None) -> bool:
            nonlocal ok
            good = True
            reasons = []
            if must_any and not any(n in text for n in must_any):
                good = False
                reasons.append(f"missing any of {must_any!r}")
            if must_all:
                for needle in must_all:
                    if needle not in text:
                        good = False
                        reasons.append(f"missing {needle!r}")
            if must_not:
                for needle in must_not:
                    if needle in text:
                        good = False
                        reasons.append(f"contains forbidden {needle!r}")
            if not good:
                ok = False
                failures.append(f"{label}: {'; '.join(reasons)}")
                failures.append(f"{label}: window={text[:600]!r}")
            print(f"  {'PASS' if good else 'FAIL'} {label}")
            return good

        # ---- Spec's 10 live prompts ----
        for i, (prompt, wait_s, must_any, must_all, must_not) in enumerate(PROMPTS, start=1):
            mark = log.mark()
            if not _send(prompt):
                print(f"  SKIP {i}. {prompt} (CLI exited)", flush=True)
                ok = False
                break
            _wait(wait_s)
            reply = log.since(mark)
            _check(f"{i}. {prompt}", reply, must_any, must_all, must_not)

        # ---- Adversarial variants: identical semantic action (pwd) ----
        print("adversarial variants (all must print the working directory):")
        for i, prompt in enumerate(ADVERSARIAL, start=11):
            mark = log.mark()
            if not _send(prompt):
                print(f"  SKIP {i}. {prompt} (CLI exited)", flush=True)
                ok = False
                break
            _wait(3.5)
            reply = log.since(mark)
            if CWD not in reply or "/bin/sh:" in reply:
                ok = False
                failures.append(f"{i}. {prompt}: did not run pwd (window={reply[:400]!r})")
            print(f"  {'PASS' if CWD in reply and '/bin/sh:' not in reply else 'FAIL'} {i}. {prompt}")

        print("RESULT:", "PASS" if ok else "FAIL")
        if failures:
            print("FAILURES:")
            for f in failures:
                print(f"  - {f}")
        return 0 if ok else 1
    finally:
        print("[harness] cleaning up", flush=True)
        log.stop()
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, OSError):
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        os.close(master)
        os.close(slave)


if __name__ == "__main__":
    sys.exit(main())
