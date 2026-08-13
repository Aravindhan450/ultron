"""Live validation of the Code Intelligence query-resolution fix through the REAL CLI.

Starts `ultron chat` in a pty (no LLM required — every prompt below routes to
a deterministic code-intelligence tool) and drives the spec's 15 live prompts.
For each prompt it verifies the *actual* tool behaviour from the assistant's
reply:

  1.  Find where TaskState is defined        -> find_definition, VERIFIED
  2.  Find where taskstate is defined        -> case-insensitive -> TaskState
  3.  Find where TASKSTATE is defined        -> case-insensitive -> TaskState
  4.  Find where Task State is defined       -> normalized -> TaskState
  5.  Find where TaskState is used           -> find_references
  6.  Find references to taskstate           -> case-insensitive references
  7.  Find where CodingExecutor is defined   -> find_definition
  8.  Find where codingexecutor is defined   -> case-insensitive
  9.  Find where coding executor is implemented -> code_search
 10.  Find where the Supervisor is defined   -> article-tolerant -> Supervisor
 11.  Find where supervisor is defined       -> normalized -> Supervisor
 12.  Where is the OrchestrationValidator implemented? -> code_search
 13.  Where is orchestrationvalidator implemented?     -> code_search
 14.  Where is task state handled?           -> code_search (normalized)
 15.  Where is command execution implemented? -> code_search / semantic

Key invariants verified:
- Equivalent queries resolve to equivalent entities (TaskState/taskstate/Task State);
- the deterministic route answers symbol questions — never the LLM fallback
  ("couldn't generate") and never a speculative "likely" filename guess;
- no shell is ever invoked for these questions.
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


PROMPTS = [
    # (prompt, wait_seconds, [must-contain-any-of], [must-contain-all-of], [must-not-contain])
    ("Find where TaskState is defined", 8.0,
     ["Definitions of 'TaskState'"], ["src/ultron"], ["/bin/sh:", "couldn't generate", "likely"]),
    ("Find where taskstate is defined", 8.0,
     ["Definitions of 'TaskState'"], ["src/ultron"], ["/bin/sh:", "couldn't generate", "likely"]),
    ("Find where TASKSTATE is defined", 8.0,
     ["Definitions of 'TaskState'"], ["src/ultron"], ["/bin/sh:", "couldn't generate", "likely"]),
    ("Find where Task State is defined", 8.0,
     ["Definitions of 'TaskState'"], ["src/ultron"], ["/bin/sh:", "couldn't generate", "likely"]),
    # Note: the reference windows include the harness file's own lines, which
    # contain the literal strings "/bin/sh:" and "couldn't generate" — so
    # those forbidden markers are not asserted for the reference prompts
    # (the /bin/sh and LLM-fallback checks are covered by the definition and
    # code-search prompts above).
    ("Find where TaskState is used", 8.0,
     ["References to 'TaskState'"], ["src/ultron"], None),
    ("Find references to taskstate", 8.0,
     ["References to 'TaskState'"], ["src/ultron"], None),
    ("Find where CodingExecutor is defined", 8.0,
     ["Definitions of 'CodingExecutor'"], ["src/ultron"], ["/bin/sh:", "couldn't generate", "likely"]),
    ("Find where codingexecutor is defined", 8.0,
     ["Definitions of 'CodingExecutor'"], ["src/ultron"], ["/bin/sh:", "couldn't generate", "likely"]),
    ("Find where coding executor is implemented", 8.0,
     ["CodingExecutor", "executor.py"], None, ["/bin/sh:", "couldn't generate"]),
    ("Find where the Supervisor is defined", 8.0,
     ["Definitions of 'Supervisor'"], ["src/ultron"], ["/bin/sh:", "couldn't generate", "likely", "probably"]),
    ("Find where supervisor is defined", 8.0,
     ["Definitions of 'Supervisor'"], ["src/ultron"], ["/bin/sh:", "couldn't generate", "likely"]),
    ("Where is the OrchestrationValidator implemented?", 8.0,
     ["OrchestrationValidator"], None, ["/bin/sh:", "couldn't generate"]),
    ("Where is orchestrationvalidator implemented?", 8.0,
     ["OrchestrationValidator"], None, ["/bin/sh:", "couldn't generate"]),
    ("Where is task state handled?", 8.0,
     ["TaskState"], None, ["/bin/sh:", "couldn't generate"]),
    ("Where is command execution implemented?", 8.0,
     ["command", "execute"], None, ["/bin/sh:", "couldn't generate"]),
]


def main() -> int:
    deadline = time.monotonic() + 240  # hard wall-clock budget

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

        for i, (prompt, wait_s, must_any, must_all, must_not) in enumerate(PROMPTS, start=1):
            mark = log.mark()
            if not _send(prompt):
                print(f"  SKIP {i}. {prompt} (CLI exited)", flush=True)
                ok = False
                break
            _wait(wait_s)
            reply = log.since(mark)
            _check(f"{i}. {prompt}", reply, must_any, must_all, must_not)

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
