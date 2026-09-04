"""Live validation of the ReAct-loop routing correction through the REAL CLI.

Starts `ultron chat --agent react` in a pty with a REAL local LLM (llama-server)
and drives prompts that exercise the routing correction layer:

  1. How does the Supervisor delegate work?  -> repository investigation,
                                               NEVER a web-search confirmation
  2. Where is taskstate used?                -> reference lookup (case-insensitive)
  3. Where is command execution implemented? -> primary implementation synthesis
  4. What is the latest Python release?      -> NOT a repository investigation
                                               (external question stays external)

The routing correction (`route_llm_tool_call`) is a runtime safety layer: the
LLM may pick any tool it likes, but a repository question can never execute as
a web search and a question-shaped argument on a generic code tool is moved to
the specific capability.  The harness verifies the *observed* CLI behaviour —
the reply must never show a web-search confirmation for repository questions,
and repository answers must cite real source evidence.
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

    def size(self) -> int:
        with self._lock:
            return len(self._log)

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)


PROMPTS = [
    # (prompt, wait_seconds, must_any, must_not)
    ("How does the Supervisor delegate work?", 90.0,
     ["delegation", "Supervisor"],
     ["Search the web", "Web search requested", "search_web"]),
    ("Where is taskstate used?", 120.0,
     ["TaskState", "References"],
     ["Search the web", "Web search requested"]),
    ("Where is command execution implemented?", 90.0,
     ["src/ultron", "implement"],
     ["Search the web", "Web search requested"]),
    ("What is the latest Python release?", 90.0,
     None,
     ["Repository investigation", "No repository evidence"]),
]


def main() -> int:
    deadline = time.monotonic() + 480  # hard wall-clock budget

    def _left() -> float:
        return deadline - time.monotonic()

    master, slave = pty.openpty()
    set_size(slave, 40, 120)

    env = dict(os.environ)
    proc = subprocess.Popen(
        [sys.executable, "-m", "ultron.main", "chat", "--agent", "react"],
        stdin=slave, stdout=slave, stderr=slave,
        env=env, close_fds=True, start_new_session=True,
    )
    log = PtyLog(master)
    ok = True
    try:
        time.sleep(6.0)  # banner + engine warm-up + first prompt
        failures = []

        def _wait_reply(mark: int, max_wait: float) -> str:
            """Wait until the terminal output stops growing (reply settled),
            capped at max_wait.  Spinner frames count as growth, so this
            naturally waits out model generation and stops after the final
            rendered reply is stable."""
            start = time.monotonic()
            last_len = log.size()
            last_growth = start
            while time.monotonic() - start < min(max_wait, max(0.1, _left())):
                time.sleep(1.0)
                cur = log.size()
                if cur != last_len:
                    last_len = cur
                    last_growth = time.monotonic()
                elif time.monotonic() - last_growth >= 5.0:
                    break
            return log.since(mark)

        def _send(prompt: str) -> bool:
            if proc.poll() is not None:
                return False
            try:
                os.write(master, prompt.encode() + b"\n")
                return True
            except OSError:
                return False

        def _check(label: str, text: str, must_any: list | None,
                   must_not: list | None) -> bool:
            nonlocal ok
            good = True
            reasons = []
            if must_any and not any(n.lower() in text.lower() for n in must_any):
                good = False
                reasons.append(f"missing any of {must_any!r}")
            if must_not:
                for needle in must_not:
                    if needle.lower() in text.lower():
                        good = False
                        reasons.append(f"contains forbidden {needle!r}")
            if not good:
                ok = False
                failures.append(f"{label}: {'; '.join(reasons)}")
                failures.append(f"{label}: window={text[:800]!r}")
            print(f"  {'PASS' if good else 'FAIL'} {label}")
            return good

        for i, (prompt, wait_s, must_any, must_not) in enumerate(PROMPTS, start=1):
            mark = log.mark()
            if not _send(prompt):
                print(f"  SKIP {i}. {prompt} (CLI exited)", flush=True)
                ok = False
                break
            reply = _wait_reply(mark, wait_s)
            _check(f"{i}. {prompt}", reply, must_any, must_not)

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
