"""End-to-end proof: the whole conversation re-renders on EVERY resize.

Starts the real `ultron chat` CLI in a pty and builds a mixed conversation
with no LLM required (greetings via the fast-path intent detector + slash
commands). A background thread continuously drains the pty master (exactly
what a real terminal emulator does — without it the child would block on a
full pty output buffer). Then the window is resized several times
(100 -> 60 -> 120 -> 75) while the prompt is live, and after EVERY resize
the entire recorded transcript must be re-rendered exactly once: every
marker present with its exact expected count, none duplicated, none lost.

Markers (all produced by our own prints, unaffected by prompt_toolkit's
prompt-area redraws):

  - "Hello! How can I help you today?"  x2  (two greeting responses)
  - "Available Commands"                x1  (/help table header)
  - "Memory graph"                      x1  (/memory stats line)
  - "Ultron AI"                         x1  (startup banner, replayed by
                                            the rebuild block on reflow)
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

MARKERS = {
    "Hello! How can I help you today?": 2,
    "Available Commands": 1,
    "Memory graph": 1,
    "Ultron AI": 1,
}


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
            return bytes(self._log[mark:]).decode(errors="replace")

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)


def check_window(text: str) -> bool:
    """Verify every marker appears exactly its expected number of times."""
    clean = ANSI.sub("", text)
    ok = True
    for marker, want in MARKERS.items():
        got = clean.count(marker)
        status = "ok" if got == want else f"expected {want}, found {got}"
        if got != want:
            ok = False
        print(f"  {marker!r}: {status}")
    return ok


def main() -> int:
    master, slave = pty.openpty()
    set_size(slave, 30, 100)

    env = dict(os.environ)
    proc = subprocess.Popen(
        [sys.executable, "-m", "ultron.main", "chat"],
        stdin=slave,
        stdout=slave,
        stderr=slave,
        env=env,
        close_fds=True,
        start_new_session=True,
    )
    log = PtyLog(master)
    ok = True
    try:
        # Mark immediately so the startup banner lands inside the first window.
        start = log.mark()
        time.sleep(2.5)  # banner + first prompt

        # Build the conversation at 100 cols.
        for line in ("hi", "hello", "/help", "/memory"):
            os.write(master, line.encode() + b"\n")
            time.sleep(0.9)
        time.sleep(0.8)
        print("conversation @100:")
        ok &= check_window(log.since(start))

        # Resize mid-conversation: the whole transcript must re-render exactly
        # once at each new width — nothing lost, nothing duplicated.
        for target in (60, 120, 75):
            mark = log.mark()
            set_size(slave, 30, target)
            time.sleep(1.5)  # watcher tick (0.2s) + reflow + prompt redraw
            time.sleep(0.5)
            print(f"resize ->{target}:")
            ok &= check_window(log.since(mark))

        print("RESULT:", "PASS" if ok else "FAIL")
        return 0 if ok else 1
    finally:
        log.stop()
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, OSError):
            pass
        proc.wait(timeout=5)
        os.close(master)
        os.close(slave)


if __name__ == "__main__":
    sys.exit(main())
