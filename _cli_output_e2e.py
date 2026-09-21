"""
End-to-End PTY test verifying CLI Output Contract and log isolation in real terminal.

Starts the real `ultron chat --no-server` CLI in a pseudo-terminal (PTY) and verifies that:
1. The conversational interface is clean.
2. Startup banner and boxed responses render canonical Ultron UI.
3. Internal observability logs (ModelRouter, Task classification, llama-server lifecycle,
   raw execution headers, etc.) are strictly isolated and NEVER leaked into the user terminal.
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

FORBIDDEN_LEAK_PATTERNS = [
    "Task classified:",
    "ModelRouter decision:",
    "Spawning llama-server:",
    "llama-server is ready:",
    "DEBUG [",
    "DEBUG:ultron",
    "INFO:ultron",
    "Executed tool '",
]


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

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)


def main() -> int:
    master, slave = pty.openpty()
    set_size(slave, 35, 110)

    root_dir = os.path.dirname(os.path.abspath(__file__))
    venv_python = os.path.join(root_dir, ".venv", "bin", "python")
    python_exe = venv_python if os.path.exists(venv_python) else sys.executable

    src_dir = os.path.join(root_dir, "src")
    env = dict(os.environ)
    if env.get("TERM") in (None, "dumb"):
        env["TERM"] = "xterm-256color"
    current_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{src_dir}:{current_pp}" if current_pp else src_dir
    env["ULTRON_NO_SERVER"] = "1"
    env["ULTRON_VERBOSE"] = "0"

    proc = subprocess.Popen(
        [python_exe, "-m", "ultron.main", "chat", "--no-server"],
        stdin=slave,
        stdout=slave,
        stderr=slave,
        env=env,
        close_fds=True,
        start_new_session=True,
    )
    log = PtyLog(master)
    ok = True

    def wait_for_pattern(pattern: str, timeout: float = 10.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            clean = ANSI.sub("", log.since(0))
            if pattern in clean:
                return True
            if proc.poll() is not None:
                return False
            time.sleep(0.05)
        return False

    try:
        # 1. Wait for startup banner
        if not wait_for_pattern("Ultron AI", timeout=12.0):
            err_text = log.since(0)
            print(f"FAILED: Timed out waiting for startup banner. PTY output:\n{err_text}")
            return 1

        time.sleep(0.5)

        # 2. Send greeting 'hii'
        os.write(master, b"hii\n")
        if not wait_for_pattern("Hello! How can I help you today?", timeout=8.0):
            print("FAILED: Greeting response was not received.")
            ok = False

        time.sleep(0.5)

        # 3. Send /help slash command
        os.write(master, b"/help\n")
        if not wait_for_pattern("Available Commands", timeout=8.0):
            print("FAILED: /help table was not received.")
            ok = False

        time.sleep(0.5)

        # 4. Read entire session transcript and check for forbidden leaks
        full_transcript = ANSI.sub("", log.since(0))

        leaks_found = []
        for forbidden in FORBIDDEN_LEAK_PATTERNS:
            if forbidden in full_transcript:
                leaks_found.append(forbidden)

        if leaks_found:
            print(f"FAILED: Leaked internal logs detected in terminal transcript: {leaks_found}")
            ok = False
        else:
            print("PASS: Zero internal log leakage detected in terminal session.")

        # 5. Check response formatting
        if not re.search(r"╭─+\s*ULTRON\s*─+╮", full_transcript):
            print("FAILED: Canonical boxed response header '╭─── ULTRON ───╮' missing from transcript.")
            ok = False
        else:
            print("PASS: Canonical UI.render_response() box detected.")

        print("OVERALL RESULT:", "PASS" if ok else "FAIL")
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
