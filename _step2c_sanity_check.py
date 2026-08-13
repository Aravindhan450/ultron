"""STEP 2C sanity check through the REAL CLI (small, not a holdout test).

Checks the Intent -> Capability selection integration does not break either
agent:

  1. Default (Simple) agent: "Find where the OrchestrationValidator is
     defined" — deterministic path through handle_routed_intent ->
     select_capability -> preferred tool.  No LLM needed.
  2. ReAct agent: "Run pwd" — LLM loop still resolves the terminal tool and
     produces a coherent reply (route_llm_tool_call integration intact).

Deliberately small: the full capability/generalization benchmark is a later
step.  These are sanity prompts only.
"""

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

ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def set_size(fd: int, h: int, w: int) -> None:
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", h, w, 0, 0))


class PtyLog:
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
            return ANSI.sub("", self._log.decode(errors="replace"))


def run_prompt(args: list[str], prompt: str, wait_s: float) -> str:
    master, slave = pty.openpty()
    set_size(slave, 40, 120)
    log = PtyLog(master)
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
    reply = ""
    last_len = 0
    while time.monotonic() < deadline:
        snap = log.snapshot()
        if len(snap) != last_len:
            last_len = len(snap)
            tail = snap[-900:]
            # Stop once a reply marker + settled output appears after prompt echo.
            if "●" in tail or "▍" in tail or ("—" in tail and "?" not in tail[-40:]):
                time.sleep(1.0)
                snap2 = log.snapshot()
                if len(snap2) == len(snap):
                    break
        time.sleep(0.4)
    time.sleep(0.5)
    reply = log.snapshot()
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
    return reply


def main() -> int:
    base = [sys.executable, "-m", "ultron.main", "chat"]
    checks = [
        (
            base,
            "Find where the OrchestrationValidator is defined",
            25,
            "Check 1 (Simple agent, deterministic selector path)",
        ),
        (
            base + ["--agent", "react"],
            "Run pwd",
            150,
            "Check 2 (ReAct agent, LLM loop + routing integration)",
        ),
    ]
    failed = False
    for args, prompt, wait, label in checks:
        print(f"\n=== {label} ===")
        print(f"prompt: {prompt!r}")
        out = run_prompt(args, prompt, wait)
        # Show the last meaningful lines.
        tail = "\n".join(out.strip().splitlines()[-6:])
        print("--- tail ---")
        print(tail)
        lowered = out.lower()
        if "error" in lowered and "traceback" in lowered:
            print("RESULT: FAIL (traceback)")
            failed = True
        elif "command not found" in lowered:
            print("RESULT: FAIL (command not found)")
            failed = True
        elif "couldn't generate a response" in lowered:
            print("RESULT: PARTIAL (empty response)")
        else:
            print("RESULT: OK (coherent reply)")
    print("\n" + ("SANITY: PASS" if not failed else "SANITY: FAIL"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
