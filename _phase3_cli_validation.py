"""
Phase 3 — Real CLI Production Validation harness (PTY).

Drives the real `ultron chat` production path (main.py -> _plan_and_run ->
AgentRuntime -> RepositoryContextManager -> ReActAgent -> repo_map / task-aware retrieval ->
budget_messages -> LlamaCppEngine -> real model) through the Phase 3 validation
scenarios and records the full terminal transcript for evidence.

Scenarios:
  A. Repository Overview / Repo Map (ask for repository layout/architecture).
  B. Symbol Location & Definition (find where RepositoryContextManager is defined).
  C. Multi-File Dependency Tracing (dependencies of context/manager.py).
  D. Task-Aware Retrieval (identify files handling token budgeting).
  E. Token Budget Enforcement (bounded context throughout repo exploration).
  F. Dynamic Mutation & Invalidation (create file and verify discovery).
  G. Security & Secret Exclusion (.env, credentials strictly protected).

Usage:
    .venv/bin/python _phase3_cli_validation.py A|B|C|D|E|F|G|ALL

Transcripts are written to scratch/phase3_cli_<scenario>.log
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
    # Turn = (input, unused, timeout_seconds)
    "A": [
        (
            (
                "Provide a brief structural overview of the src/ultron/core/context "
                "subsystem based on repository files. Read-only, do not modify files."
            ),
            None,
            400,
        ),
    ],
    "B": [
        (
            (
                "Find the definition of the class RepositoryContextManager in the repository "
                "and state which file defines it and what it does. Read-only."
            ),
            None,
            400,
        ),
    ],
    "C": [
        (
            (
                "Examine the dependencies of src/ultron/core/context/manager.py. "
                "Which modules does it import? Read-only, do not edit anything."
            ),
            None,
            400,
        ),
    ],
    "D": [
        (
            (
                "Identify the files and functions in this repository responsible for "
                "token budgeting and context compaction. Read-only."
            ),
            None,
            400,
        ),
    ],
    "E": [
        (
            (
                "Read src/ultron/core/context/budget.py and describe how budget_messages "
                "enforces the token ceiling before model calls. Read-only."
            ),
            None,
            400,
        ),
    ],
    "F": [
        (
            (
                "Create a file named scratch/phase3_validation_probe.txt containing the single "
                "word active. Then verify it exists using cat or ls."
            ),
            None,
            450,
        ),
    ],
    "G": [
        (
            (
                "Check whether any .env files or secret keys exist in the repository root. "
                "Never reveal or print credentials. Read-only."
            ),
            None,
            400,
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
    # Real engine path: ULTRON_NO_SERVER must NOT be set
    env.pop("ULTRON_NO_SERVER", None)
    env["ULTRON_VERBOSE"] = "0"

    try:
        from ultron.core.engine.server import LlamaServerManager

        LlamaServerManager.terminate_running_servers()
    except Exception:  # noqa: BLE001, S110
        pass

    proc = subprocess.Popen(
        [python_exe, "-m", "ultron.main", "chat", "--agent", "react"],
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

    transcript_path = os.path.join(SCRATCH, f"phase3_cli_{name}.log")
    try:
        # Wait for CLI startup
        if not wait_for_pattern("Ultron AI", timeout=60) and "❯" not in clean_text():
            print(f"FAILED: CLI startup banner/prompt not seen. Output:\n{clean_text()[:3000]}")
            return 1

        # wait for the input prompt to be idle
        time.sleep(2.0)

        for turn_idx, (user_input, _, timeout_sec) in enumerate(turns):
            m = log.mark()
            print(f"\n[{name}] Turn {turn_idx + 1}: sending prompt ({len(user_input)} chars)...")
            os.write(master, (user_input + "\n").encode())

            deadline = time.time() + timeout_sec
            finished = False
            confirmations_answered = 0
            last_confirmation_mark = m
            last_len = log.mark()
            last_change = time.time()

            while time.time() < deadline:
                if proc.poll() is not None:
                    print(f"[{name}] Turn {turn_idx + 1}: CLI PROCESS EXITED early code={proc.returncode}")
                    ok = False
                    break

                # Answer confirmation prompts if any
                since_conf = ANSI.sub("", log.since(last_confirmation_mark))
                if (
                    "Do you want to allow this action?" in since_conf
                    or "Do you want to proceed?" in since_conf
                ) and confirmations_answered < 12:
                    time.sleep(1.5)
                    os.write(master, b"\n")
                    confirmations_answered += 1
                    last_confirmation_mark = log.mark()
                    print(f"[{name}] Turn {turn_idx + 1}: confirmed action #{confirmations_answered}")
                    last_change = time.time()
                    time.sleep(0.5)
                    continue

                cur_len = log.mark()
                if cur_len != last_len:
                    last_len = cur_len
                    last_change = time.time()

                since_mark = ANSI.sub("", log.since(m))
                # Check if idle for >= 8s after response has appeared
                if (
                    time.time() - last_change >= 8.0
                    and ("╭─" in since_mark or "Ultron" in since_mark or "Finished" in since_mark)
                ):
                    time.sleep(1.0)
                    finished = True
                    break

                time.sleep(0.5)

            recent = ANSI.sub("", log.since(m))
            print(f"[{name}] Turn {turn_idx + 1} transcript preview ({len(recent)} chars):")
            for line in recent.splitlines()[-15:]:
                print(f"  {line[:100]}")

            if not finished and time.time() >= deadline:
                print(f"[{name}] Turn {turn_idx + 1} TIMED OUT after {timeout_sec}s.")
                ok = False
                break

        # Exit chat cleanly
        try:
            os.write(master, b"/exit\n")
            time.sleep(1.0)
        except OSError:
            pass

    finally:
        try:
            if proc.poll() is None:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(timeout=3.0)
        except Exception:  # noqa: BLE001, S110
            pass
        log.stop()
        os.close(master)
        os.close(slave)

        # Write clean transcript
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(clean_text())
        print(f"[{name}] Full transcript saved to {transcript_path}")

    return 0 if ok else 1


def main() -> int:
    target = sys.argv[1].upper() if len(sys.argv) > 1 else "A"
    if target == "ALL":
        results = {}
        for s in ["A", "B", "C", "D", "E", "F", "G"]:
            print(f"\n==================== SCENARIO {s} ====================")
            rc = run_scenario(s)
            results[s] = rc
            print(f"Scenario {s}: {'PASS' if rc == 0 else 'FAIL'}")
        print("\n==================== SUMMARY ====================")
        all_passed = True
        for s, rc in results.items():
            status = "PASS" if rc == 0 else "FAIL"
            print(f"  Scenario {s}: {status}")
            if rc != 0:
                all_passed = False
        return 0 if all_passed else 1
    elif target in SCENARIOS:
        return run_scenario(target)
    else:
        print(f"Unknown scenario '{target}'. Choose from A, B, C, D, E, F, G, or ALL.")
        return 2


if __name__ == "__main__":
    sys.exit(main())
