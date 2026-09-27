"""
Real-time Phase 3 Validation Runner.
Executes the real Ultron CLI in the fastapi-backend repository,
monitors runtime events, cache operations, and model interactions.
"""
import fcntl
import json
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
ROOT_DIR = "/Users/aravindhan/ultron"
WORKSPACE_DIR = os.path.join(ROOT_DIR, "fastapi-backend")
SCRATCH = os.path.join(ROOT_DIR, "scratch")
EVENTS_DIR = os.path.expanduser("~/.ultron/events")
os.makedirs(SCRATCH, exist_ok=True)


def set_size(fd: int, h: int, w: int) -> None:
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", h, w, 0, 0))


class PtyDrainer:
    def __init__(self, master: int) -> None:
        self.master = master
        self.buffer = bytearray()
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while not self.stop_event.is_set():
            r, _, _ = select.select([self.master], [], [], 0.1)
            if r:
                try:
                    chunk = os.read(self.master, 65536)
                except OSError:
                    break
                if not chunk:
                    break
                with self.lock:
                    self.buffer.extend(chunk)

    def mark(self) -> int:
        with self.lock:
            return len(self.buffer)

    def since(self, m: int) -> str:
        with self.lock:
            return bytes(self.buffer[m:]).decode(errors="replace")

    def all_text(self) -> str:
        with self.lock:
            return bytes(self.buffer).decode(errors="replace")

    def clean_text(self) -> str:
        return ANSI.sub("", self.all_text())

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=1.0)


def get_latest_event_file(since_timestamp: float) -> str | None:
    if not os.path.exists(EVENTS_DIR):
        return None
    candidates = []
    for f in os.listdir(EVENTS_DIR):
        if f.startswith("task_") and f.endswith(".jsonl"):
            p = os.path.join(EVENTS_DIR, f)
            mtime = os.path.getmtime(p)
            if mtime >= since_timestamp - 2.0:
                candidates.append((mtime, p))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def parse_context_built_events(event_file: str) -> list[dict]:
    events = []
    if not os.path.exists(event_file):
        return events
    with open(event_file, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                if data.get("event_type") == "context_built":
                    events.append(data)
            except Exception:  # noqa: BLE001, S112
                continue

    return events


def run_session(session_name: str, turns: list[tuple[str, int]], log_file_name: str) -> dict:
    print("\n==================================================")
    print(f"STARTING REAL CLI SESSION: {session_name}")
    print(f"Working Directory: {WORKSPACE_DIR}")
    print("==================================================")

    master, slave = pty.openpty()
    set_size(slave, 40, 140)

    env = dict(os.environ)
    env["TERM"] = "xterm-256color"
    env["PYTHONPATH"] = f"{ROOT_DIR}/src"
    env.pop("ULTRON_NO_SERVER", None)
    env["ULTRON_VERBOSE"] = "1"
    env["ULTRON_LLAMA_SERVER_PORT"] = "8085"
    env["ULTRON_LLAMA_CPP_BASE_URL"] = "http://127.0.0.1:8085"

    python_exe = os.path.join(ROOT_DIR, ".venv", "bin", "python")

    # Clean existing server processes before starting
    try:
        from ultron.core.engine.server import LlamaServerManager
        LlamaServerManager.terminate_running_servers(port=8085)
    except Exception:  # noqa: BLE001, S110
        pass

    proc = subprocess.Popen(

        [python_exe, "-m", "ultron.main", "chat", "--agent", "react"],
        stdin=slave,
        stdout=slave,
        stderr=slave,
        env=env,
        cwd=WORKSPACE_DIR,
        close_fds=True,
        start_new_session=True,
    )

    drainer = PtyDrainer(master)
    turn_results = []

    def wait_for_ready(timeout: float = 60.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            text = drainer.clean_text()
            if "Ultron AI" in text and "❯" in text:
                return True
            if proc.poll() is not None:
                return False
            time.sleep(0.5)
        return False

    try:
        if not wait_for_ready(60.0):
            print("ERROR: Ultron CLI did not reach ready prompt within 60s")
            return {"success": False, "error": "Startup timeout", "turns": []}

        print("Ultron CLI is READY. Beginning turns...")
        time.sleep(2.0)

        for turn_idx, (prompt, timeout_sec) in enumerate(turns, start=1):
            turn_start = time.time()
            m = drainer.mark()
            print(f"\n--- [Turn {turn_idx}] Sending Prompt ({len(prompt)} chars) ---")
            print(f"Prompt: {prompt}")

            os.write(master, (prompt + "\n").encode())

            deadline = time.time() + timeout_sec
            last_change = time.time()
            last_len = drainer.mark()
            confirmations_handled = 0
            last_conf_mark = m
            finished = False

            while time.time() < deadline:
                if proc.poll() is not None:
                    print(f"CLI exited prematurely with code {proc.returncode}")
                    break

                # Handle security confirmation prompts automatically
                since_conf = ANSI.sub("", drainer.since(last_conf_mark))
                if ("Do you want to allow this action?" in since_conf or "Do you want to proceed?" in since_conf):
                    time.sleep(1.0)
                    confirmations_handled += 1
                    last_conf_mark = drainer.mark()
                    last_change = time.time()
                    if "uvicorn" in since_conf:
                        print(f"  [Turn {turn_idx}] User declining unwanted daemon server (uvicorn) #{confirmations_handled}")
                        os.write(master, b"\x1b[B\n")
                    else:
                        print(f"  [Turn {turn_idx}] Auto-confirmed security prompt #{confirmations_handled}")
                        os.write(master, b"\n")
                    time.sleep(0.5)
                    continue

                cur_len = drainer.mark()
                if cur_len != last_len:
                    last_len = cur_len
                    last_change = time.time()

                # Check if idle for >= 10s and response box is rendered
                since_turn = ANSI.sub("", drainer.since(m))
                if time.time() - last_change >= 9.0 and ("╭─" in since_turn or "Finished" in since_turn or "ULTRON" in since_turn):
                    finished = True
                    break

                time.sleep(0.5)

            turn_elapsed = time.time() - turn_start
            turn_text = ANSI.sub("", drainer.since(m))
            latest_event_file = get_latest_event_file(turn_start)
            context_events = parse_context_built_events(latest_event_file) if latest_event_file else []

            print(f"--- [Turn {turn_idx}] Completed in {turn_elapsed:.1f}s (finished={finished}) ---")
            print(f"Event file: {latest_event_file}")
            print(f"CONTEXT_BUILT events: {len(context_events)}")

            turn_results.append({
                "turn": turn_idx,
                "prompt": prompt,
                "duration": turn_elapsed,
                "finished": finished,
                "event_file": latest_event_file,
                "context_events": context_events,
                "transcript_snippet": turn_text[-3000:],
            })

            time.sleep(2.0)

        # Exit chat cleanly
        try:
            os.write(master, b"/exit\n")
            time.sleep(2.0)
        except OSError:
            pass

    finally:
        try:
            if proc.poll() is None:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(timeout=3.0)
        except Exception:  # noqa: BLE001, S110
            pass

        drainer.stop()
        os.close(master)
        os.close(slave)

        out_path = os.path.join(SCRATCH, log_file_name)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(drainer.clean_text())
        print(f"Session transcript saved to {out_path}")

    return {
        "success": True,
        "turns": turn_results,
        "full_log": os.path.join(SCRATCH, log_file_name),
    }


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "TEST1_P1"

    if mode == "TEST1_P1":
        # Test 1 Process 1: Initial Investigation
        p1_prompt = (
            "I need to investigate a login reliability problem. First, inspect the repository "
            "and identify which files are responsible for authentication, session creation, "
            "credential validation, and the API endpoint that handles login. "
            "Explain how those pieces are connected. Don't modify anything."
        )
        res = run_session("TEST 1 - Process 1", [(p1_prompt, 490)], "realtime_test1_proc1.log")
        print(f"Result: {res['success']}")

    elif mode == "TEST1_P2":
        # Test 1 Process 2: Persistence & Mutation Verification
        turns = [
            (
                (
                    "Continue investigating the authentication subsystem. Without modifying anything, "
                    "identify the existing login flow and tell me which component is responsible for "
                    "validating the user's credentials before a session is created."
                ),
                400,
            ),
            (
                (
                    "Create a small documentation file named AUTH_FLOW_NOTES.md in the project root "
                    "explaining that this is a temporary investigation note. Do not modify any source code."
                ),
                400,
            ),
            (
                (
                    "Refresh your understanding of the repository and tell me whether AUTH_FLOW_NOTES.md "
                    "is now visible to repository intelligence."
                ),
                350,
            ),
            (
                "Delete the file AUTH_FLOW_NOTES.md.",
                350,
            ),
        ]
        res = run_session("TEST 1 - Process 2", turns, "realtime_test1_proc2.log")
        print(f"Result: {res['success']}")

    elif mode == "TEST2":
        # Test 2: Semantic Repository Retrieval & Control Test
        turns = [
            (
                (
                    "Users are reporting that after signing in successfully, they occasionally get treated "
                    "as unauthenticated when they make their next API request. Trace the code responsible "
                    "for preserving a user's authenticated state between the login request and subsequent "
                    "API requests. Identify the relevant implementation files and explain the flow. Do not modify anything."
                ),
                450,
            ),
            (
                (
                    "Trace why a client can successfully establish an identity with the application but "
                    "lose that identity when communicating with protected endpoints afterward. Find the "
                    "code that transfers the authenticated identity from the initial sign-in interaction "
                    "into later request handling. Do not change anything."
                ),
                450,
            ),
        ]
        res = run_session("TEST 2 - Semantic Retrieval & Control", turns, "realtime_test2.log")
        print(f"Result: {res['success']}")


