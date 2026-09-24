"""
LlamaServerManager: Process lifecycle management for local llama-server instances.

Ensures that:
  - llama-server is started cleanly before interactive chat sessions.
  - Model path and binary existence are verified up front.
  - Existing running instances on the endpoint are safely detected without hijacking.
  - Process readiness is verified before Ultron starts talking to it.
  - Process termination is strict, isolated to the spawned process group, and idempotent.
"""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx

from ultron.core.config import settings
from ultron.core.logging import get_logger

logger = get_logger("ultron.engine.server")


class LlamaServerManager:
    """Manages the subprocess lifecycle of a local llama-server instance."""

    def __init__(
        self,
        binary: str | None = None,
        model_path: str | Path | None = None,
        host: str | None = None,
        port: int | None = None,
        context_length: int | None = None,
        gpu_layers: int | None = None,
        startup_timeout: float | None = None,
        shutdown_timeout: float | None = None,
        log_file: Path | None = None,
    ) -> None:
        self.binary_name = binary or settings.llama_server_binary
        self.model_path = model_path or settings.llama_server_model_path or settings.model
        self.host = host or settings.llama_server_host
        self.port = port if port is not None else settings.llama_server_port
        self.context_length = context_length if context_length is not None else settings.llama_server_context_length
        self.gpu_layers = gpu_layers if gpu_layers is not None else settings.llama_server_gpu_layers
        self.startup_timeout = startup_timeout if startup_timeout is not None else settings.llama_server_startup_timeout
        self.shutdown_timeout = shutdown_timeout if shutdown_timeout is not None else settings.llama_server_shutdown_timeout

        self.log_file = log_file or (settings.data_dir / "llama-server.log")
        self._process: subprocess.Popen[Any] | None = None
        self._log_fh: Any = None
        self._owned: bool = False

    @property
    def endpoint_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def resolve_binary(self) -> str:
        """Finds the llama-server executable from config or PATH."""
        if os.path.isabs(self.binary_name):
            if os.path.isfile(self.binary_name) and os.access(self.binary_name, os.X_OK):
                return self.binary_name
            raise FileNotFoundError(
                f"Configured llama-server binary was not found or is not executable: '{self.binary_name}'"
            )

        resolved = shutil.which(self.binary_name)
        if not resolved:
            raise FileNotFoundError(
                f"Could not find '{self.binary_name}' in PATH. "
                "Please install llama.cpp (e.g. `brew install llama.cpp`) or configure ULTRON_LLAMA_SERVER_BINARY."
            )
        return resolved

    def validate_model_path(self) -> Path:
        """Validates that the target GGUF model file exists on disk."""
        if not self.model_path or self.model_path in ("gemini-2.5-flash", "default", "llama3"):
            # Check if default home directory models exist
            common_search = list((Path.home() / "models").glob("*.gguf")) if (Path.home() / "models").exists() else []
            if common_search:
                return common_search[0].resolve()
            raise ValueError(
                "No valid GGUF model path configured. Please set ULTRON_LLAMA_SERVER_MODEL_PATH "
                "or ULTRON_MODEL in .env pointing to a local .gguf model file."
            )

        p = Path(self.model_path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(
                f"Configured GGUF model file not found: '{p}'. "
                "Please verify ULTRON_LLAMA_SERVER_MODEL_PATH in .env."
            )
        if not p.is_file():
            raise ValueError(f"Configured model path is not a file: '{p}'")
        return p

    def check_endpoint_occupied(self) -> bool:
        """Checks if a server is already listening on the target endpoint."""
        try:
            with httpx.Client(timeout=1.0) as client:
                resp = client.get(f"{self.endpoint_url}/v1/models")
                return resp.status_code == 200
        except (httpx.HTTPError, OSError):
            return False

    def start(self) -> None:
        """
        Launches llama-server in an isolated process group and waits for readiness.
        """
        if self.is_running:
            return

        # Check if already occupied
        if self.check_endpoint_occupied():
            logger.info("Existing server detected at %s. Refusing to take ownership.", self.endpoint_url)
            raise RuntimeError(
                f"An existing server is already active at {self.endpoint_url}. "
                "Ultron will not hijack or terminate an unmanaged process. "
                "Please stop the existing server or configure a different port with ULTRON_LLAMA_SERVER_PORT."
            )

        binary = self.resolve_binary()
        model_file = self.validate_model_path()

        cmd = [
            binary,
            "-m", str(model_file),
            "--host", self.host,
            "--port", str(self.port),
            "-c", str(self.context_length),
            "-ngl", str(self.gpu_layers),
        ]

        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self._log_fh = open(self.log_file, "a", encoding="utf-8")  # noqa: SIM115
        self._log_fh.write(f"\n--- [Ultron starting llama-server at {time.strftime('%Y-%m-%d %H:%M:%S')}] ---\n")

        self._log_fh.write(f"Command: {' '.join(cmd)}\n")
        self._log_fh.flush()

        logger.info("Spawning llama-server: %s", " ".join(cmd))
        try:
            # start_new_session=True creates an isolated process group so signals
            # sent to the terminal (e.g. Ctrl+C) do not kill llama-server before
            # Ultron can gracefully shut it down.
            self._process = subprocess.Popen(
                cmd,
                stdout=self._log_fh,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
            self._owned = True
        except OSError as e:
            self._cleanup_log_handle()
            raise RuntimeError(f"Failed to spawn llama-server: {e}") from e

        self.wait_until_ready()

    def wait_until_ready(self) -> None:
        """Polls the server readiness endpoint until it responds or times out."""
        start_time = time.monotonic()
        url = f"{self.endpoint_url}/v1/models"

        while time.monotonic() - start_time < self.startup_timeout:
            if self._process is not None and self._process.poll() is not None:
                exit_code = self._process.poll()
                self.stop()
                raise RuntimeError(
                    f"llama-server exited prematurely during startup with exit code {exit_code}. "
                    f"Check server log at {self.log_file} for details."
                )

            try:
                with httpx.Client(timeout=1.0) as client:
                    resp = client.get(url)
                    if resp.status_code == 200:
                        logger.info("llama-server is ready at %s", self.endpoint_url)
                        return
            except (httpx.HTTPError, OSError):
                pass

            time.sleep(0.2)

        self.stop()
        raise TimeoutError(
            f"llama-server failed to become ready within {self.startup_timeout}s. "
            f"Check server log at {self.log_file}."
        )

    def stop(self) -> None:
        """
        Gracefully terminates ONLY the process group started by this manager.
        """
        if not self._owned or self._process is None:
            self._cleanup_log_handle()
            return

        pid = self._process.pid
        logger.info("Stopping owned llama-server process (PID %d)...", pid)

        if self._process.poll() is None:
            try:
                # Terminate the process group
                pgid = os.getpgid(pid)
                os.killpg(pgid, signal.SIGTERM)
            except (ProcessLookupError, OSError):
                try:
                    self._process.terminate()
                except OSError:
                    pass

            # Wait for graceful exit
            start_time = time.monotonic()
            while time.monotonic() - start_time < self.shutdown_timeout:
                if self._process.poll() is not None:
                    break
                time.sleep(0.1)

            # If still alive, force kill
            if self._process.poll() is None:
                logger.warning("llama-server PID %d did not exit gracefully; sending SIGKILL.", pid)
                try:
                    pgid = os.getpgid(pid)
                    os.killpg(pgid, signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    try:
                        self._process.kill()
                    except OSError:
                        pass
                try:
                    self._process.wait(timeout=2.0)
                except (subprocess.TimeoutExpired, OSError):
                    pass

        self._process = None
        self._owned = False
        self._cleanup_log_handle()

    def _cleanup_log_handle(self) -> None:
        if self._log_fh is not None:
            try:
                self._log_fh.flush()
                self._log_fh.close()
            except OSError:
                pass
            self._log_fh = None

    @classmethod
    def terminate_running_servers(
        cls,
        port: int | None = None,
        host: str | None = None,
        timeout: float = 3.0,
    ) -> list[int]:
        """
        Finds and terminates any llama-server processes listening on the target port
        or matching the configured endpoint. Ensures the port is freed.
        """
        target_port = port if port is not None else settings.llama_server_port
        killed_pids: list[int] = []

        candidate_pids: set[int] = set()

        # 1. Identify candidate PIDs listening on target port via lsof
        try:
            out = subprocess.check_output(
                ["lsof", "-ti", f":{target_port}", "-sTCP:LISTEN"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            for p in out.splitlines():
                p_str = p.strip()
                if p_str.isdigit():
                    candidate_pids.add(int(p_str))
        except (subprocess.SubprocessError, FileNotFoundError, OSError):
            pass

        # 2. Identify candidate PIDs matching llama-server via pgrep
        try:
            out = subprocess.check_output(
                ["pgrep", "-fl", "llama-server"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            for line in out.splitlines():
                parts = line.strip().split(None, 1)
                if parts and parts[0].isdigit():
                    pid = int(parts[0])
                    cmd = parts[1] if len(parts) > 1 else ""
                    if (
                        f"{target_port}" in cmd
                        or f":{target_port}" in cmd
                        or (target_port == 8080 and "--port" not in cmd and "-p" not in cmd)
                    ):
                        candidate_pids.add(pid)
        except (subprocess.SubprocessError, FileNotFoundError, OSError):
            pass

        # 3. Filter candidates to strictly verify they are llama-server processes
        llama_pids: list[int] = []
        for pid in candidate_pids:
            try:
                cmd = subprocess.check_output(
                    ["ps", "-p", str(pid), "-o", "command="],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
                if "llama-server" in cmd or "llama_server" in cmd:
                    llama_pids.append(pid)
            except (subprocess.SubprocessError, FileNotFoundError, OSError):
                pass

        # 4. Terminate each identified llama-server process
        for pid in llama_pids:
            logger.info("Terminating running llama-server process on port %d (PID %d)...", target_port, pid)
            try:
                try:
                    pgid = os.getpgid(pid)
                    os.killpg(pgid, signal.SIGTERM)
                except OSError:
                    os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                continue
            except OSError as e:
                logger.warning("Failed to send SIGTERM to llama-server PID %d: %s", pid, e)

            # Wait up to timeout for clean exit
            start = time.monotonic()
            exited = False
            while time.monotonic() - start < timeout:
                try:
                    os.kill(pid, 0)
                    time.sleep(0.1)
                except ProcessLookupError:
                    exited = True
                    break
                except OSError:
                    break

            if not exited:
                logger.warning("llama-server PID %d did not exit within %ss; sending SIGKILL.", pid, timeout)
                try:
                    try:
                        pgid = os.getpgid(pid)
                        os.killpg(pgid, signal.SIGKILL)
                    except OSError:
                        os.kill(pid, signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    pass

            killed_pids.append(pid)

        return killed_pids
