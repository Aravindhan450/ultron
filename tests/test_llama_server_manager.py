"""
Unit tests for LlamaServerManager process lifecycle.
"""
from unittest.mock import MagicMock

import httpx
import pytest

from ultron.core.engine.server import LlamaServerManager


def test_resolve_binary_custom_missing():
    manager = LlamaServerManager(binary="/nonexistent/path/to/llama-server")
    with pytest.raises(FileNotFoundError, match="Configured llama-server binary was not found"):
        manager.resolve_binary()


def test_resolve_binary_via_path(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/llama-server" if name == "llama-server" else None)
    manager = LlamaServerManager(binary="llama-server")
    assert manager.resolve_binary() == "/usr/bin/llama-server"


def test_validate_model_path_missing(tmp_path):
    manager = LlamaServerManager(model_path=str(tmp_path / "missing.gguf"))
    with pytest.raises(FileNotFoundError, match="Configured GGUF model file not found"):
        manager.validate_model_path()


def test_validate_model_path_not_a_file(tmp_path):
    manager = LlamaServerManager(model_path=str(tmp_path))
    with pytest.raises(ValueError, match="is not a file"):
        manager.validate_model_path()


def test_validate_model_path_success(tmp_path):
    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"GGUFDATA")
    manager = LlamaServerManager(model_path=str(model_file))
    assert manager.validate_model_path() == model_file.resolve()


def test_check_endpoint_occupied_true(monkeypatch):
    manager = LlamaServerManager(host="127.0.0.1", port=8080)

    def mock_get(client, url):
        return httpx.Response(status_code=200, json={"data": []})

    monkeypatch.setattr(httpx.Client, "get", mock_get)
    assert manager.check_endpoint_occupied() is True


def test_check_endpoint_occupied_false(monkeypatch):
    manager = LlamaServerManager(host="127.0.0.1", port=8080)

    def mock_get(client, url):
        raise httpx.ConnectError("Connection refused")

    monkeypatch.setattr(httpx.Client, "get", mock_get)
    assert manager.check_endpoint_occupied() is False


def test_start_refuses_when_occupied(monkeypatch, tmp_path):
    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"GGUFDATA")
    manager = LlamaServerManager(model_path=str(model_file))
    monkeypatch.setattr(manager, "check_endpoint_occupied", lambda: True)

    with pytest.raises(RuntimeError, match="An existing server is already active"):
        manager.start()


def test_start_success_and_readiness(monkeypatch, tmp_path):
    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"GGUFDATA")
    log_file = tmp_path / "llama-server.log"

    manager = LlamaServerManager(
        binary="llama-server",
        model_path=str(model_file),
        log_file=log_file,
        startup_timeout=2.0,
    )

    monkeypatch.setattr(manager, "resolve_binary", lambda: "/mock/llama-server")
    monkeypatch.setattr(manager, "check_endpoint_occupied", lambda: False)

    fake_proc = MagicMock()
    fake_proc.pid = 12345
    fake_proc.poll.return_value = None
    monkeypatch.setattr("subprocess.Popen", lambda *args, **kwargs: fake_proc)

    # Mock readiness check
    attempts = 0

    def mock_get(client, url):
        nonlocal attempts
        attempts += 1
        if attempts >= 2:
            return httpx.Response(status_code=200, json={"data": []})
        raise httpx.ConnectError("Not ready yet")

    monkeypatch.setattr(httpx.Client, "get", mock_get)

    manager.start()
    assert manager.is_running is True
    assert manager._owned is True


def test_server_exits_prematurely_during_startup(monkeypatch, tmp_path):
    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"GGUFDATA")
    log_file = tmp_path / "llama-server.log"

    manager = LlamaServerManager(
        binary="llama-server",
        model_path=str(model_file),
        log_file=log_file,
        startup_timeout=2.0,
    )

    monkeypatch.setattr(manager, "resolve_binary", lambda: "/mock/llama-server")
    monkeypatch.setattr(manager, "check_endpoint_occupied", lambda: False)

    fake_proc = MagicMock()
    fake_proc.pid = 12345
    fake_proc.poll.return_value = 1  # Exited with code 1
    monkeypatch.setattr("subprocess.Popen", lambda *args, **kwargs: fake_proc)

    with pytest.raises(RuntimeError, match="llama-server exited prematurely during startup"):
        manager.start()


def test_stop_idempotent(monkeypatch):
    manager = LlamaServerManager()
    assert manager.is_running is False
    # Stopping an unowned / unstarted server is safe and does nothing
    manager.stop()
    manager.stop()


def test_stop_owned_process(monkeypatch):
    manager = LlamaServerManager()
    fake_proc = MagicMock()
    fake_proc.pid = 9999
    # poll returns None (alive initially), then 0 (exited after sigterm)
    fake_proc.poll.side_effect = [None, 0, 0, 0]

    manager._process = fake_proc
    manager._owned = True

    monkeypatch.setattr("os.getpgid", lambda pid: pid)
    monkeypatch.setattr("os.killpg", lambda pgid, sig: None)

    manager.stop()
    assert manager._process is None
    assert manager._owned is False

