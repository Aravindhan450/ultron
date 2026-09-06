"""Tests for ultron.core.intelligence.model_lifecycle.

Validates Phase 3 Model Lifecycle Manager: state transitions, one-active-model
enforcement, idempotency, failure recovery, concurrency, and shutdown.
"""
from __future__ import annotations

import concurrent.futures
import time
from unittest.mock import MagicMock, patch

import pytest

from ultron.core.intelligence.model_catalog import get_default_catalog
from ultron.core.intelligence.model_lifecycle import (
    LifecycleState,
    ModelLifecycleManager,
)


@pytest.fixture
def catalog():
    return get_default_catalog()


@pytest.fixture
def gemma_spec(catalog):
    return catalog.get("gemma-3-4b-it")


@pytest.fixture
def coder_spec(catalog):
    return catalog.get("qwen2.5-coder-7b-instruct")


@pytest.fixture
def qwen_spec(catalog):
    return catalog.get("qwen3-8b")


@pytest.fixture
def manager():
    return ModelLifecycleManager()


class TestLifecycleManager:

    @patch("ultron.core.intelligence.model_lifecycle.LlamaServerManager")
    def test_initial_state(self, mock_server, manager, gemma_spec):
        assert manager.get_status(gemma_spec.model_id) == LifecycleState.UNLOADED
        assert manager.get_loaded_models() == []

    @patch("ultron.core.intelligence.model_lifecycle.LlamaServerManager")
    def test_ensure_loaded_success(self, mock_server_class, manager, gemma_spec):
        mock_server = MagicMock()
        mock_server.is_running = True
        mock_server.endpoint_url = "http://127.0.0.1:8080"
        mock_server_class.return_value = mock_server

        handle = manager.ensure_loaded(gemma_spec)
        assert handle.state == LifecycleState.LOADED
        assert handle.endpoint_url == "http://127.0.0.1:8080"
        assert manager.get_status(gemma_spec.model_id) == LifecycleState.LOADED
        
        loaded = manager.get_loaded_models()
        assert len(loaded) == 1
        assert loaded[0].model_id == gemma_spec.model_id
        
        mock_server.start.assert_called_once()

    @patch("ultron.core.intelligence.model_lifecycle.LlamaServerManager")
    def test_ensure_loaded_idempotent(self, mock_server_class, manager, gemma_spec):
        mock_server = MagicMock()
        mock_server.is_running = True
        mock_server.endpoint_url = "http://127.0.0.1:8080"
        mock_server_class.return_value = mock_server

        # First load
        manager.ensure_loaded(gemma_spec)
        assert mock_server.start.call_count == 1
        
        # Second load
        manager.ensure_loaded(gemma_spec)
        # Should not start again
        assert mock_server.start.call_count == 1
        assert manager.get_status(gemma_spec.model_id) == LifecycleState.LOADED

    @patch("ultron.core.intelligence.model_lifecycle.LlamaServerManager")
    def test_switching_models(self, mock_server_class, manager, gemma_spec, coder_spec):
        # We need to simulate different server instances for Gemma and Coder.
        mock_gemma_server = MagicMock()
        mock_gemma_server.is_running = True
        mock_gemma_server.endpoint_url = "http://127.0.0.1:8080"
        
        mock_coder_server = MagicMock()
        mock_coder_server.is_running = True
        mock_coder_server.endpoint_url = "http://127.0.0.1:8080"
        
        mock_server_class.side_effect = [mock_gemma_server, mock_coder_server]

        # Load Gemma
        manager.ensure_loaded(gemma_spec)
        assert manager.get_status(gemma_spec.model_id) == LifecycleState.LOADED
        
        # Load Coder
        manager.ensure_loaded(coder_spec)
        
        # Gemma should be stopped and unloaded
        mock_gemma_server.stop.assert_called_once()
        assert manager.get_status(gemma_spec.model_id) == LifecycleState.UNLOADED
        
        # Coder should be loaded
        mock_coder_server.start.assert_called_once()
        assert manager.get_status(coder_spec.model_id) == LifecycleState.LOADED
        
        loaded = manager.get_loaded_models()
        assert len(loaded) == 1
        assert loaded[0].model_id == coder_spec.model_id

    @patch("ultron.core.intelligence.model_lifecycle.LlamaServerManager")
    def test_load_failure(self, mock_server_class, manager, gemma_spec):
        mock_server = MagicMock()
        mock_server.start.side_effect = RuntimeError("Simulated startup failure")
        mock_server.endpoint_url = "http://127.0.0.1:8080"
        mock_server_class.return_value = mock_server

        with pytest.raises(RuntimeError, match="Simulated startup failure"):
            manager.ensure_loaded(gemma_spec)
            
        assert manager.get_status(gemma_spec.model_id) == LifecycleState.FAILED
        assert manager.get_loaded_models() == []
        mock_server.stop.assert_called_once() # Ensure cleanup occurred

    @patch("ultron.core.intelligence.model_lifecycle.LlamaServerManager")
    def test_health_check_crash(self, mock_server_class, manager, gemma_spec):
        mock_server = MagicMock()
        mock_server.is_running = True
        mock_server.endpoint_url = "http://127.0.0.1:8080"
        mock_server_class.return_value = mock_server

        manager.ensure_loaded(gemma_spec)
        assert manager.get_status(gemma_spec.model_id) == LifecycleState.LOADED
        
        # Simulate process crash
        mock_server.is_running = False
        assert manager.get_status(gemma_spec.model_id) == LifecycleState.FAILED
        assert manager.get_loaded_models() == []

    @patch("ultron.core.intelligence.model_lifecycle.LlamaServerManager")
    def test_release_and_shutdown(self, mock_server_class, manager, gemma_spec):
        mock_server = MagicMock()
        mock_server.is_running = True
        mock_server.endpoint_url = "http://127.0.0.1:8080"
        mock_server_class.return_value = mock_server

        manager.ensure_loaded(gemma_spec)
        manager.shutdown()
        
        mock_server.stop.assert_called_once()
        assert manager.get_status(gemma_spec.model_id) == LifecycleState.UNLOADED
        assert manager.get_loaded_models() == []
        
        # Repeated shutdown should be safe
        manager.shutdown()
        assert mock_server.stop.call_count == 1

    @patch("ultron.core.intelligence.model_lifecycle.LlamaServerManager")
    def test_release_failure(self, mock_server_class, manager, gemma_spec):
        mock_server = MagicMock()
        mock_server.is_running = True
        mock_server.endpoint_url = "http://127.0.0.1:8080"
        mock_server.stop.side_effect = RuntimeError("Stop failed")
        mock_server_class.return_value = mock_server

        manager.ensure_loaded(gemma_spec)
        manager.release(gemma_spec.model_id)
        
        assert manager.get_status(gemma_spec.model_id) == LifecycleState.FAILED

    @patch("ultron.core.intelligence.model_lifecycle.LlamaServerManager")
    def test_concurrency_race(self, mock_server_class, manager, gemma_spec):
        """Test that rapid concurrent loads of the same model don't spin up multiple servers."""
        mock_server = MagicMock()
        mock_server.endpoint_url = "http://127.0.0.1:8080"
        
        # We simulate a slow start to encourage a race condition if locking fails
        def slow_start():
            time.sleep(0.1)
        mock_server.start.side_effect = slow_start
        mock_server.is_running = True
        mock_server_class.return_value = mock_server

        def load_it():
            return manager.ensure_loaded(gemma_spec)

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(load_it) for _ in range(3)]
            results = [f.result() for f in futures]
            
        assert len(results) == 3
        # Should only have instantiated and started ONE server due to lock + idempotency check
        assert mock_server_class.call_count == 1
        assert mock_server.start.call_count == 1
