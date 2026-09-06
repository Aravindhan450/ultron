"""ultron.core.intelligence.model_lifecycle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Model Lifecycle Manager for Ultron's multi-model architecture.

Responsible for safely loading and unloading models, managing the actual
llama.cpp inference processes, and enforcing the one-active-model memory
policy for the M4 16GB unified-memory hardware profile.
"""
from __future__ import annotations

import threading
from enum import Enum, unique
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from ultron.core.engine.server import LlamaServerManager
from ultron.core.intelligence.model_catalog import ModelSpec
from ultron.core.logging import get_logger

logger = get_logger("ultron.engine.lifecycle")


@unique
class LifecycleState(str, Enum):
    """Observable state of a model's runtime residency."""
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    UNLOADING = "unloading"
    FAILED = "failed"


class ModelHandle(BaseModel):
    """
    A handle representing an actively loaded model runtime.
    Encapsulates endpoint routing without leaking raw subprocess details.
    """
    model_config = ConfigDict(frozen=True)

    model_spec: ModelSpec
    endpoint_url: str
    state: LifecycleState


class ModelLifecycleManager:
    """
    Manages model loading and guarantees safe process boundaries.
    Enforces a strict ONE ACTIVE MODEL policy to protect system memory.
    """

    def __init__(self, models_dir: Path | None = None, no_server: bool = False, endpoint_url: str = "http://127.0.0.1:8080") -> None:
        self.no_server = no_server
        self.default_endpoint_url = endpoint_url
        self.models_dir = models_dir
        self._lock = threading.RLock()
        self._states: dict[str, LifecycleState] = {}
        
        self._active_spec: ModelSpec | None = None
        self._active_server: LlamaServerManager | None = None

    def get_status(self, model_id: str) -> LifecycleState:
        """Return the current lifecycle state of a specific model."""
        with self._lock:
            # If a model claims to be LOADED but the server process died unexpectedly,
            # update the state to FAILED truthfully.
            current_state = self._states.get(model_id, LifecycleState.UNLOADED)
            if (current_state == LifecycleState.LOADED 
                and self._active_spec and self._active_spec.model_id == model_id
                and self._active_server and not self._active_server.is_running):
                self._states[model_id] = LifecycleState.FAILED
                current_state = LifecycleState.FAILED
            
            return current_state

    def get_loaded_models(self) -> list[ModelSpec]:
        """Return a list of currently loaded models (always 0 or 1 for this phase)."""
        with self._lock:
            if self._active_spec and self.get_status(self._active_spec.model_id) == LifecycleState.LOADED:
                return [self._active_spec]
            return []

    def ensure_loaded(self, model_spec: ModelSpec) -> ModelHandle:
        """
        Make the specified model available for inference.
        If another model is loaded, it will be safely unloaded first.
        This operation is idempotent.
        """
        with self._lock:
            status = self.get_status(model_spec.model_id)
            
            # Idempotency check
            if status == LifecycleState.LOADED and self._active_server and self._active_server.is_running:
                logger.info("Model %s is already loaded.", model_spec.model_id)
                return ModelHandle(
                    model_spec=model_spec,
                    endpoint_url=self._active_server.endpoint_url,
                    state=LifecycleState.LOADED
                )
            
            # If another model is active (or failed/partially active), release it
            if self._active_spec and self._active_spec.model_id != model_spec.model_id:
                logger.info("Switching models: unloading %s to make room for %s.", self._active_spec.model_id, model_spec.model_id)
                self.release(self._active_spec.model_id)
            elif self._active_spec and self._active_spec.model_id == model_spec.model_id and status == LifecycleState.FAILED:
                # Same model, but in a failed state. Needs clean up before retry.
                self.release(model_spec.model_id)

            self._states[model_spec.model_id] = LifecycleState.LOADING
            self._active_spec = model_spec
            
            if self.no_server:
                self._states[model_spec.model_id] = LifecycleState.LOADED
                logger.info("no_server is active; skipping actual model load for %s.", model_spec.model_id)
                return ModelHandle(
                    model_spec=model_spec,
                    endpoint_url=self.default_endpoint_url,
                    state=LifecycleState.LOADED
                )
            
            try:
                # Use LlamaServerManager to manage the subprocess
                model_path = model_spec.resolve_path(self.models_dir)
                
                # Create a new server manager pointing to the resolved path
                self._active_server = LlamaServerManager(
                    model_path=model_path,
                    context_length=model_spec.recommended_context_length
                )
                
                logger.info("Starting model server for %s...", model_spec.model_id)
                self._active_server.start()
                
                self._states[model_spec.model_id] = LifecycleState.LOADED
                logger.info("Model %s successfully loaded.", model_spec.model_id)
                
                return ModelHandle(
                    model_spec=model_spec,
                    endpoint_url=self._active_server.endpoint_url,
                    state=LifecycleState.LOADED
                )
                
            except Exception as e:
                # Loading failed. Clean up and mark as failed.
                logger.error("Failed to load model %s: %s", model_spec.model_id, e)
                self._states[model_spec.model_id] = LifecycleState.FAILED
                
                if self._active_server:
                    try:
                        self._active_server.stop()
                    except (OSError, RuntimeError) as stop_err:
                        logger.warning("Cleanup stop failed for %s: %s", model_spec.model_id, stop_err)
                
                raise RuntimeError(f"Failed to load model {model_spec.model_id}: {e}") from e

    def release(self, model_id: str) -> None:
        """Safely unload the specified model and terminate its runtime."""
        with self._lock:
            if self._active_spec and self._active_spec.model_id == model_id:
                self._states[model_id] = LifecycleState.UNLOADING
                if self.no_server:
                    self._states[model_id] = LifecycleState.UNLOADED
                    self._active_spec = None
                    return
                if self._active_server:
                    logger.info("Stopping model server for %s...", model_id)
                    try:
                        self._active_server.stop()
                    except (OSError, RuntimeError) as e:
                        logger.warning("Error while stopping model server for %s: %s", model_id, e)
                        # State remains accurate: it failed to unload cleanly
                        self._states[model_id] = LifecycleState.FAILED
                        return
                    
                    self._active_server = None
                    
                self._active_spec = None
                self._states[model_id] = LifecycleState.UNLOADED
                logger.info("Model %s successfully unloaded.", model_id)

    def shutdown(self) -> None:
        """Unload any actively loaded model and cleanly shut down the manager."""
        with self._lock:
            if self._active_spec:
                self.release(self._active_spec.model_id)
