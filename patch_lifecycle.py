with open("src/ultron/core/intelligence/model_lifecycle.py", "r") as f:
    content = f.read()

# Replace __init__
content = content.replace(
    "def __init__(self, models_dir: Path | None = None) -> None:",
    "def __init__(self, models_dir: Path | None = None, no_server: bool = False, endpoint_url: str = \"http://127.0.0.1:8080\") -> None:\n        self.no_server = no_server\n        self.default_endpoint_url = endpoint_url"
)

# Replace ensure_loaded
old_loading = """            self._states[model_spec.model_id] = LifecycleState.LOADING
            self._active_spec = model_spec
            
            try:"""
new_loading = """            self._states[model_spec.model_id] = LifecycleState.LOADING
            self._active_spec = model_spec
            
            if self.no_server:
                self._states[model_spec.model_id] = LifecycleState.LOADED
                logger.info("no_server is active; skipping actual model load for %s.", model_spec.model_id)
                return ModelHandle(
                    model_spec=model_spec,
                    endpoint_url=self.default_endpoint_url,
                    state=LifecycleState.LOADED
                )
            
            try:"""
content = content.replace(old_loading, new_loading)

# Replace release
old_release = """        with self._lock:
            if self._active_spec and self._active_spec.model_id == model_id:
                self._states[model_id] = LifecycleState.UNLOADING
                if self._active_server:"""
new_release = """        with self._lock:
            if self._active_spec and self._active_spec.model_id == model_id:
                self._states[model_id] = LifecycleState.UNLOADING
                if self.no_server:
                    self._states[model_id] = LifecycleState.UNLOADED
                    self._active_spec = None
                    return
                if self._active_server:"""
content = content.replace(old_release, new_release)

with open("src/ultron/core/intelligence/model_lifecycle.py", "w") as f:
    f.write(content)
