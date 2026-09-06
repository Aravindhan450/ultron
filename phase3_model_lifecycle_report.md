# Phase 3 Model Lifecycle Manager Report

## 1. Implementation Summary
Implemented `ModelLifecycleManager` inside `src/ultron/core/intelligence/model_lifecycle.py`. 
It establishes a strict boundary for safely loading models, tracking their lifecycle state, and enforcing the M4 16GB memory constraint (exactly one model active at a time). It defines standard observable states (`LifecycleState`) and encapsulates the loaded process via a typed `ModelHandle`. It uses `threading.RLock()` to prevent race conditions during concurrent loads or model switching.

## 2. Existing Runtime Reuse
Discovered and fully reused the existing `LlamaServerManager` in `src/ultron/core/engine/server.py`. 
This existing abstraction already securely manages isolated process groups (`start_new_session=True`), verifies model binaries and GGUF paths, and provides a readiness polling mechanism (`wait_until_ready()` checking `/v1/models`). The `ModelLifecycleManager` orchestrates instances of `LlamaServerManager` instead of inventing a new loader.

## 3. Lifecycle State Machine
```text
UNLOADED
   ↓
LOADING
   ↓
LOADED
   ↓
UNLOADING
   ↓
UNLOADED

* Failure paths transition to FAILED.
* Health check failure during load transitions to FAILED and triggers resource cleanup.
* Process crash while LOADED transitions truthfully to FAILED.
```

## 4. Memory Strategy
The architecture guarantees the **One-Active-Model Baseline** required for Apple Silicon M4 16GB.
When `ensure_loaded(model_spec)` is called, the manager inspects the current `_active_spec`. If another model is actively loaded or partially active/failed, it immediately calls `release()` on that model. This guarantees that `LlamaServerManager.stop()` strictly cleans up the old process before allocating memory for the new model.

## 5. Model Switching
Switching is fully automated and idempotent:
* **Gemma → Coder:** `ensure_loaded(coder)` detects Gemma is active, issues `stop()` to Gemma's server, updates Gemma state to `UNLOADED`, sets Coder to `LOADING`, spawns Coder's server, and finally sets Coder to `LOADED`.
* **Coder → Qwen3:** Similar flow. The lock serializes this safely.
* **Qwen3 → Qwen3:** Safe. Returns the existing `ModelHandle` immediately without a process restart.

## 6. Failure Handling
* **Load Failure:** If `LlamaServerManager.start()` fails (e.g., file missing or port in use), the error is caught, the state transitions to `FAILED`, the server is forcibly stopped, and the exception is bubbled up.
* **Process Crash:** `get_status()` does a live check (`is_running`) on the underlying process. If a model was `LOADED` but the process died, the manager self-corrects the state to `FAILED`.
* **Unload Failure:** If `stop()` raises an `OSError`, the state updates to `FAILED` instead of falsely claiming it was cleanly `UNLOADED`.

## 7. Concurrency
The `ModelLifecycleManager` uses a `threading.RLock()`.
When requests race (e.g., Thread A asks for Gemma, Thread B asks for Gemma), Thread A acquires the lock, loads the model, and completes. Thread B acquires the lock, observes the model is already `LOADED` and active, and immediately returns the cached handle, entirely preventing duplicate server instantiations.

## 8. Files Changed
* `src/ultron/core/intelligence/model_lifecycle.py` (Created)
* `tests/test_model_lifecycle.py` (Created)

## 9. Tests
```text
ModelCatalog: PASS (43 tests)
ModelRouter: PASS (15 tests)
LifecycleManager: PASS (9 tests)
Intelligence: PASS
AgentRuntime: PASS
ReAct/SimpleAgent: PASS
llama.cpp: PASS
Full suite: PASS (1741 passed, 6 deselected)
```
Exact commands used:
* `pytest tests/test_model_lifecycle.py -v` (9 passed in 0.27s)
* `pytest -q` (1741 passed in ~30s)

## 10. Model-in-the-Loop
Model-in-the-loop validation unavailable in this environment (the unit tests use `unittest.mock.MagicMock` to stub out `LlamaServerManager` as requested to avoid requiring actual local GGUFs). Real hardware validation was deliberately skipped in Phase 3 unit testing per instruction #29.

## 11. Diff Scope
`git diff` confirms that absolutely no changes were made to the `AgentRuntime`, `SecurityBoundary`, `LlamaServerManager`, `ModelRouter`, or `ModelCatalog`. Only `model_lifecycle.py` and its test suite were introduced.

## 12. Known Issues
None.

## 13. Completion
Phase 3 complete.
Waiting for human review before Phase 4.
