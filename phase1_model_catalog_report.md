# Phase 1: Model Catalog Final Report

## Changed

* `src/ultron/core/intelligence/model_catalog.py` (Modified: Implemented full catalog logic)
* `tests/test_model_catalog.py` (Created: Unit tests for the new catalog)

*(Note: Unrelated pre-existing changes to `conftest.py` from the Phase 3.3 test run were reverted to keep the diff clean).*

## Model catalog

The following three canonical models were successfully registered:

**Qwen3-8B**
* ID: `qwen3-8b`
* Role: `PRIMARY`
* Quantization: `Q5_K_M`
* Filename: `Qwen3-8B-Q5_K_M.gguf`
* Capabilities: `GENERAL`, `REASONING`, `PLANNING`, `TOOL_USE`, `AGENT`, `STRUCTURED_OUTPUT`

**Gemma 3 4B IT**
* ID: `gemma-3-4b-it`
* Role: `FAST`
* Quantization: `Q8_K_M`
* Filename: `gemma-3-4b-it-Q8_K_M.gguf`
* Capabilities: `GENERAL`, `LIGHTWEIGHT_REASONING`, `SUMMARIZATION`, `STRUCTURED_OUTPUT`, `VISION`

**Qwen2.5-Coder-7B-Instruct**
* ID: `qwen2.5-coder-7b-instruct`
* Role: `CODING`
* Quantization: `Q8_0`
* Filename: `qwen2.5-coder-7b-instruct-q8_0.gguf`
* Capabilities: `CODING`, `DEBUGGING`, `REPOSITORY_ANALYSIS`, `CODE_GENERATION`, `STRUCTURED_OUTPUT`, `TOOL_USE`

## Tests

```text
catalog tests: PASS
intelligence tests: PASS
llama.cpp tests: PASS
agent/runtime tests: PASS
full suite: PASS
```

**Commands Used:**
* `pytest tests/test_model_catalog.py -v` (43 passed in 0.17s)
* `pytest -q` (1717 passed, 6 deselected in 31.51s)
* `ruff check .` (All checks passed!)

## Existing issues

**Introduced by this change:** None.
**Pre-existing:** None identified during the full test suite run. The suite cleanly passes.

## Architecture impact

The newly introduced `ModelCatalog` lays the static foundation for the multi-model architecture. It introduces strong typing (`ModelSpec`, `ModelCapability`, `ModelRole`) for models.

In the future:
* **ModelRouter** will query the catalog using capability lookups (e.g., `catalog.models_with_capability(ModelCapability.VISION)`) to dynamically match user tasks to the best-suited model.
* **ModelLifecycleManager** will use the catalog's `resolve_path()` and `recommended_context_length` to seamlessly locate, load, and unload the underlying GGUF binaries into the llama.cpp engine.
* **AgentRuntime** will coordinate between the Router and the Lifecycle manager, swapping out the execution engine on-the-fly without needing to hardcode any model names.

This change is small, focused, typed, heavily tested, and completely reversible, with absolutely zero impact on the existing agent runtime.
