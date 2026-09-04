# PROJECT ULTRON — ARCHITECTURAL & MIGRATION ANCHOR STATE

> **Purpose of this document:**
> This file is the single source of truth for the ongoing development, architectural state, and migration progress of Project Ultron.
> **Rule for all coding agents:** Read this file FIRST before taking any action to understand what has been completed, what is in progress, current architectural constraints, and test baselines to prevent hallucinations or regressing completed work. Update this document immediately after completing each migration phase or architectural change.

---

## 1. System Overview & Active Migration Context

Project Ultron is undergoing an **Ollama → llama.cpp / llama-server** migration as defined in `migration_plan.md`.

- **Target Local LLM Backend**: `llama-server` (OpenAI-compatible HTTP API at `http://127.0.0.1:8080`) backed by native Apple Silicon Metal GPU acceleration (`-ngl 99`).
- **Active Model Running Locally**: `Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf` stored at `/Users/aravindhan/models/qwen2.5-coder-7b-instruct-q4_k_m.gguf`.
- **Inference Pipeline**:
  $$\text{Ultron Python process} \longrightarrow \text{LlamaCppEngine} \longrightarrow \text{llama-server HTTP API} \longrightarrow \text{GGUF Model} \longrightarrow \text{Apple Silicon Metal GPU}$$

---

## 2. Non-Negotiable Architectural Rules

1. **BaseEngine Contract Preservation**: Agents (`SimpleAgent`, `ReActAgent`) interact only with `BaseEngine` (`generate(...)`, `stream(...)`). Agents and tools must NEVER be directly coupled to `llama.cpp`.
2. **No `llama-cpp-python`**: Communication with `llama-server` must strictly use HTTP (`httpx` async calls) to avoid native C bindings or crashing Python processes.
3. **Zero Weakening of Security**: The `SecurityBoundary` (`src/ultron/core/security/`) and its 3-tier gating system must remain 100% intact.
4. **Zero Capability Routing Changes**: `Intent -> Capability -> Tool` routing logic and all 38 registered tools must not be modified.
5. **No Blind Deletion of Ollama**: Ollama files and references are only removed after the replacement is fully validated across unit, integration, and live model-in-the-loop tests.
6. **No Machine-Specific Hardcoded Paths**: Model paths and URLs must remain cleanly configurable via environment variables / `.env` / configs.

---

## 3. Migration Phase Progress Tracking

### Phase 0: Baseline Audit & Validation — [COMPLETED]
- **Pytest Baseline**: 1,590 / 1,590 tests passing (100%).
- **Ruff Lint Baseline**: 0 errors, 0 warnings.
- **Stress & Security Audit**: 29 / 29 checks passing (100%).
- **Ollama Dependency Map**: Completed and categorized.

### Phase 1: Implement `LlamaCppEngine` — [COMPLETED]
- **File**: `src/ultron/core/engine/llama_cpp.py`
- **Implementation**:
  - `LlamaCppEngine(BaseEngine)` implemented using `httpx.AsyncClient`.
  - Full non-streaming generation via `POST /v1/chat/completions`.
  - Real SSE streaming via `POST /v1/chat/completions` using `aiter_lines()`.
  - Model discovery via `GET /v1/models`.
  - Multimodal capability detection via `GET /props`.
- **Targeted Test Suite**: `tests/test_llamacpp_engine.py` (8/8 unit tests passing with mock HTTP layer).
- **Full Test Suite**: 1,598 / 1,598 tests passing (100%).
- **Ruff Status**: Clean (0 errors).
- **Live Model-in-the-Loop Test**: Verified against real local `llama-server` running Qwen2.5-Coder on Metal GPU (successful model discovery, generation, and streaming).

### Phase 2: Switch Factory & Default Configuration — [PENDING]
- Switch `src/ultron/core/engine/__init__.py` to return `LlamaCppEngine` from `get_engine()`.
- Update `configs/models.yaml` to set `llama_cpp` as enabled/default.
- Update `/model` slash command in `src/ultron/main.py`.

### Phase 3: Multimodal & Agent Compatibility — [PENDING]
- Update `handle_image` in `src/ultron/core/agents/simple.py` and `tests/test_multimodal.py`.

### Phase 4: Clean Ollama References & Deprecation — [PENDING]
- Remove `src/ultron/core/engine/ollama.py`.
- Update docs (`README.md`, `AGENTS.md`, `PROJECT_CONTEXT.md`, `docs/multimodal.md`).
- Repository-wide search ensuring 0 Ollama runtime references remain.

### Phase 5: Final Validation — [PENDING]
- End-to-end live CLI interactive check (`ultron chat`) and full regression test suite.

---

## 4. Current Test & Verification Snapshot

- **Pytest**: `1598 passed in 28.40s`
- **Ruff**: `All checks passed!`
- **Harness Verification**: `29/29 checks passed`

### Phase 2: Make LlamaCpp the Active Ultron Engine — [COMPLETED]
- **Files Modified**:
  - `src/ultron/core/config.py`: Added `llama_cpp_base_url` (default `"http://127.0.0.1:8080"`) and `timeout` (default `120.0s`).
  - `src/ultron/core/engine/__init__.py`: Updated `get_engine()` to return `LlamaCppEngine` as default local LLM backend.
  - `configs/models.yaml`: Enabled `llama_cpp` runtime configuration by default.
  - `tests/test_engine_factory.py`: Added 5 targeted unit tests for factory behavior, default engine type, streaming, generation, and agent binding.
- **Test Results**:
  - `test_engine_factory.py` + `test_llamacpp_engine.py`: 13/13 passed.
  - Full pytest suite: 1,603 / 1,603 passed (100%).
  - Ruff lint: Clean (0 errors).
- **Real Model-in-the-Loop CLI Test**:
  - SimpleAgent routed real prompt queries ("hello", "Explain recursion in Java with a simple example.") through `LlamaCppEngine` to local `llama-server` running Qwen2.5-Coder on Metal GPU, generating full coherent responses without errors or Ollama dependencies.

### Phase 3: Migrate Model Management from Ollama Semantics — [COMPLETED]
- **Files Modified**:
  - `src/ultron/main.py`: Updated `/model` slash command handler to remove Ollama `/api/tags` assumptions and error messages. Displays clean status showing active model identifier, `llama.cpp (llama-server)` backend, and server URL (`http://127.0.0.1:8080`). Supports both interactive model selection and inline switches (`/model <model_name>`), syncing session, state, `.env` file, and `engine.model`.
  - `tests/test_main_slash.py`: Added 3 unit tests verifying `/model` display status, inline switching, questionary picker selection, and offline/empty server handling with zero Ollama dependencies.
- **Test Results**:
  - `tests/test_main_slash.py`: 14 / 14 tests passed (100%).
  - Full pytest suite: 1,606 / 1,606 tests passed (100%).
  - Ruff lint: Clean (0 errors).
- **Real Model-in-the-Loop CLI Test**:
  - Validated live `/model` slash command output: accurately displayed `llama.cpp (llama-server)` backend, server endpoint `http://127.0.0.1:8080`, and successfully switched active model state.

### Phase 3 Verification Gate — [COMPLETED & VERIFIED]
- **Inspection**:
  - `llama-server` is single-model per process. `/v1/models` reports the loaded GGUF. Passing different model names in `/v1/chat/completions` does NOT change server weights or switch models.
- **Truthful Semantics Implementation**:
  - `/model` queries the live server via `get_active_model()` and displays the true active GGUF path, backend name, and endpoint URL.
  - Rejects attempts to dynamically switch to a different/unloaded model and provides instructions on restarting `llama-server` with the target `-m <model.gguf>`.
  - Guarantees state (`session.active_model`, `engine.model`, `settings.model`) is strictly synced with reality.
- **Verification Metrics**:
  - `tests/test_main_slash.py`: 15/15 passed.
  - Full pytest suite: 1,607 / 1,607 passed (100%).
  - Ruff: Clean (0 errors).
  - Real Model-in-the-Loop Test: Verified truthful active model display, rejection of unloaded model requests without corrupting state, and subsequent truthful generation.

### Phase 4: Multimodal / Vision Migration to llama.cpp — [COMPLETED]
- **Files Modified**:
  - `src/ultron/core/engine/llama_cpp.py`: Updated `supports_images()` to parse `modalities.vision` and `has_mmproj` from `llama-server /props`. Fully handles translating image messages into OpenAI-compatible `image_url` data URIs for `/v1/chat/completions`.
  - `src/ultron/core/agents/simple.py`: Removed all `ollama pull llava` error strings from `handle_image`. Provides truthful, backend-neutral instructions referencing `--mmproj` and vision GGUF models.
  - `tests/test_multimodal.py`: Migrated tests from Ollama `/api/show` mocks to `LlamaCppEngine` and `/props` mocks.
  - `tests/test_llamacpp_engine.py`: Added tests for multimodal payload formatting and `supports_images()` modalities parsing.
  - `docs/multimodal.md`: Updated architecture design documentation to reflect `LlamaCppEngine` and `llama-server`.
- **Test Results**:
  - `test_multimodal.py` + `test_llamacpp_engine.py`: 34 / 34 passed.
  - Full pytest suite: 1,609 / 1,609 passed (100%).
  - Ruff lint: Clean (0 errors).
- **Runtime Verification**:
  - Validated live text-only model detection (`modalities.vision: false`).
  - Tested live image analysis attempt against text-only Qwen2.5-Coder model: correctly detected unsupported vision, clearly and truthfully informed the user without failing or crashing.

### Phase 5: Real Model-in-the-Loop Validation Migration — [COMPLETED]
- **Harnesses Audited & Migrated**:
  - `_react_routing_live_check.py`: Updated PTY harness from Ollama to `llama-server`.
  - Created `_phase5_live_validation.py`: Comprehensive real-runtime test harness executing all 10 live model scenarios against Metal GPU `llama-server`.
- **Live Scenarios Validated (10/10 PASS)**:
  1. Basic conversation (`SimpleAgent`, Qwen2.5-Coder text generation).
  2. Repository intelligence (`ReActAgent`, located `BaseEngine` in repository).
  3. ReAct routing (`ReActAgent`, directory listing tool intent).
  4. Tool execution (`SimpleAgent`, `run pwd` executed real command).
  5. Web routing (`SimpleAgent`, external search routing).
  6. Security boundary (`SimpleAgent`, denied credential command, required confirmation for state-change).
  7. Multi-step task (`SimpleAgent`, sequential tool steps).
  8. Streaming (`LlamaCppEngine.stream()`, real token chunks).
  9. `/model` status (`handle_slash_command`, synchronized with live server).
  10. Unsupported vision handling (`SimpleAgent`, graceful rejection on text-only model).
- **Test Metrics**:
  - Full pytest regression suite: 1,609 / 1,609 passed (100%).
  - Ruff lint: Clean (0 errors).
  - Real Model-in-the-Loop: 10 / 10 passed (100%).

### Phase 6: Complete Ollama Eradication — [COMPLETED]
- **Deletions & Cleanups**:
  - `src/ultron/core/engine/ollama.py`: Completely deleted.
  - `src/ultron/core/engine/__init__.py`: Removed `OllamaEngine` export. Public API strictly exports `BaseEngine`, `LlamaCppEngine`, and `get_engine`.
  - `configs/models.yaml`: Removed dead `ollama:` configuration block.
  - Cleaned all docstrings and operational documentation (`README.md`, `docs/multimodal.md`, `semantic.py`, `simple.py`, `test_planner.py`).
- **Scan Verification**:
  - Zero active Ollama runtime dependencies, imports, or configuration parameters remain across `src/`, `configs/`, `docs/`, `tests/`, and `README.md`.
- **Regression & Model-in-the-Loop Metrics**:
  - Pytest suite: 1,609 / 1,609 passed (100%).
  - Ruff lint: Clean (0 errors).
  - Live Model-in-the-Loop (`_phase5_live_validation.py`): 10 / 10 scenarios passed against live Metal GPU `llama-server`.

### Phase 7: Final Hostile Migration Audit — [COMPLETED & VERIFIED]
- **Forensics & Audit Findings**:
  - Full codebase scan: Zero active Ollama runtime dependencies, imports, configuration parameters, or operational docs.
  - Negative assertion in `tests/test_main_slash.py:313` ensures zero Ollama mentions surface in CLI error strings.
  - Import forensics: `LlamaCppEngine` and `BaseEngine` import cleanly; `OllamaEngine` correctly triggers `ImportError`.
  - Offline server handling: Gracefully catches and reports network connection failure without silent fallbacks.
  - Model-in-the-loop: 10 / 10 scenarios passed against live Metal GPU `llama-server`.
  - Full pytest suite: 1,609 / 1,609 passed (100%).
  - Ruff lint: Clean (0 errors).
