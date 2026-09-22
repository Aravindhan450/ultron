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

- **Pytest**: `1,779 passed, 6 deselected in 35.21s` (100% pass rate)
- **Ruff**: `All checks passed!` (0 lint errors)
- **Harness & Security Verification**: 100% checks passed, live model-in-the-loop validated across all active tiers
- **Active Model Fleet**: Gemma-3-4B (`fast`), Qwen3-8B (`primary`), Qwen2.5-Coder-7B (`coder`) dynamically managed by `ModelLifecycleManager`

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

### Phase 8: Llama-Server Session Lifecycle Implementation — [COMPLETED & VERIFIED]
- **Component Added**:
  - `src/ultron/core/engine/server.py`: Implemented `LlamaServerManager` to manage `llama-server` subprocess lifecycle.
  - Safe binary & model path resolution, isolated process group spawning (`start_new_session=True`), readiness polling via `GET /v1/models`, safe refusal to hijack existing running servers, and idempotent process-group termination (`SIGTERM` -> timeout -> `SIGKILL`).
- **CLI Wiring**:
  - `src/ultron/main.py`: Wrapped `async_chat()` in a `try...finally` block ensuring `server_manager.start()` executes before chat initialization and `server_manager.stop()` cleanly terminates ONLY the started instance upon exit, interrupt (Ctrl+C), or exception.
- **Settings Added**:
  - `ULTRON_LLAMA_SERVER_BINARY`, `ULTRON_LLAMA_SERVER_MODEL_PATH`, `ULTRON_LLAMA_SERVER_HOST`, `ULTRON_LLAMA_SERVER_PORT`, `ULTRON_LLAMA_SERVER_CONTEXT_LENGTH`, `ULTRON_LLAMA_SERVER_GPU_LAYERS`, `ULTRON_LLAMA_SERVER_STARTUP_TIMEOUT`, `ULTRON_LLAMA_SERVER_SHUTDOWN_TIMEOUT`.
- **Test Results**:
  - Unit tests in `tests/test_llama_server_manager.py`: 12 / 12 passed.
  - Full pytest suite: 1,621 / 1,621 passed (100%).
  - Ruff lint: Clean (0 errors).
  - Real Model-in-the-Loop: Verified auto-spawning, prompt interaction, and clean teardown upon exit.

---

## 5. Freebuff-Standards Coding Agent Upgrade Tracking

### Phase 1: AgentRuntime Foundation — [COMPLETED & VERIFIED]
- **Package Created**: `src/ultron/core/runtime/`
  - `__init__.py`: Public API exports.
  - `runtime.py`: `AgentRuntime` lifecycle coordinator.
  - `state.py`: Explicit `RunState` and `RuntimeStatus` state machine with deterministic transition verification (`assert_runtime_transition`).
  - `budget.py`: `RuntimeBudget` enforcing iterations, tool calls, delegations, and wall-clock timeouts.
  - `cancellation.py`: Cooperative `CancellationToken` mechanism.
  - `events.py`: Lightweight in-process `EventBus` with typed `RuntimeEvent` history.
  - `result.py`: Structured `RunResult` containing outcome status, evidence, changed files, and termination reason.
- **Integration**:
  - `src/ultron/main.py`: Wired CLI execution paths through `AgentRuntime.execute(...)` for all `ReActAgent` and `SimpleAgent` runs while preserving `SecurityBoundary` authorization.
- **Test Suite**:
  - Created `tests/test_agent_runtime.py` covering lifecycle, budget limits, timeout, cancellation, and event emissions.
  - Pytest baseline: **1,632 / 1,632 passed (100%)**.
  - Ruff lint: Clean (0 errors).
- **Model-in-the-Loop Validation**:
  - Verified repository inspection task and safe coding task against live local `llama-server` on Apple Metal GPU.

### Phase 2: Repository-Aware ContextManager Foundation — [COMPLETED & VERIFIED]
- **Package Created**: `src/ultron/core/context/`
  - `__init__.py`: Public exports.
  - `models.py`: `ContextItem`, `ContextPriority`, `ContextRetrievalResult`, `ContextRetrievalStatus`, `ContextSnapshot`, `ContextSourceType`.
  - `retrieval.py`: `RepositoryRetriever` querying files (with region bounding), AST symbols/references, lexical code search, and Git state with explicit `NOT_FOUND` contracts and path safety.
  - `manager.py`: `RepositoryContextManager` assembling multi-source context, enforcing `ContextBudgetConfig`, signature-based deduplication, prioritized compaction, and emitting observable snapshots.
- **Integration**:
  - `src/ultron/core/runtime/runtime.py`: Integrated `RepositoryContextManager` into `AgentRuntime` lifecycle coordinator.
- **Test Suite**:
  - Created `tests/test_context_manager.py` (9/9 unit tests passing).
  - Pytest suite upgraded: **1,641 / 1,641 passed (100%)**.
  - Ruff status: Clean (0 errors).
- **Model-in-the-Loop Validation**:
  - 4 real live model scenarios executed against Metal GPU `llama-server` (Repository discovery, context retrieval, missing resource `NOT_FOUND` reporting, and automated tool/test execution).

---

## 6. Real-World Capability & Reliability Hardening Tracking

### Phase 4: Real-World Capability Validation & Model Fleet Routing — [COMPLETED & VERIFIED]
- **Documentation**: Detailed findings in `ultron_phase4_real_world_capability_validation_report.md`.
- **Fleet Orchestration**: Live transitions between `gemma-3-4b-it` (`fast`), `Qwen3-8B` (`primary`), and `qwen2.5-coder-7b-instruct` (`coder`) orchestrated by `ModelLifecycleManager` without zombie processes or port conflicts.
- **Task Precedence**: Hardened deterministic classifier in `task_classification.py` (`_SE_CREATION_RE`) to ensure software creation directives take precedence over incidental explanation keywords.
- **Security & Guardrail Verification**: 100% boundary integrity under adversarial tests; interactive confirmation modals correctly gate state-modifying actions (`write_file`, `run_command`).

### Phase 5: Research Pipeline Hardening & Real Artifact Execution Layer — [COMPLETED & VERIFIED]
- **Documentation**:
  - Research Pipeline Report: `ultron_research_pipeline_validation_report.md`
  - Artifact Execution Report: `ultron_artifact_execution_validation_report.md`
  - Failure Analysis: `ultron_phase5_failure_analysis.md`
- **Workstream A — Research Pipeline Hardening**:
  - Implemented `src/ultron/core/intelligence/research.py`: multi-tier evidence extraction, domain quality classification (Tier 1–4), freshness tagging, and conflict detection.
  - Integrated into `src/ultron/core/intelligence/synthesis.py`: synthesizes answers strictly grounded in normalized evidence with verified bracketed inline citations (`[1]`, `[2]`).
  - Unit tests: `tests/test_research_pipeline.py` (7/7 passed).
- **Workstream B — Real Artifact Creation & Execution Layer**:
  - Implemented `build_artifact_plan` in `src/ultron/core/intelligence/task_planning.py`: outcome-oriented plan enforcing project scaffolding on disk (`write_file`), runtime execution & testing (`run_command`), and empirical evidence verification.
  - Interactive continuity in `src/ultron/main.py`: `react_exec_agent` state preserved across user confirmation turns so multi-step build/test cycles resume without context loss.
  - Unit tests: `tests/test_artifact_execution.py` (4/4 passed).
  - Empirical verification: Scaffolding and execution of SQLite expense tracker application (`expenses.db`, `expense_app/app.py`) verified live.
- **Quality Baseline**:
  - Pytest suite: **1,779 / 1,779 passed (100%)**.
  - Ruff lint: Clean (0 errors).

### Phase 6: Dedicated External Workspace Confinement (`ULTRON_WORKSPACE`) — [COMPLETED & VERIFIED]
- **Documentation**: Detailed findings in `ultron_workspace_sandbox_validation_report.md`.
- **Configuration & Dynamic Resolution**:
  - `src/ultron/core/config.py`: Added `workspace: str | None = None` to `Settings`, dynamically configurable via `ULTRON_WORKSPACE=~/UltronWorkspace`.
  - `src/ultron/core/tools/paths.py`: Added `get_configured_workspace()`, `get_allowed_base_dirs()`, `set_active_project_dir()`, `get_active_project_dir()`, and `resolve_project_path()`. Resolves `~` and env vars, normalizes absolute paths.
- **Security & Path Confinement**:
  - `src/ultron/security/file_policy.py` & `src/ultron/core/tools/paths.py`: Updated path confinement (`is_path_safe`) to validate against both `ALLOWED_BASE_DIR` and `get_configured_workspace()`. Strict protection against traversal (`../`, `../../`, absolute escapes, symlinks).
- **Tool Execution Context**:
  - `src/ultron/core/tools/builtin/command_runner.py`: Updated `_run_one` and `run_command` with `cwd` parameter, defaulting to `get_active_project_dir()`.
  - `src/ultron/core/tools/builtin/file_writer.py`, `file_reader.py`, `src/ultron/core/coding/edits.py`, and `workspace.py`: Relative paths resolve cleanly against active project directory within the external workspace.
- **Artifact Integration & Isolation**:
  - `src/ultron/core/intelligence/task_planning.py`: Updated `build_artifact_plan` and `_attach_coding_context` to derive project directory slugs (`~/UltronWorkspace/<slug>/`), create directory, and set active project directory context. TaskPlan tracks `project_dir`.
  - `src/ultron/main.py`: Final response reports project location (`Location: ~/UltronWorkspace/<project>`).
- **Quality Baseline**:
  - Pytest suite: **1,785 / 1,785 passed (100%)**.
  - Ruff lint: Clean (0 errors).
  - Live model validation: Demonstrated project directory creation and execution in `~/UltronWorkspace/python-expense-tracker-application/` outside Ultron repo root.

### Phase 7: Real-World Capability Validation Protocol (Tests 01–10) — [COMPLETED & VERIFIED]
- **Documentation**: Comprehensive report in [`ultron_real_world_capability_validation.md`](file:///Users/aravindhan/ultron/ultron_real_world_capability_validation.md).
- **Protocol Execution**:
  - 10 empirical tests executed autonomously under `~/UltronWorkspace/test-01-tamil-nadu-weather/` through `~/UltronWorkspace/test-10-student-expense-manager/`.
  - Level 0–4 evidence captured across filesystem artifacts, AST compilation, SQLite schema inspection, pytest test runs, and live API queries.
- **Autonomous Repair Benchmark (Test 05)**:
  - 100% autonomous reproduction, root-cause identification, and file patch via `replace_in_file` with verified 2/2 `pytest` pass.
- **Independent Ground Truth Verification**:
  - Open-Meteo live weather data gathered simultaneously across 5 key Tamil Nadu districts (Chennai 33.1°C, Coimbatore 29.8°C, Madurai 34.2°C, Salem 32.5°C, Tiruchirappalli 33.8°C) for direct output verification.
- **Security & Confinement**:
  - 100% strict confinement inside `~/UltronWorkspace/`; zero escapes; zero source repository pollution.

### Phase 8: Autonomous Coding Agent v1 (Reliability, Real Application Launch, Verification & Repair) — [COMPLETED & VERIFIED]
- **Explicit Acceptance Criteria & Evidence Levels**:
  - Implemented `AcceptanceCriterion`, `AcceptanceCriterionStatus`, and `EvidenceLevel` (Level 0: Generated, Level 1: Created, Level 2: Executed, Level 3: Launched, Level 4: Interacted, Level 5: Verified, Level 6: Repaired) in `src/ultron/core/types.py`.
  - `TaskState` and `TaskPlan` now track structured acceptance criteria and enforce `all_required_criteria_satisfied()` before `mark_complete()` allows task completion.
- **Application Lifecycle Tracking**:
  - Added `ApplicationLifecycleState` (`CREATED`, `EXECUTED`, `STARTED`, `READY`, `INTERACTED`, `VERIFIED`) across `TaskState` and `Observation`.
- **Failure Classification & Loop Circuit Breaking**:
  - Expanded `FailureCategory` (`SYNTAX`, `COMPILATION`, `TEST_ASSERTION`, `DEPENDENCY`, `CONFIGURATION`, `ENVIRONMENT`, `RUNTIME`, `PERMISSION`, `TIMEOUT`, `NETWORK_API`, `DATA`, `UI`, `UNKNOWN`) in `src/ultron/core/coding/executor.py`.
  - Added macOS Tkinter virtualenv diagnosis: automatically identifies `_tkinter` omission and provides actionable strategy repair guidance (`/usr/bin/python3 <app.py>`).
  - Gated repetition detection: blocks identical repeated failures and injects root cause diagnosis and strategy-shift guidance directly into agent observations.
- **Autonomous Multi-Step Plan Architecture**:
  - Upgraded `build_artifact_plan` in `src/ultron/core/intelligence/task_planning.py` to the full 4-step autonomous lifecycle (Scaffold -> Launch -> Interact -> Verify with independent evidence).
- **Quality Baseline**:
  - Pytest suite: **1,792 / 1,792 passed (100%)**.
  - Ruff lint: Clean (0 errors).
  - Real macOS application launch verified: Tamil Nadu Weather Desktop GUI running live on macOS screen via native Aqua bindings.



