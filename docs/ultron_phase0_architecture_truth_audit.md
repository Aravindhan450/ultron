# Ultron Phase 0 — Architecture Truth Audit Report

**Date:** 23 September 2026  
**Document Version:** 1.0  
**Target Specification:** `ULTRON_Autonomous_Coding_Harness_Gap_Closure_Plan.pdf` (Version 1.0)  
**Status:** COMPLETE  

---

## 1. Executive Summary

This architecture truth audit establishes the empirical baseline of Project Ultron prior to executing the 11-phase Autonomous Coding Harness Gap Closure Plan. The audit was conducted strictly against the executable source code, test suites, and live harnesses in `/Users/aravindhan/ultron`. No production code was modified during this phase.

### Key Audit Discoveries

1. **Repository Reality vs. Documentation**:
   - Tool count: Documentation claims 38 registered tools (`AGENTS.md`, `PROJECT_CONTEXT.md`). The canonical registry (`src/ultron/core/tools/definitions.py`) actually defines **58 tools**.
   - Model Infrastructure: The Gap Closure Plan noted that `model_catalog.py` might be a scaffold. In reality, `model_catalog.py` (278 lines), `model_router.py` (244 lines), and `model_lifecycle.py` (186 lines) are **implemented** and integrated into `AgentRuntime`.
   - Orchestration Subsystem: An entire 2,000+ line module exists at `src/ultron/core/orchestration/` (implementing `Supervisor`, `DelegationRequest`, `AgentRegistry`, `Workflow`, and `Validation`), yet it is **completely disconnected** from `main.py`, `AgentRuntime`, `SimpleAgent`, and `ReActAgent`.
   - Scaffolds: All four specialized agents (`orchestrator.py`, `codeact.py`, `operative.py`, `monitor.py`), all sandboxing modules (`security/sandbox/`), peripheral subsystems (`mcp/`, `voice/`, `platform/`, `ui/web`, `ui/desktop`, `ui/api`), and memory modules (`faiss.py`, `hybrid.py`) are **7-to-8-line scaffold placeholders**.

2. **Core Operational Bottlenecks**:
   - **Unbounded In-Loop Context**: While `RepositoryContextManager` enforces a 4,000-token budget on static repo context, conversation history (`messages`) inside `ReActAgent.run()` accumulates every thought and tool observation without token budgeting, summarization, or truncation.
   - **Self-Referential Prompt Verification**: Task verification (`_verify_task` and `_verify_plan_task` in `react.py`) relies almost entirely on LLM self-evaluation via JSON output rather than independent, deterministic test execution (`VerificationSpec`).
   - **No Safe Mutation Recovery**: There is no Git checkpointing, dirty-worktree protection, or rollback engine (`core/vcs/` does not exist).
   - **Direct Host Execution**: Commands run directly on the host via `subprocess.Popen(shell=True)` without execution backend abstractions or OS-level sandboxing.

---

## 2. Repository Snapshot

- **Total Python Modules in `src/ultron`**: 98 files
- **Total Unit & Integration Tests**: 1,845 collected (1,839 passed, 6 deselected `mitl` tests)
- **Linter Status**: `ruff check .` passed with 0 errors (clean)
- **Active Supported Agents**: `simple`, `react` (declared in `src/ultron/core/agents/__init__.py:SUPPORTED_AGENTS`)
- **Default Engine Backend**: `LlamaCppEngine` (HTTP client targeting `llama-server` OpenAI endpoint at `http://127.0.0.1:8080`)
- **Active Model Catalog**: `qwen3-8b` (PRIMARY), `gemma-3-4b-it` (FAST), `qwen2.5-coder-7b-instruct` (CODING)

---

## 3. Actual Runtime Flow (`ultron chat`)

Tracing execution from CLI prompt submission to final response:

```text
CLI (ultron.main:async_chat)
  │
  ├─> PromptSession (prompt_toolkit, width-adaptive ChatSession)
  │     └─> user inputs text
  │
  ├─> Slash Command Handler (handle_slash_command)
  │     └─> /security, /agent, /model, /tools, /resume, etc.
  │
  ├─> Task Preparation (prepare_task_for_execution in task_planning.py)
  │     ├─> Deterministic classification (classify_task_deterministic)
  │     ├─> Plan generation (plan_task / LLM)
  │     ├─> Plan validation (validate_plan)
  │     └─> Returns TaskState (or prompts for clarification)
  │
  ├─> Agent Runtime (AgentRuntime.execute in runtime.py)
  │     ├─> Model Routing (ModelRouter.route)
  │     ├─> Lifecycle check (ModelLifecycleManager.ensure_loaded)
  │     ├─> RunState initialized (CREATED -> INITIALIZING -> RUNNING)
  │     ├─> In-memory EventBus emits RUN_STARTED
  │     ├─> Context Snapshot Assembly (RepositoryContextManager.assemble_snapshot)
  │     └─> Invokes agent.run(user_input, history, task=..., ...)
  │
  ├─> ReAct Agent Execution Loop (ReActAgent.run in react.py)
  │     ├─> Injects system prompt + schema + context_snapshot
  │     ├─> While loop: 1 .. active_budget.max_iterations
  │     │     ├─> Emits STEP_STARTED & MODEL_CALLED to in-memory EventBus
  │     │     ├─> engine.generate(messages) (OpenAI format via LlamaCppEngine)
  │     │     ├─> extract_tool_call(response)
  │     │     │
  │     │     ├─[If tool_call is None]─> Completion Verification:
  │     │     │     ├─> Calls _verify_task() or _verify_plan_task()
  │     │     │     ├─> Generates LLM verification prompt
  │     │     │     ├─> Parses requirement JSON satisfaction
  │     │     │     └─> If satisfied: returns final ChatMessage; Else: continues loop
  │     │     │
  │     │     └─[If tool_call extracted]─> Tool Execution Pipeline:
  │     │           ├─> Coding gate: checks failure thrashing (_coding_gate)
  │     │           ├─> Route tool call: route_llm_tool_call()
  │     │           ├─> Tool Dispatch: _route_tool(tool_name, arguments)
  │     │           ├─> Security Boundary: boundary.check()
  │     │           │     ├─> DENY: hard-blocked, returns error observation
  │     │           │     ├─> CONFIRM: returns PendingAction -> hands to CLI
  │     │           │     └─> ALLOW: executes tool directly on host
  │     │           ├─> Observation returned and appended to messages
  │     │           ├─> Updates TaskState, CodeContext, ModificationTracker
  │     │           └─> Next iteration
  │     │
  │     └─> If PendingAction yielded:
  │           └─> main.py confirms with questionary -> executes action -> resumes agent.run()
  │
  └─> CLI renders response via UI.render_response()
```

---

## 4. Agent Implementation Status

| Agent | Location | Class / Symbol | Status | Reality & Responsibility |
|---|---|---|---|---|
| **BaseAgent** | `src/ultron/core/agents/base.py` | `BaseAgent` | IMPLEMENTED | Abstract contract: `__init__(engine)`, `run(user_input, history) -> ChatMessage`. Lacks runtime/event/state abstractions. |
| **SimpleAgent** | `src/ultron/core/agents/simple.py` | `SimpleAgent` | IMPLEMENTED | 3,251 lines. 24 deterministic `detect_*` regex/intent routers + tool handlers + single-shot LLM fallback. |
| **ReActAgent** | `src/ultron/core/agents/react.py` | `ReActAgent` | IMPLEMENTED | 2,109 lines. Core multi-turn Reason+Act execution loop. Handles prompt building, tool routing, LLM verification. |
| **Orchestrator** | `src/ultron/core/agents/orchestrator.py` | N/A | SCAFFOLD | 8-line placeholder file with docstring only. |
| **CodeAct** | `src/ultron/core/agents/codeact.py` | N/A | SCAFFOLD | 8-line placeholder file with docstring only. |
| **Operative** | `src/ultron/core/agents/operative.py` | N/A | SCAFFOLD | 8-line placeholder file with docstring only. |
| **Monitor** | `src/ultron/core/agents/monitor.py` | N/A | SCAFFOLD | 8-line placeholder file with docstring only. |
| **Supervisor** | `src/ultron/core/orchestration/delegation.py` | `Supervisor` | UNUSED | Implemented in disconnected orchestration module. Never called from CLI or `AgentRuntime`. |

---

## 5. Model Infrastructure Status

| Component | File Path | Status | Reality & Capabilities |
|---|---|---|---|
| **ModelCatalog** | `src/ultron/core/intelligence/model_catalog.py` | IMPLEMENTED | 278 lines. Strongly typed catalog defining `ModelSpec`, `ModelCapability`, `ModelRole`. Pre-configures `qwen3-8b`, `gemma-3-4b-it`, `qwen2.5-coder-7b-instruct`. |
| **ModelRouter** | `src/ultron/core/intelligence/model_router.py` | IMPLEMENTED | 244 lines. Deterministic routing based on `ComplexityLevel`, `ContextSize`, `TaskRoutingState`, `MemoryPressure`. Supports escalation. |
| **ModelLifecycle** | `src/ultron/core/intelligence/model_lifecycle.py` | IMPLEMENTED | 186 lines. Handles dynamic loading/unloading of GGUF models via `LlamaServerManager`. |
| **LlamaCppEngine** | `src/ultron/core/engine/llama_cpp.py` | IMPLEMENTED | 231 lines. HTTP client implementing `BaseEngine`, targeting OpenAI-compatible endpoints (`/v1/chat/completions`). |
| **LlamaServerManager**| `src/ultron/core/engine/server.py` | IMPLEMENTED | 255 lines. Subprocess manager for local `llama-server` instances. |
| **vLLM Engine** | `src/ultron/core/engine/vllm.py` | SCAFFOLD | 7-line placeholder. |
| **MLX Engine** | `src/ultron/core/engine/mlx.py` | SCAFFOLD | 7-line placeholder. |
| **Cloud Engine** | `src/ultron/core/engine/cloud.py` | SCAFFOLD | 7-line placeholder. |
| **Hardware Aware** | `src/ultron/core/intelligence/hardware_aware.py` | IMPLEMENTED | System memory, CPU core, and load detection. |

---

## 6. Tool / Execution Architecture

- **Tool Catalog**: Single canonical source of metadata at `src/ultron/core/tools/definitions.py` with **58 registered tools** (e.g. `run_command`, `read_file`, `write_file`, `find_definition`, `web_search`, `fetch_page_text`, `query_triples`, etc.).
- **Missing Action Lifecycle**:
  - Tools do **not** use a unified `ActionRequest` $\rightarrow$ `SecurityBoundary` $\rightarrow$ `Execution` $\rightarrow$ `ActionResult/Observation` lifecycle.
  - Tools are individual functions returning raw primitives (`str`, `dict`, `bool`, or `ChatMessage`).
  - Observation structuring (`Observation` model in `src/ultron/core/coding/observations.py`) is applied retrospectively inside `CodingExecutor` rather than being the universal output contract of tool execution.
  - No action IDs, no execution IDs, and no timeout/cancellation propagation into tool functions.

---

## 7. Task State Analysis

Ultron suffers from **state fragmentation** across 5 distinct components:

1. **`TaskState`** (`src/ultron/core/types.py:546`):
   - The primary task model: owns `task_id`, `goal`, `status` (`TaskStatus`), `requirements`, `current_step`, `total_steps`, `execution_history`, `errors`, `context`, `plan`, `code_context`.
2. **`RunState`** (`src/ultron/core/runtime/state.py:64`):
   - Runtime lifecycle tracking: owns `run_id`, `task_id`, `status` (`RuntimeStatus`: `CREATED`, `INITIALIZING`, `RUNNING`, `VERIFYING`, `COMPLETED`, `FAILED`, `BUDGET_EXCEEDED`, `TIMED_OUT`, `CANCELLED`), `budget`.
3. **`CLIState`** (`src/ultron/core/state.py:10`):
   - Chat session state: owns `active_model`, `current_dir`, `status` (`"Ready"`, `"Thinking..."`).
4. **`AgentStatus`** (`src/ultron/core/orchestration/lifecycle.py:27`):
   - Lifecycle state for orchestrated agents (`PENDING`, `INITIALIZING`, `RUNNING`, `PAUSED`, `WAITING_FOR_INPUT`, `COMPLETED`, `FAILED`, `CANCELLED`, `TIMED_OUT`).
5. **`PlanStep`** (`src/ultron/core/types.py:465`):
   - Individual step state within `TaskPlan` (`status`: `PENDING`, `RUNNING`, `SUCCEEDED`, `FAILED`, `SKIPPED`).

**State Flow Conflict**: `AgentRuntime` tracks execution in `RunState`, but `ReActAgent` operates on `TaskState` and mutates it directly. On turn continuation, `main.py` serializes `TaskState` to `.ultron/task.json` via `task_store.py`, while `RunState` and its events are discarded.

---

## 8. Context Management & Token Budgeting Analysis

- **RepositoryContextManager** (`src/ultron/core/context/manager.py`):
  - Implemented with token budget configuration (`ContextBudgetConfig`: max 4,000 tokens total).
  - Prioritizes context: task goal > active files > symbol references > git diff > observations.
- **Critical Context Gap**:
  - The context snapshot is formatted once at turn start and injected into `system_prompt`.
  - Inside `ReActAgent.run()`, the `messages` list grows indefinitely on every iteration as LLM thoughts and tool observations are appended.
  - **Zero in-loop token budgeting**: No check is performed against the model's context window (`16,384` or `8,192` tokens) before calling `self.engine.generate()`.
  - In tasks with extensive tool outputs (e.g. large file reads, long command outputs, multiple failed retries), prompt length easily blows past model context limits, causing silent truncation or inference failure.

---

## 9. Repository Intelligence Analysis

- **Current Implementation**:
  - `CodeIndex` (`src/ultron/core/coding/intelligence/index.py`, 626 lines) builds an in-memory symbol and dependency index.
  - Parsers (`parsers.py`): `PythonAstParser` (uses Python stdlib `ast`) and `RegexParser` (heuristic for JS, TS, Go, Rust, Java, C/C++).
- **Gaps**:
  - **No tree-sitter integration**: Non-Python languages rely on brittle regex heuristic parsing.
  - **No RepoMap**: Ultron lacks an Aider-style token-budgeted repository map that condenses directory structure and key symbol signatures into a compact prompt block.
  - **Brittle Natural Language Resolution**: Symbol queries that deviate from exact patterns fail deterministic routing (confirmed by `_ci_query_live_check.py` and `_react_routing_live_check.py` failures).

---

## 10. Verification & Autonomous Repair Analysis

- **Verification Reality**:
  - `src/ultron/core/verification/` **does not exist**.
  - Verification is handled inside `ReActAgent` via `_verify_task()` and `_verify_plan_task()`.
  - **Self-referential prompt verification**: Verification consists of prompting the LLM with its own output and asking it to output JSON confirming if requirements were satisfied.
  - Runtime acceptance criteria (`ApplicationLifecycleState` and `verify_acceptance_criterion`) rely on regex keyword matching on executed command strings (`"pytest"`, `"server"`, `"app.py"`).
- **Repair Reality**:
  - `CodingExecutor` (`src/ultron/core/coding/executor.py`) provides `classify_failure()` and `RepairBudget` (limits retries and detects identical repeated tool arguments).
  - However, there is no autonomous loop that compiles failures into a structured `FailureEvidence` object and drives a deterministic `RepairController`.

---

## 11. Git / Recovery Analysis

- **Current Implementation**:
  - `src/ultron/core/vcs/` **does not exist**.
  - `workspace.py` and `edits.py` execute read-only queries (`git status --short`, `git diff`).
  - `ModificationTracker` tracks in-memory file modification events during a turn.
- **Gaps**:
  - No pre-mutation Git checkpoints.
  - No task-scoped commit or stash creation.
  - No rollback mechanism to restore workspace state after a failed autonomous modification.
  - No dirty worktree protection (user changes risk being overwritten or conflated with agent edits).

---

## 12. Execution Isolation & Sandboxing Analysis

- **Current Implementation**:
  - `src/ultron/security/sandbox/` contains only 8-line scaffolds (`agent.py`, `container.py`, `wasm.py`).
  - `command_runner.py` executes commands directly on the host operating system using `subprocess.Popen(shell=True)`.
  - `src/ultron/core/execution/` **does not exist**.
- **Security Boundary**:
  - Host execution is gated purely by `src/ultron/security/boundary.py` (`SecurityBoundary.check`), which classifies commands into risk tiers and runs `GuardrailsEngine` to detect secrets, PII, path traversal, and dangerous command patterns (`rm -rf /`, `mkfs`, etc.).
  - There is no container, jail, or sandbox isolating the filesystem, process tree, or network.

---

## 13. Parallel Execution Analysis

- **Current Implementation**:
  - `parallel_tools.py` (`run_tool_batch`) executes up to 8 independent tool calls concurrently using `ThreadPoolExecutor`.
  - Each tool in the batch is security-checked via `check_action()`.
- **Gaps**:
  - **No Dependency DAG**: Batch execution is an un-ordered list; it cannot represent dependencies (e.g. `write_file` must precede `pytest`).
  - **No Conflict Detection**: No mutual exclusion on conflicting file writes.
  - **No Failure Cancellation**: If tool 1 fails, dependent tool 2 is not aborted.

---

## 14. Observability & Replay Analysis

- **Current Implementation**:
  - `EventBus` (`src/ultron/core/runtime/events.py`) provides in-memory publish/subscribe for runtime events (`RUN_STARTED`, `STEP_STARTED`, `TOOL_STARTED`, etc.).
  - `AuditLog` (`src/ultron/security/audit.py`) writes security verdicts to `~/.ultron/audit.jsonl`.
- **Gaps**:
  - `EventBus` history is **never persisted to disk**; all events are lost upon CLI process exit.
  - No append-only task event store (`core/runtime/event_store.py`).
  - No task replay command or capability (`ultron replay` does not exist).
  - Post-mortem debugging relies solely on log files.

---

## 15. Empirical Test Results

### 1. Pytest Full Suite
- **Command**: `.venv/bin/pytest -q`
- **Result**: `1839 passed, 6 deselected in 35.87s` (100% pass rate for non-MITL tests).
- **Deselected Tests**: 6 Model-in-the-Loop tests (`tests/model_in_loop/test_real_coding_agent.py`) marked with `@pytest.mark.mitl`.

### 2. Code Quality & Linting
- **Command**: `.venv/bin/ruff check .`
- **Result**: `All checks passed!` (0 errors).

### 3. Live Verification & Stress Harnesses
- `_stress_audit.py`: **29/29 checks passed** (covers parallel tool batch, security boundary, planning preflight, debug context, learning, structured output, resource monitor).
- `_reflow_e2e.py`: **PASS** (width-adaptive terminal reflow at 60, 75, 100, 120 columns).
- `_cli_output_e2e.py`: **PASS** (clean CLI output isolation and brand response panels).
- `_orchestration_live_check.py`: **109/109 checks passed** (proves orchestration contracts pass in isolation).
- `_ci_query_live_check.py`: **FAIL (12 passed, 3 failed)**. Fails on prompts 13, 14, 15 due to rigid regex-based symbol matching for conversational queries.
- `_react_routing_live_check.py`: **FAIL (3 passed, 1 failed)**. Fails on `"Where is command execution implemented?"`.

---

## 16. Gap Matrix

| # | Capability | Current Reality | Evidence | Status | Gap | Target Phase |
|---|---|---|---|---|---|---|
| 1 | **TaskState** | Exists in `core/types.py`, but competes with `RunState`, `CLIState`, `AgentStatus`. | `types.py:546`, `state.py:64` | PARTIAL | Unify into single authoritative task state with explicit lifecycle transitions. | Phase 1 |
| 2 | **Event Model** | In-memory `EventBus` only; events discarded on exit. | `runtime/events.py:63` | PARTIAL | Durable, append-only event store (JSONL/SQLite) with replay capability. | Phase 1 |
| 3 | **Context Manager** | `RepositoryContextManager` exists for initial snapshot. | `context/manager.py:44` | PARTIAL | Enforce bounded context budget inside active ReAct while loop. | Phase 2 |
| 4 | **Token Budgeting** | Budget exists on initial snapshot, absent in conversation loop. | `react.py:1184` | BROKEN | Active token accounting before every LLM call; context condensation. | Phase 2 |
| 5 | **Repository Intelligence** | Python AST index and regex fallback exist. | `parsers.py:13`, `index.py:30` | PARTIAL | Tree-sitter integration; robust provenance tracking. | Phase 3 |
| 6 | **Repo Map** | Completely absent. | `grep -rn "repo_map" src/` | MISSING | Configurable token-budgeted tree/symbol map. | Phase 3 |
| 7 | **Typed Actions** | Tool definitions exist, but actions are raw dicts. | `definitions.py:40` | PARTIAL | Formal `ActionRequest` protocol with typed schemas and action IDs. | Phase 4 |
| 8 | **Typed Observations** | `Observation` model exists, but tools return strings. | `observations.py:39`, `react.py:1286` | PARTIAL | Tools return structured observations with exit code, stdout/err, duration. | Phase 4 |
| 9 | **Execution Orchestrator** | Disconnected in `core/orchestration/`; CLI bypasses it. | `main.py:937`, `react.py:1030` | BROKEN | Connect agent runtime to unified action executor. | Phase 4 |
| 10 | **Verification Engine** | LLM self-evaluation prompt; no independent test runner. | `react.py:1455`, `react.py:1575` | BROKEN | Deterministic `VerificationSpec` engine executing real commands and assertions. | Phase 5 |
| 11 | **Autonomous Repair** | Failure classification and retry gating exist. | `executor.py:58`, `react.py:1250` | PARTIAL | Bounded `RepairController` with structured `FailureEvidence`. | Phase 5 |
| 12 | **Git Checkpoints** | Completely absent. | `ls -d src/ultron/core/vcs` | MISSING | Pre-mutation checkpoints before first autonomous change. | Phase 6 |
| 13 | **Rollback** | Completely absent. | `core/coding/edits.py:62` | MISSING | Task-scoped rollback restoring workspace to clean pre-task checkpoint. | Phase 6 |
| 14 | **Execution Backend** | Host `subprocess.Popen` hardcoded in `command_runner.py`. | `command_runner.py:45` | MISSING | `ExecutionBackend` interface (`HostBackend` vs `SandboxBackend`). | Phase 7 |
| 15 | **Sandbox** | Scaffold files only (`agent.py`, `container.py`, `wasm.py`). | `security/sandbox/container.py:7` | SCAFFOLD | Isolated container/sandbox backend for untrusted execution. | Phase 7 |
| 16 | **Dependency Scheduler** | Batch parallel execution in thread pool; no DAG. | `parallel_tools.py:37` | PARTIAL | Action dependency DAG, write-conflict serialization, failure abort. | Phase 8 |
| 17 | **Model Router** | Implemented, but partially integrated into chat loop. | `model_router.py:24`, `main.py:875` | IMPLEMENTED | Refine routing matrix and wire directly into task execution state. | Phase 9 |
| 18 | **Model Lifecycle** | `ModelLifecycleManager` implemented and functional. | `model_lifecycle.py:35` | IMPLEMENTED | Make loading observable and resource-aware. | Phase 9 |
| 19 | **Agent Delegation** | Implemented in `core/orchestration/`, but unused. | `orchestration/delegation.py:15` | PARTIAL | Integrate bounded single-level delegation into active runtime. | Phase 10 |
| 20 | **Observability** | In-memory events and audit JSONL exist. | `runtime/events.py:71`, `audit.py:40`| PARTIAL | Per-task metrics (duration, model calls, tool calls, token usage). | Phase 11 |
| 21 | **Replay** | Completely absent. | `main.py:60` | MISSING | Deterministic replay from persisted event stream. | Phase 11 |
| 22 | **Benchmarks** | MITL harness exists; benchmark suite missing. | `tests/model_in_loop/` | PARTIAL | Standardized regression coding benchmark suite. | Phase 11 |
| 23 | **Security Policy** | `SecurityBoundary` and `GuardrailsEngine` fully implemented. | `boundary.py:43`, `guardrails.py:42`| IMPLEMENTED | Keep 100% intact; make action-aware. | Invariant |
| 24 | **Tool Registry** | Canonical definitions table with 58 registered tools. | `definitions.py:7` | IMPLEMENTED | Single source of truth. | Invariant |
| 25 | **Planning** | `task_planning.py` and `plan_validation.py` implemented. | `task_planning.py:30` | IMPLEMENTED | Mature; promote into runtime subsystem. | Phase 1/4 |
| 26 | **Memory** | SQLite graph triples and flat facts implemented. | `tools/memory/graph.py:20` | IMPLEMENTED | Separate long-term memory from task execution context. | Phase 2 |
| 27 | **Retrieval** | `RepositoryRetriever` implemented for file/symbol context. | `context/retrieval.py:35` | IMPLEMENTED | Extend into repository intelligence layer. | Phase 3 |

---

## 17. Architectural Conflicts & Duplicate Responsibilities

1. **Disconnected Parallel Orchestration Subsystem**:
   - `src/ultron/core/orchestration/` provides an extensive implementation of agent roles (`Supervisor`, `Coder`, `Researcher`, `Tester`, `Reviewer`), artifacts, validation, and lifecycles.
   - However, `main.py` and `AgentRuntime` do not use this system; instead, `main.py` dispatches directly to `ReActAgent.run()`.
2. **Competing State Machines**:
   - `TaskState` (`core/types.py`) vs. `RunState` (`core/runtime/state.py`) vs. `AgentStatus` (`core/orchestration/lifecycle.py`) vs. `CLIState` (`core/state.py`).
   - Multiple status enums (`TaskStatus`, `RuntimeStatus`, `AgentStatus`) can represent conflicting states during the same execution.
3. **Scattered Security Dispatch**:
   - Both `SimpleAgent` (30+ scattered `check_action` calls) and `ReActAgent` (`_route_tool` with 300+ lines of dispatch) manually check the security boundary and handle confirmation prompts, rather than routing through a unified execution runtime.
4. **Prompt Growth vs. Token Limits**:
   - `RepositoryContextManager` bounds static context to 4,000 tokens, but `ReActAgent.run()` appends every tool result into `messages` indefinitely, guaranteeing context overflow on long tasks.
5. **Self-Referential Verification vs. Deterministic Ground Truth**:
   - `_verify_task` asks the LLM to self-certify whether requirements were satisfied, creating false completion claims when the model hallucinated completion without running tests.
6. **Host Subprocess Execution without Abstraction**:
   - `command_runner.py` directly executes host shell commands, coupling tool logic directly to local OS execution without an `ExecutionBackend` protocol.

---

## 18. Recommended Phase Ordering & Alignment

The 11-phase sequence defined in `ULTRON_Autonomous_Coding_Harness_Gap_Closure_Plan.pdf` is fully validated by this audit:

1. **Phase 1 (TaskState + Event Model)**: MUST come first to resolve state fragmentation and establish a durable event store.
2. **Phase 2 (Context Manager + Token Budgeting)**: MUST precede repository intelligence to ensure that expanded repo context does not overflow model context windows.
3. **Phase 3 (Repository Intelligence / RepoMap)**: Bridges the gap between bare file searching and structured symbol context.
4. **Phase 4 (Typed Tool / Action / Observation Runtime)**: Decouples tool execution from `ReActAgent` and standardizes the request-authorize-execute-observe lifecycle.
5. **Phase 5 (Verification Engine + Autonomous Repair)**: Replaces LLM self-evaluation with deterministic `VerificationSpec` test execution and structured `FailureEvidence`.
6. **Phase 6 (Git Checkpoints & Rollback)**: Provides safety and recoverability for autonomous mutations.
7. **Phase 7 (Execution Backends & Sandbox)**: Introduces the `ExecutionBackend` abstraction, separating host execution from future containers.
8. **Phase 8 (Dependency-Aware Parallel Execution)**: Upgrades batch execution into a DAG-aware scheduler with write conflict serialization.
9. **Phase 9 (Model Router Integration)**: Consolidates existing `ModelRouter` and `ModelLifecycleManager` into active runtime execution.
10. **Phase 10 (Bounded Delegation / Subagents)**: Reconnects the dormant `core/orchestration/` capabilities into bounded subtask delegation.
11. **Phase 11 (Observability, Replay & Capability Benchmarks)**: Delivers per-task metrics, deterministic replay, and standardized coding benchmarks.

---

## 19. Explicit Unknowns

1. **Tree-Sitter Binary Packaging on macOS**: Need to verify whether pre-built wheels for `tree-sitter` and language grammars install cleanly across all target Apple Silicon / Intel macOS environments without requiring external compiler dependencies.
2. **Concurrent Model Latency & Memory Headroom**: Need empirical measurement of memory overhead when running `qwen2.5-coder-7b-instruct` under heavy compilation/test execution on 16GB unified memory machines.

---

## 20. Files Inspected

The following primary files and symbols were directly inspected and verified during Phase 0:

- `src/ultron/main.py` (`async_chat`, `handle_slash_command`, `_plan_and_run`)
- `src/ultron/core/agents/__init__.py` (`SUPPORTED_AGENTS`, `get_agent`)
- `src/ultron/core/agents/base.py` (`BaseAgent`)
- `src/ultron/core/agents/simple.py` (`SimpleAgent`, `handle_*`, `detect_*`)
- `src/ultron/core/agents/react.py` (`ReActAgent.run`, `_route_tool`, `_verify_task`, `_verify_plan_task`, `_coding_gate`)
- `src/ultron/core/agents/orchestrator.py` (scaffold)
- `src/ultron/core/agents/codeact.py` (scaffold)
- `src/ultron/core/agents/operative.py` (scaffold)
- `src/ultron/core/agents/monitor.py` (scaffold)
- `src/ultron/core/runtime/runtime.py` (`AgentRuntime.execute`, `_build_routing_request`)
- `src/ultron/core/runtime/state.py` (`RunState`, `RuntimeStatus`)
- `src/ultron/core/runtime/events.py` (`EventBus`, `RuntimeEvent`, `RuntimeEventType`)
- `src/ultron/core/runtime/budget.py` (`RuntimeBudget`, `BudgetExceededError`)
- `src/ultron/core/types.py` (`TaskState`, `TaskPlan`, `PlanStep`, `TaskRequirement`, `ToolExecution`)
- `src/ultron/core/state.py` (`CLIState`)
- `src/ultron/core/intelligence/model_catalog.py` (`ModelCatalog`, `ModelSpec`, `ModelCapability`, `ModelRole`)
- `src/ultron/core/intelligence/model_router.py` (`ModelRouter`, `RoutingRequest`, `RoutingDecision`)
- `src/ultron/core/intelligence/model_lifecycle.py` (`ModelLifecycleManager`)
- `src/ultron/core/intelligence/task_planning.py` (`prepare_task_for_execution`, `plan_task`)
- `src/ultron/core/intelligence/plan_validation.py` (`validate_plan`)
- `src/ultron/core/intelligence/parallel_tools.py` (`run_tool_batch`, `plan_tool_batch`)
- `src/ultron/core/tools/definitions.py` (`TOOL_DEFINITIONS`, `ToolDefinition` — 58 registered tools)
- `src/ultron/core/tools/registry.py` (`get_tools_schema`, `execute_tool`)
- `src/ultron/core/tools/paths.py` (`get_configured_workspace`, `is_path_safe`, `resolve_project_path`)
- `src/ultron/core/tools/builtin/command_runner.py` (`run_command`, `_run_one`)
- `src/ultron/core/coding/executor.py` (`CodingExecutor`, `classify_failure`, `RepairBudget`)
- `src/ultron/core/coding/observations.py` (`Observation`, `ObservationKind`)
- `src/ultron/core/coding/intelligence/parsers.py` (`PythonAstParser`, `RegexParser`)
- `src/ultron/core/coding/intelligence/index.py` (`CodeIndex`)
- `src/ultron/core/context/manager.py` (`RepositoryContextManager`, `ContextBudgetConfig`)
- `src/ultron/core/context/retrieval.py` (`RepositoryRetriever`)
- `src/ultron/core/orchestration/delegation.py` (`Supervisor`, `DelegationRequest`)
- `src/ultron/core/orchestration/workflow.py` (`WorkflowEngine`)
- `src/ultron/core/orchestration/lifecycle.py` (`AgentStatus`)
- `src/ultron/security/boundary.py` (`SecurityBoundary`, `check_action`, `classify_action`)
- `src/ultron/security/guardrails.py` (`GuardrailsEngine`)
- `src/ultron/security/audit.py` (`AuditLog`)
- `src/ultron/security/sandbox/container.py` (scaffold)
- `src/ultron/security/sandbox/agent.py` (scaffold)
- `src/ultron/security/sandbox/wasm.py` (scaffold)
- `src/ultron/validation/evaluate.py`, `runner.py`, `model.py`, `audit.py` (validation harness)
- `tests/test_ui_session.py`, `tests/test_external_workspace.py`
- Root scripts: `_stress_audit.py`, `_reflow_e2e.py`, `_cli_output_e2e.py`, `_ci_query_live_check.py`, `_react_routing_live_check.py`, `_orchestration_live_check.py`, `_step3_validation.py`

---
*Report completed and verified without production code modifications.*
