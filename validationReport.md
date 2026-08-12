# Ultron FIX Series — Validation & Implementation Report

**Purpose:** This report documents, per section, what was implemented and exactly what
was tested/validated, so an independent AI (or human) can cross-verify the current
implementation status without re-reading the full conversation.

**How to verify anything in this report:** every number below was produced by an actual
command run in `/Users/aravindhan/ultron` (see the "Reproduction commands" in each section).
The repository root is the working directory for all commands.

---

## 0. Executive summary (as of this report)

| Metric | Value |
|---|---|
| Full automated test suite | **1287 passed, 0 failed** (`pytest -q`) |
| Lint | **clean** (`ruff check .` → "All checks passed!") |
| Live orchestration validation harness | **109/109 checks passed** (`python _orchestration_live_check.py`) |
| FIX #7 uncommitted change set | 9 entries (see §9.1) |
| Earlier fixes (FIX #1–#6) | committed in git history (`2eead6e` = FIX #1/#2, `d94f067` = FIX #3/#4, `d4d2747` = FIX #5/#6) |

Verified 12 Aug 2026. Full suite runtime ≈ 22 s.

**Fix series overview:**

- **FIX #1** — TaskState + completion enforcement (committed)
- **FIX #2** — General task understanding + structured planning (committed)
- **FIX #3** — Coding workspace + execution context + CodingExecutor (committed)
- **FIX #4** — Codebase intelligence (lexical/symbol/AST/LSP/semantic/dependency) (committed)
- **FIX #5** — Test intelligence + verification + controlled self-repair (committed)
- **FIX #6** — Memory hierarchy + ContextManager (committed)
- **FIX #7** — Multi-agent orchestration (agent contract §7.1 → registry §7.2 → artifacts §7.3
  → supervisor delegation §7.4 → validation layer → workflow engine §7.6) (**uncommitted**)

---

## 1. FIX #1 — TaskState + Completion Enforcement

### What was implemented
- `TaskState` in `src/ultron/core/types.py` with status machine
  (`TASK_PENDING/TASK_IN_PROGRESS/TASK_WAITING_CONFIRMATION/TASK_COMPLETED/TASK_FAILED/TASK_BLOCKED`),
  requirements list (`add_requirement`), completion enforcement (`is_complete()`,
  `remaining_requirements()`, `remaining_steps()`), and `TaskError` recording (step + message).
- `TaskType` classification enum.
- Completion is *enforced*: a task can only be marked complete when requirements are satisfied —
  the LLM saying "done" is not sufficient.

### Test phases
- **Unit tests** — `tests/test_task_state.py` (25 tests), `tests/test_task_validation.py`
  (11 tests), `tests/test_task_execution.py` (9 tests).
  Coverage: status transitions, requirement completion, completion guards (cannot complete with
  unresolved requirements), `TaskError` recording, TaskState serialization.
- **Result:** all pass; part of the 1287-suite.

### Reproduction
```bash
.venv/bin/python -m pytest tests/test_task_state.py tests/test_task_validation.py tests/test_task_execution.py -q
```

---

## 2. FIX #2 — General Task Understanding + Structured Planning

### What was implemented
- Task classification (`test_task_classification.py` era logic), structured `TaskPlan`
  (`src/ultron/core/types.py`) with ordered `PlanStep`s, planner in
  `src/ultron/core/intelligence/planning.py`, and plan→TaskState integration
  (`attach_plan`, step completion tracking, `remaining_steps()`).

### Test phases
- **Unit tests** — `tests/test_planner.py` (10), `tests/test_planning.py` (28),
  `tests/test_plan_validation.py` (13), `tests/test_task_planning.py` (15),
  `tests/test_task_classification.py` (27), `tests/test_task_plan_integration.py` (15).
  Coverage: goal→plan generation, step ordering, step status, plan validation (well-formed plans),
  plan/state integration, classification of task types.
- **Result:** all pass.

### Reproduction
```bash
.venv/bin/python -m pytest tests/test_planner.py tests/test_planning.py tests/test_plan_validation.py tests/test_task_planning.py tests/test_task_classification.py tests/test_task_plan_integration.py -q
```

---

## 3. FIX #3 — Coding Workspace + Execution Context + CodingExecutor

### What was implemented
New package **`src/ultron/core/coding/`**:
- `workspace.py` — `CodingWorkspace`/project detection: working dir, project root, git status,
  detected project type, languages, package manager, build system, test framework, source/test
  dirs, `.gitignore`-aware exclusions (`.git`, `node_modules`, `.venv`, `__pycache__`, `dist`, `build`).
- `observations.py` — structured `Observation` model distinguishing file content / search result /
  command result / test result / build result / error / diff / repository state.
- `command.py` — structured `CommandResult`: command, exit code, stdout, stderr, duration,
  timeout status.
- `edits.py` — safe edit operations: create, replace, targeted edit, append, delete, rename;
  modification tracking (path, action, timestamp, success/failure) with git diff/status
  integration where available (git not required).
- `context.py` — `CodingContext` / execution context: workspace, relevant files, recent
  observations, current task, current plan step, previous modifications, command/test results,
  errors. Survives confirmation.
- `executor.py` — **CodingExecutor**: observe → decide next action → tool → observation →
  update TaskState → validate → next step/repair → verify. Budgets (max reasoning iterations,
  max repair attempts, repeated-identical-action gating, command timeout). Failure classification
  (syntax/compilation/test-assertion/dependency/config/environment/runtime/permission/unknown),
  controlled repair loop, regression testing, build validation, diff awareness, confirmation
  resume (no plan restart), false-completion rejection.

All writes/commands route through the existing security/permission systems — no privileged bypass.

### Test phases
- **Unit tests** — `tests/test_coding_workspace.py` (18), `tests/test_coding_edits.py` (21),
  `tests/test_coding_context.py` (26), `tests/test_coding_executor.py` (35).
  Coverage: workspace discovery, project detection, repository inspection, relevant-file
  discovery, command-result capture, file creation, targeted modification, deletion,
  modification tracking, git diff/status integration, observation creation, confirmation-context
  preservation, TaskState integration, executor budgets, failure classification, repair gating,
  false-completion rejection.
- **Stress/end-to-end validation (20 scenarios)** — created temporary repositories and validated:
  create project; modify existing project; fix intentionally-broken code; fix compilation failure;
  multiple independent failures; failure requiring code reading; refactoring; add feature+tests;
  preserve existing green tests; unrelated-file protection; confirmation interruption; user denial;
  command failure; repeated failure (budget stops loop); false completion rejection; security
  interception; git-diff verification; minimal typo fix (no over-engineering); complex JWT task;
  adaptive debugging (plan adaptation on new information).
- **Result:** all pass.

### Reproduction
```bash
.venv/bin/python -m pytest tests/test_coding_workspace.py tests/test_coding_edits.py tests/test_coding_context.py tests/test_coding_executor.py -q
```

---

## 4. FIX #4 — Codebase Intelligence

### What was implemented
New package **`src/ultron/core/coding/intelligence/`** (layered, language-agnostic):
- `index.py` — repository index: file→language→symbols→imports→searchable content→mtime/hash;
  incremental updates (re-index only changed files); dependency relationships (imports, dependents),
  distinguishing EXACT vs INFERRED.
- `symbols.py` — `Symbol` representation: name, kind, language, file, location, scope, parent,
  signature.
- `parsers.py` — modular `SourceParser` protocol + registry; **Python AST parser** (stdlib `ast` —
  classes, functions, methods, imports); regex-based fallback for other languages; malformed-source
  degradation; unsupported-language handling.
- `search.py` — filesystem + lexical search (file names, extensions, text, regex, directory
  filtering, ignored-dir exclusion).
- `semantic.py` — semantic search foundation integrated with the **existing** memory/vector
  infrastructure (no duplicate system); returns file/symbol/language/chunk/line-range metadata;
  embeddings are never the source of truth (the file remains authoritative).
- `lsp.py` — LSP abstraction + server manager: initialize/shutdown/definition/references/hover/
  document-symbols/workspace-symbols/implementation/call-hierarchy; graceful handling of
  unsupported operations; server detection, lifecycle, timeouts, crashes.
- `facade.py`/`intelligence_bridge.py` — coherent `CodeIntelligence` interface:
  `search_code`, `find_symbol`, `find_definition`, `find_references`, `find_implementations`,
  `find_callers`, `find_callees`, `inspect_file`, `inspect_symbol`, `search_semantically`,
  `get_imports`, `get_dependents`. Fallback ladder: LSP → AST → lexical → semantic.

### Test phases
- **Unit tests** — `tests/test_code_intelligence.py` (34): symbol search, definition, references,
  callers, import graph, semantic search, repository exploration, bug fix via intelligence,
  refactor via references, stale-index invalidation, modified-file re-indexing, targeted context,
  large repository.
- **Integration** — `tests/test_intelligence_executor.py` (29): CodingExecutor using code
  intelligence tools; LSP-fallback, semantic-fallback, feature implementation, code review.
- **Stress** — `tests/test_intelligence_stress.py` (23): 20-scenario stress test on a
  representative multi-module repository — symbol search, definition/reference/caller lookup,
  import graph, semantic search (no keyword overlap), repo exploration, bug fix, refactor
  (`UserService`→`AccountService`), same-name disambiguation (`auth.UserService` vs
  `billing.UserService`), large-repo behavior (not reading every file), LSP available/unavailable,
  stale index, malformed code, unsupported language, code review, feature implementation, false
  context, security (workspace boundary).
- **Result:** all pass. Known limitation reported honestly: symbol extraction is **AST-based for
  Python + lexical fallback** — it is *not* full IDE-level (no tree-sitter, no real LSP server
  bundled); LSP is an abstraction with mocks/fakes in CI.

### Reproduction
```bash
.venv/bin/python -m pytest tests/test_code_intelligence.py tests/test_intelligence_executor.py tests/test_intelligence_stress.py -q
```

---

## 5. FIX #5 — Test Intelligence + Verification + Self-Repair

### What was implemented
- `tests/test_test_intelligence.py` era modules — deterministic:
  - **Test discovery** — multi-language test-file inventory (`discover_test_files`).
  - **Affected-test selection** — `select_affected_tests` maps changed source files to
    likely-affected tests by convention (mirror/sibling/module/node variants).
  - **Failure localization** — `FailureAnalysis`/`FailureCategory` (syntax, compilation,
    test-assertion, dependency, configuration, environment, runtime, permission, timeout,
    unknown); deterministic file/line extraction from output.
  - **Verification** — targeted-then-broad test running, build/lint/type validation selection
    from project context, final-diff inspection, false-completion rejection
    (LLM "done" ≠ verified).
  - **Repair** — controlled loop: run → analyze → locate → modify → rerun, gated by progress
    and budgets; deterministic mechanisms (discovery/parsing/selection) + LLM for root-cause
    reasoning.

### Test phases
- **Unit tests** — `tests/test_test_intelligence.py` (34): test discovery, affected-test
  selection, failure localization, false-completion rejection, result parsing, regression
  selection. Also `tests/test_plan_execution.py` (28) covering executor-driven plan execution.
- **Result:** all pass.

### Reproduction
```bash
.venv/bin/python -m pytest tests/test_test_intelligence.py tests/test_plan_execution.py -q
```

---

## 6. FIX #6 — Memory Hierarchy + Context Management

### What was implemented
New package **`src/ultron/core/memory/`**:
- `models.py` — `MemoryConfidence` semantics (DIRECT_OBSERVATION / HIGH_CONFIDENCE /
  INFERRED / USER_PROVIDED), `MemorySource` tracking (user, repository inspection, tool result,
  code intelligence, test result, explicit system knowledge, LLM inference); structured memory
  records (id, type, content, source, project/workspace, timestamp, confidence, validity,
  metadata).
- Memory hierarchy: **WorkingMemory** (ephemeral, per-reasoning-cycle), **TaskMemory** (tied to
  TaskState — does not duplicate it), **SessionMemory** (current session), **ProjectMemory**
  (project-scoped facts, keyed by workspace identity), **LongTermMemory** (persistent, policy-
  gated).
- `context.py` — **ContextManager**: assembles only the context needed for the current reasoning
  step (user request → task state → plan step → recent observations → relevant code → relevant
  memory); priority hierarchy (current source > old memory; TaskState > conversation history;
  Code Intelligence > stale memory); configurable `ContextBudget` (recent messages, tool
  observations, code snippets, memory records, total size); context compression/summarization
  that never drops critical evidence (exact failing test, exact error, changed files); large
  tool-output handling (summary + relevant matches + reference, not 20k raw lines).
- Persistence: JSONL/sqlite-backed stores; restart-safe; **secret redaction** before persistence
  (no API keys/passwords/tokens); invalidation (current repository evidence marks stale memory);
  project isolation (Project A memory never leaks into Project B).

### Test phases
- **Unit** — `tests/test_memory_foundation.py` (29): working/task/session/project memory,
  persistence, retrieval, source tracking, confidence semantics, stale-memory invalidation,
  project scoping, context prioritization, context budget, context assembly, restart persistence,
  task/session separation, secret redaction.
- **Integration** — `tests/test_memory_integration.py` (20): project memory reused on follow-up
  tasks, current code overrides stale memory, task state survives confirmation, task survives
  process restart, interrupted task resumes, unrelated new task gets separate TaskState,
  meaningful observation becomes project memory, irrelevant observation not persisted, long
  context compressed, large command output summarized, stale memory invalidated, project
  isolation, secrets not persisted, context priority, Code Intelligence outranks stale memory.
- **Stress** — `tests/test_memory_stress.py` (20): 20-scenario stress test (project discovery
  memory, follow-up task, stale memory, project isolation, task continuation, new task after
  completion, large tool output, long session, memory formation policy, false memory rejection,
  memory conflict resolution, secret redaction, context priority, compression, repository
  change, git branches, memory failure fallback, code-intelligence failure fallback, session
  boundary, performance).
- `tests/test_memory_graph.py` (27) covers the pre-existing knowledge-graph memory.
- **Result:** all pass.

### Reproduction
```bash
.venv/bin/python -m pytest tests/test_memory_foundation.py tests/test_memory_integration.py tests/test_memory_stress.py -q
```

---

## 7. FIX #7 — Multi-Agent Orchestration (uncommitted)

New package **`src/ultron/core/orchestration/`** — 9 modules + `__init__.py`, 5,545 lines:

| Module | Lines | Contents |
|---|---|---|
| `lifecycle.py` | 153 | `AgentStatus` (PENDING→ASSIGNED→RUNNING→WAITING→COMPLETED; FAILED/BLOCKED/CANCELLED) + `can_transition()` + `AgentStatusChange` |
| `contract.py` | 134 | `Agent` base contract, `AgentIdentity`, `run_with_state` |
| `models.py` | 596 | `AgentState` (lifecycle + metadata), `AgentResult` (SUCCESS/FAILED/BLOCKED/CANCELLED/NEEDS_INPUT + summary/evidence/artifact/changed_files/tests/blockers/recommendations/metadata), `ExecutionContext` (task_id, agent_id, workspace, allowed tools, permissions, budget, plan step, cancellation) |
| `permissions.py` | 299 | `AgentPermissions` (frozen profile: allowed_tools, read/write/shell/network, max budget, risk), `PermissionCategory`, `classify_tool` |
| `registry.py` | 458 | `AgentRegistry`: register/retrieve/validate/list/instantiate; duplicate rejection; unknown type → safe failure; 6 agent types (supervisor/researcher/coder/tester/reviewer/security) with explicit metadata |
| `artifacts.py` | 486 | `AgentArtifact` base + `ResearchFinding`/`ImplementationResult`/`TestResult`/`ReviewResult`/`SecurityFinding`; discriminated union; `ArtifactStore` (JSONL, thread-safe, duplicate-id rejection, corrupt-record skipping) |
| `delegation.py` | 637 | `DelegationRequest` + `Supervisor` (create/dispatch/cancel/decide, timeout via `asyncio.wait_for`, budget enforcement, TaskState recording, context isolation) |
| `validation.py` | 1,373 | `OrchestrationValidator` — 11 check families, 13 violation codes (see §7.5) |
| `workflow.py` | 1,196 | `Workflow`/`WorkflowStep` models + transition tables, `WorkflowEngine` (create/validate/start/pause/resume/cancel/execute_next_step/execute_until_blocked/trace), dependency validation, Supervisor+Validator+TaskState integration (see §7.6) |

### 7.1 — Agent Contract + Lifecycle (TEST A–J + critical separation)

**Implemented:** `Agent` contract, `AgentIdentity`, `AgentState` lifecycle with transition table,
`ExecutionContext` (scoped view, frozen permissions), `AgentResult` (structured, not strings),
budget on the context.

**Test phases:**
- **Unit** — `tests/test_agent_contract.py` (**33 tests**): lifecycle transitions, invalid
  transitions rejected, terminal-state guards, `AgentResult` shape per status, `ExecutionContext`
  permission scoping, cancellation, failure, blocked state, budget handling.
- **10-min live** — `_orchestration_live_check.py` §7.1 (**42 checks**): TEST A create agent,
  B assign task, C run, D successful result, E failure, F blocked, G cancel, H invalid
  transitions rejected, I context contains only permitted tools, J result is structured —
  **plus the critical invariant: an AgentResult does NOT automatically complete the TaskState**
  (agent completion ≠ task completion).

### 7.2 — Agent Registry + Permissions

**Implemented:** 6 registered agent types with explicit metadata and frozen permission profiles.
Permissions come from the registry/runtime — the LLM can never modify its own permissions.
Read/write/shell/network per agent type (researcher read-only; coder read/write/test/shell
subject to security; tester read/search/test but no application-code writes; reviewer
read/search/git-diff only; security read/search/analysis only; supervisor read/search only).

**Test phases:**
- **Unit** — `tests/test_agent_registry.py` (**27 tests**): all six types, lookup, duplicate
  registration rejected, unknown agent rejected, capability lookup, permission lookup,
  unauthorized tool/write/shell, security enforcement, profile attachment on instantiation.
- **10-min live** — §7.2 (**13 checks**): researcher read succeeds / write blocked; coder read
  succeeds / write → security evaluation; tester test command allowed / source write blocked;
  reviewer git diff ok / write blocked; unknown type rejected; duplicate registration rejected.

### 7.3 — Structured Agent Results + Artifacts

**Implemented:** five artifact types with provenance (`task_id`, `agent_id`, `artifact_id`,
`timestamp`, `summary`, `evidence`, `source`, `confidence` reusing FIX #6 `MemoryConfidence`,
`related_files/symbols`, `metadata`). `TestResult` **reuses FIX #5's** `CommandResult` +
`FailureAnalysis` (no second test-result system). `AgentResult.artifact` is a discriminated
union (restored from JSON as the concrete type). `ArtifactStore` JSONL persistence.
`to_agent_result()` folds the artifact into the compact inter-agent envelope.

**Test phases:**
- **Unit** — `tests/test_agent_artifacts.py` (**25 tests**): serialization/deserialization,
  validation (missing required fields), ownership, type validation, malformed artifacts,
  persistence across instances, corrupt-record skipping, duplicate-id rejection, TaskState
  association (`task_key`), FIX #5 reuse, GuardrailFinding + ReviewFinding round-trips.
  **CRITICAL test:** a researcher generating a large internal trajectory passes only its
  structured artifact onward — the internal history is never injected.
- **10-min live** — §7.3 (**14 checks**): each artifact type round-trips between components;
  artifact size/metadata/task-association/source/evidence inspected; trajectory-isolation check.
- **Fixes during review:** `model_rebuild(_types_namespace=...)` for the discriminated union
  import cycle; `threading.Lock` on the store; dead imports removed; duplicate-id → hard error
  (was a silent data-loss risk).

### 7.4 — Supervisor → Specialist Delegation

**Implemented:** `DelegationRequest` (delegation_id, task_id/parent_task_id, agent_type,
objective, input_artifacts, constraints, frozen permissions, expected_output, budget,
timeout_seconds) with lifecycle reusing `AgentStatus`. `Supervisor`: `create_delegation`
(unknown type → KeyError), `dispatch` (fresh run or WAITING resume), `cancel_delegation`,
`decide`, TaskState recording — deterministic, no LLM. **Context isolation:** specialists
receive only the string-only `task_brief` (goal/type/steps/requirements/errors) + artifact
summaries + constraints — never transcript, trajectory, or full TaskState.

**Test phases:**
- **Unit** — `tests/test_supervisor_delegation.py` (**20 tests**): delegation creation, budget/
  timeout folding, lifecycle transitions + result-status lock, dispatch success/failure/
  NEEDS_INPUT-resume/timeout, cancellation (pre-run + mid-run), invalid agent, missing factory,
  permission propagation, **context isolation** (brief delivered; transcript/history/trajectory
  excluded), artifact return + ownership, TaskState updated-but-never-completed, `decide`
  mapping for all statuses.
- **10-min live** — §7.4 (**10 checks**): "Understand authentication" supervisor→researcher
  flow (search → symbol lookup → file inspection → ResearchFinding returned, read-only enforced,
  trajectory NOT injected, TaskState updated, delegation status correct) + researcher timeout /
  failure / cancellation each verified.
- **Fixes during review:** `task_state` was not reaching the context builder (specialists never
  received the brief) — wired through; regression test asserts the brief IS delivered; dead
  `boundary` param removed.

### 7.5 — Orchestration Validation Layer

**Implemented:** `OrchestrationValidator` — deterministic, read-only, no LLM, no side effects,
idempotent. Validates: lifecycle transitions + result-status match; delegation validity +
registered agent type; permissions/tool authorization (inspected, never re-issued); budget
(within/exceeded/**at-limit = warning**); timeout (within/exceeded/execution-flagged/missing
data); AgentResult schema per status; artifacts (schema/provenance/ownership/task); workspace
scope (workspace escape = CRITICAL, out-of-scope = HIGH); TaskState consistency (blocked/failed
task claiming success; completed task with unresolved requirements = corruption); **completion
claims** (agent claims are NOT proof — SUCCESS is checked against task requirements, remaining
plan steps, and required evidence; test-claim contradiction detection); ownership
(delegation↔task↔run cross-refs).

**Violation codes (13):** `INVALID_LIFECYCLE_TRANSITION`, `UNAUTHORIZED_TOOL`,
`UNAUTHORIZED_FILE_ACCESS`, `BUDGET_EXCEEDED`, `TIMEOUT_EXCEEDED`, `INVALID_AGENT_RESULT`,
`INVALID_ARTIFACT`, `ARTIFACT_OWNERSHIP_VIOLATION`, `ARTIFACT_TASK_MISMATCH`, `TASK_STATE_CONFLICT`,
`WORKSPACE_SCOPE_VIOLATION`, `FALSE_COMPLETION_CLAIM`, `TEST_CLAIM_CONTRADICTION`.

**Test phases:**
- **Unit** — `tests/test_orchestration_validation.py` (**52 tests**): every check family per the
  spec's exhaustive list (lifecycle valid/invalid/terminal/retry; permissions authorized/
  unauthorized/write/shell; budget within/exceeded/exact-boundary; timeout within/exceeded/
  missing-data; result valid+invalid success/failure/blocked/needs-input; artifacts valid/
  malformed/wrong-task/wrong-agent/missing-provenance; task state consistent/conflicting/
  cancelled-success/unresolved-steps; completion valid/false/missing-test-evidence/
  missing-verification/incomplete-plan/negation; workspace allowed/disallowed/mixed;
  **idempotency** — two runs identical; **no side effects** — TaskState/artifacts/permissions
  snapshotted unchanged).
- **10-min live** — §7.5 (**12 checks**, V1–V12): valid research → PASS; researcher write →
  permission violation; valid coder implementation → PASS; SUCCESS without evidence →
  result/completion failure; agent claims tests passed while Test Intelligence says failed →
  contradiction; out-of-scope file → workspace violation; budget exceeded; timeout exceeded;
  Task-A artifact into Task-B → mismatch; cancelled task returns SUCCESS → conflict; validate
  twice → identical + zero mutation; all required evidence → PASS.
- **Fixes during review (code-reviewer-deepseek-flash):** the completion-claim heuristic regex
  matched substrings — `"task completely broken"`, `"undone everything"`, `"not all done yet"`
  were misclassified as completion claims. Verified empirically, fixed with word boundaries
  (`\b`) + negation lookbehinds + comma disambiguation; **empirically verified zero false
  positives and zero missed legitimate claims**; 2 regression tests added.

### 7.6 — Workflow Engine + Sequential Execution

**Implemented** (`src/ultron/core/orchestration/workflow.py`, 1,196 lines):

- **Models** — `WorkflowStatus` / `WorkflowStepStatus` enums + transition tables; `Workflow`
  (workflow_id, task_id, ordered steps, status, current_step, `WorkflowEvent` observability
  list); `WorkflowStep` (step_id, agent_type, objective, dependencies, input artifacts,
  `required_evidence`, `claims_completion`, `allowed_scope`, `timeout_seconds`).
  `Workflow.complete()` refuses while any step is unfinished (state-corruption guard).
- **Dependency validation** — `create_workflow` rejects (ValueError): empty workflows,
  duplicate step/workflow ids, unknown/missing/self/circular dependencies (DFS cycle
  detection), cross-workflow dependencies, unregistered agent types.
- **Sequential engine** — `create_workflow / validate_workflow / start / execute_next_step /
  execute_until_blocked (auto-starts PENDING) / pause (PENDING or RUNNING) / resume (PAUSED
  or WAITING) / cancel / get_status / trace_rows`. Steps run only when all dependencies are
  COMPLETED; WAITING steps resume through the supervisor's existing WAITING-dispatch path
  (same delegation, run record preserved); completed steps are never repeated, including
  across `model_dump_json()` → new engine + same `ArtifactStore` reload (Live test 76.11).
- **Supervisor integration** — the engine never instantiates agents; every step becomes a
  `DelegationRequest` (objective, constraints, expected output, per-step timeout) dispatched
  through `Supervisor`; new `Supervisor.get_run()` accessor exposes the run record.
- **Validator gate** — every AgentResult is validated via `OrchestrationValidator`
  (`ValidationContext` with agent_state/result/delegation/task_state/artifacts/workspace/
  allowed_scope/required_evidence/test_results/claims_completion) PLUS deterministic engine
  gates: `_missing_evidence` (step `required_evidence` always enforced) and failing
  `TestResult` artifacts fail their step. A SUCCESS without evidence never completes a step.
- **Failure mapping** — agent FAILED/CANCELLED → step + workflow FAILED/CANCELLED; agent
  NEEDS_INPUT → step + workflow WAITING (resumable); agent BLOCKED → step + workflow
  BLOCKED (**terminal by design**, mirroring `TaskState.block()`; only PAUSED/WAITING are
  resumable). No automatic retries.
- **TaskState integration** — workflow completion invokes the existing
  `TaskState.mark_complete()` API inside a guard: on `ValueError` (requirements/plan
  unresolved) the task stays incomplete and the reason is recorded in
  `workflow.metadata["task_completion"]` — the engine never forces `task.status`. Step
  failures are recorded via the existing `errors.append(TaskError(...))` mechanism (and the
  observation surface is overwritten so the failure wins over the supervisor's success note).
- **Context isolation** — a step receives only the result artifacts of its completed
  dependencies, optionally filtered by `input_artifact_types`; never trajectories.
- **Observability** — `WorkflowEvent`s (WORKFLOW_CREATED/STARTED/PAUSED/RESUMED/
  CANCELLED/COMPLETED, STEP_READY/STARTED/COMPLETED/FAILED/BLOCKED/WAITING/CANCELLED)
  logged via `get_logger`; `trace_rows()` links workflow → task → step → delegation → agent
  → artifact.

**Test phases:**
- **Unit/integration** — `tests/test_workflow_engine.py` (**51 tests**): creation +
  structural validation (missing task id, duplicate ids, unregistered agent type, missing /
  self / circular / cross-workflow dependencies, empty workflows), dependency enforcement
  (multiple sequential dependencies), lifecycle (valid + invalid transitions, pause-before-
  first-step, cancel-before/during, complete-guard), sequential execution (order, no-repeat,
  `execute_next_step` before start / while paused), delegation wiring (agent type, objective,
  constraints, expected output, task binding, artifact flow + type filtering), validator gate
  (valid / invalid / false-completion / missing-evidence), failures (research / implementation
  / test / review), blocked + needs-input + resume, pause/resume + serialization reload,
  cancellation preserving artifacts, completion + TaskState enforcement (resolved → completed;
  unresolved → NOT completed), context isolation (no trajectory), traceability, determinism.
- **10-min live** — `_orchestration_live_check.py` §7.6 (**15 live tests / 18 checks**):
  happy path 4-step chain; dependency enforcement; agent failure; validation failure
  (SUCCESS without evidence); test failure; NEEDS_INPUT → WAITING; pause/resume without
  repeating research; cancellation preserving artifacts; context isolation; false completion;
  restart via serialization in a fresh engine; end-to-end traceability; circular workflow
  rejected; multi-artifact filtering; TaskState invariant (all SUCCESS but requirement open
  → task NOT completed).
- **Fixes during testing:** fake agents must key behavior on their own identity (not the
  context); `execute_until_blocked` now auto-starts PENDING workflows; dependency artifacts
  are wired into `step.input_artifacts` (with `input_artifact_types` filtering) before
  delegation; needs-input-once detection uses `state.status is WAITING` at factory time;
  reviewer fixes: BLOCKED-terminal documented, `resume()` no-op no longer logs
  WORKFLOW_RESUMED, failure observation wins on the TaskState surface.

---

## 8. Test-count progression (suite grew monotonically, no regressions)

| Stage | Suite count | Δ |
|---|---|---|
| Baseline before FIX #7 (incl. §7.1 contract 33 + §7.2 registry 27) | 1139 | — |
| +§7.3 artifacts (23) | 1162 | +23 |
| +review regression tests (2) | 1164 | +2 |
| +§7.4 supervisor/delegation (20) | 1184 | +20 |
| +validation layer (50) | 1234 | +50 |
| +regex regression tests (2) | 1236 | +2 |
| +workflow engine §7.6 (51) — **current** | **1287** | +51 |

Pre-FIX #7 sections contributed earlier (already committed): FIX #3 coding (100 tests),
FIX #4 intelligence (86), FIX #5 test intelligence (62), FIX #6 memory (69:
foundation 29 + integration 20 + stress 20; `test_memory_graph.py` 27 is the pre-existing
knowledge-graph memory), FIX #1/#2 planning/state (~153).

---

## 9. Reproducing the full verification

### 9.1 Current change set (uncommitted)
```
 M docs/agents.md
?? _orchestration_live_check.py
?? src/ultron/core/orchestration/
?? tests/test_agent_artifacts.py
?? tests/test_agent_contract.py
?? tests/test_agent_registry.py
?? tests/test_orchestration_validation.py
?? tests/test_supervisor_delegation.py
?? tests/test_workflow_engine.py
```

### 9.2 Commands a verifier should run (all from repo root)
```bash
# 1. Full automated suite (expect: 1287 passed, 0 failed)
.venv/bin/python -m pytest -q

# 2. Lint (expect: "All checks passed!")
.venv/bin/ruff check .

# 3. Live orchestration harness (expect: "Live validation: 109/109 checks passed.", exit 0)
.venv/bin/python _orchestration_live_check.py

# 4. Per-section unit tests (expect: each file's count passes)
.venv/bin/python -m pytest tests/test_agent_contract.py tests/test_agent_registry.py -q      # 33 + 27
.venv/bin/python -m pytest tests/test_agent_artifacts.py -q                                   # 25
.venv/bin/python -m pytest tests/test_supervisor_delegation.py -q                             # 20
.venv/bin/python -m pytest tests/test_orchestration_validation.py -q                          # 52
.venv/bin/python -m pytest tests/test_workflow_engine.py -q                                   # 51
```

### 9.3 Live-harness section breakdown (109 checks)
| Section | Checks |
|---|---|
| §7.1 contract + lifecycle | 42 |
| §7.2 registry + permissions | 13 |
| §7.3 artifacts | 14 |
| §7.4 supervisor → delegation | 10 |
| validation layer | 12 |
| §7.6 workflow engine (15 live tests) | 18 |

---

## 10. Known limitations (do NOT over-claim)

1. **Orchestration is sequential only** — §7.6 workflows execute one step at a time through
   the supervisor (dependencies enforced, no concurrent steps; `Supervisor.decide()` never
   claims task completion — the workflow engine owns that decision). Parallel execution,
   dynamic routing, workspace locks, retries beyond existing behavior, and agent-to-agent
   messaging are still not implemented.
2. **No real specialist agents yet** — §7.4 and §7.6 run on fakes via `agent_factory`; dispatch
   without a factory fails loudly. The real research/coder/tester/reviewer/security agents
   (wrapping FIX #3–#6 capabilities) are future work.
3. **Code intelligence is AST-based (Python) + lexical fallback** — not tree-sitter, no bundled
   real LSP servers; LSP is an abstraction validated with mocks/fakes. Semantic search is
   metadata-aware but embeddings are never the source of truth.
4. **Validation reports, it does not enforce** — timeout cancellation, retries, and violation
   handling remain the Supervisor/execution layer's responsibility.
5. **No git-diff-based workspace verification yet** — files are compared against workspace/scope
   prefixes, not `git status` (a suggested follow-up).
6. **Memory formation is policy-gated, not autonomous** — "intelligent memory formation" (auto-
   promoting observations) is listed as remaining work in FIX #6's report.
7. **Default artifact_id is `task:agent:type`** — an agent producing two artifacts of the same
   type for one task must pass explicit ids (the store enforces uniqueness).
8. **Heuristic completion phrasing is conservative** and only consulted when the explicit
   `claims_completion` flag is absent; it applies to SUCCESS results only (erring toward stricter
   validation — correct direction).

## 11. Architectural invariants preserved (per spec)

- TOOL SUCCESS ≠ STEP SUCCESS ≠ TASK SUCCESS
- LLM CLAIM OF SUCCESS ≠ VERIFIED SUCCESS (validation layer + completion enforcement)
- FAILED TEST ≠ COMPLETED TASK
- PENDING CONFIRMATION ≠ TASK TERMINATION (confirmation resume preserves state)
- PLAN ≠ EXECUTION ≠ VERIFICATION
- SECURITY remains outside the LLM's authority (frozen profiles, registry-controlled; validation
  inspects verdicts, never re-issues or bypasses them)
- CURRENT SOURCE > STALE MEMORY; TASKSTATE > CONVERSATION HISTORY; PROJECT MEMORY is
  project-scoped; SECRET never becomes persistent memory.
- STEP SUCCESS ≠ WORKFLOW SUCCESS ≠ TASK SUCCESS — the workflow engine gates every step
  through the validator and invokes TaskState's own `mark_complete()` (guarded); it never
  forces `task.status`. A workflow may complete while the task legitimately stays incomplete.
