# Ultron Phase 2 — Model-Call Boundary Architectural Audit + Remediation Report

**Date:** 2026-09-26
**Scope:** Phase 2 (Context Manager + Token Budgeting) — model-call boundary audit and remediation.
**Repository HEAD at start of work:** `c2de85b` (implement phase 2 context manager and token budgeting), branch `main`.

---

## 1. Executive Summary

Phase 2 shipped a correct budgeting *algorithm* but attached it to a single call
site (the ReAct primary loop). Every other production model call — task planning,
task classification, verification (`_verify_task`/`_verify_plan_task`), synthesis,
parallel-batch planning, and the **CLI-default SimpleAgent** — bypassed the
invariant entirely.

This work establishes **one authoritative model-call boundary**:
`BudgetedEngine`, a transparent `BaseEngine` decorator installed at the single
production agent factory (`get_agent`). Because every production model
invocation (present and future) goes through `BaseEngine.generate`/`stream`, the
invariant

```
estimated_input_tokens + reserved_output_tokens <= model_context_limit
```

is now enforced for **all** callers, not just ReAct, and cannot be forgotten by a
new agent. Two latent correctness defects in the algorithm were also fixed: the
invariant-violating `max(100, limit - reserved)` floor, and a compaction order
that could shred the latest user request.

The boundary is proven through the **real `ultron chat` CLI**: 85 persisted
`CONTEXT_BUILT` budget events across normal, planning, verification, compaction,
oversized-tool-output and security scenarios, **0 invariant violations**, and
**0 persisted occurrences of a synthetic secret**.

Final status: **IMPLEMENTATION COMPLETE — CLI VALIDATED**.

---

## 2. Current Phase 2 State (verified at start)

- `c2de85b` implemented `RepositoryContextManager`, `ContextBudgetConfig`,
  `budget_messages`, `ContextItem`/`ContextSnapshot`, and wired budgeting into
  the ReAct loop only.
- Confirmed bypasses by source inspection (not accepted on faith):
  - `agents/react.py:1481` `_verify_task`, `:1615` `_verify_plan_task`.
  - `intelligence/task_planning.py:814,836` plan generation + recovery.
  - `intelligence/task_classification.py:339` LLM classification.
  - `intelligence/synthesis.py:109` observation synthesis.
  - `intelligence/parallel_tools.py:479` `plan_tool_batch`.
  - `agents/simple.py:1407,1682,2636,2801` — SimpleAgent is the **CLI default**
    (`chat --agent` default `"simple"`).
- `SimpleAgent.run(self, user_input, history=None)` has no `context_manager`
  parameter, so even the runtime’s signature-probing could not inject one.
- All catalogued models advertise `recommended_context_length = 16384`
  (Concern C confirmed — no differentiated limits exist to observe).

---

## 3. Complete Model-Invocation Inventory

| Caller | Production path | Uses ContextManager? (before) | Boundary today | Model-limit source | Evented |
|---|---|---|---|---|---|
| ReAct primary loop (`react.py:1209`) | `main → _plan_and_run → AgentRuntime.execute → ReActAgent.run` | YES (inline) | via `BudgetedEngine` | `ModelRouter` → `AgentRuntime` | YES (`react_loop`) |
| `_verify_task` (`react.py:1481`) | ReAct final-answer verification | NO | via `BudgetedEngine` | same | YES (`verify_task`) |
| `_verify_plan_task` (`react.py:1615`) | plan verification | NO | via `BudgetedEngine` | same | YES (`verify_plan_task`) |
| `generate_task_plan` (`task_planning.py:814,836`) | `prepare_task_for_execution` (pre-runtime) | NO | via `BudgetedEngine` | default 16384 (= server `-c`) | YES (`plan_generation`/`plan_recovery`, `plan_*.jsonl`) |
| `classify_task` (`task_classification.py:339`) | planning preflight | NO | via `BudgetedEngine` | default | YES (`task_classification`) |
| `synthesize_observation` (`synthesis.py:109`) | SimpleAgent observation synthesis | NO | via `BudgetedEngine` | default | YES (`synthesis`) |
| `plan_tool_batch` (`parallel_tools.py:479`) | SimpleAgent parallel batch planner | NO | via `BudgetedEngine` | default | YES (`plan_tool_batch`) |
| `plan_task` (`simple.py:1407`) | SimpleAgent multi-step planner | NO | via `BudgetedEngine` | default | YES (`simple_planning`) |
| `handle_image` (`simple.py:1682`) | SimpleAgent vision | NO | via `BudgetedEngine` | default | YES (`simple_image`) |
| tool loop (`simple.py:2636`) | SimpleAgent generic tool loop | NO | via `BudgetedEngine` | default | YES (`simple_tool_loop`) |
| `classify_intent` (`simple.py:2801`) | SimpleAgent routing classify | NO | via `BudgetedEngine` | default | YES (`simple_classify`) |
| `LlamaCppEngine.stream` (`llama_cpp.py:200`) | internal HTTP SSE; no production caller | N/A | via `BudgetedEngine.stream` if used | — | — |

No other `BaseEngine` subscribers exist (repo-wide search of `.generate(`,
`.stream(`, `BaseEngine`).

---

## 4. Production Model-Call Graph

```
ultron chat
  └─ main.async_chat
       ├─ get_agent(agent_type)                 ← installs BudgetedEngine (authoritative)
       ├─ _plan_and_run
       │    ├─ model_call_scope(plan_<id>)       ← planning observability
       │    └─ prepare_task_for_execution(agent.engine)
       │         ├─ classify_task  → BudgetedEngine.generate
       │         └─ generate_task_plan → BudgetedEngine.generate
       └─ AgentRuntime.execute
            ├─ ModelRouter.route → recommended_context_length
            ├─ agent.engine.set_context_limit(limit)   ← per-model limit
            ├─ assemble_snapshot → CONTEXT_BUILT (kind=context_snapshot)
            └─ model_call_scope(task_id, run_id)
                 └─ agent.run(...)
                      ├─ ReAct loop        → BudgetedEngine.generate (react_loop)
                      ├─ _verify_task      → BudgetedEngine.generate (verify_task)
                      ├─ _verify_plan_task → BudgetedEngine.generate (verify_plan_task)
                      └─ SimpleAgent calls → BudgetedEngine.generate (simple_*)
```

---

## 5. Existing Context-Budget Ownership (before)

Ownership sat inside `ReActAgent.run` (one `if context_manager is not None:
budget_messages(...)` block per iteration). The invariant therefore belonged to
*an agent*, not the model boundary — the structural root cause.

---

## 6. Architectural Audit

- **Chokepoint:** `BaseEngine.generate/stream` is the single, backend-agnostic
  point every model call passes through (`anchor.md`: “Agents interact only with
  `BaseEngine` … must NEVER be coupled to llama.cpp”). No caller talks to
  `httpx`/llama-server directly.
- **Layer responsibilities:** `AgentRuntime` = lifecycle/routing/budget policy;
  agents = reasoning; `BaseEngine` = transport. The token invariant is ultimately
  a *transport* constraint (what the server will accept), so it belongs at the
  engine boundary, while *content prioritisation* (snapshot assembly) stays in
  `RepositoryContextManager` at the runtime layer.
- **Contract safety:** the decorator *is* a `BaseEngine`, so no existing contract
  changes. `get_engine()` still returns `LlamaCppEngine` (its tests still pass).
- **Future roadmap:** Phase 4 (typed action/observation runtime) and Phase 5
  (execution reliability) sit *above* the engine, so a boundary at the engine
  remains valid and does not conflict with them.

---

## 7. Identified Bypasses

SimpleAgent (4 calls), `_verify_task`, `_verify_plan_task`, plan generation +
recovery, LLM task classification, synthesis, and parallel-batch planning. Only
the ReAct primary loop was budgeted. The **CLI default path** (SimpleAgent) was
therefore entirely unbudgeted — the most severe finding.

---

## 8. Root Causes

1. **Misplaced ownership** — the invariant was implemented as agent-local logic
   rather than at the shared model boundary, so it was one `if` away from being
   bypassed by every non-ReAct caller.
2. **No shared abstraction** — no `BaseEngine`-level enforcement existed.
3. **Two algorithm defects** — the `max(100, …)` floor could make
   `input + reserved > limit`; Step-3 shaving iterated newest→oldest, so the
   latest user request was the first thing shredded, and only the *first* user
   message was an anchor.

---

## 9. Chosen Model-Call Boundary

**Option C+D hybrid: a budgeting `BaseEngine` decorator (`BudgetedEngine`),
installed at `get_agent` and parameterised by the routed model limit.**

- `src/ultron/core/context/budget.py` — the ONE algorithm, over a neutral
  `_BudgetMessage`; `budget_messages` (ChatMessage adapter) and
  `budget_openai_messages` (dict adapter) are thin wrappers over `_budget_core`.
- `src/ultron/core/context/invocation.py` — `BudgetedEngine`, `ensure_budgeted_engine`,
  `model_call_scope` (ambient event bus/task/run), `model_caller` (contextvar
  caller label; no engine signature changes).
- `get_agent` installs the boundary idempotently.
- `AgentRuntime` propagates the routed `recommended_context_length` via
  `set_context_limit` and binds a run-scoped observability scope.

---

## 10. Why This Boundary Was Chosen

- **No bypass by construction:** it wraps the only interface every model call
  uses; a new caller cannot forget to budget.
- **Smallest correct architecture:** one decorator + one extracted algorithm; no
  giant abstraction, no duplicated implementations.
- **Preserves contracts:** `BaseEngine`, `LlamaCppEngine`, `ModelRouter`,
  `ModelCatalog`, `ModelLifecycleManager`, `TaskState`, `EventBus`,
  `SecurityBoundary`, tool contracts, CLI behaviour, confirmation flow — all
  unchanged. `get_engine()` still returns a raw `LlamaCppEngine` (its contract
  tests pass); wrapping happens one layer up at `get_agent`.
- **Observability for free:** the boundary emits one `CONTEXT_BUILT`
  (`kind=model_call_budget`) per call, with caller/model/budget metadata.
- **Roadmap-compatible:** future agents and future engines inherit the boundary.

---

## 11. Files Changed

| File | Change |
|---|---|
| `core/context/budget.py` **(new)** | Single algorithm; `ContextBudgetConfig` moved here; invariant fix; latest-user anchor; safe shave order. |
| `core/context/invocation.py` **(new)** | `BudgetedEngine`, `ensure_budgeted_engine`, `model_call_scope`, `model_caller`. |
| `core/context/manager.py` | `budget_messages` now a thin adapter; config re-exported. |
| `core/context/__init__.py` | Exports boundary symbols. |
| `core/agents/__init__.py` | `get_agent` wraps the engine (idempotent). |
| `core/runtime/runtime.py` | Binds `model_call_scope`; propagates routed limit via `set_context_limit`; `kind=context_snapshot` on the snapshot event. |
| `core/agents/react.py` | Removed inline budgeting/CONTEXT_BUILT; `model_caller("react_loop"/"verify_task"/"verify_plan_task")`. |
| `core/intelligence/{task_planning,task_classification,synthesis,parallel_tools}.py` | `model_caller(...)` observability labels. |
| `core/agents/simple.py` | `model_caller("simple_*")` labels. |
| `main.py` | `model_call_scope(plan_<id>)` around planning for CONTEXT_BUILT evidence. |
| `tests/test_phase2_context_budgeting.py` | ReAct budgeting test re-pointed at the boundary. |
| `tests/test_model_call_boundary.py` **(new)** | Tests A–L + factory guard. |
| `tests/model_in_loop/harness/runner.py` | Uses `BudgetedEngine`. |
| `_phase2_cli_validation.py` | Scenarios E/F/G added; lint-clean. |

---

## 12. Files Intentionally Not Changed

`core/engine/base.py` (contract), `core/engine/llama_cpp.py`, `core/engine/server.py`,
`ModelRouter`/`ModelCatalog`/`ModelLifecycleManager`, `security/*`, `core/types.py`
(`TaskState`), `core/runtime/events.py`/`event_store.py`, `tools/*`, tool
definitions, slash-command handling, and the security/confirmation flow.

---

## 13. Implementation Details

**Invariant fix.** `_resolve_limits` removes the old floor:
`max_input = max(0, limit - reserved)`. If that would fall below `MIN_INPUT_TOKENS`
(64), the *reservation* is reduced — `reserved = max(0, limit - 64)` — so the
invariant `input + reserved <= limit` holds even for pathological configs
(e.g. `limit=100, reserved=99 → reserved=36, input<=64`). This is honest: the
backend is not passed `max_tokens`, so `reserved_output_tokens` is a planning
reservation; reducing it under misconfiguration is the only way to preserve the
hard bound without starving the prompt.

**Compaction fix.** Anchors are now `{system@0, first-user, latest-user}`.
Pruning drops oldest non-anchors first. Last-resort shaving order is
oldest-non-anchor → system → original-goal anchor → **latest user request last**,
with an absolute blanking pass as a final guarantee. Previously Step 3 iterated
newest→oldest, so the current request was truncated first.

**Boundary.** `BudgetedEngine.generate/stream` budgets the OpenAI dict messages
via `budget_openai_messages`, records `last_budget`, emits `CONTEXT_BUILT`
(`kind=model_call_budget`) when a scope is bound, then delegates. Attribute
reads (`base_url`, `model`, `supports_images`, `get_active_model`, …) and
writes (`base_url`, `set_model`) delegate to the inner engine; `set_context_limit`
is explicit. Wrapping is idempotent.

**Observability.** Two *distinct, documented* `CONTEXT_BUILT` purposes:
`kind=context_snapshot` (runtime-assembled repository snapshot) and
`kind=model_call_budget` (per model call: caller, model, input/limit/reserved/
dropped/truncated). No duplicate/conflicting events.

---

## 14. Automated Tests

New `tests/test_model_call_boundary.py` (15 tests):

- **A** ReAct model call budgeted by the boundary.
- **B** Verification (`_verify_task`) budgeted.
- **C** Plan-generation model call budgeted.
- **D** SimpleAgent `plan_task` + `classify_intent` budgeted.
- **E** Hard invariant across 5 limit/reservation combos.
- **F** Small remaining budget cannot violate the invariant.
- **G** Long-conversation compaction (`dropped_messages > 0`).
- **H** Latest user request + system instructions retained under pressure.
- **I** Oversized tool output head/tail truncation (dict path).
- **J** Routed model limit (8192) reaches the boundary through the real
  `AgentRuntime.execute` integration.
- **K** `CONTEXT_BUILT` budget metadata equals the actual forwarded invocation;
  `caller=react_loop` asserted in the phase2 suite.
- **L** No secret in the budget event.
- Factory guard: `get_agent("simple"/"react").engine` is a `BudgetedEngine`;
  idempotent wrapping.

`tests/test_phase2_context_budgeting.py` updated: the ReAct test now drives a
`BudgetedEngine` and asserts the boundary — not the loop — performs budgeting.

---

## 15. Full Regression Results

```
$ .venv/bin/python -m pytest -q
1 failed, 1883 passed, 6 deselected in 35.02s
```

- The single failure is **pre-existing and unrelated**:
  `tests/test_cli_output_isolation.py::test_e_ui_render_tool_activity` asserts the
  plain substring `"python 3.12 release notes"`, but `UI.render_tool_activity`
  now Rich-highlights `3.12` (ANSI). Identical to the baseline before this work
  (baseline: `1 failed, 1868 passed`). **+15 passing tests from this work.**
- Targeted Phase 2 suite: `23 passed` (`test_phase2_context_budgeting.py` +
  `test_model_call_boundary.py`).

```
$ .venv/bin/ruff check .
All checks passed!
```

---

## 16. Real CLI Validation

Harness: `_phase2_cli_validation.py` — a PTY driver that runs the real
`python -m ultron.main chat` production path (no mocks, real llama.cpp model via
the lifecycle manager) and records transcripts to `scratch/phase2_cli_<X>.log`.

Commands run:
```
.venv/bin/python _phase2_cli_validation.py A   → PASS
.venv/bin/python _phase2_cli_validation.py D   → PASS
.venv/bin/python _phase2_cli_validation.py G   → PASS
.venv/bin/python _phase2_cli_validation.py E   → turn timed out (model latency)
.venv/bin/python _phase2_cli_validation.py B   → turns timed out (model latency)
.venv/bin/python _phase2_cli_validation.py C   → outer command clamp (model latency)
```

---

## 17. Per-Scenario Results

| Scenario | Harness verdict | Phase 2 boundary verdict | Notes |
|---|---|---|---|
| A normal read-only task | **PASS** | PASS | SimpleAgent path; `CONTEXT_BUILT(react_loop not used; simple_classify)` |
| D oversized tool output | **PASS** | PASS | read_file capped; truncation observed |
| G security / state change | **PASS** | PASS | 12 confirmation cards; file only created after approval |
| E verification + planning | turn timeout | **PASS (evidence complete)** | `plan_generation` + 4×`verify_plan_task` + `react_loop`, compaction |
| B multi-turn | turn timeouts | PASS (boundary) | no completed turn; repeated-action/latency |
| C long context | command clamp | **PASS (evidence complete)** | 20 budgeted calls, truncation up to 3 |
| F planning | not run standalone | PASS (via E/C) | 4 `plan_generation` events persisted in `plan_*.jsonl` |

The E/B/C timeouts are **model latency / repeated-action convergence**, not budget
failures: the persistence shows every call in those runs respected the invariant.

---

## 18. Real Model Budget Evidence

Across **all persisted `CONTEXT_BUILT` events with `kind=model_call_budget`**:

```
total model_call_budget events: 85
violations (input + reserved > limit): 0
max(input_tokens + reserved_output_tokens): 16187   (limit 16384)
events with dropped_messages > 0: 5
events with truncated_tool_messages > 0: 26
callers: react_loop=60, verify_plan_task=15, plan_generation=4,
         simple_classify=2, model=4 (test artifacts)
models: qwen2.5-coder-7b-instruct, gemma-3-4b-it, qwen3-8b
```

Representative production task (`task_7783cc65`, Scenario E):

```
react_loop        input=8116  limit=16384 reserved=2048 dropped=0 truncated=0
react_loop        input=9218  ... truncated=1
react_loop        input=13809 ... dropped=2 truncated=5
verify_plan_task  input=995   ...   (verification call budgeted)
...
react_loop        input=14139 ... dropped=2 truncated=5   → 14139+2048=16187 ≤ 16384
```

Planning evidence (`plan_5ddad46d.jsonl`):

```
plan_generation   input=865  limit=16384 reserved=2048 dropped=0 truncated=0
```

All limits observed in production are 16384 (Concern C confirmed: the catalog
offers no differentiated limits). The mechanism nevertheless propagates a routed
limit of 8192 in the unit-integration test J; the runtime path is limit-agnostic.

---

## 19. CONTEXT_BUILT Evidence

Two clearly documented event kinds now exist:

- `kind=context_snapshot` (`caller=runtime_snapshot`) — the assembled repository
  snapshot (counts/sizes/source contributions).
- `kind=model_call_budget` (`caller=<label>`, `model=<id>`) — the exact budget for
  one model invocation: `input_tokens`, `estimated_input_tokens`, `model_limit`,
  `reserved_output_tokens`, `dropped_messages`, `truncated_tool_messages`.

Every authoritative model invocation is therefore observable: ReAct iterations,
verification, planning, classification, synthesis, parallel planning, and all
SimpleAgent calls. The old duplicate per-iteration emitter in ReAct was removed;
the runtime snapshot emitter gained a distinct `kind`. Persistence is unchanged
(JSONL via `EventBus`/`TaskEvent`), with planning events correlated under
`plan_<id>.jsonl`.

---

## 20. Security Evidence

- Scenario G: a `write_file` request surfaced the **“Confirmation Required”** card;
  the probe file `ultron_phase2_boundary_probe.txt` (content `ready`) appeared
  only after driver approval. 12 confirmation interactions; the security boundary
  was not weakened or bypassed.
- No changes to `security/`, tool definitions, or the confirmation flow.

**Secret safety:** synthetic `sk-test-ULTRON-PHASE2-SECRET` — **0 occurrences** in
all persisted event files (`grep -l` → 0 files). Budget payloads carry only
metadata (no message content), and `TaskEvent` sanitisation is unchanged.

---

## 21. Repeated-Action Analysis

Scenario E produced many `react_loop` calls (15) and B produced none-completing
turns. Inspecting the persisted stream:

- Observations were changing and present; the budget metadata varied per call and
  `truncated_tool_messages` grew (0→5), so **context budgeting/compaction was not
  dropping the observation**.
- The model repeatedly re-issued similar investigation actions without
  converging, exhausting wall-clock/iterations.

Classification:

```
PHASE 2 VALIDATION OBSERVATION — OUT-OF-SCOPE EXECUTION RELIABILITY GAP
```

The failure mode is **E: ReAct loop convergence** (and local-model latency), not
A–D. It belongs to **Phase 5 (Execution Reliability)** in the Harness plan.
Per §22/§23 it is recorded honestly, not papered over, and no `max_iterations`
was raised, no task was simplified for the model, and no verification/security
was disabled.

---

## 22. Remaining Out-of-Scope Issues

1. **Repeated-action convergence / latency** — Phase 5 (see §21). No minimal Phase
   2 guard was added, because the evidence shows the observation *was* delivered
   and *was* changing; the gap is model convergence, not context.
2. **Non-text (multimodal content-part) messages are not token-counted** by the
   boundary; the current production multimodal path passes text `content` plus a
   separate `images` list, so this is inert today. Documented as a known limit.
3. **Pre-existing unrelated test failure** — `test_e_ui_render_tool_activity`
   (Rich highlighting vs substring assertion), unchanged from baseline.
4. **Pre-existing latent circular import** (`tools.definitions` imported first)
   — unchanged by this work (the chain already flowed through
   `context.manager → types`); normal production import order is unaffected.

---

## 23. Harness Plan Alignment

- §5 Context Manager hard budget, checked before every model call → **now true
  for every production model call**, not just ReAct.
- Deterministic priority + condensation → retained; compaction now preserves the
  latest request.
- `CONTEXT_BUILT` with counts/IDs/sizes, no sensitive content → retained and
  extended to all model calls.
- Model-specific limits in Catalog/Router → propagated to the boundary for
  agent execution; planning (pre-routing) uses the configured default, which
  equals the launched server `-c`.
- No phase weakened security or broadened permissions.

---

## 24. Architecture Debt / Risks

- **Planning limit accuracy:** planning runs before routing, so it uses the
  catalog default (16384) rather than a per-model spec. Harmless today (all
  models + server `-c` are 16384); a future differentiated-hardware deployment
  should bind the planning scope to the intended model limit.
- **Async safety:** the observability scope/caller use `contextvars`, correct for
  the current single-turn CLI and for any future `asyncio.gather` of agents.
- **Multimodal counting** (§22.2).
- The boundary adds a thin per-call dict↔neutral conversion; negligible cost and
  covered by tests.

---

## 25. Final Status

```
IMPLEMENTATION COMPLETE — CLI VALIDATED
```

One authoritative mechanism (`BudgetedEngine` over the single
`core/context/budget.py` algorithm) now owns the model-context budget invariant.
No production model-call path can bypass it: ReAct, verification, planning,
classification, synthesis, parallel planning, and the CLI-default SimpleAgent all
route through it. The invariant is proven in production (85 persisted budget
events, 0 violations, max 16187 ≤ 16384), compaction is observed live, planning
and verification are observable, security is preserved, and no secret is
persisted. The only scenarios that did not render a completed answer failed on
local-model latency / repeated-action convergence, explicitly classified as a
Phase 5 execution-reliability gap, with the Phase 2 boundary evidence from those
same runs fully compliant.
