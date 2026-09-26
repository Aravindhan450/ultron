# ULTRON Phase 2 — Real CLI Production Validation Report

**Date:** 24 September 2026
**Phase under validation:** Phase 2 — Context Manager + Token Budgeting (committed `c2de85b`)
**Target specification:** `ULTRON_Autonomous_Coding_Harness_Gap_Closure_Plan.pdf` (v1.0), §5
**Production entry point used:** `.venv/bin/python -m ultron.main chat` (real `llama-server`, real GGUF model)
**Harness:** `_phase2_cli_validation.py` (pty driver; same conventions as `_cli_output_e2e.py`)
**Raw evidence:** `scratch/phase2_cli_{A,B,C,D}.log` (+ `*_clean.log`), `~/.ultron/events/*.jsonl`, `~/.ultron/ultron.log`

---

## 1. Harness requirements checked

Requirements taken from the Harness Plan §5 (Phase 2) — no invented requirements:

| # | Requirement (Harness Plan §5) | How checked | Result |
|---|---|---|---|
| R1 | `ContextManager` with a hard token/character budget | Source inspection `core/context/manager.py`; in-CLI `CONTEXT_BUILT` payloads | PASS |
| R2 | Deterministic context priority (task > step > failure > repo > recent > older) | Source inspection of `budget_messages` anchors + priority logic | PASS (static) |
| R3 | Condensation/summarization for long histories | Real CLI long turn C: `dropped_messages=4` observed | PASS |
| R4 | Reserve output budget | `reserved_output_tokens=2048` in every in-loop event | PASS |
| R5 | Bounded model context before **every** model call in the agent loop | `react.py:1185-1202` + 36 persisted in-loop `CONTEXT_BUILT` events | PASS |
| R6 | Oversized tool-output handling (head/tail truncation) | Real CLI: `truncated_tool_messages` up to 4 with `max_tool_output_tokens=1000` | PASS |
| R7 | Preservation of critical task/system context (anchors) | Source inspection: system prompt + first user goal protected | PASS (static) |
| R8 | Model-specific context limits from the selected model | `runtime.py` assigns `budget.model_context_limit` from `handle.model_spec.recommended_context_length`; observed 16384 for routed coder model | PASS (wiring) / PARTIAL (see §5) |
| R9 | `CONTEXT_BUILT` observability | Persisted JSONL events (snapshot + per-iteration) | PASS |
| R10 | Integration with the real runtime/agent path | Real CLI runs A & D completed end-to-end (`task_completed`) | PASS |
| R11 | No model call exceeds configured context budget | 36 in-loop events, invariant checked, 0 violations | PASS |

Out-of-scope for Phase 2 per the plan (recorded only): SimpleAgent has no context-manager budgeting; verification/planning model calls are unbudgeted (see §7).

---

## 2. Production call graph (observed, not assumed)

```
ultron chat  (main.py:790 async_chat)
  └─ ChatSession prompt (prompt_toolkit)
  └─ handle_slash_command / truncate_history(history, 10)
  └─ AgentRuntime(router=…, lifecycle_manager=…, event_bus=EventBus(get_default_event_store()))
        (main.py:935-943 — one runtime per turn)
  └─ _plan_and_run (main.py:946)
        └─ prepare_task_for_execution(input, engine)   [task_planning.py — 2 model calls]
        └─ AgentRuntime.execute(…)                      [runtime.py: execute()]
              ├─ ModelRouter.route(_build_routing_request)   → qwen2.5-coder-7b-instruct (coding)
              ├─ ModelLifecycleManager.ensure_loaded → engine.base_url / set_model swapped
              ├─ context_manager.budget.model_context_limit = model_spec.recommended_context_length
              ├─ TaskEvent MODEL_SELECTED  → persisted (JsonlEventStore)
              ├─ RepositoryContextManager.assemble_snapshot → TaskEvent CONTEXT_BUILT (snapshot metadata)
              └─ agent.run(..., context_manager=…, event_bus=…, run_state=…)
                    └─ ReActAgent.run loop (react.py:1162)
                          ├─ RuntimeEvent STEP_STARTED
                          ├─ context_manager.budget_messages(messages)   ← PHASE 2
                          ├─ RuntimeEvent CONTEXT_BUILT(budget_meta)     ← persisted via emit_sync
                          ├─ RuntimeEvent MODEL_CALLED
                          ├─ engine.generate(history_to_openai_format(model_messages))
                          ├─ tool_call → _coding_gate → boundary.check() → ALLOW / CONFIRM / DENY
                          └─ none → _verify_task / _verify_plan_task → accept iff task.is_complete()
  └─ PendingAction loop → questionary confirm → execute_pending_action (main.py:1170+)
  └─ UI.render_response()
```

Phase 2 is invoked at **`react.py:1187`** (budget) and emits at **`react.py:1189-1200`** (event), plus the
runtime-level snapshot emission at **`runtime.py` (`CONTEXT_BUILT` snapshot)**. Both were observed in the real CLI session.

---

## 3. Automated tests (exact commands, exact results)

```
.venv/bin/python -m pytest -q tests/test_phase2_context_budgeting.py
→ 8 passed in 0.04s

.venv/bin/python -m pytest -q
→ 1 failed, 1868 passed, 6 deselected in 34.54s
   FAILED tests/test_cli_output_isolation.py::test_e_ui_render_tool_activity

.venv/bin/ruff check .
→ All checks passed!
```

The single failure is **pre-existing and unrelated to Phase 2**: `UI.render_tool_activity` now applies Rich
syntax highlighting to the action target, so `python 3.12 release notes` renders as
`python \x1b[1;36m3.12\x1b[0m release notes` and the test's plain-substring assertion fails. The 6 deselected
tests are the `mitl`-marked model-in-the-loop suite (run separately below via the CLI).

---

## 4. CLI scenarios

Harness: `_phase2_cli_validation.py A|B|C|D` (pty; real server; confirmations answered with Enter = "Yes, allow").
Scenario tasks were sized smaller than the protocol's example prompts because the local 7B model needs 25–45 s per
reasoning step and each turn allows up to `max_iterations=20`; the protocol's prompts are explicitly illustrative
("such as" / "for example"), and the same production path is exercised regardless of question breadth.

### Scenario A — normal read-only task
| Field | Value |
|---|---|
| Command | `.venv/bin/python _phase2_cli_validation.py A` |
| Input | "Read the file src/ultron/core/context/manager.py and summarize what it is responsible for. Do not modify any files." |
| Observed | CLI started; task classified; router selected `qwen2.5-coder-7b-instruct` (coding); `CONTEXT_BUILT` emitted; tool `read_file` executed; response rendered in the ULTRON box; CLI still alive |
| Evidence | `scratch/phase2_cli_A_clean.log`; task `task_55c84a76` → 8 events, `task_created → … → run_completed → task_completed` |
| Result | **PASS** |

### Scenario B — multi-turn context retention
| Field | Value |
|---|---|
| Command | `.venv/bin/python _phase2_cli_validation.py B` |
| Input | T1 remember codename + synthetic credential; T2 read `main.py`; T3 no-tools explanation + codename recall |
| Observed | **T1 ran away**: `list_directory` → `pyproject.toml` → repeated `code_investigation`/`read_file orchestration/permissions.py` ×3 → `Execution exceeded max_iterations (20)` → `run_failed`. T2 timed out; T3 responded. |
| Evidence | `scratch/phase2_cli_B_clean.log`; task `task_edca73aa` = 97 events, 20 `step_started`, 20 `model_called`, 16 tool events, `run_failed` |
| Result | **FAIL (validation blocked)** — model-level multi-turn recall not demonstrated; see §9 root cause |

### Scenario C — long-context / compaction stress
| Field | Value |
|---|---|
| Command | `.venv/bin/python _phase2_cli_validation.py C` |
| Input | T1 read manager.py + runtime.py; T2 read boundary.py; T3 summarize task + evidence |
| Observed | T1 timed out (model latency); T2 **responded successfully** (17.9k transcript); T3 issued two confirmation cards (auto-approved) then timed out. **Compaction engaged**: in-loop event `dropped_messages=4` at `input_tokens=13862`, `truncated_tool_messages=3`. No overflow crash. |
| Evidence | `scratch/phase2_cli_C_clean.log`; task `task_2bedce28` (38 events, 10 `context_built`, 8 `model_called`, 1 `confirmation_requested`); task `task_bcf104f6` (38 events, `truncated_tools` up to 4) |
| Result | **PARTIAL** — bounded/condensed context proven; final summary answer not completed within the harness window |

### Scenario D — oversized tool output
| Field | Value |
|---|---|
| Command | `.venv/bin/python _phase2_cli_validation.py D` |
| Input | "Read the file src/ultron/core/agents/simple.py and summarize its structure. Read-only…" |
| Observed | CLI completed; `read_file` returns 5,000 chars (~1,250 tokens) > `max_tool_output_tokens=1000`, and the budgeted observation was truncated (head/tail) before the model call; response rendered; no crash |
| Evidence | `scratch/phase2_cli_D_clean.log`; task `task_b74163a5` → `task_completed`; `truncated_tool_messages` up to 4 across the session tasks |
| Result | **PASS** |

### Scenario E/F + secret-safety (event/log evidence)
| Field | Value |
|---|---|
| Model-specific limit | `model_selected = qwen2.5-coder-7b-instruct (role=coding)`; every `CONTEXT_BUILT` snapshot carries `model_context_limit: 16384` matching the coder model's `recommended_context_length` |
| `CONTEXT_BUILT` persisted | 41 events across 5 task files (snapshot form + per-iteration form) in `~/.ultron/events/<task_id>.jsonl` |
| Secret safety | `sk-test-ULTRON-PHASE2-SECRET` → **0 occurrences** in any persisted event file |
| Result | **PASS** (secret + persistence + event correctness), **PARTIAL** for "model-specific" (§5) |

---

## 5. Actual context-budget evidence

Observed values from persisted per-iteration `CONTEXT_BUILT` events (`~/.ultron/events/*.jsonl`):

```
model_context_limit       : 16384      (from selected model spec; identical for all 3 catalog models)
reserved_output_tokens    : 2048
max_input_tokens (derived): 16384 - 2048 = 14336
observed input_tokens     : 7808 → 13943   (min → max across 36 real model calls)
observed dropped_messages : 0 (35 calls) → 4 (1 call, input_tokens=13862)
observed truncated_tools  : 0 → 4
budget invariant          : input_tokens + reserved_output_tokens <= model_limit
                            → 36 OK / 0 VIOLATED
```

Sample (worst case observed):
`input=13943 + reserved=2048 = 15991 <= 16384 → OK`

**Test E caveat (evidence-backed):** the runtime→budget wiring is present
(`runtime.py`: `self.context_manager.budget.model_context_limit = handle.model_spec.recommended_context_length`)
and unit-covered (`test_runtime_updates_context_budget_for_routed_model`), but at CLI level the value cannot be
*distinguished* from a hardcoded 16384 because `qwen3-8b`, `gemma-3-4b-it`, and `qwen2.5-coder-7b-instruct`
currently all declare `recommended_context_length = 16384`. Distinguishing it requires a catalog entry with a
different context length (e.g. an 8192 model) or a unit assertion on a model whose value differs.

---

## 6. Event evidence

```
task_id (Scenario A)   : task_55c84a76…   goal="Read the file src/ultron/core/context/manager.py…"
task_id (Scenario D)   : task_b74163a5…   goal="Read the file src/ultron/core/agents/simple.py…"
task_id (Scenario C)   : task_2bedce28…, task_bcf104f6…   (no goal field; tasks created pre-runtime)
task_id (Scenario B)   : task_edca73aa…   (runaway; run_failed)
CONTEXT_BUILT events   : 41 total — snapshot form (total_estimated_tokens, total_characters, items_count,
                         dropped_items_count, compacted, source_contributions, model_context_limit)
                         + in-loop form (input_tokens, model_limit, reserved_output_tokens,
                         dropped_messages, truncated_tool_messages)
persisted              : YES — ~/.ultron/events/<task_id>.jsonl (JsonlEventStore, monotonic sequence,
                         RuntimeEvents persisted via EventBus.emit_sync → to_task_event)
secret-safe            : YES — 0 synthetic-secret occurrences in all event files
lifecycle coverage     : task_created, task_state_changed, model_selected, context_built, step_started,
                         model_called, tool_started, tool_completed, confirmation_requested,
                         run_started, run_completed, run_failed, task_completed, task_failed
```

Security-gate evidence (real CLI, scenario C): the model attempted `write_file project_summary.md` and
`run_command git …`; neither executed silently — both produced `Confirmation Required` cards
(`✻ Create new file — project_summary.md`) and ran only after approval. Task state and events reflect
`confirmation_requested`.

---

## 7. Architecture audit — Phase 2 coverage of `engine.generate()` paths

| # | Call site | Budgeted by Phase 2? | Classification |
|---|---|---|---|
| 1 | `core/agents/react.py:1209` (primary ReAct loop) | **YES** — `budget_messages` at :1187, `CONTEXT_BUILT` at :1189 | In scope, validated |
| 2 | `core/agents/react.py:1481` (`_verify_task`) | NO | **Integration gap (Phase 2 scope)** — `context_manager` is available in `run()`; verification prompt (which embeds the model's proposed answer) is unbounded |
| 3 | `core/agents/react.py:1615` (`_verify_plan_task`) | NO | **Integration gap (Phase 2 scope)** — same as #2 |
| 4 | `core/intelligence/task_planning.py:814, 836` (plan + recovery) | NO | **Integration gap (upstream)** — prompts contain the user goal only; no token accounting |
| 5 | `core/agents/simple.py:1407, 1682, 2636, 2801` | NO | **Out of Phase 2 scope as specified** (plan names `agents/react.py`); SimpleAgent is bounded only by `truncate_history(history, 10)` |

Explicit statement: **not every production `engine.generate()` path is budgeted.** The primary ReAct loop is;
verification and planning calls are not. These are recorded architectural integration gaps, not silent exclusions.

---

## 8. Files changed during validation

**No source modifications were required** — `git status` shows zero tracked modifications
(`ruff` clean, Phase 2 tests green before and after).

New untracked artifacts produced by this validation:
```
_phase2_cli_validation.py         (validation harness; pty driver, scenario-scoped)
scratch/phase2_cli_A.log / _clean.log
scratch/phase2_cli_B.log / _clean.log
scratch/phase2_cli_C.log / _clean.log
scratch/phase2_cli_D.log / _clean.log
scratch/phase2_driver_A.out       (superseded launch output)
scratch/gap_closure_plan.txt      (extracted text of the Harness Plan PDF, used for phase requirements)
project_summary.md                (created BY the production CLI in scenario C — gated by a confirmation card
                                   and approved by the validation driver; the prompt had requested read-only,
                                   so the model over-stepped and the security boundary caught it)
```

`project_summary.md` is a validation side effect, not a source change; delete it if unwanted.

---

## 9. Failures and fixes

### F1 — Scenario B runaway loop (validation blocker)
```
Failure        : Scenario B turn 1 exhausted max_iterations (20) with repeated identical tool calls
                 (code_investigation + read_file orchestration/permissions.py ×3) and ended in
                 "Execution exceeded max_iterations (20)"; task_edca73aa → run_failed.
Production path: AgentRuntime.execute → ReActAgent.run loop → _coding_gate → repeat
Root cause     : (a) model thrash (7B coder re-issuing an identical tool call after an unhelpful
                 observation); (b) budget governance fired correctly (RuntimeBudget stopped at 20), but
                 repeated-identical-action detection exists only in CodingExecutor/RepairBudget
                 (core/coding/executor.py) and is not applied uniformly at the ReAct loop layer.
Fix applied    : none to source (per protocol, capture → trace first).
Recommended    : Phase 5 work — promote repeated-identical-failure detection into a ReAct-loop-level breaker
                 with a bounded, explicit stop reason; regression test with a scripted repeating engine.
```

### F2 — Scenario C turn 1/3 timeouts; Scenario B turn 2 timeout
```
Failure        : Turn exceeded the harness window (190-220 s).
Root cause     : local model latency (25-45 s per generate call) × up to 20 iterations; not a Phase 2 defect.
                 No wall-clock timeout is configured (RuntimeBudget.timeout_seconds=None).
Fix applied    : harness-side timeout sizing only (protocol-compliant; no implementation change).
Recommended    : optional per-task wall-clock budget to avoid unbounded interactive turns.
```

### F3 — Pre-existing unrelated test failure
```
Failure        : tests/test_cli_output_isolation.py::test_e_ui_render_tool_activity
Root cause     : UI.render_tool_activity now syntax-highlights the target text, so "3.12" carries ANSI
                 bold-cyan codes and the plain-substring assertion fails.
Fix applied    : none (out of Phase 2 scope; no source change made during this validation).
Recommended    : assert on ANSI-stripped output (or on the semantic fields) rather than raw text.
```

**Required remediation before Phase 2 can be declared fully CLI-validated:**
1. Route `_verify_task` and `_verify_plan_task` model calls through `budget_messages` (Phase 2 scope, small generic change + regression test).
2. Record the SimpleAgent/planning bypass as an acknowledged gap or bring them under the ContextManager.
3. Add the ReAct-level repeated-identical-action breaker (Phase 5 dependency that currently blocks multi-turn CLI validation).

---

## 10. Final status

```
IMPLEMENTATION REQUIRES REMEDIATION
```

Rationale — the Phase 2 mechanism itself **is** proven through the real CLI:
- Scenarios **A** and **D** completed end-to-end (`task_completed`) with the real `llama-server` model.
- **36/36** real in-loop model calls satisfied `input_tokens + reserved_output_tokens <= model_context_limit`.
- Compaction engaged naturally (`dropped_messages=4`) and oversized tool outputs were truncated in production.
- `CONTEXT_BUILT` is emitted and **persisted** (snapshot + per-iteration), with zero synthetic-secret leakage.
- The security boundary surfaced confirmation cards for the model's attempted write/command instead of executing them silently.

But a full `IMPLEMENTATION COMPLETE — CLI VALIDATED` verdict is **not** justified today because:
- Scenario **B failed** (runaway iteration loop → `run_failed`), so multi-turn retention was not demonstrated at model level.
- Scenario **C is partial** (bounded context/compaction proven; final summary turn not completed in the harness window).
- **Phase 2 does not cover every production model call** — `_verify_task`, `_verify_plan_task`, planning, and SimpleAgent `generate()` paths bypass token budgeting.
- The model-specific limit requirement is wired and unit-verified but not independently distinguishable at CLI level with the current catalog (all models = 16384).

Next step per the Strict protocol (§14): implement remediation item 1 (smallest generic fix — budget the verification model calls), add its regression test, re-run `pytest tests/test_phase2_context_budgeting.py` + full suite + `ruff`, then re-run the real CLI (`_phase2_cli_validation.py A` and `C`) and update this report.
