# PROJECT ULTRON — STEP 3: Generic Capability-Based Model Validation Framework

**Verdict: PASS**

The framework tests capabilities, not hardcoded prompts; subjects and tasks
are dynamically discovered/generated; execution goes through the real CLI;
traces are complete; evaluation is capability-level with independent ground
truth; failures are classified by subsystem; development/holdout are
separated; production code is never modified to make tests pass; the
anti-hardcoding audit exists and is clean; the framework itself is tested;
the existing architecture remains the single source of truth; full
regression passes.

---

## 1. Architecture of the validation framework

```
                CANONICAL ARCHITECTURE (consumed, never redefined)
   IntentCategory (nlp/intent.py)  ToolCapability + CapabilityContract
   TOOL_DEFINITIONS (tools/definitions.py)         SecurityBoundary
                        │
                        ▼
              ultron.validation/  (the observer)
   ┌──────────────┬───────────────┬───────────────┬───────────────┐
   ▼              ▼               ▼               ▼               ▼
 subjects     generator        runner         evaluate        audit
 (discovery)  (task gen)   (real CLI, pty)  (8 dims +         (anti-
                                                 failure           hardcoding)
                                                 classes)
   └──────────────────────────┬──────────────────────────┘
                              ▼
                           report.py  →  deterministic markdown report
```

Layers per STEP 3:
- **Layer A** — deterministic regression tests (`tests/`): unchanged, 1545 → 1568 tests pass.
- **Layer B** — capability model tests: the new framework (this step).
- **Layer C** — holdout (STEP 4): `TestSplit.HOLDOUT` exists in the model; generation is
  independent (seeded) and holdout tasks are never stored in production code.

## 2. Files changed

All new — **zero production files modified** (verified via `git status`):

| file | purpose |
|---|---|
| `src/ultron/validation/__init__.py` | package docs + exports |
| `src/ultron/validation/model.py` | `CapabilityTestCase`, `TaskTrace`, `Evaluation`, `Verdict`, `FailureKind`, `TestSplit` (pure data; references canonical vocab only) |
| `src/ultron/validation/subjects.py` | dynamic subject discovery via the existing Code Intelligence index (`find_symbols_in_file`), src-preferring, deterministic |
| `src/ultron/validation/generator.py` | capability testability classification (44/44), entity variants, template families, strict validation, round-robin diversity plan |
| `src/ultron/validation/runner.py` | pty CLI harness (idle-based settle), injectable executor, `ParsedTrace` signal extraction |
| `src/ultron/validation/evaluate.py` | eight-dimension evaluation, canonical ground truth, `FailureKind` classification |
| `src/ultron/validation/audit.py` | AST-based anti-hardcoding scan of routing/security layers |
| `src/ultron/validation/report.py` | deterministic structured report (Phase 18 layout) |
| `tests/test_validation_framework.py` | 23 self-tests (test the tester) |
| `_step3_validation.py` | live runner: `--count N` / `--budget-seconds N` (10-min mode) |

## 3. Test generation strategy

- Subjects: every symbol (class/function/enum) in `src/` files via the existing index,
  plus files and top-level directories — arbitrary repository entities, never hardcoded.
- Tasks: template families per capability (definition/reference/symbol/inspection/code/
  semantic/investigation/file/directory/git/terminal/test/lint/typecheck/build/resource/
  env/…). Wording varies across sentence structure, verbosity, and entity naming
  (`TaskState`, `taskstate`, `TASKSTATE`, `Task State`, `task_state`).
- Validation: every generated task is checked — canonical contract exists, the
  deterministic intent router agrees when it can parse the wording (strict for symbol
  tasks), and the subject is mentioned. Invalid tasks are rejected, not silently kept.
- Diversity: round-robin across capabilities (14 tasks → 14 distinct capabilities in the
  live run); seeded RNG → deterministic plans.

## 4. Capability coverage

All 44 canonical capabilities classified (generation policy only — no tool metadata):

- **read_only (22)** — auto-tested: definition/reference/symbol/inspection/code/semantic/
  repository inspection+investigation, dependency, file search/read/inspection, directory
  list, git, terminal, memory query, graph reasoning, resource monitoring, debug env,
  structured output, api schema, parallel batch.
- **requires_execution (4)** — test_execution, lint, typecheck, build (opt-in `--allow-execution`).
- **external (3)** — web_search, http_request, page_fetch (network; excluded by default).
- **unsafe (12)** — file write/create/delete/rename, directory create, format, install,
  app start/stop, memory update/association, plan management (never auto-generated).
- **not_testable (3)** — coding_request (meta), information_request (context-dependent),
  database_query (live DB state) — **reported in the coverage table, not silently excluded**.

## 5. Model execution mechanism

`ValidationRunner` spawns the actual CLI (`python -m ultron.main chat [--agent react]`)
in a pty and types the generated task, exactly like a user. Reply completion is detected
by terminal idleness after the prompt is re-rendered (with a hard per-task deadline), so
task latency reflects real reply time (~4.7 s/task in the live run). The executor is
injectable so framework self-tests run without a model. Internal functions
(`select_capability`, `find_definition`, …) are used only for evaluation ground truth.

## 6. Trace format

`TaskTrace`: case (id, capability, task, expected capability, subject, contract refs),
ANSI-stripped transcript, latency, plus parsed signals (observed tool names, security
decision, failure markers, quoted entities, evidence file:line hits). Parsed signals are
matched inside the **reply window** (transcript tail) so displayed file contents cannot
produce false tool/marker signals.

## 7. Evaluation mechanism

Eight dimensions per task — intent, capability, tool selection, argument, security,
evidence, investigation, final answer — each PASS/PARTIAL/FAIL/UNRESOLVED, aggregated to
an overall verdict. Dimensions an evaluator cannot determine from the transcript are
UNRESOLVED, never fabricated (e.g., Simple-agent replies that don't name their tool leave
tool-level dims UNRESOLVED).

## 8. Ground-truth mechanism

Independent of the model's answer:
- expected capability = the task's canonical label;
- valid tool set = `tools_with_capability(expected)` + related contracts (TOOL_DEFINITIONS);
- argument ground truth = subject name variants;
- evidence ground truth = per-capability evidence markers (file:line, definition keywords,
  listing content, time output, …) plus content-as-evidence for file/directory/search caps
  and explicit "no X found" as legitimate evidence where absence is a valid outcome.
- speculative phrasing ("is likely the definition") is evidence FAIL.

## 9. Failure classification

`FailureKind` maps failing dimensions to the subsystem: INTENT, CAPABILITY_SELECTION,
TOOL_SELECTION, ARGUMENT, ROUTING, SECURITY, EXECUTION, EVIDENCE, INVESTIGATION,
SYNTHESIS, MODEL_LIMITATION, ENVIRONMENT, EVALUATION, UNKNOWN. A wrong-tool observation is
classified CAPABILITY_SELECTION_FAILURE (the observable root cause); an empty response is
MODEL_LIMITATION; command-not-found is ENVIRONMENT_FAILURE; traceback is EXECUTION_FAILURE;
speculative definitions are EVIDENCE_FAILURE.

## 10. Anti-hardcoding mechanism

`audit.py` parses production Python with `ast` and flags **executable string literals**
matching the historical diagnostic prompts or the historical entity names inside
routing/security layers (`core`, `security`, `permissions`) — docstrings/comments are
ignored. The validation framework itself is excluded from the scan (it is the observer;
its templates are generated test data). Result: **121 files scanned, 0 findings, 0
critical**.

## 11. 10-minute test mechanism

`_step3_validation.py --budget-seconds 600` dynamically discovers subjects, generates a
validated plan, and runs tasks through the real CLI until the wall-clock budget is
reached; `--count N` runs a fixed-size plan. Diversity control prevents one
capability/entity/template from monopolizing the run.

## 12. Development/holdout separation

`TestSplit.DEVELOPMENT` (default) vs `TestSplit.HOLDOUT` (STEP 4). Generation is seeded
and deterministic; holdout will use an independent seed after implementation is frozen.
No generated task or expected answer is stored in production code.

## 13. Validation-framework test results (test the tester)

`tests/test_validation_framework.py` — **23/23 pass**, covering the Phase 19 checklist:
1. arbitrary entity discovery (fixture repo) + determinism
2. multi-capability generation + wording/entity variation
3. invalid-task rejection (mislabeled capability, missing subject)
4. complete trace capture (transcript + latency)
5. ground truth independent of a confident-but-unsupported model answer
6. deliberately wrong tool → TOOL_SELECTION/CAPABILITY_SELECTION failure classification
7. deliberately wrong capability → classified failure
8. missing-evidence detection (no file:line; speculative "likely the definition")
9. no production mutation during validation (sentinel file unchanged)
10. deterministic structured reports (byte-identical for identical traces)

Plus invariants: capability testability covers all 44 canonical capabilities; default
plans are read-only only (unsafe/external never generated); every task references the
canonical vocabulary with a contract and discoverable tools.

## 14. Full regression results

- `tests/test_validation_framework.py`: 23/23 PASS
- **Full suite: 1568 passed** (was 1545 at STEP 2C; +23), 0 failures
- `ruff check .`: clean

## 15. Live sanity check (limited, NOT the holdout benchmark)

`_step3_validation.py --count 14 --wait 30 --agent simple` (model `gemma4`):
- **13/14 PASS, 0 FAIL, 1 PARTIAL, 0 UNRESOLVED**
- 14 tasks across 14 distinct read-only capabilities, generated on the fly
- per-task latency ≈ 4.7 s (reply-time bound, not fixed timeouts)
- the single PARTIAL: `repository_investigation` — the model produced a VERIFIED
  synthesized answer, but no tool name is visible in Simple-agent output, so
  tool-level dimensions stay UNRESOLVED and the task can't reach 5+ PASS dimensions.
- report written to `validationReport.md` (executive summary, capability matrix, routing/
  evidence/security/ReAct results, generalization indicators, failures, anti-hardcoding,
  coverage table, recommendations).

## 16. Known limitations

- Tool-level dimensions (intent/capability/tool) are UNRESOLVED for Simple-agent replies
  that don't name their tool; ReAct traces expose tool names and will score these dims.
- Evaluation is heuristic and conservative by design: markers are matched in the reply
  window to avoid file-content false positives, which can occasionally miss a real signal.
- The 10-minute mode is a wall-clock budget on generated tasks, not a fixed prompt list —
  it exercises the real model but is not the STEP 4 holdout benchmark.
- `database_query`, `information_request`, and `coding_request` are classified
  not-testable in this environment and are reported rather than generated.
- No production code was changed during this step (git status: only new framework files).

**STOPPING here per STEP 3 — the holdout benchmark is STEP 4 and was not started.**
