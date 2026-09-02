# PROJECT ULTRON — STEP 3.1 Report
## Validation Framework Refinement Before Holdout

**Verdict: PASS** ✅

---

## 1. What was wrong with the original validation methodology

Three methodological weaknesses were identified and confirmed in the framework:

1. **Task validity was partially gated by the deterministic intent router.** The generator
   accepted tasks partly based on whether the deterministic router agreed with the expected
   capability — circular validation that could hide legitimate natural-language variation.
2. **Diversity was template-driven.** Tasks varied mostly by substituting the entity name into
   a small set of per-capability templates.
3. **Evaluation was heuristic and single-path.** One evaluation blended capability truth,
   execution/evidence truth, and final-answer truth; markers (file:line, "definition",
   tool names) could drive PASS without repository verification.

Additionally, live sanity testing exposed **evaluation artifacts** that had to be fixed for
the framework to be trustworthy (see §16).

## 2. How task validity is now determined

`validate_task()` in `generator.py` verifies, **without invoking the production intent
router**:

- the capability contract exists;
- the subject exists in the repository (or is subject-free and valid for the capability);
- the subject kind is appropriate for the capability;
- the required repository evidence is obtainable (subject file/dir/definition present);
- the generated request contains sufficient information and does not contradict the capability;
- the success criteria are evaluable by the capability's contract.

The deterministic intent router (`select_capability`/`route_request`) is **never called as part
of the validity decision**.

## 3. How deterministic-router disagreement is handled

- The router is still executed, but **only as a diagnostic signal**, recorded on the case:
  `router_capability` + `router_agreement`.
- Disagreement **keeps the task** and is surfaced in the report's Router-diagnostics table
  (expected vs router vs model).
- Disagreement is never treated as task invalidity, and never penalizes the model (see §12).

## 4. How development generation differs from holdout generation

`GenerationStrategy` (DEVELOPMENT_DIRECT vs HOLDOUT_INDIRECT) selects **structurally
different template tables** (`_STRATEGY_TABLES`), not just different seeds:

- **Development**: direct requests, explicit symbol lookup, explicit capability wording
  ("Find the definition of X", "Where is X used?").
- **Holdout**: indirect requests, contextual/conversational phrasing, multi-clause requests,
  implicit capability ("Can you trace where X is referenced in the project?", "Someone
  mentioned X; tell me about it.").

Each strategy has its own seed space and its own template set; the holdout cannot be
reached by re-seeding development.

## 5. How semantic entities are counted

`Subject` carries a **canonical entity id** (`entity_id`, derived from the symbol's module +
declared name). `TaskGenerator` tracks:

- `entity_counts` — unique **semantic entities** (TaskState, taskstate, TASKSTATE, Task State,
  task_state are ONE entity);
- `surface_counts` — surface-form variation per entity (counted, not treated as new entities);
- `capability_counts` — tasks per capability.

Coverage reporting and round-robin budget allocation operate on **capability + entity**,
so one entity cannot dominate a plan.

## 6. How task diversity is measured

The report's Generalization-indicators section reports: unique entities, unique surface
forms, unique capabilities, unique task forms (template ids), and wording variation per
capability. The generator round-robins across capabilities and prefers under-tested
entities (coverage-aware selection), not PASS-rate optimization.

## 7. The three evaluation layers

`evaluate.py` now produces three **independent** layers per trace:

- **Layer A — CAPABILITY TRUTH** (`_layer_capability`): the task's true requirement, from
  case metadata + contract + repository subject existence. Never inferred from the model.
- **Layer B — EXECUTION / EVIDENCE TRUTH** (`_layer_execution`): security decision, tool
  selection vs canonical `tools_with_capability`, argument correctness, and evidence that
  **verifies against repository state** (file content shown, directory entries listed,
  claimed paths exist + mention the subject).
- **Layer C — FINAL ANSWER TRUTH** (`_layer_answer`): five sub-dimensions — factual
  correctness, task relevance, evidence grounding, completeness, uncertainty calibration —
  each PASS/PARTIAL/FAIL/UNRESOLVED, evaluated from the reply window independent of the
  tool trace.

## 8. How repository ground truth is obtained

Deterministic, read-only repository checks against the actual tree:

- file existence (`_repo_file_exists` — files **and** directories);
- file content actually displayed (`_repo_content_shown` — distinctive chunks of the real
  file appear in the transcript, markdown/box-normalized);
- directory listing verified against actual entries (`_verify_directory` — including honest
  empty-directory answers);
- claimed paths verified to exist and to mention the subject (`_verified_claim`);
- git/terminal/test tasks verified by the actual tool outputs in the transcript.

The model's final answer is never used as ground truth anywhere in the pipeline.

## 9. How final answers are independently evaluated

The reply window (last 800 chars — where the rendered box appears) is scored on the five
sub-dimensions in §7. Key rules:

- **task_relevance**: the final response must name the subject; content-returning
  capabilities (file read/list) are exempt because the content IS the answer;
- **evidence_grounding / factual_correctness**: delegate to `_evidence_verdict` — markers
  alone cannot pass; a claimed location must verify;
- **uncertainty_calibration**: speculative phrasing ("probably", "likely") on a definitive
  claim fails; honest "could not find" with absent entity passes;
- **completeness**: expected result kind present in the reply window.

## 10. How overall PASS/PARTIAL/FAIL is calculated

Explicit aggregation (`_aggregate`, documented in the module docstring):

```
any layer FAIL                          -> FAIL
capability PASS and execution PASS
    and answer PASS                     -> PASS
at least one layer PASS, others
    UNRESOLVED                          -> PARTIAL
no layer PASS                           -> UNRESOLVED
```

Examples from the acceptance criteria: capability PASS + execution PASS + answer FAIL →
**FAIL**; capability PASS + execution UNRESOLVED + answer PASS → **PARTIAL**. No single
successful dimension can hide a failure.

## 11. Failure attribution logic

`_attribution` uses the (expected, deterministic-router, model) triple + parsed signals:

| failure kind | trigger |
|---|---|
| SECURITY_FAILURE | denied/deny decision |
| MODEL_LIMITATION | empty response; clarification fallback when the router was clear |
| EXECUTION_FAILURE | traceback / tool error |
| ENVIRONMENT_FAILURE | command not found |
| ROUTING_FAILURE | repo question routed to web **and** the router also chose external; clarification fallback when the router was also confused |
| CAPABILITY_SELECTION_FAILURE | wrong capability while the router was clear (incl. repo→web-search when the router knew it was internal) |
| TOOL_SELECTION_FAILURE | right capability, wrong tool |
| ARGUMENT_FAILURE | wrong arguments/symbol |
| EVIDENCE_FAILURE | speculative claim, unverifiable claim, or claiming answer without evidence |
| SYNTHESIS_FAILURE | correct capability/tool/evidence but wrong final explanation |

The evaluator never blames the model when infrastructure is responsible (e.g. router
reinforced the wrong choice → ROUTING_FAILURE, not MODEL_LIMITATION).

## 12–13. Router diagnostics + model vs routing failures

The report's Router-diagnostics table records expected / deterministic-router / model for
every task. E.g. `expected=reference_lookup, router=unknown, model=reference_lookup` would
be reported as model success + router weakness. Conversely `expected=repository_investigation,
router=unknown, model=web_search` is attributed as a capability-selection failure. Router
disagreement never fails the model.

## 14. Tests added / extended

`tests/test_validation_framework.py` (now **45 tests**; +3 this cycle):

- Part 14 (refinement): valid task with router-disagreement accepted; disagreement recorded
  not invalid; invalid task rejected without router; dev/holdout structurally different;
  surface forms count as one entity; different entities count separately; marker presence
  alone cannot pass; correct-looking unsupported answer fails; correct tool + wrong final
  answer fails overall; correct answer + wrong execution cannot pass; model answer never
  ground truth; repository state used as ground truth.
- Part 15 (adversarial, all 7 cases): A correct-answer-no-evidence → EVIDENCE_FAILURE;
  B correct-tool-wrong-arguments → ARGUMENT_FAILURE; C correct-capability-wrong-tool →
  CAPABILITY_SELECTION_FAILURE; D wrong-capability-correct-looking-answer →
  CAPABILITY_SELECTION_FAILURE; E covered by A + the wrong-final-answer test (SYNTHESIS);
  F router-unknown-model-success → PASS (model not blamed for router weakness);
  G invalid task rejected.
- New attribution cases: repo question → web search (router clear) →
  CAPABILITY_SELECTION_FAILURE; repo question → web search (router confused) →
  ROUTING_FAILURE; clarification fallback (router confused) → ROUTING_FAILURE.

## 15. Live sanity results (Part 16)

`_step3_validation.py --strategy holdout --count 12 --seed 17` through the real CLI
(model gemma4, Simple agent):

```
tasks=12  PASS=0  PARTIAL=3  FAIL=9  UNRESOLVED=0
failure kinds: routing_failure=2, capability_selection_failure=2, evidence_failure=3,
               argument_failure=1, model_limitation=1
```

This run used ≥5 capabilities, ≥5 semantic entities, ≥3 task forms, and produced 11/12
router disagreements (indirect holdout phrasing) — the required diversity and
router-disagreement coverage.

**Genuine findings surfaced (not fixed — production frozen):**
1. Holdout indirect phrasing defeats the deterministic router + Simple agent: directory
   listing / file search requests fell back to "It sounds like you want to run a command —
   which command?" (routing_failure).
2. Repository questions ("Someone mentioned FAILURECATEGORY; tell me about it", "What
   implementation handles IndexSummary?") were routed to **web search** by the model
   (capability_selection_failure) — the STEP 2C-era repository-gate weakness persists for
   indirect phrasing.
3. Definition query "DEPENDENCYEDGE. Where is it defined?" produced "Definitions of 'it'
   found via source search" — the pronoun-extraction weakness persists (evidence_failure).
4. "Can you trace where Embedder is referenced?" produced an empty response
   (model_limitation).
5. Repository-inspection question hit the memory store instead of the repo
   (argument_failure).

These are exactly the kind of generalization findings STEP 4 is meant to quantify — the
framework now attributes them correctly instead of lumping them as generic evidence failures.

## 16. Evaluation-artifact fixes (framework-only, production untouched)

Live testing exposed and we fixed four framework bugs:

1. `_repo_file_exists` used `.is_file()`, so **directory subjects** (directory_list) always
   failed Layer A even when the directory exists → now `.exists()`.
2. Empty-directory verification passed unconditionally — a "which file?" clarification
   passed evidence → now requires the reply to actually say it's empty.
3. Content-returning capabilities verified by looking for the **filename** in the reply
   window; the read_file tool returns raw content with no filename header → now verifies
   actual file **content** appears (box/markdown/whitespace normalized).
4. Evidence and relevance checks scanned the whole transcript, so the **echoed user prompt**
   satisfied subject checks → now scoped to the reply window.

## 17. Full regression results

- `tests/test_validation_framework.py`: **45/45 pass**
- Full suite: **1590 passed** (was 1568; +22), 0 failures
- `ruff check .`: **clean**

## 18. Production-freeze verification

```
git status --short:
  M validationReport.md                 (framework output, regenerated)
 ?? _step3_validation.py                (live harness)
 ?? src/ultron/validation/              (framework: model/subjects/generator/runner/evaluate/audit/report)
 ?? step3Report.md                      (STEP 3 report, prior step)
 ?? step3-1Report.md                    (this report)
 ?? tests/test_validation_framework.py  (framework tests)
```

**Zero production files changed.** `IntentCategory`, intent detection, the Intent→Capability
selector, `CapabilityContract`, `TOOL_DEFINITIONS`, ReAct routing, Simple routing, and
`SecurityBoundary` are all untouched. The only production-adjacent change is the addition of
*observable* failure markers in the validation runner (web-search routing, clarification
prompt) — validation-only, no production behavior change.

## 19. Remaining limitations

- Layer B tool-observation is UNRESOLVED under the Simple agent when the reply doesn't
  name the tool (Simple output doesn't echo tool names) — execution truth still verifies
  evidence, but tool-level attribution is richer under ReAct (STEP 4 should run both).
- Web-search detection relies on the "Search the web" confirmation string — if the model
  performs web search without that string the marker may miss (fallback: tool-name scan).
- The live sanity run is stochastic per model; counts vary run to run, the attribution
  classes are stable.
- Content verification (`_repo_content_shown`) requires ≥2 distinctive lines of the file
  to appear; extremely short or heavily truncated files may not verify (UNRESOLVED, not
  falsely FAIL).
- A small number of genuine production findings are recorded above and intentionally NOT
  fixed, per the production freeze.

---

**Final verdict: PASS** — task validity is decoupled from the deterministic router, router
disagreement is diagnostic data, development/holdout generation is structurally distinct,
entities vs surface forms are separated, three-layer evaluation is independent, markers
cannot produce PASS alone, repository ground truth is used, adversarial tests pass,
production routing is unchanged, and the full regression + ruff are green.
