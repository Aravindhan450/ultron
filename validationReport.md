# Ultron Capability Validation Report

> Generated 2026-08-14 07:26 UTC · framework: `ultron.validation` (STEP 3) · deterministic aggregation

## Executive summary

| metric | value |
|---|---|
| model | gemma4 |
| agent | simple |
| duration | 266.7s (budget n/a) |
| tasks executed | 12 |
| capabilities tested | 12 |
| PASS | 0 |
| PARTIAL | 3 |
| FAIL | 9 |
| UNRESOLVED | 0 |

## Capability matrix

| capability | PASS | PARTIAL | FAIL | UNRESOLVED |
|---|---|---|---|---|
| code_search | 0 | 0 | 1 | 0 |
| definition_lookup | 0 | 0 | 1 | 0 |
| directory_list | 0 | 0 | 1 | 0 |
| file_inspection | 0 | 0 | 1 | 0 |
| file_read | 0 | 1 | 0 | 0 |
| file_search | 0 | 0 | 1 | 0 |
| reference_lookup | 0 | 1 | 0 | 0 |
| repository_inspection | 0 | 0 | 1 | 0 |
| repository_investigation | 0 | 0 | 1 | 0 |
| semantic_search | 0 | 0 | 1 | 0 |
| symbol_inspection | 0 | 0 | 1 | 0 |
| symbol_search | 0 | 1 | 0 | 0 |

## Three-layer verdicts (Part 7-11)

Overall rule: any layer FAIL -> FAIL; all three PASS -> PASS; otherwise PARTIAL/UNRESOLVED.

| layer | PASS | PARTIAL | FAIL | UNRESOLVED |
|---|---|---|---|---|
| capability | 12 | 0 | 0 | 0 |
| execution | 0 | 3 | 9 | 0 |
| answer | 0 | 3 | 9 | 0 |

### Final-answer sub-dimensions (Part 10)
| dimension | PASS | PARTIAL | FAIL | UNRESOLVED |
|---|---|---|---|---|
| completeness | 1 | 10 | 1 | 0 |
| evidence_grounding | 4 | 0 | 8 | 0 |
| factual_correctness | 3 | 0 | 9 | 0 |
| task_relevance | 6 | 1 | 3 | 2 |
| uncertainty_calibration | 11 | 0 | 0 | 1 |

## Router diagnostics (Part 12)

Router agreement with expected capability: **1/12** (disagreements are evaluation data, never task invalidity).

| case | expected | deterministic router | model (observed tool) | agreement |
|---|---|---|---|---|
| holdout.indirect.file_read-2 | file_read | unknown | unobserved | DISAGREE |
| holdout.indirect.directory_list-5 | directory_list | unknown | unobserved | DISAGREE |
| holdout.indirect.file_search-8 | file_search | unknown | unobserved | DISAGREE |
| holdout.indirect.repository_inspection-12 | repository_inspection | unknown | unobserved | DISAGREE |
| holdout.indirect.code_search-17 | code_search | unknown | unobserved | DISAGREE |
| holdout.indirect.symbol_search-22 | symbol_search | unknown | unobserved | DISAGREE |
| holdout.indirect.definition_lookup-25 | definition_lookup | definition_lookup | unobserved | agree |
| holdout.indirect.reference_lookup-29 | reference_lookup | unknown | unobserved | DISAGREE |
| holdout.indirect.symbol_inspection-34 | symbol_inspection | unknown | unobserved | DISAGREE |
| holdout.indirect.file_inspection-37 | file_inspection | unknown | unobserved | DISAGREE |
| holdout.indirect.semantic_search-40 | semantic_search | unknown | unobserved | DISAGREE |
| holdout.indirect.repository_investigation-43 | repository_investigation | unknown | unobserved | DISAGREE |

## Routing results

- **intent**: PASS=0 PARTIAL=0 FAIL=0 UNRESOLVED=12
- **capability**: PASS=0 PARTIAL=0 FAIL=0 UNRESOLVED=12
- **tool_selection**: PASS=0 PARTIAL=0 FAIL=0 UNRESOLVED=12
- **argument**: PASS=10 PARTIAL=0 FAIL=1 UNRESOLVED=1

## Evidence results

- verified evidence: **3**
- insufficient evidence: **9**
- unsupported/speculative claims: **0**
- UNRESOLVED: **0**

## Security results

- security dimension PASS: **11**
- security dimension FAIL: **0**
- security dimension UNRESOLVED: **1**

## ReAct results (tool loop signals)

- **tool_selection**: PASS=0 PARTIAL=0 FAIL=0 UNRESOLVED=12
- **investigation**: PASS=0 PARTIAL=0 FAIL=0 UNRESOLVED=0
- **final_answer**: PASS=0 PARTIAL=3 FAIL=9 UNRESOLVED=0

## Generalization indicators

- unique semantic entities: **12** (surface forms: **12**)
- unique task forms (template ids): **12**
- unique capabilities: **12**
- wording diversity: **12** distinct task strings over 12 tasks
- repeated-task rate: **0** exact duplicates
- max tasks per single entity: **1** (coverage-aware: no entity may dominate)


## Failures

| id | capability | request | tool | root cause | failing dims |
|---|---|---|---|---|---|
| holdout.indirect.directory_list-5 | directory_list | `Can you show me what's sitting in MobileApp?` | - | routing_failure | evidence=FAIL, final_answer=FAIL |
| holdout.indirect.file_search-8 | file_search | `Do we have any files named README around here?` | - | routing_failure | evidence=FAIL, final_answer=FAIL |
| holdout.indirect.repository_inspection-12 | repository_inspection | `What can you tell me about the structure of baseagent?` | - | argument_failure | argument=FAIL, evidence=FAIL, final_answer=FAIL |
| holdout.indirect.code_search-17 | code_search | `I'd like to see every place that mentions capabilityselection — can you search the source?` | - | model_limitation | evidence=FAIL, final_answer=FAIL |
| holdout.indirect.definition_lookup-25 | definition_lookup | `I'm trying to figure out where this component comes from: DEPENDENCYEDGE. Where is it defined?` | - | evidence_failure | evidence=FAIL, final_answer=FAIL |
| holdout.indirect.symbol_inspection-34 | symbol_inspection | `Someone mentioned FAILURECATEGORY; tell me about it.` | - | capability_selection_failure | evidence=FAIL, final_answer=FAIL |
| holdout.indirect.file_inspection-37 | file_inspection | `Could you give me some details on the file README.md?` | - | evidence_failure | evidence=FAIL, final_answer=FAIL |
| holdout.indirect.semantic_search-40 | semantic_search | `What implementation handles IndexSummary? Look it up.` | - | capability_selection_failure | evidence=FAIL, final_answer=FAIL |
| holdout.indirect.repository_investigation-43 | repository_investigation | `I'm trying to wrap my head around how LSPOPERATION works — can you break it down?` | - | evidence_failure | evidence=FAIL, final_answer=FAIL |

## Anti-hardcoding

_clean: 121 production files scanned, 0 findings_

## Capability coverage classification

All 44 canonical capabilities classified for automated testing:

| capability | testability |
|---|---|
| api_schema_learning | read_only |
| application_start | unsafe |
| application_stop | unsafe |
| build | requires_execution |
| code_search | read_only |
| coding_request | not_testable |
| database_query | not_testable |
| debug_environment | read_only |
| definition_lookup | read_only |
| dependency_analysis | read_only |
| directory_create | unsafe |
| directory_list | read_only |
| file_create | unsafe |
| file_delete | unsafe |
| file_inspection | read_only |
| file_read | read_only |
| file_rename | unsafe |
| file_search | read_only |
| file_write | unsafe |
| format | unsafe |
| git_operation | read_only |
| graph_reasoning | read_only |
| http_request | external |
| information_request | not_testable |
| install | unsafe |
| lint | requires_execution |
| memory_association | unsafe |
| memory_query | read_only |
| memory_update | unsafe |
| page_fetch | external |
| parallel_batch | read_only |
| plan_management | unsafe |
| reference_lookup | read_only |
| repository_inspection | read_only |
| repository_investigation | read_only |
| resource_monitoring | read_only |
| semantic_search | read_only |
| structured_output | read_only |
| symbol_inspection | read_only |
| symbol_search | read_only |
| terminal_execution | read_only |
| test_execution | requires_execution |
| typecheck | requires_execution |
| web_search | external |

## Recommendations

- **Most common failure kinds:** evidence_failure (3), routing_failure (2), capability_selection_failure (2) — classify each as implementation bug vs model limitation before patching.
- **Anti-hardcoding:** audit clean — no executable historical literals found.
