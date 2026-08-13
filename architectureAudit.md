# Ultron Architecture Audit — STEP 1

**Scope:** Complete architectural audit of the `USER REQUEST → reasoning → tool → execution`
pipeline, capability inventory, capability contracts, duplication analysis, and STEP 2
recommendations. **Analysis only — no production code was modified, no tests were added.**

**Method:** every claim below was verified against the actual source in `/Users/aravindhan/ultron`
(`src/ultron`, 32,858 lines of Python) and the test suite (25,679 lines, 1489 tests).
Line references point at the audited state.

---

## 1. Current architecture (top level)

```
┌──────────────────────────────────────────────────────────────────────┐
│ UI layer        src/ultron/ui/ (session.py, theme.py, responsive.py) │
│                 src/ultron/main.py — Typer CLI: chat / run / logs    │
└───────────────────────────┬──────────────────────────────────────────┘
                            │ prompt (ChatSession), confirmation via questionary
┌───────────────────────────▼──────────────────────────────────────────┐
│ AGENT LAYER     core/agents/                                        │
│   get_agent("simple" | "react")  (agents/__init__.py, SUPPORTED_AGENTS)│
│   SimpleAgent.run()  (simple.py:2873)   — 24 detect_* + 23 handle_*  │
│   ReActAgent.run()   (react.py:887)     — Thought→Action→Observation │
└───────────────┬───────────────────────────────────┬──────────────────┘
                │ deterministic detectors            │ LLM-driven loop
┌───────────────▼───────────────┐   ┌────────────────▼──────────────────┐
│ NLP ROUTING   core/nlp/      │   │ ROUTING CORRECTION react.py        │
│   intent.py route_request    │   │   route_llm_tool_call (react.py:110)│
│   normalize.py, workspace.py │   │   _coding_gate (react.py:1648)     │
│   project.py, capabilities.py│   │   _route_tool  (react.py:1502)     │
└───────────────┬───────────────┘   └────────────────┬──────────────────┘
                │ tool + arguments                    │ corrected tool + arguments
┌───────────────▼────────────────────────────────────▼──────────────────┐
│ SECURITY      agents/security.py::check_action                        │
│               → security/boundary.py::SecurityBoundary.check          │
│                 (risk tiers + guardrails.py + audit.py)               │
│               → deny | confirm (PendingAction → CLI) | allow          │
└───────────────┬───────────────────────────────────────────────────────┘
┌───────────────▼───────────────────────────────────────────────────────┐
│ TOOL EXECUTION  core/tools/registry.py — TOOLS dict (58 registered)   │
│   builtin/ (command_runner, file_reader/writer, http, web, database)  │
│   coding/   (edits, executor, workspace, intelligence/*, test_selection)│
│   intelligence/ (planning, structured_output, parallel_tools, debug)  │
│   learning/ (api_schema, associations)  memory/ (graph, sqlite)       │
└───────────────┬───────────────────────────────────────────────────────┘
                │ observation string
┌───────────────▼───────────────────────────────────────────────────────┐
│ REASONING     ReAct: Observation appended to messages → next iteration│
│               Simple: single-shot response                            │
│               TaskState integration (core/types.py), ContextManager   │
│               (core/memory/context_manager.py) for context assembly   │
└───────────────┬───────────────────────────────────────────────────────┘
                │ final answer
┌───────────────▼───────────────────────────────────────────────────────┐
│ SEPARATE ORCHESTRATION LAYER (FIX #7, not on the live agent path)     │
│   core/orchestration/: Supervisor→Delegation→AgentResult→Validation→  │
│   WorkflowEngine. Own AgentRegistry, AgentPermissions, ArtifactStore. │
│   Driven by fake agents in tests; not wired into `ultron chat`.       │
└───────────────────────────────────────────────────────────────────────┘
```

**Key structural fact:** there are **two independent execution stacks** in the tree:

1. **Live agent stack** (what `ultron chat` actually runs): `SimpleAgent` / `ReActAgent`
   → `agents/security.py` → `tools/registry.py` → tools. Deterministic detectors + one
   routing-correction layer (`route_llm_tool_call`).
2. **Orchestration stack** (FIX #7, uncommitted, test-only): `Supervisor` /
   `WorkflowEngine` / `AgentRegistry` / `OrchestrationValidator` with *its own*
   permission vocabulary (`orchestration/permissions.py`), *its own* `check_action`
   implementations, and fake agents. **Not connected to the live agent path.**

---

## 2. Request lifecycle (verified stages)

| # | Stage | File:function | Responsibility | Input → Output |
|---|---|---|---|---|
| 1 | Entry | `main.py:711 async_chat` | Read prompt, dispatch to agent | text → `agent.run(text)` |
| 2 | Pending-clarification | `simple.py:2873` Step −1 | One-turn memory for "which file?" replies | reply → resumed handler |
| 3 | Multi-step | `simple.py:1394 detect_multistep_intent` → `handle_multistep` | "write X then run Y" as a unit | text → plan → sequential handles |
| 4 | Specialized detectors | `simple.py` Steps 0.5–4.97 (24 detectors) | image, file read/write, memory, tests, git, lint, api-schema, resources, web, http, debug… | text → handler (most-specific first) |
| 5 | Deterministic NLP routing | `nlp/intent.py:587 route_request` (Step 4.98) | filesystem + code-intel + project-command + terminal normalization | text → `UserIntent` (category, tool, arguments) |
| 6 | Routed dispatch | `simple.py:404 handle_routed_intent` | resolve workspace root, `_gated_readonly` → boundary | `UserIntent` → ChatMessage/action |
| 7 | Generic command | `simple.py:390 detect_command_intent` → `handle_command` (Step 5) | `Execute: pwd` → `pwd` via `nlp/normalize.py` | text → run_command |
| 8 | LLM fallback | `simple.py:2805 classify_intent` + `handle_llm_fallback` (Step 5.5/6) | AI picks category, code extracts args | text → category → handler or free-form |
| 9 | Security | `agents/security.py:44 check_action` → `SecurityBoundary.check` | tier + guardrails → allow/confirm/deny | (tool, target, content) → verdict |
| 10 | Confirmation | `main.py:983` `while response_msg.pending_action` | questionary prompt, `execute_pending_action` | PendingAction → user → result |
| 11 | Execution | `tools/registry.py:87 TOOLS` → tool function | perform operation | arguments → observation string |
| 12 | Observation → model | ReAct `react.py:1044` append `ChatMessage(role=TOOL)` | feed result back | observation → next iteration (max 10) |
| 13 | Final answer | ReAct `_verify_plan_task`/`_verify_task` or Simple fallback | task verification before accepting | response → `ChatMessage(task_state=…)` |

**ReAct-specific path** (replaces stages 4–6): the LLM emits a JSON tool call
(`react.py:246 extract_tool_call`), the **coding gate** (`react.py:1648`) blocks
repeat-failed/exhausted-budget actions, **`route_llm_tool_call`** (`react.py:110`)
corrects misrouted calls (turn-level on first call + argument-level), then `_route_tool`
(`react.py:1502`) runs the security boundary and executes.

---

## 3. Capability inventory (verified against the live registry)

**58 registered tools** (`tools/registry.py:87`). Grouped by what they actually do:

| Capability | Tool(s) | Read-only | Risk tier (boundary) | Used for |
|---|---|---|---|---|
| File read | `read_file` | yes | LOW | view file |
| File write/create/append/replace/edit/delete/rename | `write_file`, `create_file`, `replace_file`, `replace_in_file`, `append_to_file`, `delete_file`, `rename_file` | no | HIGH (CRITICAL on system paths) | modify source |
| Directory listing | `list_directory`, `discover_workspace_summary` | yes | LOW | browse tree |
| File search | `search_files` | yes | LOW | name/content search |
| Lexical code search | `code_search` | yes | LOW | regex/text search |
| Symbol lookup | `find_symbol` | yes | LOW | symbol index |
| Definition lookup | `find_definition` | yes | LOW | `resolve_definition` cascade |
| Reference lookup | `find_references` | yes | LOW | `resolve_references` cascade |
| Semantic search | `semantic_search` | yes | LOW | embedding search |
| Repository investigation | `code_investigation` | yes | LOW | `resolve_investigation` synthesis |
| Imports/dependents | `get_imports`, `get_dependents` | yes | LOW | dependency edges |
| Symbol/file report | `report_symbol`, `report_file` | yes | LOW | combined report |
| Index status | `code_index_status` | yes | LOW | health |
| Terminal | `run_command`, `run_parallel` | no | HIGH | shell |
| Parallel tool batch | `run_tool_batch`, `synthesize_analysis` | mixed | LOW (gated per member) | batch reads |
| HTTP | `make_http_request` | GET yes / others no | LOW/HIGH by method | API calls |
| Web search | `search_web` | yes | LOW | external info |
| Page fetch | `fetch_page_text`, `retrieve`, `check_connectivity` | yes | LOW | web reading |
| Database | `run_query` | SELECT yes / others no | LOW/HIGH/CRITICAL by verb | SQL |
| Memory (flat + graph) | `add_memory`, `search_memories`, `add_triple`, `query_triples`, `search_triples`, `get_all_triples`, `store_memory_text` | mixed | LOW | recall |
| Knowledge-graph reasoning | `query_chain` (via memory), `memory_connections`, `related_facts`, `discover_connections`, `explain_relation` | yes | LOW | deduction |
| API schema learning | `learn_api_schema`, `api_usage_hint`, `get_api_knowledge`, `forget_api` | mixed | LOW | API knowledge |
| Resource monitor | `check_resources`, `resource_forecast` | yes | LOW | system metrics |
| Structured output | `enforce_schema`, `schema_validate`, `list_schemas` | yes | LOW | JSON/XML enforcement |
| Plan tools | `preflight_plan`, `analyze_dependencies`, `list_plan_actions` | yes | LOW | plan safety |
| Debug context | `get_debug_context`, `diagnose_failure`, `check_dependency` | yes | LOW | env diagnosis |

**Capabilities the audit found NOT directly reachable from the live agent path:**
- Test **intelligence** (discovery/affected-selection) exists in `coding/test_selection.py`
  but is invoked only by `handle_relevant_tests()`/`handle_test()` and the CodingExecutor
  verify flow — **not registered as tools**.
- `CodeIntelligenceBridge` (per-operation query API with query history) is attached to
  `CodeContext` but the live agents call the *registered tools* (`find_definition`, …)
  directly; the bridge duplicates the same resolution logic via `resolve.py`.

---

## 4. Capability contracts (behavioral, symbol-agnostic)

Each contract states the *general* behavior for an arbitrary request — no symbol is named.

### 4.1 Repository definition lookup
**Given** a natural-language request asking where an arbitrary entity is *defined/declared/
implemented as a definition*: classify as definition intent → extract the entity phrase →
resolve (exact → case-insensitive → normalized identifiers → lexical definition lines →
references → semantic, in that order) → return **VERIFIED** evidence (symbol, type,
file:line) or explicitly **UNKNOWN**. **Never** infer a file path from a filename/convention.
*Implemented:* `nlp/intent.py:285 detect_definition_lookup` → `resolve.py:171 resolve_definition`.

### 4.2 Repository reference investigation
**Given** a request asking for *uses/usages/references/callers* of an arbitrary entity:
extract ONLY the entity (never the grammatical wrapper) → resolve the entity → perform
reference lookup (exact → CI → normalized → lexical) → return verified reference locations;
do not substitute raw lexical dumps for verified references unless explicitly labeled as a
fallback. *Implemented:* `intent.py:318` → `resolve.py:228 resolve_references`.

### 4.3 Repository semantic search
**Given** a conceptual question about the codebase ("where is X handled", "how does X
work"): route to repository investigation, not web search; identify primary implementation
(`src/`-first ranking), supporting components, relevant tests; label INFERRED vs VERIFIED.
*Implemented:* `intent.py:471` → `resolve.py:472 resolve_investigation`.

### 4.4 File system operations
**Given** "list files here / read X / create Y / delete Z": map NL location phrases
("current directory", "here", "./") to the **actual workspace root** via `nlp/workspace.py`;
prefer dedicated tools over `ls`/`cat` via terminal; every mutation goes through
`check_action` (deny/confirm/allow). *Implemented:* `handle_routed_intent` + `resolve_location_path`.

### 4.5 Terminal execution
**Given** "Run X / Execute: X / run the command `X`": strip ONLY the outer NL wrapper,
never inner content (`echo "Execute: pwd"` stays intact); refuse prose that is not
command-shaped; security evaluates the **normalized** command. *Implemented:*
`nlp/normalize.py:152 normalize_terminal_command` → `handle_command`.

### 4.6 Project-aware test execution
**Given** "run the tests / run the relevant tests / run pytest X": resolve the project
environment (venv detection, pyproject/package.json/Makefile) and produce the actual
command — never a bare `pytest` on PATH; "relevant tests" = affected-test selection from
changed files. *Implemented:* `nlp/project.py:72 resolve_test_command` +
`coding/test_selection.py:183 select_affected_tests`.

### 4.7 Web / external information
**Given** an explicit external query ("search the web for X", "what is the latest Y"):
`search_web`/`fetch_page_text`, LOW/allow with guardrail denial for unsafe URLs/credentials.
Repository questions must **never** reach this capability. *Implemented:*
`route_llm_tool_call` web gate + Simple `detect_web_search_intent`.

### 4.8 Code modification (agent-facing, FIX #7 contract)
**Given** a state-changing edit by a specialist agent: only whitelisted tools, frozen
permission profile (LLM can never modify), CONFIRM-level actions delegate to the real
boundary, deny-by-default for everything else. *Implemented:*
`orchestration/permissions.py` (not wired to the live loop).

---

## 5. Existing routing mechanisms (in priority order)

**SimpleAgent (deterministic):** 24 `detect_*` functions checked in fixed priority order
(`simple.py:2873`), then `route_request` (Step 4.98), then `detect_command_intent`, then an
**LLM two-stage fallback** (`classify_intent` picks a category, code extracts arguments),
then free-form `handle_llm_fallback`. Observed property: ~40 deterministic detectors across
two vocabularies (Simple's detectors + `nlp/intent.py`'s detectors) both decide "what the
user wants" for overlapping domains (e.g. file read exists in both `detect_file_read_intent`
and `detect_file_delete`'s siblings; test intent in `detect_test_intent` and
`detect_project_command_request`).

**ReActAgent (LLM-driven):** the model selects the tool from the registry JSON schema
(`react.py:184 build_system_prompt` → `registry.get_tools_schema`); three deterministic
guards: (1) `_coding_gate` — repair budget / identical-failure ban, (2) `route_llm_tool_call`
— repo-question/web gate + symbol-argument normalization + turn-level correction on the
first tool call, (3) `_route_tool` — security boundary + command normalization for
`run_command`.

---

## 6. Existing ReAct routing protection (verified)

`route_llm_tool_call(tool, args, user_input)` (react.py:110):
- turn-level: `user_input` classifies to `find_definition`/`find_references` and the model
  used `code_search`/`semantic_search`/`search_web`/`web_search` → redirect to the specific
  tool with the correctly extracted symbol (first tool call only);
- argument-level: the call's `query`/`name` is itself a NL question → redirect to the
  capability `route_request` picks;
- web gate: `search_web` on a repository question → `code_investigation`;
- redirects are restricted to read-only code-intel tools; the corrected call still flows
  through `check_action` — no security downgrade, no state-modifying redirect.

---

## 7. Existing Code Intelligence architecture (verified)

```
resolve.py  (VERIFIED/INFERRED/UNKNOWN, evidence-grounded)
   ▲                    ▲
   │ registered tools   │ CodeIntelligenceBridge.query()
   ▼                    ▼
tools.py (code_search, find_definition, …)   intelligence_bridge.py (formatters → resolve)
   ▼
coding/intelligence/facade.py::CodeIntelligence (refresh, search, symbol, deps, semantic, LSP)
   ├── index.py::RepositoryIndex        (sqlite symbol index, incremental refresh)
   ├── search.py::search_code           (lexical, gitignore-aware)
   ├── symbols.py / parsers.py          (AST Python + regex parsers, Symbol models)
   ├── semantic.py::SemanticSearch      (embedding, degrades to lexical)
   ├── dependencies.py::DependencyGraph (imports/dependents/callers)
   └── lsp.py::LSPFacade                (graceful NoLSPServers)
```

The cascade is **exact → case-insensitive → normalized identifier → lexical → references →
semantic** (`resolve.py:171`), never the reverse; results are classified VERIFIED /
INFERRED / UNKNOWN with no speculative paths.

---

## 8. Existing security ordering (verified)

**Live agent path:** `_coding_gate` (repair budget) **→** `route_llm_tool_call` (correction)
**→** `_route_tool` → `check_action` → `SecurityBoundary.check` → deny (guardrails) /
confirm (PendingAction → CLI questionary) / allow (execute). `SecurityBoundary` classifies
by action name (LOW set / HTTP method / SQL verb / write actions / shell metacharacters)
then runs `GuardrailsEngine` (secrets, PII, unsafe URL, path escape) and audits to JSONL.

**Orchestration path (FIX #7):** `AgentPermissions.check_action` — deny-by-default on
`allowed_tools`, ALLOW/DENY category levels decide directly, CONFIRM levels delegate to the
same `SecurityBoundary`. `OrchestrationValidator` additionally checks lifecycle transitions,
budget, timeout, workspace scope, artifact ownership — read-only, never executes.

---

## 9. Duplicate logic (inventory — NOT refactored)

| # | Duplication | Locations | Notes |
|---|---|---|---|
| D1 | **Tool → capability metadata** | `nlp/capabilities.py::TOOL_CAPABILITIES` (19 entries) vs `registry.py::TOOLS` (58) + `boundary.py` LOW set + `classifier.py::_ACTION_LABELS` | The capability table covers only **19 of 58** tools and its own docstring says "kept in sync with registry" — a manual, incomplete mirror of tool metadata. |
| D2 | **Tool-name lists** | `react.py::_CODE_INTEL_TOOLS/_TURN_CORRECTABLE_TOOLS/_SPECIFIC_SYMBOL_TOOLS`, `simple.py` `_generic_target_content` set, `boundary.py` LOW set, `classifier.py` labels, `capabilities.py` table | 5+ hand-maintained sets of the same tool names; a new tool must be added to several files or routing/security silently diverge. |
| D3 | **Permission/`check_action` implementations** | `agents/security.py::check_action` (live) vs `orchestration/permissions.py::AgentPermissions.check_action` + `orchestration/registry.py` (×2) + `orchestration/models.py::ExecutionContext.check_action` | Four+ wrappers with different verdict types (`BoundaryResult` vs `PermissionCheck` vs `Decision`). |
| D4 | **Intent classifiers** | `simple.py` 24 detectors + `nlp/intent.py` 14 detectors + `intelligence/task_classification.py::classify_task_deterministic` + `simple.py::classify_intent` (LLM) + ReAct LLM | Four deterministic/LLM "what does the user want" vocabularies: tool routing (`route_request`), task typing (`TaskType`), Simple handlers, ReAct prompt. Overlapping domains (file/test/git). |
| D5 | **Symbol resolution** | `resolve.py::resolve_definition/references/symbol/investigation` vs `facade.py::find_definition/find_references` vs `index.py` | Partially consolidated — the bridge now delegates to `resolve.py`, but facade methods (`ci.find_definition`) remain a second entry point with the raw (case-sensitive) contract. |
| D6 | **Result formatting** | `resolve.py` formatters vs `facade.py::report_symbol/report_file` vs `intelligence_bridge.py` formatters vs `tools.py` | Three layers producing "tool-friendly strings" for the same operations; bridge/formatters were unified onto `resolve.py`, `report_*` remain separate. |
| D7 | **Workspace discovery** | `nlp/workspace.py::resolve_workspace` vs `coding/workspace.py::discover_workspace` vs `main.py::_short_cwd`/`CLIState` | Two workspace abstractions with different field sets (`project_root` vs `workspace` vs `current_dir`). |
| D8 | **Test command resolution** | `nlp/project.py::resolve_test_command` vs `coding/executor.py::infer_validation_commands` vs `CodingExecutor` verify flow | Both infer project test/build commands from the same project profile. |
| D9 | **Command normalization** | `nlp/normalize.py::normalize_terminal_command` (Simple + ReAct `_route_tool`) vs `simple.py::detect_command_intent` | The detector and the normalizer both extract commands; ordering interactions are subtle. |
| D10 | **Context assembly** | `memory/context_manager.py::ContextManager` vs `react.py::_build_*_context_block` helpers | ReAct builds its own per-task context blocks (plan, memory, verification) alongside the ContextManager assembly point. |

---

## 10. Architectural weaknesses

| # | Weakness | Evidence |
|---|---|---|
| W1 | **Two stacks, one live** — the entire FIX #7 orchestration layer (Supervisor, Delegation, Validation, Workflow, AgentRegistry) is not wired into `ultron chat`; only Simple/ReAct run live. Capability inventory must be reconciled. | `get_agent` returns only `simple`/`react`; `orchestration/` referenced only by tests. |
| W2 | **Metadata is scattered and partial** — tool risk, read-only-ness, category, description, argument names, labels, redirect sets live in ≥5 files, with `capabilities.py` covering 19/58 tools. New tools require multi-file edits; divergence risk is real (a live `search_web` vs `web_search` asymmetry was found and fixed in a prior cycle). | D1/D2. |
| W3 | **Detector sprawl + fixed priority order** — 24 + 14 detectors, each hand-written regex, ordered in two long chains. Generalization means adding phrasings to regexes, not composing capabilities. The LLM two-stage fallback's category vocabulary (12 categories) is disjoint from `route_request`'s 31 categories. | §5, D4. |
| W4 | **No single capability contract layer** — there is no runtime object describing "what a capability is" (inputs, outputs, security, when-to-use) that both agents and the schema generation consume; `get_tools_schema()` derives only name/description/params from signatures+docstrings. | `registry.py:159`; `capabilities.py` is advisory-only. |
| W5 | **Bridge/tool duplication** — `CodeIntelligenceBridge` re-implements per-operation dispatch and formatting that the registered tools already provide (now sharing `resolve.py`, but still two call surfaces). | D5/D6. |
| W6 | **Observation-level telemetry is partial** — `nlp/observe.py::record_action` records routed Simple actions only; the ReAct loop records via TaskState/`record_tool_result`; no single end-to-end trace structure is shared. | `observe.py` call sites (simple.py only); react.py uses task context. |
| W7 | **Security vocabularies are duplicated** (agent runtime vs orchestration) with different result types; validation inspects verdicts but the two worlds never meet at runtime. | D3. |
| W8 | **LLM fallback is a weak tail** — anything unmatched goes to free-form generation; ambiguity is handled per-detector (`_pending_clarification`) rather than by a shared clarification policy. | `simple.py:2805`/`handle_llm_fallback`. |

---

## 11. Which weaknesses matter for generalization

Relevant to the STEP-2 goal (general-purpose NL → tool → args, arbitrary requests):

- **W2/D1/D2 (metadata) — most relevant.** Generic routing cannot work from hardcoded
  regex sets; it needs one authoritative, machine-readable capability registry
  (name, category, read_only, risk, args, examples, aliases) that schema generation,
  the boundary, the classifier, and the redirect sets all derive from.
- **W3/D4 (detector sprawl)** — the specific examples (TaskState, Supervisor…) were only
  diagnostics; the real cost is that routing knowledge is spread across regexes in two
  vocabularies. Generalization = capability composition driven by metadata + a
  schema-aware selector, not more phrasings.
- **W4 (no contract layer)** — STEP 2 should introduce capability contracts (the §4
  section above) as runtime-queryable objects so tests can assert *contracts hold for
  arbitrary entities*, not example prompts.
- **W6 (telemetry)** — a shared `Intent→Tool→Args→Security→Execution→Result` trace for
  BOTH agents is needed to validate generalization empirically (the user's earlier
  reporting standard: evidence transcripts, not counts).
- **W8 (ambiguity)** — a single clarification policy (when to ask, what to ask) is
  required for arbitrary requests; today it's per-detector.
- **W5/D7/D8** — consolidation matters for maintainability but is secondary to the
  capability-contract goal.
- **W1** — wiring orchestration into the live path is a separate FIX #7 item; the audit
  recommends keeping it separate and not coupling it into the routing work.

---

## 12. Recommended changes for STEP 2 (analysis recommendation — not implemented)

1. **Authoritative ToolDefinition metadata** — one dataclass/pydantic model
   (name, description, category, capabilities, input_schema, output_schema, risk_level,
   read_only, requires_confirmation, aliases, examples) sourced from a single table;
   derive `get_tools_schema()`, the boundary LOW set, classifier labels, and the ReAct
   redirect frozensets from it (kill D1/D2).
2. **Capability-contract layer** — make §4 contracts executable objects with
   `matches(request) -> Intent` + `validate(evidence) -> verdict`, so both agents and
   tests drive routing by contracts, not by accumulated regexes.
3. **Unify the two intent vocabularies** — one `route_request` owning all deterministic
   routing (absorb or alias Simple's 24 detectors' domains into capability contracts);
   keep the LLM fallback as a *category classifier over the same vocabulary*.
4. **Shared action trace** — extend `nlp/observe.py` (or TaskState) so every tool action
   from both agents records the same structured record (intent, tool, extracted args,
   normalized args, security decision, result) for empirical validation.
5. **Schema-aware selector** — `select_tool` becomes contract-driven with preference
   ranking (dedicated tool > intelligence subsystem > structured executor > terminal),
   consuming the metadata from (1) instead of the manual `_CATEGORY_PREFERENCE` map.
6. **Ambiguity policy** — one clarification handler with confidence thresholds
   (HIGH/MEDIUM/LOW), replacing per-detector `_pending_clarification`.
7. **Defer**: orchestrator-to-live wiring, parallel execution, real specialist agents
   (later FIX #7 sections); bridge/workspace/test-command consolidation can follow once
   contracts exist.

**Stop condition for STEP 2:** contracts hold for arbitrary entities (property tests over
random symbol/request shapes), the full 1489-test suite passes, ruff clean, and live
harness evidence (Simple + ReAct) shows equivalent phrasing → equivalent tool actions.

---

## Appendix A — test coverage inventory (relevant areas)

| Area | Test files |
|---|---|
| NLP routing / intent | `test_nlp_pipeline.py` (574), `test_nlp_routing_fixes.py`, `test_repository_question_routing.py`, `test_react_routing.py` |
| Code intelligence | `test_code_intelligence.py` (692), `test_code_intelligence_resolution.py` |
| Coding executor / context | `test_coding_executor.py` (1482), `test_coding_context.py` (584), `test_coding_workspace.py`, `test_coding_edits.py` |
| Test intelligence | `test_test_intelligence.py` (731) |
| Security | `test_security.py`, `test_security_audit.py`, `test_file_policy.py`, `test_agent_security.py`, `test_permissions.py` |
| Orchestration (FIX #7) | `test_agent_contract.py` (736), `test_agent_registry.py`, `test_agent_artifacts.py`, `test_supervisor_delegation.py` (567), `test_orchestration_validation.py` (759), `test_workflow_engine.py` (977) |
| Planning / tasks | `test_planner.py`, `test_planning.py`, `test_plan_execution.py` (1418), `test_plan_validation.py`, `test_task_*.py`, `test_fix2_validation.py` (1208) |
| Memory | `test_memory_*.py` (foundation/integration/stress/graph), `test_retrieval.py` |
| CLI/UI | `test_main_slash.py`, `test_ui_session.py`, `test_ui_responsive.py` |

Current suite: **1489 passed, 0 failed**, ruff clean (verified 13 Aug 2026).

## Appendix B — files audited (primary)

`main.py`; `core/agents/{base,simple,react,security}.py`; `core/nlp/{intent,normalize,
workspace,project,capabilities,observe,interpret}.py`; `core/tools/registry.py` +
`core/tools/builtin/*`; `core/coding/{executor,workspace,test_selection,context,
edits,intelligence_bridge}.py` + `core/coding/intelligence/{facade,resolve,index,search,
semantic,symbols,parsers,dependencies,lsp,tools}.py`; `core/intelligence/{task_classification,
task_planning,planning,plan_validation,structured_output,parallel_tools,debug_context,
prompt_assembly}.py`; `core/memory/{context_manager,working_memory,session_memory,
project_memory,task_store}.py`; `core/orchestration/{models,registry,permissions,lifecycle,
contract,delegation,artifacts,validation,workflow}.py`; `security/{boundary,guardrails,
file_policy,audit,models}.py`; `permissions/classifier.py`; `core/types.py`;
`.github/workflows/ci.yml`.
