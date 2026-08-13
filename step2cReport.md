# STEP 2C — Unified Intent → Capability Selection: Final Report

**Verdict: PASS** — one generic Intent → Capability selection path connects the existing
NLP intent system to the existing capability contracts; no new vocabulary, no tool
metadata, no security bypass. Verified by `tests/test_capability_selection.py` (17 tests),
the STEP 2A/2B suites, the full regression suite (**1545 passed, 0 failures, ruff clean**),
and a real-CLI sanity check (both agents, PASS).

---

## The one selection path

```
USER REQUEST
    │
    ▼
Intent Detection          nlp/intent.py (IntentCategory, route_request)
    │
    ▼
Intent → Capability       capabilities/selector.py::select_capability()
    │                        (RESOLVED / AMBIGUOUS / UNKNOWN)
    ▼
CapabilityContract        capabilities/contracts.py (behavior: inputs, evidence,
    │                        success, failure, related capabilities)
    │
    ▼
TOOL_DEFINITIONS          tools/definitions.py (execution_tools / preferred_tool
    │                        — live canonical queries, never local metadata)
    │
    ▼
Tool choice → Security → Execution   (agents route through the boundary)
```

`select_for_request(text)` is the NLP entry point: `route_request` → `select_capability`.
Both agents consume the same result.

---

## 1. Where is Intent defined?

`src/ultron/core/nlp/intent.py` — `IntentCategory` enum (31 members) + `UserIntent` +
deterministic detectors + `route_request`. **Unchanged.**

## 2. Where is Capability defined?

`src/ultron/core/tools/definitions.py` — `ToolCapability` enum (44 members, canonical
vocabulary). Behavioral contracts: `src/ultron/core/capabilities/contracts.py`. **Unchanged.**

## 3. Where is Intent → Capability mapping defined?

`src/ultron/core/capabilities/selector.py`:
- `_INTENT_TO_CAPABILITY: dict[IntentCategory, ToolCapability]` — 29 explicit entries
  (IntentCategory values deliberately align with ToolCapability values; the table is
  written out explicitly so the layer reads as a single authoritative connection)
- `_AMBIGUOUS_INTENTS` — context-dependent intents (currently `INFORMATION_REQUEST` →
  `{information_request, web_search}` — the repository-vs-external case)
- `select_capability(intent)` / `select_for_request(text)` — the ONE path

## 4. Where is Tool metadata defined?

`src/ultron/core/tools/definitions.py::TOOL_DEFINITIONS` (58 tools). **Unchanged and still
the single source of truth.** The selector never names a tool (verified by static scan).

## 5. Does Intent → Capability mapping contain tool names?

**No.** The mapping is IntentCategory → ToolCapability only. Static scan
(`test_selector_contains_no_tool_name_metadata`) proves no registered tool name or alias
appears in `selector.py`; no `risk=`/`read_only=` data either.

## 6. Does CapabilityContract contain tool metadata?

**No.** Established in STEP 2B: the model has no risk/read-only/confirmation/alias/schema
fields, and no tool identifier appears in contract text (17 contract tests still pass).

## 7. Does capability → tool discovery still use TOOL_DEFINITIONS?

**Yes.** `CapabilitySelection.execution_tools` → `tools_with_capability(primary)`;
`preferred_tool` → `preferred_tool_for(primary)` — thin wrappers over canonical queries.
`test_execution_tools_come_from_canonical_registry` asserts equality for every mapped
intent. There is no `capability → tool` table anywhere.

## 8. Do SimpleAgent and ReAct use the same capability-selection mechanism?

**Yes.**
- **SimpleAgent** (`handle_routed_intent`): replaced the ad-hoc `_Capability(cat.value)` →
  `preferred_tool_for` coercion with `select_capability(cat)` → `selection.preferred_tool`.
- **ReAct** (`route_llm_tool_call`): the turn-level correction now determines the
  "specific symbol capability" via `select_capability(turn_intent.intent_type)`, then
  obtains the corrected tool from `selection.preferred_tool` (canonical registry).
  The redirect set is now capability-level (`_SPECIFIC_SYMBOL_CAPABILITIES =
  {DEFINITION_LOOKUP, REFERENCE_LOOKUP}`) instead of a tool-name set.
- `test_simple_and_react_share_the_same_selection` asserts both paths resolve the same
  preferred tool for the same turn.

## 9. Can an unknown intent execute a tool?

**No.** UNKNOWN selection (`None` / `IntentCategory.UNKNOWN` / unmapped / malformed) has
`primary=None`, `preferred_tool=None`, `execution_tools=()`. `test_unknown_intent_selects_no_tools`
and `test_unknown_request_selects_no_tools` prove it. Unknown intents never silently route
to web search (`test_repository_capabilities_never_route_to_external_search`).

## 10. Can a new tool supporting an existing capability be added without changing the mapping?

**Yes.** The mapping is Intent → Capability; tool discovery is a live canonical query. A
new tool tagged with an existing capability is immediately visible via
`execution_tools`/`preferred_tool` — proven by the STEP 2B mutation tests (still passing).

## 11. Can a new capability be added without modifying unrelated intent mappings?

**Yes.** A new capability = a `ToolCapability` member + a contract (STEP 2B); the
intent mapping only needs a new entry if a new *intent* should select it. Existing
intent mappings are untouched (proven by `test_new_capability_needs_no_tool_metadata_changes`
in the STEP 2B suite).

## 12. Is SecurityBoundary still authoritative?

**Yes.** The selector is selection-only: it returns capability/tool names, never executes.
Execution in both agents still flows through `check_action` → SecurityBoundary
(allow/confirm/deny). The ReAct correction re-routes every corrected call through the
boundary (existing behavior, unchanged). `test_selection_does_not_bypass_security` asserts
the boundary's canonical risk/read-only still governs the tool.

---

## State behavior (Phase 5/12)

| condition | state |
|---|---|
| known intent + known capability | RESOLVED (primary + related from contract) |
| context-dependent intent (repo vs external) | AMBIGUOUS (candidates in `ambiguity`, no tool) |
| known intent + no mapping | UNKNOWN (explicit reason) |
| malformed / empty / no deterministic intent | UNKNOWN (explicit reason) |

Related capabilities come **from the contract** (`contract.related_capabilities`), never a
second table — `test_multiple_related_capabilities_are_representable` asserts
`selection.related == selection.contract.related_capabilities` (e.g. CODING_REQUEST → 7
related capabilities).

## Anti-hardcoding audit (Phase 15)

`test_selector_has_no_historical_symbol_hardcoding` — no TaskState/Supervisor/CodingExecutor
anywhere in the selector. The mapping remains Intent → Capability (29 entries), not
Intent → dozens of tools.

## Files changed

| file | change |
|---|---|
| `src/ultron/core/capabilities/selector.py` | **new** — selection layer (state enum, selection dataclass, mapping tables, `select_capability`/`select_for_request`) |
| `src/ultron/core/capabilities/__init__.py` | exports selector |
| `src/ultron/core/agents/react.py` | turn-level correction consumes `select_capability`; `_SPECIFIC_SYMBOL_CAPABILITIES` (capability-level); dropped unused import |
| `src/ultron/core/agents/simple.py` | `handle_routed_intent` code-intel branch uses `select_capability` |
| `tests/test_capability_selection.py` | **new** — 17 tests |
| `tests/test_tool_metadata_consolidation.py` | updated the ReAct redirect test to the capability-level set |

## Regression (all actually executed)

```
pytest tests/test_capability_selection.py -q                17 passed
pytest tests/test_capability_selection.py tests/test_capability_contracts.py \
      tests/test_tool_metadata_consolidation.py -q          56 passed
pytest -q                                                   1545 passed (was 1528; +17), 0 failures
ruff check .                                                All checks passed!
```

## Real-CLI sanity check (Phase 17)

`_step2c_sanity_check.py` through the actual `ultron chat` (real Ollama):

1. **Simple agent** — "Find where the OrchestrationValidator is defined" →
   returned the **verified definition** (`src/ultron/core/orchestration/validation.py:264-1364`,
   class) — the selector → preferred-tool path works end-to-end. OK
2. **ReAct agent** — "Run pwd" → "The current working directory is /app." — LLM loop +
   routing integration intact. OK

**SANITY: PASS.** (Deliberately small; the full capability/generalization holdout
benchmark remains STEP 3/STEP 4.)

## Remaining limitations

- `INFORMATION_REQUEST` is declared AMBIGUOUS (repo vs external) — no current detector
  emits it, so today the ambiguity is latent; the mechanism is exercised by tests and
  ready for later context-aware routing.
- The Simple agent's 24 `detect_*`/`handle_*` pairs still bypass the selector (they are
  per-domain deterministic handlers, not NLP-routed); the NLP-routed path now uses the
  selector. Unifying every detector through the selector is a later-step concern.
- `AMBI/UNKNOWN` handling is selection-level only; a clarification UI/flow is explicitly
  out of scope (later step).

## Reproduction

```bash
.venv/bin/python -m pytest tests/test_capability_selection.py -q   # 17 passed
.venv/bin/python -m pytest -q                                      # 1545 passed
.venv/bin/ruff check .                                             # clean
.venv/bin/python -u _step2c_sanity_check.py                        # SANITY: PASS
```
