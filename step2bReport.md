# STEP 2B — Generic Capability Contracts: Final Report

**Verdict: PASS** — generic behavioral capability contracts exist, keyed by the canonical
`ToolCapability` vocabulary; `TOOL_DEFINITIONS` remains the single authoritative source of
tool metadata; no duplicate metadata was introduced. Verified by
`tests/test_capability_contracts.py` (17 tests) plus the STEP 2A consolidation suite, and
the full regression suite (**1528 passed, 0 failures, ruff clean**).

---

## 1. What exactly is a capability?

A **capability** is a system-level ability the agent can bring to bear on a user request —
a behavioral goal, not an executable. It is identified by a member of the canonical
vocabulary `ToolCapability` (44 values, defined in `tools/definitions.py`): e.g.
`definition_lookup`, `reference_lookup`, `repository_investigation`, `terminal_execution`,
`coding_request`.

Implemented at: `src/ultron/core/tools/definitions.py` (the vocabulary) and
`src/ultron/core/capabilities/contracts.py` (the behavioral contracts describing each one).

## 2. What exactly is a tool?

A **tool** is an executable mechanism — a registered function with a JSON schema, risk,
read-only status, aliases, etc. There are 58 registered tools, each defined exactly once in
`TOOL_DEFINITIONS`.

Implemented at: `src/ultron/core/tools/definitions.py` (metadata) +
`src/ultron/core/tools/builtin/` (executable functions) + `src/ultron/core/tools/registry.py`
(registry/schema derived from the metadata).

**Capability ≠ tool.** One capability may be served by several tools (`file_write` → 4,
`memory_query` → 7, `information_request` → 5); one tool may serve several capabilities.
`coding_request` is a capability with **zero** direct tools — it is a multi-step ability
expressed through `related_capabilities` + the multi-step flags.

## 3. What exactly is an intent?

An **intent** is what the user wants, extracted from natural language by the NLP layer
(`IntentCategory` + `UserIntent` in `src/ultron/core/nlp/intent.py`, `route_request`).
Intent is **detection**; capability is **ability**; tool is **execution**.

```
user request → intent (nlp/intent.py) → capability (capabilities/contracts.py)
             → canonical registry (tools/definitions.py) → tool → execution
```

The intent layer is a *consumer* of the capability vocabulary (its category values match
`ToolCapability` values so routing maps intent → capability without a second vocabulary).
Contracts describe the capability's *meaning* — they contain no intent-detection logic.

## 4. Where is each concept implemented?

| concept | implementation | role |
|---|---|---|
| capability vocabulary | `tools/definitions.py::ToolCapability` | canonical identifiers (44) |
| capability contracts | `capabilities/contracts.py::CAPABILITY_CONTRACTS` | behavioral description (44) |
| intent | `nlp/intent.py::IntentCategory` / `UserIntent` / `route_request` | user-request detection |
| tool metadata | `tools/definitions.py::TOOL_DEFINITIONS` (58) | single source of truth |
| tool execution | `tools/builtin/` + `tools/registry.py` | executable + derived schema |

## 5. Can a capability use multiple tools?

**Yes.** `CapabilityContract.execution_tools()` queries `tools_with_capability(capability)`
from the canonical registry — it returns *all* registered tools serving that capability.
Verified: `file_write` (4 tools), `memory_query` (7), `information_request` (5),
`graph_reasoning` (4), `api_schema_learning` (4), `parallel_batch` (2), etc.
`preferred_tool()` returns the first (insertion order decides).

## 6. Does any capability contract duplicate tool metadata?

**No.** Three layers of proof:

1. **Model proof** — `CapabilityContract` has no tool-metadata fields (no `risk`,
   `read_only`, `requires_confirmation`, `aliases`, `domain`, `schema`, `tool`).
   `test_contracts_do_not_duplicate_tool_metadata` asserts the dataclass fields and that
   no underscore-bearing tool identifier or alias appears anywhere in contract text.
2. **Static proof** — `test_contract_source_has_no_tool_metadata_tables` scans
   `contracts.py` for any `TOOL_DEFINITIONS`-style table, `CAPABILITY_TOOLS` map, or
   literal tool list; the module's only import from definitions is the capability
   vocabulary + discovery queries.
3. **Semantic proof** — `test_contracts_do_not_duplicate_risk_or_read_only_sets` — the
   only failure-like data in a contract are `CapabilityFailure` classes; risk/read-only
   membership comes exclusively from `TOOL_DEFINITIONS`.

## 7. Can a new tool be added without modifying capability contracts?

**Yes.** `test_mutation_add_tool_propagates_to_contract_discovery` monkeypatches a
synthetic `ToolDefinition` serving an existing capability into `TOOL_DEFINITIONS` — the
contract's `execution_tools()` sees it immediately, with zero contract changes.
`test_mutation_remove_tool_propagates_to_contract_discovery` proves removal propagates
too (no stale mapping).

## 8. Can a new capability be added without modifying TOOL_DEFINITIONS metadata?

**Yes.** `test_new_capability_needs_no_tool_metadata_changes` registers a synthetic
capability purely at the contract layer (an extended enum member + a contract entry);
existing tool metadata is untouched. The new capability correctly reports
`execution_tools() == []` (explicit "no execution mechanism" condition) rather than
inventing a tool.

## 9. Does routing obtain tool information from TOOL_DEFINITIONS?

**Yes.** Contract discovery methods are thin wrappers over the canonical queries:
`execution_tools()` → `tools_with_capability()`, `preferred_tool()` →
`preferred_tool_for()`. `test_capabilities_discover_tools_through_canonical_registry`
asserts the contract result equals the canonical query result for every capability.
(Existing routing — `simple.py` / `react.py` — likewise obtains tool info from
`TOOL_DEFINITIONS` via `preferred_tool_for` / derived sets, established in STEP 2A.)

## 10. Is there exactly ONE source of truth for tool metadata?

**Yes.** `TOOL_DEFINITIONS`. Contracts reference capabilities (a vocabulary, not tool
metadata), and never tool names/aliases/risk/read-only/schemas. The STEP 2A mutation
tests still pass (schema, capability lookup, security tiers, orchestration all observe a
single synthetic definition change), and the STEP 2B tests prove contracts observe
canonical changes with no independent mapping to go stale.

---

## What was built

**`src/ultron/core/capabilities/contracts.py`** (+ package `__init__.py`):

- `EvidenceKind` — VERIFIED / INFERRED / SOURCE_MATCH / SEMANTIC_MATCH / UNKNOWN
  (preserves the code-intelligence resolver's existing evidence semantics)
- `CapabilityFailure` — 8 meaningful failure classes (invalid input, unresolved entity,
  no evidence, tool unavailable, permission denied, timeout, execution failure,
  insufficient context) — no generic "operation failed"
- `CapabilityContract` — capability, purpose, user_intent, required_inputs,
  required_context, evidence_required, success_criteria, failure_classes,
  may_require_investigation, may_require_multiple_calls, related_capabilities
  (behavior only — no tool metadata fields)
- `CAPABILITY_CONTRACTS` — 44 contracts, one per canonical `ToolCapability` value
  (100% enum coverage; `coding_request` is the multi-step meta-capability with no
  direct tool, by design)
- Discovery methods `execution_tools()` / `preferred_tool()` / `has_execution_tools()`
  — live queries against `TOOL_DEFINITIONS`
- `contract_for(cap)` / `capability_names()` helpers

**Multi-step contracts** (may_require_investigation + may_require_multiple_calls +
related_capabilities): `repository_investigation` (9 related capabilities),
`coding_request` (7), `test_execution` (3), `web_search` (→ page_fetch),
`memory_association`, `information_request`, `parallel_batch`.

**Generic:** no contract mentions `TaskState` / `Supervisor` / `CodingExecutor` /
`OrchestrationValidator` (case-insensitive scan) — every contract describes behavior for
arbitrary repository entities (`"repository entity"`, `"symbol"`, `"entity or topic"`).

## Tests

`tests/test_capability_contracts.py` — 17 deterministic tests:

1. every canonical capability has a contract (44/44)
2. identifiers unique + registered identically
3. contracts contain required behavioral information
4. **no tool metadata duplication** (model fields + identifier scan)
5. no risk/read-only duplication
6. discovery through canonical registry (equals `tools_with_capability`)
7. multi-tool capabilities exist
8. generic (no project symbols)
9. success criteria evidence-grounded
10. failure states explicit (no generic placeholder)
11. multi-step capabilities represented
12. related capabilities are capabilities (not tools)
13. single-step defaults (no investigation)
14. **mutation: add tool → contract sees it**
15. **mutation: remove tool → contract sees it**
16. **new capability needs no tool-metadata changes**
17. static scan: contracts.py has no metadata tables

## Regression

```
pytest tests/test_capability_contracts.py tests/test_tool_metadata_consolidation.py -q
39 passed          (17 new + 22 STEP 2A)

pytest -q
1528 passed        (was 1511; +17), 0 failures

ruff check .
All checks passed!
```

## Reproduction

```bash
.venv/bin/python -m pytest tests/test_capability_contracts.py -q   # 17 passed
.venv/bin/ruff check .                                             # clean
.venv/bin/python -m pytest -q                                      # 1528 passed
```

## Known limitations

- Contracts are a definition + validation layer; routing integration (consuming
  `contract_for` / evidence requirements during dispatch) is deliberately deferred to the
  next step per scope control (no new semantic selector / unified intent vocabulary yet).
- The 1:1 capabilities whose value coincides with a tool name (`code_search`,
  `semantic_search`) are canonical vocabulary by design — verified non-duplicating, but
  worth remembering when reading routing code.
- `coding_request` has no direct execution tool; its contract expresses the multi-step
  ability and correctly reports an empty `execution_tools()`.
