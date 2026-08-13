# STEP 2A — Tool/Capability Metadata Consolidation: Final Report

**Verdict: PASS** — ONE authoritative metadata source exists; all consumers derive from it.
No independent authoritative metadata tables remain. Verified by automated tests
(`tests/test_tool_metadata_consolidation.py`, 22 tests) and the full regression suite
(**1511 passed, 0 failures, ruff clean**).

---

## 1. The ONE canonical metadata source

**`src/ultron/core/tools/definitions.py`** — module-level `TOOL_DEFINITIONS: dict[str, ToolDefinition]`.

`ToolDefinition` owns, per tool, exactly once:

| field | meaning |
|---|---|
| `name` | canonical tool identity |
| `capabilities` | capability tags (`ToolCapability`), 0..N per tool |
| `domain` | `ToolDomain` (filesystem / code_intelligence / execution / …) |
| `description` | human-readable purpose (auto-derived signature docstring, `resolved_description`) |
| `read_only` | read-only status |
| `risk` | declared risk (`ToolRisk`) |
| `requires_confirmation` | confirmation requirement |
| `aliases` | alternative action spellings (e.g. `web_search` for `search_web`) |
| `target_arg` / `action_label` | routing classification + confirmation label |
| `internal` | internal/external classification |

The executable function definition stays with the tool implementation in
`core/tools/builtin/` — `definitions.py` references it once and derives the input
schema from the actual callable (no duplicate function definitions, no duplicate schemas).

Derived queries exposed to consumers (single implementation each):

- `canonical_action_name(name)` — bidirectional alias canonicalization
- `get_tool_definition(name)`
- `tools_with_capability(cap)`, `preferred_tool_for(cap)`, `tools_with_any_capability(caps)`
- `readonly_tool_names()`, `code_intel_tool_names()`, `generic_code_tool_names()`, `web_tool_names()`
- `action_label_for(action)`, `tool_aliases()`

---

## 2. Metadata sources removed

| removed structure | file | disposition |
|---|---|---|
| `TOOL_CAPABILITIES` / capability table | `core/nlp/capabilities.py` | **file deleted** (`nlp/__init__.py` updated) |
| `_CODE_INTEL_TOOLS`-style hardcoded redirect sets | `core/agents/react.py` | replaced by derived frozensets |
| hardcoded read-only/code-intel/redirect lists | `core/agents/simple.py` | replaced by `readonly_tool_names()` / `code_intel_tool_names()` / `preferred_tool_for()` |
| boundary tier lists (LOW/MEDIUM/HIGH/CRITICAL) | `security/boundary.py` | replaced by canonical `risk` lookup + internal non-tool action policy |
| `_ACTION_ALIASES` duplicate alias map | `permissions/classifier.py` | replaced by canonical `tool_aliases()` / `action_label_for()` |
| alias spellings map | `core/intelligence/parallel_tools.py` | replaced by canonical `canonical_action_name()` (lazy) |
| orchestration `classify_tool` static sets + coder/analyst baseline whitelists | `core/orchestration/permissions.py`, `core/orchestration/registry.py` | now call-time derived from canonical |
| `_INTELLIGENCE_TOOLS` literal in executor | `core/coding/executor.py` | replaced by canonical-derived `_code_intel_tools()` (runtime-lazy, cycle-safe) |

**Deleted files:** `src/ultron/core/nlp/capabilities.py` (the entire duplicate capability table).

---

## 3. Metadata sources now derived

- **Registry / schema generation** — `core/tools/registry.py` builds `TOOLS` and the JSON
  Schema from `TOOL_DEFINITIONS` (tool names, signatures, schemas all flow from canonical).
- **ReAct routing** — `core/agents/react.py` redirect/allow sets are computed from
  `readonly_tool_names()` / `code_intel_tool_names()` / `web_tool_names()`.
- **Simple/NLP routing** — `core/agents/simple.py` `_generic_target_content` and
  `handle_routed_intent` resolve content defaults and preferred tools via canonical queries.
- **Security classification** — `security/boundary.py` derives declared risk/read-only/
  confirmation from `get_tool_definition()`; aliases canonicalized at the top of
  `classify_action`.
- **Classifier labels/aliases** — `permissions/classifier.py` consumes canonical
  `action_label_for()` / `tool_aliases()`.
- **Orchestration permissions** — `classify_tool` derives write/search categories at call
  time from canonical domain/read-only metadata (+ one documented policy carve-out:
  `check_connectivity` stays NETWORK).
- **Orchestration agent baselines** — coder/analyst tool whitelists derived from canonical
  membership queries.
- **Executor observability** — `_code_intel_tools()` / `_inspection_tools()` derive from
  canonical `code_intel_tool_names()`.

---

## 4. Structures that remain, and why they are NOT authoritative

Per PHASE 5/10, remaining tool-name lists were classified; metadata-bearing lists were
eliminated. What remains is **policy / execution ordering / unrelated constants** — never
independent tool metadata:

| remaining structure | file | classification |
|---|---|---|
| `_STATE_CHANGING_TOOLS` | `core/coding/executor.py` | policy — identical-action gating scope (deliberately a subset of non-read-only tools; documented) |
| `BATCH_READONLY_TOOLS` | `core/intelligence/parallel_tools.py` | policy — concurrency wave scheduling (excludes code-intel tools for deterministic exploration; `run_query` treated as read for batching; documented) |
| `PLAN_ACTION_SPECS` | `core/intelligence/planning.py` | policy — planner action vocabulary + required fields per action |
| `COMPLEX_TASK_TYPES`, `_STOPWORDS`, topic word-banks | `plan_validation.py`, `parallel_tools.py`, `learning/associations.py` | unrelated constants |
| guardrail pattern sets | `security/guardrails.py` | policy — must remain independent by design (guardrails override metadata) |
| `run_command`/`run_parallel` policy branches | `security/boundary.py` | policy — execution rules (verified to run *before* canonical lookup) |

`test_remaining_policy_sets_reference_known_canonical_tools` guarantees every tool name in
these policy sets resolves to a canonical tool or alias (or a documented internal action
like `query_chain`) — policy can never silently drift from the canonical source.

---

## 5. Adding a new tool: define metadata ONCE

A developer adds a tool by:

1. implementing the function in `core/tools/builtin/` (as before), **and**
2. adding one `_define(...)` / `_reg(...)` entry in `definitions.py` (name, capability,
   domain, risk, read_only, aliases, …).

`registry.py` picks up the tool + schema; routing, ReAct redirects, security tiers,
classifier labels/aliases, orchestration classification, and executor observability all
follow automatically. `test_every_registered_tool_has_canonical_metadata` fails if a
registered tool lacks a canonical definition.

---

## 6. Risk-level change propagation

`test_mutation_propagates_to_schema_capability_security_and_orchestration` (monkeypatch)
mutates a synthetic canonical definition and asserts the consumers observe the new
`category`, `read_only`, `risk`, and `aliases` — schema generation, capability lookup,
security boundary tier, and orchestration classification all reflect the change.
(Security policy branches and guardrails intentionally re-evaluate and may still override
declared risk — that is by design, see §7.)

---

## 7. Tool removed from registry → stale metadata

Cannot go stale undetected:

- `test_canonical_definitions_are_complete` — every canonical entry resolves to a real
  registered callable.
- `test_remaining_policy_sets_reference_known_canonical_tools` — policy sets cannot
  reference names that no longer canonicalize to a registered tool.
- `test_boundary_tiers_match_canonical_declared_risk` + runtime consistency assertions —
  security cannot disagree with canonical metadata.

A removed tool immediately fails the "canonical definition references an unknown tool /
policy references unknown tool" checks.

---

## 8. Duplicate-authoritative-metadata detection

`tests/test_tool_metadata_consolidation.py` (22 tests) is the automated detector:

- `test_every_registered_tool_has_canonical_metadata` — no registered tool lacks metadata
- `test_canonical_definitions_are_complete` — no canonical entry references an unknown tool
- `test_canonical_definitions_are_unique` — no duplicate definitions
- `test_schema_generation_consumes_canonical` — schemas come from canonical
- `test_old_duplicate_structures_are_removed` — the old tables (`TOOL_CAPABILITIES`,
  `_CODE_INTEL_TOOLS`, …) no longer exist as authoritative data
- `test_react_redirect_sets_are_derived_from_canonical` — ReAct sets equal the derived ones
- `test_boundary_tiers_match_canonical_declared_risk` / `test_boundary_resolves_aliases`
- `test_capability_preference_maps_routing_categories` / `test_action_labels_come_from_canonical`
- `test_classifier_consumes_canonical_labels_and_aliases`
- `test_orchestration_classification_derives_from_canonical`
- `test_mutation_propagates_…` — single-source proof
- `test_guardrails_override_…` (×2) — security special case
- `test_no_exact_match_routing_on_historical_symbols` / `test_generic_routing_works_for_arbitrary_symbols` — anti-hardcoding
- `test_remaining_policy_sets_reference_known_canonical_tools` — policy drift guard

---

## 9. Coverage: 58/58 tools

Verified at runtime:

```
registered tools:          58
canonical definitions:     58
registered missing canonical: NONE
canonical not registered:  []
read-only tools:           42
code-intelligence tools:   11
tools with no capability tag: []
```

Every registered tool has exactly one canonical `ToolDefinition` with capability,
domain, risk, read-only, and aliases. 76 capability tags span 58 tools (multi-capability
tools allowed without duplicated metadata). **No unclassified registered tools.**

---

## 10. Dependency direction

```
                  TOOL_DEFINITIONS (src/ultron/core/tools/definitions.py)
                              │  (the ONLY authoritative metadata)
      ┌──────────────┬────────┼────────────────┬───────────────┐
      ▼              ▼        ▼                ▼               ▼
 tools/registry   nlp/intent  agents/react   agents/simple   security/boundary
 (TOOLS +         (routing)   (redirect/      (_generic_       (risk tiers, aliases,
  schemas)                    allow sets)     content,        confirmation)
                                              preferred tools)
      │              │        │                │               │
      ▼              ▼        ▼                ▼               ▼
   ReAct loop     capabilities  classifier   orchestration    guardrails
                  lookups     (labels/       (permissions,    (policy — override)
                              aliases)       baselines)
                                              │
                                              ▼
                                           executor
                                         (observability)
```

There are **no arrows back** from consumers into independent metadata: every consumer's
metadata comes from `definitions.py` queries; remaining local lists are classified policy
or constants, validated against canonical by the drift-guard test.

---

## Phase-by-phase evidence

- **PHASE 1 inventory** — every metadata location enumerated (registry, `nlp/capabilities.py`,
  boundary tier lists, classifier aliases, react/simple redirect sets, parallel_tools
  aliases, orchestration permissions + baselines, executor sets).
- **PHASE 2** — canonical model in `definitions.py` (`ToolCapability`, `ToolDomain`,
  `ToolRisk`, `ToolDefinition`, derived queries).
- **PHASE 3** — `registry.py` rewritten to derive `TOOLS` + schemas from canonical.
- **PHASE 4** — consumers migrated one by one (schema gen → capability lookup → ReAct →
  Simple/NLP → classifier → orchestration → parallel_tools).
- **PHASE 5** — `nlp/capabilities.py` deleted; all old tables eliminated or derived.
- **PHASE 6** — boundary derives tiers from canonical; `run_command`/`run_parallel` policy
  branches verified to run before canonical lookup; guardrail-override regression tests added.
- **PHASE 7** — 22-test validation suite.
- **PHASE 8** — mutation-proof test propagates one synthetic definition to all consumers.
- **PHASE 9** — 58/58 coverage verified.
- **PHASE 10** — anti-hardcoding sweep; executor sets derived (cycle-safe lazy imports);
  remaining lists classified policy/constants; drift guard added.
- **PHASE 11** — full suite **1511 passed** (was 1510; +1 new test), 0 failures, ruff clean.

## Known remaining limitations

- `executor.py` and `boundary.py` import the canonical module lazily to break a pre-existing
  import cycle (`definitions → intelligence → types → coding.context → executor`). The lazy
  import is cached after first use; hot paths are unaffected.
- `BATCH_READONLY_TOOLS` and `PLAN_ACTION_SPECS` remain literal policy lists by design;
  they are not metadata but are validated against canonical so they cannot drift.
- Guardrails keep their own pattern sets (security policy) — intentionally independent of
  tool metadata, and required to be so by the security architecture.

## Reproduction

```bash
.venv/bin/python -m pytest tests/test_tool_metadata_consolidation.py -q   # 22 passed
.venv/bin/ruff check .                                                     # clean
.venv/bin/python -m pytest -q                                              # 1511 passed
```
