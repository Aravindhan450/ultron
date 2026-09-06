# Phase 2: Model Router Corrections Report

## Corrections Made

As requested, I applied a targeted correction pass to resolve two semantic issues in the `ModelRouter`:

1. **Correction 1: Removed Arbitrary Primary Fallback.** 
   The router no longer silently degrades to the first catalog model if the `PRIMARY` model is missing. Instead, it explicitly raises a `RuntimeError` (`"Cannot route: Required PRIMARY fallback model is missing from catalog."`). The `fallback_model` contract is now guaranteed to return a valid `PRIMARY` ModelSpec or fail cleanly.
2. **Correction 2: Made Coding Escalation Explicit.** 
   The escalation policy is no longer handled implicitly as a loophole inside the hard-constraints filter. Instead, it is treated as a top-level **Explicit Routing Mode**. When `coding=True` and `task_state=ESCALATION`, the router skips standard capability filtering/scoring and immediately returns a deterministic routing decision selecting the `PRIMARY` model, paired with an explicit reason (`"Escalation state overrides coding specialist preference; PRIMARY selected for deeper diagnosis."`).

No security code was touched, and the architectural boundary between authorization and model selection remains fully intact. No extraneous abstraction layers or dependencies were added.

## Final Routing Behavior

* **simple general** → `FAST` (Gemma 3 4B IT)
* **complex general** → `PRIMARY` (Qwen3-8B)
* **coding initial** → `CODING` (Qwen2.5-Coder-7B-Instruct)
* **coding repair** → `CODING` (Qwen2.5-Coder-7B-Instruct)
* **coding escalation** → `PRIMARY` (Qwen3-8B)

## Fallback Behavior

* **PRIMARY exists** → `fallback_model` = `PRIMARY`
* **PRIMARY missing** → Clear routing error (`RuntimeError`)

## Tests

New tests were successfully integrated into `tests/test_model_router.py` to assert correct `IN_PROGRESS`, `REPAIR`, and `ESCALATION` coding states, verify the returned `reason` string reflects deep diagnosis, and confirm the strict `PRIMARY` fallback requirement.

```text
ModelRouter: PASS (15 explicit tests)
ModelCatalog: PASS (43 tests)
Intelligence: PASS
AgentRuntime: PASS
ReAct/SimpleAgent: PASS
llama.cpp: PASS
Full suite: PASS (1732 passed, 6 deselected)
```

**Exact Commands:**
* `pytest tests/test_model_router.py -v` (15 passed in 0.15s)
* `pytest -q` (1732 passed, 6 deselected in ~30s)

## Diff Scope

* **Files Modified:** `src/ultron/core/intelligence/model_router.py` and `tests/test_model_router.py`.
* **Verification:** `git diff` confirms that absolutely no changes were made to the `AgentRuntime`, `SecurityBoundary`, existing test fixtures, or any component other than the targeted fixes in the pure-decision router.
