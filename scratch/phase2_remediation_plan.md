# Phase 2 — Model-Call Boundary Remediation: Internal Implementation Plan

## 1. Current ownership
`RepositoryContextManager.budget_messages()` (core/context/manager.py) implements the
algorithm, but it is invoked from exactly ONE production site: the ReAct primary loop
(`agents/react.py:1187`). Every other model call bypasses it:
verification (`_verify_task`/`_verify_plan_task`), planning/classification
(`task_planning.py`, `task_classification.py`), synthesis (`synthesis.py`),
parallel planning (`parallel_tools.py`), and SimpleAgent (the CLI *default* agent).

## 2. Problem
No single authoritative owner of `input + reserved_output <= model_limit`.
Budgeting is a per-call-site responsibility, so it is trivially forgotten.

## 3. Root cause
`BaseEngine.generate/stream` IS the one chokepoint every model call passes through,
but the invariant was placed one layer too high (inside one agent), so any caller
that is not that agent is unbudgeted by construction.

## 4. New ownership (chosen boundary: C+D hybrid — a budgeting BaseEngine decorator)
- Extract the budgeting algorithm into `core/context/budget.py` (single implementation),
  operating on a neutral `_BudgetMessage`; `budget_messages` (ChatMessage) and
  `budget_openai_messages` (OpenAI dicts) are thin adapters over it.
- Add `core/context/invocation.py::BudgetedEngine(BaseEngine)` — a decorator that
  wraps any engine and applies the invariant to every `generate()`/`stream()` call.
  It preserves the BaseEngine contract (it IS a BaseEngine), so callers are unchanged.
- Install it at the single production agent factory `core/agents/__init__.py::get_agent`
  (idempotently). ReAct, SimpleAgent, `/agent`, `/reload`, repair-agent and planning
  (which uses `agent.engine`) all receive the same wrapped engine.
- `AgentRuntime` binds a model-call observation scope (event_bus/task_id/run_id) around
  agent execution and propagates the routed model's `recommended_context_length` into
  the boundary via `set_context_limit()`.
- The ReAct loop's inline budgeting + CONTEXT_BUILT emission is removed (the boundary
  now owns it). The runtime's snapshot-form CONTEXT_BUILT is retained with a distinct
  `kind` so the two event purposes are unambiguous.

## 5. Fixes folded in
- Remove the invariant-violating `max(100, limit - reserved)` floor: clamp
  `max_input = max(0, limit - reserved)` and, when the reservation would starve the
  prompt, *reduce the reservation* (never exceed the model limit).
- Protect the LATEST user request as an anchor (not just the first user message), and
  shave oldest/low-priority material first so the current request survives compaction.

## 6. Files that must change
- `core/context/budget.py` (new) — single algorithm + config + limits.
- `core/context/invocation.py` (new) — BudgetedEngine, scope, ensure_budgeted_engine.
- `core/context/manager.py` — delegate budget_messages; re-export config.
- `core/context/__init__.py` — export boundary symbols.
- `core/agents/__init__.py` — wrap engine in get_agent.
- `core/runtime/runtime.py` — bind scope, propagate limit, `kind` on snapshot event.
- `core/agents/react.py` — drop inline budgeting; pass `caller` kwarg.
- `core/intelligence/task_planning.py`, `task_classification.py`, `synthesis.py`,
  `parallel_tools.py`, `agents/simple.py` — pass `caller` kwarg (observability only;
  budgeting is automatic).
- tests + harness.

## 7. Files that must NOT change
BaseEngine contract, LlamaCppEngine, ModelRouter/Catalog/Lifecycle, SecurityBoundary,
TaskState, EventBus/EventStore, tool contracts, CLI behaviour, confirmation flow.

## 8. Tests required
A–L per prompt §18, plus an architectural guard test asserting no direct
`engine.generate(`/`engine.stream(` call remains in agent/intelligence production code
outside the boundary.

## 9. CLI validation required
Scenarios A–G via `_phase2_cli_validation.py` against the real `ultron chat` path.
