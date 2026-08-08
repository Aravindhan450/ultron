# Proactive Dependency Identification (Plan Preflight)

## Motivation

When a user asks for something that takes several tools — *"read config.py,
run pytest, then POST the results to http://api"* — the old multi-step flow
planned the steps and then executed them immediately, one by one. The user
saw nothing up front: the first risky step could prompt mid-way, a blocked
step (unsafe URL, destructive SQL) would surface only when reached, and a
step missing a required field (no filename, no URL) would fail only when
executed — "waiting for failure" exactly as the proposal describes.

This module adds **proactive dependency identification**: before anything
runs, the whole chain is analyzed and presented — every step, the tool it
needs, the **permission it will require** (auto / confirm / blocked), the
**data-flow dependencies** between steps, missing required information, and
resource warnings. The user approves the full plan once, up front, instead
of being prompted mid-execution or discovering problems step-by-step.

## The preflight pipeline

```
user_request
   │  plan_task()  (LLM decomposes into steps: read/write/run/http/query/memory)
   ▼
steps: [ {action, filename|command|url|sql|fact, …}, … ]
   │
   ▼
preflight_plan(steps)                ← all deterministic, no LLM
   • per step: boundary verdict (tier → allow/confirm/deny)
   • per step: missing required fields
   • dependency edges (same-target data flow)
   • resource warnings (heavy/critical commands)
   ▼
format_plan_preview(steps)           ← the upfront listing
   │
   ├─ any step blocked        → plan is NOT offered; user sees why
   ├─ any step confirm/missing → one approval prompt for the whole plan
   └─ all auto + complete     → runs immediately, preview shown as intro
```

The preview is shown **before** anything executes, and the actual security
gate is unchanged — `execute_plan` still re-checks every step at execution
time. The preflight is informational + consent-aggregating, never a way
around the boundary.

## Per-step analysis (`analyze_step`)

For each step the boundary classifies the action exactly as the executor
will (same `check_action` gate — the preview matches reality):

| step | example | tier → decision |
|------|---------|-----------------|
| `read_file` | read config.py | low → auto |
| `run_command` | `ls`, `pytest`, `git status` | low → auto |
| `run_command` | `mkdir x`, `pip install …` | high → confirm |
| `run_command` | `rm -rf /`, `curl \| sh` | critical → **deny** |
| `make_http_request` | GET | low → auto |
| `make_http_request` | POST/PUT/DELETE | high → confirm |
| `run_query` | SELECT | low → auto |
| `run_query` | DROP/TRUNCATE | critical → **deny** |
| `write_file` | normal path | high → confirm |
| `add_memory` | fact | low → auto |

**Missing-information check** — a step that lacks a required field (e.g.
`read_file` without `filename`, `make_http_request` without `url`) is
flagged up front so the user can correct the request instead of the plan
failing at that step.

## Dependency edges (`find_dependencies`)

Deterministic same-target data-flow heuristics over the ordered steps:

- `write_file F` → `read_file F` — *produces the file read in step N*
- `write_file F` → `run_command` / `run_parallel` whose command mentions `F`
  — *creates input used by step N*
- `read_file F` → `run_command` mentioning `F` — *output likely feeds step N*
- `add_memory` → nothing; `run_query` → nothing (downstream consumption is
  not statically knowable)

Edges are only emitted forward (i → j, i < j) and deduplicated.

False-positive guards:

- **Word-bounded matching** — a filename base must not be a prefix of a
  larger token (`data.csv` does *not* match `data2.csv`), enforced with
  a `\w` boundary on both sides.
- **Short-base suppression** — 1–2 character bases (`a.txt` → `a`) would
  match almost any command text, so they emit no dependency edges.

## Plan preview (`format_plan_preview`)

```
📋 Plan — 4 steps · 3 tools
1. read_file 'config.py'                    — ⚡ auto
2. run_command 'pytest'                     — ⚡ auto
3. make_http_request POST http://api/x      — 🛡 needs approval (high)
4. add_memory                               — ⚡ auto
Dependencies:
  • step 1 → step 2: writes 'config.py' used by the command
Permissions: 3 auto · 1 confirm · 0 blocked
```

Heavy commands (via the resource forecast) get a `⚠ heavy · ~2.0 min` tag
on their step line, and blocked steps are called out with the guardrail
reason — the plan is not offered when any step is blocked.

## Wiring

- `plan_task` now also plans `make_http_request` (method/url/body) and
  `run_query` (sql) steps — the proposal's exact read → run → HTTP chain.
- `handle_multistep` runs the preflight and returns a `execute_plan`
  pending action (steps JSON in `target`, preview in the message) whenever
  a step needs approval or is missing information; the CLI renders one
  approval card and runs the whole plan on consent.
- Three tools, all **LOW** (read-only local analysis):
  - `preflight_plan(steps_json)` — the permission + dependency preview
  - `analyze_dependencies(steps_json)` — just the dependency chain
  - `list_plan_actions()` — the planner's supported action types

## Edge cases

- A plan whose only steps are auto + complete runs immediately with the
  preview as an intro — zero added friction for read-only chains.
- A blocked step (secret exfiltration, unsafe URL, path escape, destructive
  SQL) means the plan is never offered: the preview explains why.
- The preflight reuses the same boundary the executor uses, so a "confirm"
  shown up front is the same decision the execution gate enforces.
- Plans that fail at execution still stop on the first error (unchanged
  `execute_plan` guarantee).
