# Parallel Tool Processing & Result Synthesis

**Goal:** when a single request needs the results of several *different* tools
(e.g. read a file *and* search the web *and* query the database), Ultron runs
those tools **at the same time** and weaves their outputs into **one coherent
analysis** — instead of executing them one after another and dumping three
disconnected replies.

The existing `run_parallel` tool only parallelizes *shell commands* (same tool,
many commands). This module generalizes the idea to the whole tool surface:
many tools, one request, one synthesized answer.

---

## The gap

A request like *"read config.json, then search the web for the pandas release
notes, then check if the docs site is up"* would otherwise execute:

```
read_file            → done (waits)
web_search           → done (waits)
check_connectivity   → done (waits)
```

Total wall time = the **sum** of the three. With parallel dispatch the wall
time is roughly **the slowest** of the three, and the user gets a single report
that ties the answers together instead of three disconnected replies.

```
⚡ Parallel batch — 3 calls · 3 executed · 0.42s

[1] ✅ read_file file_path=config.json
    [build-system] requires = ["hatchling"] ...

[2] ✅ web_search query=pandas release notes
    pandas 2.0.1 released with new features ...

[3] ✅ check_connectivity url=https://pandas.pydata.org
    https://pandas.pydata.org is online (200 OK, 41ms)

🧠 Combined analysis
• Shared across sources: pandas, release
```

---

## How it works

### The engine — `core/intelligence/parallel_tools.py`

**`run_tool_batch(calls_json)`** — the registered tool. Takes a JSON array of
`{"tool": ..., "arguments": {...}}` calls, gates **each** call through the
security boundary (`check_action`), executes every auto-allowed call
**concurrently** in a thread pool, and returns a synthesized report.

```json
[
  {"tool": "read_file", "arguments": {"file_path": "config.json"}},
  {"tool": "web_search", "arguments": {"query": "pandas release notes"}},
  {"tool": "check_connectivity", "arguments": {"url": "https://pandas.pydata.org"}}
]
```

**`synthesize_results(results)` / `synthesize_analysis(results_json)`** —
deterministic synthesis. Combines per-tool outputs into one analysis:
cross-tool keyword connections (terms appearing in more than one result) plus
per-source summaries. No LLM is involved — the synthesis is instant and
grounded in the actual tool outputs.

**`plan_tool_batch(user_input, engine)`** — the LLM planner (like `plan_task`)
that turns an ambiguous request (e.g. *"gather everything relevant about X"*)
into the batch of `{tool, arguments}` calls. It is restricted to a read-mostly
allowlist (`PLANNER_TOOLS`) and never suggests state-changing work.

### Safety: what may run concurrently

Concurrency changes *timing*, never *permission*. Each call in the batch still
passes through the exact same `check_action` gate as a single call would:

- **`deny`** (guardrail hard block: secret exfiltration, unsafe URL, path
  escape) → the call is *never executed*; it is reported as blocked.
- **`confirm`** (state-modifying, or HIGH/CRITICAL under the active mode) →
  the call is *never executed* silently; it is reported as *"needs approval"*
  so the caller can run it through the normal `PendingAction` flow.
- **`allow`** → executed concurrently with the other allowed calls.
- **Unknown tool names** → reported as errors; the rest of the batch still
  runs.

The batch is only as safe as its most dangerous member: every call is
classified individually before anything executes. A batch can never execute
`write_file`, `run_command`, destructive SQL, or POST/PUT/DELETE requests
without approval — in fact those verdicts keep them out of the batch entirely.

**Read/write waves.** Only read-only tools (`BATCH_READONLY_TOOLS`) run
concurrently. State-writing LOW tools (e.g. `add_memory`) still execute, but
**sequentially after the read wave** — concurrent SQLite writers would collide
with *"database is locked"*. `make_http_request` is deliberately excluded from
the concurrent set: a state-changing method (POST/PUT/DELETE/PATCH) is
auto-allowed under permissive mode and must never run in parallel.

**Dedup.** Identical calls (same tool + same arguments) execute once, in order
of first occurrence, so a batch is never run twice by accident. The dedup key
is robust to non-JSON-serializable argument values.

**Name canonicalization.** The security layer and the agent detectors speak
*action* names (`web_search`), while the tool registry keys the function under
its own name (`search_web`). Batch calls may arrive in either spelling — the
ReAct agent emits registry names (from `get_tools_schema`), detectors/planner
emit action names. Every name is canonicalized to the action name for gating
and wave classification, and only resolved to the registry key at execution
time, so both spellings gate identically.

### The agent flows

**SimpleAgent (deterministic detectors).** `detect_tool_batch_intent` fires on
requests that clearly want *several independent sources*:

- Multi-file reads: *"read config.json and notes.txt"* → two `read_file` calls
- Multi-URL checks: *"check example.com and example.org"* (bare domains and
  explicit URLs both work, TLD-gated so `config.json` is never a domain)
- Repeated searches: *"search for pandas and search for numpy"* → two
  `web_search` calls
- Explicit parallelism markers: *"at the same time"*, *"in parallel"*,
  *"simultaneously"* — with concrete targets → deterministic batch; without →
  the LLM planner

Single-target requests ("read config.json") and shell-command parallel batches
("run X and Y in parallel") are exempt — they keep their existing paths.

**ReActAgent (model-driven).** `_route_tool` has an explicit `run_tool_batch`
branch so the model can request a concurrent batch in one step. It routes
directly (like `run_parallel`) because the per-member gates inside the tool are
authoritative — this also avoids an outer content scan that could false-block
a valid batch whose member carries an `http://` URL. Both `calls_json` (JSON
string) and `calls` (structured list) argument spellings are accepted. The
system prompt instructs the model: *prefer a single `run_tool_batch` call for
independent read-only lookups*. The synthesized report flows back as one
Observation.

---

## Usage from the CLI

Just ask in natural language — the detectors route to the batch automatically:

```
> read README.md and pyproject.toml
⚡ Parallel batch — 2 calls · 2 executed · 0.01s
...

> check example.com and example.org
⚡ Parallel batch — 2 calls · 2 executed · 0.12s
...

> search for pandas and search for numpy
⚡ Parallel batch — 2 calls · 2 executed · 0.35s
...

> gather info about the project at the same time
⚡ Parallel batch — 2 calls · 2 executed · 0.02s   (planned via LLM)
...
```

With the ReAct agent (`ultron chat --agent react`), the model can also call the
tool directly whenever it spots independent lookups:

```text
Thought: I need two independent facts — the file contents and whether the site is up.
```

```json
{"tool": "run_tool_batch", "arguments": {"calls_json": "[{\"tool\": \"read_file\", ...}, ...]"}}
```

---

## Report format

Every batch returns one report:

```
⚡ Parallel batch — N calls · M executed · 0.42s

[i] ✅ read_file file_path=config.json
    <result, truncated to 240 chars>
[i] ⚠️ run_command command=rm -rf / — needs approval (not run in batch): Needs approval: ...
[i] ⛔ read_file file_path=/etc/shadow — blocked (never ran): Blocked by security: ...

🧠 Combined analysis
• Shared across sources: keyword1, keyword2

ℹ️ Calls needing approval were NOT run — say 'yes' to execute them through the normal confirmation flow.
```

Badges: ✅ executed · ⚠️ executed with an error · ⛔ blocked (never ran) · 🛡
needs approval (never ran). The combined analysis only adds the cross-source
view — it never repeats the per-call results.

---

## Tools

| Tool | Description | Risk |
|------|-------------|------|
| `run_tool_batch` | Run several tool calls concurrently, gated per-call, returns a synthesized report | LOW (members gated individually) |
| `synthesize_analysis` | Combine a list of `{tool, result}` outputs into one analysis block | LOW (pure text) |

Both are auto-allowed (LOW) by the boundary; the real safety decision happens
per-member inside the batch.

---

## Limits

- **`MAX_BATCH_CALLS = 8`** — a batch is capped at 8 calls; the rest are
  ignored.
- Read-only tools run in parallel; state-writing LOW tools run sequentially
  after (wave split).
- Results are truncated to 240 chars per call in the report.
- Only tools in `BATCH_READONLY_TOOLS` / `PLANNER_TOOLS` participate in the
  concurrent / planner paths — everything else is either gated out or executed
  sequentially.

---

## Testing

- **Unit tests** (`tests/test_parallel_tools.py`, 40): concurrency, per-call
  deny/confirm gating, unknown-tool isolation, malformed input, dedup
  (including non-serializable arguments), read/write wave separation,
  bare-domain TLD gating, no double URL match, name-canonicalization for both
  spellings, and the regression test that would catch a `web_search` /
  `search_web` mismatch.
- **ReAct tests** (`tests/test_react_agent.py`): full loop with observation
  flowback, direct-route spy proving no outer URL scan, missing-args
  observation, `calls`-list spelling.
- **Stress harness** (`_stress_audit.py`, run in CI as the `stress` job): 29
  checks including 24 concurrent batches from 8 threads, 20 concurrent memory
  writes with no SQLite locks, and a simulated-bug check proving the harness
  fails when the action-name resolution is broken.
- Agent-flow tests use a fake engine — no live model needed.
