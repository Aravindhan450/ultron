# Parallel Command Execution

> Design doc for Ultron's parallel-command tool: dispatch many shell commands
> at once and wait for the collective results, instead of queueing them.

## Motivation

Ultron's `run_command` tool executes one command at a time and blocks until it
finishes. That is fine for single actions, but inefficient for batch work:
spellcheck ten files, format-check ten files, lint ten files, then collect
every result. Done sequentially that is 3×N round-trips of wall-clock time,
most of it waiting on I/O-bound processes.

Running the commands **concurrently** moves Ultron from sequential computing
to parallel computation:

- Ten independent checks complete in roughly the time of the slowest one,
  not the sum of all ten.
- Each command keeps its own timeout, so one hung process cannot stall the
  batch.
- A failure in one command does not cancel the rest — the user gets every
  result in a single report.

## Design

### The tool — `run_parallel`

Registered in the tool registry as `run_parallel`:

```
run_parallel(commands: list[str], timeout: int = 15) -> str
```

Semantics:

- **Concurrency** — commands are dispatched on a thread pool (one worker per
  command), each running `subprocess.run(..., shell=True)` exactly like the
  existing `run_command` tool. Because each worker blocks on its own
  subprocess, the batch wall-clock time is ~the slowest command, not the sum.
- **Per-command isolation** — each command gets its own result block
  (`Exit code`, `Output`, `Error Output`) and its own timeout. A timeout or
  crash in one command never affects the others.
- **Collective report** — the return value is a single summary plus one block
  per command:
  ```
  3/3 commands succeeded in 0.42s

  [1] OK — ruff check src/ultron
  Exit code: 0
  Output: ...

  [2] OK — python -m pytest -q
  ...
  ```
- **Empty input** — `run_parallel([])` returns an error string; the tool
  never runs an empty batch silently.

### Security model

This is the important part. A batch is *at least as dangerous as its most
dangerous command*, so the batch is gated command-by-command:

- **Classification** — `classify_action("run_parallel", target)` splits the
  newline-joined command batch back into individual commands and classifies
  each one with the exact same rules as `run_command` (read-only allow-list,
  shell-metacharacter escalation, dangerous-pattern escalation). The batch
  takes the **worst** tier: one `rm -rf /` in a batch of ten `ls` calls
  makes the whole batch CRITICAL.
- **Guardrails** — the guardrail engine scans *each* command in the batch for
  dangerous shell patterns and credential-like strings, exactly as it scans a
  single `run_command`. A credential embedded in any command denies the batch
  outright.
- **Agent-level gate** — `handle_parallel` calls `check_action("run_command",
  cmd)` for every command in the batch:
  - any `deny`  → the whole batch is blocked (nothing runs),
  - any `confirm` → the whole batch is offered as a single interactive
    `PendingAction` confirmation, listing every command, so the user approves
    the batch once,
  - every `allow` → the batch executes immediately.

  This reuses the existing confirmation flow: `PendingAction.action_type` is
  `run_parallel` and `target` is the newline-joined command list.

### Detector — `detect_parallel_intent`

Deterministic regex detector (matching the project's "code detects, AI only
falls back" philosophy), triggered by an explicit parallelism marker:

- `run <c1> and <c2> [and <c3>] in parallel`
- `execute a, b, c simultaneously`
- `run X at the same time` / `concurrently`
- front-marked form: `in parallel, run X and Y`

The command portion is split on commas, `" and "`, and newlines. A bare
`run ls` (no marker) stays sequential — the marker is required so nothing is
silently parallelized that the user did not ask to parallelize.

The LLM fallback path can also emit `run_parallel` as a tool call
(`{"tool": "run_parallel", "arguments": {"commands": [...]}}`); the same
per-command gate applies there and in the ReAct agent's tool executor.

## CLI flow

1. User: *"run ruff check src and python -m pytest in parallel"*
2. `detect_parallel_intent` → `["ruff check src", "python -m pytest"]`
3. `handle_parallel` gates each command. Both are read-only (LOW) → the batch
   auto-executes via the registry tool; the combined report is shown inline.
4. If any command were state-changing (e.g. `git push`) → a single
   confirmation card lists all commands; approving it runs the whole batch.

## Edge cases

- **Mixed-risk batches** — the whole batch takes the worst verdict. Low
  commands inside a confirmed batch only run if the user approves the batch.
- **Empty segments** — `run a,  , b in parallel` drops blank segments.
- **Quoted commands** — surrounding quotes on a segment are stripped only
  when they wrap the whole segment; quotes that are part of the command
  (`echo "fish and chips"`) are preserved, and separators inside quotes are
  never treated as command boundaries.
- **Embedded newlines** — a "command" carrying newlines (e.g. from an LLM
  tool call) is normalized in `handle_parallel`: each line is gated and
  executed as its own command, so classification always matches execution
  exactly.
- **A single command "in parallel"** — `run pytest in parallel` yields a
  one-command batch; harmless and still routed through the same gate.
- **Timing flakiness** — the concurrency guarantee is best-effort on
  I/O-bound subprocesses; the tool reports measured elapsed time so users can
  verify the speedup.

## Future work

- Streaming: show output per command as it completes instead of one combined
  report at the end.
- `parallel: true` flag on `run_command` so the LLM can annotate single
  commands for batching.
- A job-scoped timeout for the whole batch (currently each command has its
  own).
- Apply the same concurrent-dispatch pattern to HTTP fetches and web
  searches, which are equally I/O-bound.
