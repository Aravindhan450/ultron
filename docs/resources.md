# Resource Constraint Awareness

## Motivation

Ultron runs arbitrary shell commands and multi-step plans on the user's
machine. Some commands are cheap (`ls`), some are quietly expensive (`pip
install`, `find /`, a long test suite, a parallel batch of dozens of
commands). Today there is no way to know *before* running which ones will
saturate the CPU, blow through memory, or drag on for minutes — the user
finds out after the fact.

This feature adds a built-in resource monitor that:

- **measures** what a command actually consumed (wall time, CPU time, peak
  memory) and reports it right in the command output;
- **learns** from history — past runs of the same command family feed the
  next forecast;
- **warns before the bottleneck** — a command whose forecast is heavy or
  critical is surfaced *before* execution, converting silent auto-runs into
  an informed confirmation, and adding warnings to confirmation prompts.

## The three signals

| Signal   | Source (stdlib-first)                                        |
|----------|--------------------------------------------------------------|
| Time     | `time.monotonic()` around the run                            |
| CPU      | `resource.getrusage(RUSAGE_CHILDREN)` deltas (`ru_utime` + `ru_stime`) |
| Peak RSS | live sampling of the child pid (psutil if installed, else `/proc/<pid>/status` on Linux, `ps -o rss=` on macOS) |

All three degrade gracefully: on platforms where a signal is unavailable
(no `resource` module, no `ps`, unknown kernel) it is simply omitted from
the report rather than crashing the command.

## Forecast engine

`forecast_command(command)` combines two evidence sources:

1. **Static patterns** — a table of command families with known resource
   profiles (`pip/npm install` → heavy, `find /` → critical, `pytest` →
   moderate, `grep -r /` → heavy, compilers/builds → moderate–heavy,
   `git clone`/dumps → moderate, …).
2. **History** — the last runs of the same command family from the SQLite
   store (`~/.ultron/.ultron_resources.db`); measured duration and peak
   memory override the static guess when present. Families are *granular*
   for package managers (`pip install` ≠ `pip list`, `npm run` ≠ `npm
   install`), so one heavy install never taints unrelated subcommands, and
   the store keeps only the newest 50 runs per family so anomalous runs
   age out.

Severity ladder: `light` → `moderate` → `heavy` → `critical`, with
thresholds on expected duration (60s/5min) and peak memory (1.5GB/4GB).
The forecast carries human-readable reasons ("dependency installs download
and compile packages", "last run took 3m 12s").

## Pre-execution policy

| Forecast        | Confirm-required action         | Auto-allowed action            |
|-----------------|---------------------------------|--------------------------------|
| `light`         | normal prompt                   | run silently                   |
| `moderate`      | prompt + note                   | run; note appended to reply    |
| `heavy`/`critical` | prompt + warning            | **escalate to confirmation** with the warning |

In **permissive** security mode the escalation never fires — that mode
promises no prompts, so a heavy command still runs, with the resource
warning attached to the reply instead.

The escalation is the proactive half of the feature: a LOW-risk command
that would otherwise auto-run (e.g. `find /`) is offered for confirmation
with its resource forecast shown. Parallel batches escalate when any
command is heavy/critical *or* when the batch is large (more than 8
commands — parallel spikes multiply CPU and memory).

## Tools

- `check_resources` — a system snapshot (CPU cores + load, memory used /
  total, optional psutil detail). Read-only, LOW risk, auto-allowed.
- `resource_forecast(command)` — returns the forecast for one command
  (pattern + history). Read-only, LOW risk.

Both are registered and gated like every other tool, so the ReAct agent
gets them for free.

## Reporting

Every executed command (single or parallel) ends with a compact measured
line:

```
[resources] finished in 3.2s · CPU 1.8s · peak ~145 MB
[resources] batch: 6 commands in 8.4s · peak ~430 MB
```

The same measurement is written to the history store, so the *next* run of
that command family forecasts from reality, not guesses.

## Agent wiring

- `detect_resource_intent` handles "check system resources", "how much
  memory is free", "will this command be heavy?", "resource forecast for
  <cmd>" (with command extraction from quotes / "for" / "of").
- `handle_command` / `handle_parallel` consult the forecast before deciding
  auto-run vs confirm and embed warnings in confirmation prompts.
- Multi-step plans (`execute_plan`) annotate heavy `run_command` steps with
  a resource warning before they execute.

## Edge cases

- A DB failure degrades to "no history" — forecasting and measuring never
  break a command.
- `shell=True` means the sampled pid is the shell; `ps`/`/proc` follow it,
  and a timeout kills the shell exactly as before.
- The report line never changes the command's own stdout/stderr formatting;
  existing "Exit code: N" contract is preserved.
- Historical estimates only apply within the same command family — `ls`
  never inherits `pip install`'s profile.
