# Environmental-State Debugging

When Ultron debugs user code, a plain "here's the traceback" answer is only half
a fix: the same code can fail on one machine and run on another because of the
**environment**. This layer makes every debugging answer carry the exact
environmental state that produced the error — OS version, Python runtime,
installed library versions, and the declared-versus-installed dependency
picture — plus a deterministic diagnosis of the failure and an
expected-vs-actual reconciliation.

## Design goals

1. **No guessing** — the diagnosis is derived from the *actual* command output
   (exit code, stderr, traceback patterns), never from the model's recollection
   of what "usually" happens.
2. **Nothing fabricated** — every fact in the report (OS, Python, versions) is
   read from the live process at report time.
3. **Fast fixes** — the report turns "ModuleNotFoundError" into "the module
   `pandas` is not installed (declared in pyproject.toml as `pandas>=2.0`, but
   only 25 of 40 declared packages are present)" style answers.

## The environment snapshot (`capture_environment`)

Deterministic, read-only, in-process where possible:

- **OS** — `platform.system()`, `release()`, `machine()`, full `platform()`
  string.
- **Python** — version, implementation, executable, and the current working
  directory the failing command ran in.
- **Key tools** — best-effort `--version` probes for a small curated set
  (`git`, `ruff`, `pytest`, `node`, `npm`, `gcc`) with a short timeout; a probe
  that fails is simply omitted, never an error.
- **Installed packages** — via `importlib.metadata`, with a curated
  "interesting packages" list (numpy, pandas, requests, httpx, flask, fastapi,
  django, pytest, …) always shown with their versions, plus the total
  distribution count.
- **Declared requirements** — `pyproject.toml` (`[project].dependencies`,
  read with `tomllib`) and a top-level `requirements.txt` when present. The
  report flags **declared-but-missing** and **version-mismatched** packages —
  the single most common cause of "works on my machine".

## Failure diagnosis (`diagnose_failure`)

Takes the raw command result (the same string `run_command` returns) and
classifies it:

| Cause | Signal in output | Suggested fix |
|---|---|---|
| `missing_dependency` | `ModuleNotFoundError` / `ImportError` | `pip install <module>` (with declared version when known) |
| `syntax_error` | `SyntaxError` | check the flagged line/character |
| `name_error` | `NameError` | variable/function spelled as used |
| `missing_file` | `FileNotFoundError`, "No such file or directory" | verify the path exists |
| `permission` | "Permission denied" | check file/executable permissions |
| `command_not_found` | "command not found" | install the tool or fix `PATH` |
| `network` | `ConnectionError`, "Connection refused", timeout on HTTP | check the endpoint/URL |
| `tests_failed` | pytest `X failed` summary | open the failing assertion |
| `timeout` | "command timed out" | split the command or raise the limit |
| `unknown` | — | read the output below |

The diagnosis extracts the **exit code**, the **stderr tail**, and any module
names appearing in traceback lines (so the report can dependency-check exactly
what the traceback imports). If the command output contains a pytest summary,
`tests_failed` is preferred and the pass/fail counts are parsed.

## Expected-vs-actual reconciliation

The user may state what they expected ("expected 3 tests to pass", "should
print 42"). The report records the expectation verbatim and marks it
**satisfied** / **not satisfied** by a simple containment/equality check against
the actual output. When no expectation is given, the report says so instead of
inventing one.

## Report shape (`format_debug_report`)

```
🔍 Debug report
Command: python main.py
──────────────
🏷 Exit code: 1
📋 Diagnosis: missing_dependency — ModuleNotFoundError: No module named 'pandas'
💡 Suggested fix: pip install pandas   (declared: pandas>=2.0, installed: missing)

Expected: "3 rows printed"          → not satisfied (output contains no "3 rows")

🌍 Environment
  OS       macOS 15.x (arm64)
  Python   3.12.x · cpython
  CWD      /Users/you/project
  Tools    git 2.47, ruff 0.8, pytest 8.3
  Packages pytest 8.3.4, httpx 0.27 … (42 installed)
  ⚠ Declared-but-missing: pandas (pyproject.toml: pandas>=2.0)
  ⚠ Version mismatch: requests declared >=2.32, installed 2.31

Error output (tail):
  ModuleNotFoundError: No module named 'pandas'
```

## Tools

All three are **LOW** risk (pure read-only inspection, auto-allowed):

- `get_debug_context` — print the full environment snapshot.
- `diagnose_failure(text, command?)` — classify a command result.
- `check_dependency(name)` — is a package installed, at what version, and does
  it match the declared requirement?

## Agent flow

`detect_debug_intent` catches debugging phrasing ("debug this", "why is my
script failing", "diagnose this error: …", "help me fix") and optionally
extracts the failing command (quoted, or after "run") and any stated
expectation (after "expected"). `handle_debug` then:

- pasted error text  → diagnose it directly (no execution),
- a command          → gate it through the security boundary, run it, diagnose
  the result,
- neither            → return the environment snapshot and ask what to debug.

The command is always gated exactly like a normal `run_command` (path-escape,
secrets and dangerous-pattern guardrails still apply); a denied command is
reported as blocked and never executed.
