# Ultron — Stress Test & Audit Report

**Date:** 2026-08-08
**Scope:** All recently implemented intelligence, learning, security, and tool features.
**Harness:** `_stress_audit.py` (reproducible; 29/29 checks) + full pytest suite + ruff.

---

## 1. Summary

| Metric | Result |
|---|---|
| Full test suite | **600 passed** (3 consecutive runs, no flakes) |
| Lint (ruff, with & without config) | **Clean** |
| Stress harness (`_stress_audit.py`) | **29/29 checks passed** |
| Registered tools | 38 |
| Feature code under audit | ~4,070 LOC across 7 modules |
| Feature test code | ~3,040 LOC across 7 test files |
| **Bugs found by this audit** | **1 real bug (fixed)** + 2 latent inconsistencies (documented) |

**Verdict:** The implementations are production-grade for local, single-user,
permission-gated use. The stress run surfaced exactly one real defect — an
action-name vs. tool-name mismatch that silently broke batched web searches —
which is now fixed and covered by a regression test.

---

## 2. Methodology

Each subsystem was exercised under conditions the unit tests don't fully cover:

- **Concurrency:** many tool calls in parallel (threads), repeated batches, and
  mixed read/write waves against a throwaway SQLite DB.
- **Hostile input:** malformed JSON, path escapes (`../..`, `~/.ssh/id_rsa`,
  `/etc/shadow`), secret-shaped content, unknown tools.
- **Consistency:** the same boundary verdict repeated 50× must be identical
  every time.
- **Flakiness:** the full pytest suite ran 3× back-to-back.
- **Timing:** wall-clock for concurrent batches (should be ~the slowest call,
  not the sum).
- **Isolation:** stress runs used `tempfile` DBs; the real memory DB, audit
  trail, and API-schema store were never used except where noted.

---

## 3. Per-Subsystem Results

### 3.1 Inter-tool parallel processing (`core/intelligence/parallel_tools.py`) — ✅

| Check | Result |
|---|---|
| 6 concurrent calls (reads + connectivity + search + fetch) | 6/6 executed |
| Timing | 6 calls in ~0.1s local (network calls dominate when included) |
| Deny (`/etc/shadow`) | blocked, never ran |
| Confirm (`rm -rf /`) | needs approval, never ran silently |
| Identical calls | deduplicated to 1 execution |
| 20 concurrent memory writes (write wave) | 0 errors, no SQLite lock |
| 24 batches from 8 threads | 0 errors |
| Hostile/malformed input (6 cases) | all handled as errors/reports |

**Bug found & fixed:** `web_search` (boundary action name) vs. `search_web`
(registry tool name). A batched `web_search` previously failed with
`unknown tool 'web_search'` — unit tests missed it because they only exercised
`read_file`/`check_connectivity`. Fixed with bidirectional name canonicalization
(`_canonical_action_name` / `_registry_tool_name`) so **either** spelling gates
correctly (LOW → allow) and executes. Regression test covers both spellings
with a stubbed backend (no network in CI).

### 3.2 Security boundary + guardrails + audit (`security/`) — ✅

| Check | Result |
|---|---|
| Verdict consistency (12 probes × 50 runs) | identical every time |
| Audit log (isolated, 600 records) | 600 valid JSONL lines; allow/confirm/deny all recorded |
| Secret exfiltration (`AKIA...` in write) | denied (aws_access_key rule) |
| `run_tool_batch` / `synthesize_analysis` classification | LOW / allow |

**Observation (minor):** the stress harness writes to the *shared* boundary for
the consistency probe, appending a few hundred lines to
`~/.ultron/security_audit.jsonl` (a gitignored runtime artifact). Audit-log
integrity itself was verified against an isolated `AuditLog`.

### 3.3 Task planning & preflight (`core/intelligence/planning.py`) — ✅

| Check | Result |
|---|---|
| 6-step plan (write → read → command → query → HTTP → memory) | auto=4, confirm=2, blocked=0, missing=0 |
| Dependency edges | 3 found (producer/producer/feeds) |
| Missing-field detection | caught (`read_file` without filename) |
| Prefix-match false positive guard | `rm data2.csv` does **not** match `data.csv` |
| Preview render | OK |

### 3.4 Environmental-state debugging (`core/intelligence/debug_context.py`) — ✅

| Check | Result |
|---|---|
| Environment snapshot | OS=Darwin, Python=3.12.13, 43 packages, CWD |
| Failure-cause matrix (9 inputs) | all classified (missing_dependency, name, import, tests_failed, database, network, key, type, unknown) |
| Dependency check | `pytest 9.1.1 is installed` |
| Full report render | OK |

### 3.5 Personalized learning (`core/learning/associations.py`) — ✅

- 10 facts stored; connections discovered; `memory_connections` renders.
- Threshold behavior holds (unrelated facts do not fabricate connections).

### 3.6 Structured output (`core/intelligence/structured_output.py`) — ✅

- 7 enforce cases (json/markdown/xml/table × valid + invalid) all return strings
  without raising; `schema_validate` returns a verdict.

### 3.7 Resource monitor (`core/tools/resource_monitor.py`) — ✅

- `check_resources` returns CPU/load/memory snapshot (10 cores, live load).
- `resource_forecast` returns a forecast string.

### 3.8 API schema learning (`core/learning/api_schema.py`) — ✅

- `learn_api_schema`, `get_api_knowledge`, `api_usage_hint` all work against a
  throwaway DB.

### 3.9 Agent dispatch integrity (`core/agents/simple.py`) — ✅

- 10 varied inputs (batch, greeting, search, command, memory, connectivity,
  tests, git, debug) dispatched without crashes; batch detection fired for
  multi-source requests only.

---

## 4. Latent Inconsistencies (not bugs — documented)

1. **`query_chain` is classified by the boundary and mapped in
   `_generic_target_content` but is *not* a registered tool.** It is a
   knowledge-graph helper used internally (`graph.query_chain`). Any batch or
   ReAct call to it reports `unknown tool` — correct, safe behavior, but the
   boundary entry is misleading. Low priority: either register it or drop the
   boundary/action references.
2. **`make_http_request` is deliberately excluded from the concurrent
   read-wave set** even for GET. Slightly conservative (a batched GET runs in
   the sequential wave), but strictly safer — a POST is auto-allowed in
   permissive mode and must never run concurrently.

---

## 5. Security Review Notes

- **Deny/confirm contract holds under concurrency.** No stress path allowed a
  deny or confirm verdict to execute; the wave split (read-only concurrent,
  writers sequential) prevents SQLite "database is locked" races.
- **Name canonicalization is symmetric** — both `web_search` and `search_web`
  gate identically (the earlier one-way fix would have let `search_web` slip
  into a HIGH→confirm skip; caught in review and corrected).
- **Hostile paths and secrets are hard-blocked** before any tool runs, in both
  single-call and batch paths.

---

## 6. Reproduction

```bash
# Full suite + lint
.venv/bin/python -m pytest -q          # 596 passed
.venv/bin/ruff check . --isolated      # clean

# Stress harness (29 checks, uses throwaway temp DBs only)
.venv/bin/python _stress_audit.py
```

---

## 7. Follow-up hardening (post-report)

1. **ReAct parallel batching** — the ReAct agent now routes `run_tool_batch`
   directly (like `run_parallel`) so a valid batch whose member carries an
   `http://` URL is never false-blocked by an outer content scan; inner
   per-member gating remains authoritative (deny never runs, confirm never
   runs silently). Accepts both `calls_json` (string) and `calls` (list)
   spellings, and the system prompt instructs the model to prefer a single
   batch for independent read-only lookups. 4 tests added.
2. **CI stress job** — `.github/workflows/ci.yml` gained a `stress` job that
   runs `_stress_audit.py` (which now exits non-zero on any failed check). A
   post-review hardening pass strengthened the 6-call batch assertion from
   `executed >= 4` to `"unknown tool" not in report and executed >= 3` — the
   original check would *not* have caught the `web_search`/`search_web`
   regression (unknown-tool members count as ⚠️, not ✅), and the new form is
   network-robust in CI. Simulated-bug verification confirms the harness now
   fails when the regression is reintroduced.

## 8. Conclusion

The audited implementations are **stable, fast, and safe** for their intended
scope. The audit found one real bug (now fixed with a regression test), two
documented latent inconsistencies, and no concurrency, integrity, or security
failures under load. The test suite is green across repeated runs and the lint
surface is clean. The ReAct agent and CI now exercise the parallel-batch path
and the stress harness automatically.
