"""STEP 3 — live capability validation runner.

Exercises the ACTUAL Ultron CLI with dynamically generated capability tasks:

    discover subjects (current repository)
        -> generate validated tasks (capability contracts + canonical vocabulary)
        -> run each through the real CLI (pty)
        -> evaluate each trace on eight dimensions
        -> audit production for anti-hardcoding
        -> write a deterministic structured report

Usage:

    # limited sanity run (a handful of tasks, all read-only capabilities)
    .venv/bin/python _step3_validation.py --count 8 --wait 50

    # approximate 10-minute dynamic mode (respects a wall-clock budget)
    .venv/bin/python _step3_validation.py --budget-seconds 600 --wait 50

The tasks are generated — never a static prompt list.  This script never
modifies production code; it only observes and reports.
"""

from __future__ import annotations

import argparse
import time

from ultron.validation.audit import audit_production
from ultron.validation.evaluate import evaluate_many
from ultron.validation.generator import TaskGenerator
from ultron.validation.model import GenerationStrategy
from ultron.validation.report import build_report
from ultron.validation.runner import ValidationRunner
from ultron.validation.subjects import discover_subjects

DEFAULT_MODEL = "gemma4"


def _summary(traces, evaluations) -> str:
    from collections import Counter

    verdicts = Counter(e.overall.value for e in evaluations)
    kinds = Counter(e.failure_kind.value for e in evaluations if e.failure_kind)
    lines = [f"tasks={len(traces)} verdicts={dict(verdicts)}"]
    if kinds:
        lines.append(f"failure kinds={dict(kinds)}")
    return " ".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="STEP 3 live capability validation")
    parser.add_argument("--count", type=int, default=0, help="exact number of tasks (limited run)")
    parser.add_argument("--budget-seconds", type=float, default=0, help="wall-clock budget (10-min mode)")
    parser.add_argument("--wait", type=float, default=50.0, help="per-task CLI wait (seconds)")
    parser.add_argument("--agent", default="simple", help="agent type (simple|react)")
    parser.add_argument("--seed", type=int, default=17, help="generation seed (deterministic)")
    parser.add_argument("--per-capability", type=int, default=3)
    parser.add_argument(
        "--strategy",
        choices=["development", "holdout"],
        default="development",
        help="development-direct or holdout-indirect generation",
    )
    parser.add_argument("--report", default="validationReport.md")
    args = parser.parse_args()

    if not args.count and not args.budget_seconds:
        parser.error("pass --count N or --budget-seconds N")

    print("== STEP 3 validation ==")
    print(f"agent={args.agent} seed={args.seed} wait={args.wait}s")

    print("discovering subjects from the current repository...")
    subjects = discover_subjects(".", max_symbols=60, max_files=12)
    print(f"  {len(subjects)} subjects "
          f"({sum(1 for s in subjects if s.kind in ('class', 'function', 'enum'))} symbols, "
          f"{sum(1 for s in subjects if s.kind == 'file')} files, "
          f"{sum(1 for s in subjects if s.kind == 'directory')} dirs)")

    strategy = (
        GenerationStrategy.HOLDOUT_INDIRECT
        if args.strategy == "holdout"
        else GenerationStrategy.DEVELOPMENT_DIRECT
    )
    print(f"generating validated capability tasks (strategy={strategy.value})...")
    gen = TaskGenerator(subjects, seed=args.seed, strategy=strategy)
    if args.count:
        plan = gen.generate_plan(max_tasks=args.count, per_capability=args.per_capability)
    else:
        plan = gen.generate_plan(per_capability=args.per_capability)
    # Every task is validated at generation time against the canonical
    # vocabulary (capability exists, contract exists, router agreement).
    print(f"  {len(plan)} tasks across "
          f"{len({c.expected_capability.value for c in plan})} capabilities")
    for c in plan:
        print(f"    [{c.expected_capability.value}] {c.task!r}")

    runner = ValidationRunner(agent=args.agent, wait_s=args.wait)
    traces = []
    budget = args.budget_seconds or float("inf")
    start = time.monotonic()
    print(f"running through the real CLI (budget={budget:.0f}s if set)...")
    for i, case in enumerate(plan, 1):
        if time.monotonic() - start >= budget:
            print(f"  budget reached after {i - 1} tasks")
            break
        print(f"  [{i}/{len(plan)}] {case.expected_capability.value}: {case.task!r}")
        trace = runner.run_one(case)
        traces.append(trace)
        print(f"      latency={trace.latency_s:.1f}s")

    print("evaluating traces (eight dimensions per task)...")
    evaluations = evaluate_many(traces)
    print("  " + _summary(traces, evaluations))

    print("running anti-hardcoding audit on production code...")
    audit = audit_production()
    print(f"  {audit.files_scanned} files scanned, {audit.count} findings "
          f"({len(audit.critical())} critical)")

    report = build_report(
        traces,
        evaluations,
        model=DEFAULT_MODEL,
        agent=args.agent,
        audit=audit,
        duration_budget_s=args.budget_seconds or None,
    )
    with open(args.report, "w", encoding="utf-8") as fh:
        fh.write(report)
    print(f"report written to {args.report}")

    passed = sum(1 for e in evaluations if e.overall.value == "PASS")
    print(f"\nRESULT: {passed}/{len(evaluations)} PASS, "
          f"{sum(1 for e in evaluations if e.overall.value == 'FAIL')} FAIL, "
          f"{sum(1 for e in evaluations if e.overall.value == 'PARTIAL')} PARTIAL, "
          f"{sum(1 for e in evaluations if e.overall.value == 'UNRESOLVED')} UNRESOLVED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
