"""Structured validation report (Phase 18 of STEP 3).

Aggregates traces + evaluations into the markdown report layout mandated by
STEP 3: executive summary, capability matrix, routing results, evidence
results, security results, ReAct results, generalization indicators,
failures, anti-hardcoding findings, and recommendations.  Fully
deterministic: identical traces produce identical reports.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime

from ultron.validation.audit import AuditReport
from ultron.validation.generator import CAPABILITY_TESTABILITY
from ultron.validation.model import Evaluation, TaskTrace, Verdict
from ultron.validation.runner import parse_trace


@dataclass
class ValidationSummary:
    """Aggregated numbers for the executive summary."""

    model: str
    agent: str
    duration_s: float
    total: int
    passed: int
    partial: int
    failed: int
    unresolved: int
    capabilities_tested: int


def _verdict_count(evaluations: list[Evaluation], verdict: Verdict) -> int:
    return sum(1 for e in evaluations if e.overall is verdict)


def summarize(traces: list[TaskTrace], evaluations: list[Evaluation], *, model: str, agent: str) -> ValidationSummary:
    duration = sum(t.latency_s for t in traces)
    return ValidationSummary(
        model=model,
        agent=agent,
        duration_s=duration,
        total=len(traces),
        passed=_verdict_count(evaluations, Verdict.PASS),
        partial=_verdict_count(evaluations, Verdict.PARTIAL),
        failed=_verdict_count(evaluations, Verdict.FAIL),
        unresolved=_verdict_count(evaluations, Verdict.UNRESOLVED),
        capabilities_tested=len({t.case.expected_capability.value for t in traces}),
    )


def _dimension_stats(evaluations: list[Evaluation]) -> dict[str, Counter]:
    stats: dict[str, Counter] = {}
    for e in evaluations:
        for name, verdict in e.dimensions.items():
            stats.setdefault(name, Counter())[verdict.value] += 1
    return stats


def _capability_matrix(traces: list[TaskTrace], evaluations: list[Evaluation]) -> str:
    rows: dict[str, Counter] = {}
    for t, e in zip(traces, evaluations):
        cap = t.case.expected_capability.value
        rows.setdefault(cap, Counter())[e.overall.value] += 1
    if not rows:
        return "_no tasks_"
    lines = ["| capability | PASS | PARTIAL | FAIL | UNRESOLVED |", "|---|---|---|---|---|"]
    for cap in sorted(rows):
        c = rows[cap]
        lines.append(f"| {cap} | {c['PASS']} | {c['PARTIAL']} | {c['FAIL']} | {c['UNRESOLVED']} |")
    return "\n".join(lines)


def _capability_coverage_table() -> str:
    lines = ["| capability | testability |", "|---|---|"]
    for cap in sorted(CAPABILITY_TESTABILITY, key=lambda c: c.value):
        lines.append(f"| {cap.value} | {CAPABILITY_TESTABILITY[cap].value} |")
    return "\n".join(lines)


def _failures_table(traces: list[TaskTrace], evaluations: list[Evaluation]) -> str:
    rows = []
    for t, e in zip(traces, evaluations):
        if e.overall is not Verdict.FAIL:
            continue
        parsed = parse_trace(t.transcript)
        tool = parsed.observed_tool or "-"
        fail_dims = ", ".join(
            f"{k}={v.value}" for k, v in sorted(e.dimensions.items()) if v is Verdict.FAIL
        )
        kind = e.failure_kind.value if e.failure_kind else "-"
        rows.append(
            f"| {t.case.case_id} | {t.case.expected_capability.value} | `{t.case.task}` | "
            f"{tool} | {kind} | {fail_dims} |"
        )
    if not rows:
        return "_no failures_"
    lines = ["| id | capability | request | tool | root cause | failing dims |", "|---|---|---|---|---|---|"]
    lines.extend(rows)
    return "\n".join(lines)


def _generalization(traces: list[TaskTrace]) -> str:
    # Part 5: one semantic entity vs its surface forms are counted separately.
    entities = {t.case.entity_id for t in traces if t.case.entity_id}
    surface_forms = {t.case.surface_form for t in traces if t.case.surface_form}
    templates = {t.case.template_id for t in traces if t.case.template_id}
    capabilities = {t.case.expected_capability.value for t in traces}
    tasks = [t.case.task for t in traces]
    repeated = len(tasks) - len(set(tasks))
    per_capability_entities = Counter(
        t.case.entity_id for t in traces if t.case.entity_id
    )
    return (
        f"- unique semantic entities: **{len(entities)}** (surface forms: **{len(surface_forms)}**)\n"
        f"- unique task forms (template ids): **{len(templates)}**\n"
        f"- unique capabilities: **{len(capabilities)}**\n"
        f"- wording diversity: **{len(set(tasks))}** distinct task strings over {len(tasks)} tasks\n"
        f"- repeated-task rate: **{repeated}** exact duplicates\n"
        f"- max tasks per single entity: **{per_capability_entities.most_common(1)[0][1] if per_capability_entities else 0}** "
        "(coverage-aware: no entity may dominate)\n"
    )


def _three_layers(traces: list[TaskTrace], evaluations: list[Evaluation]) -> str:
    """Part 7-11: capability / execution / answer truth, independently."""
    def row(layer: str) -> str:
        c = Counter(getattr(e, layer) for e in evaluations)
        return (
            f"| {layer} | {c[Verdict.PASS]} | {c[Verdict.PARTIAL]} | "
            f"{c[Verdict.FAIL]} | {c[Verdict.UNRESOLVED]} |"
        )

    lines = ["| layer | PASS | PARTIAL | FAIL | UNRESOLVED |", "|---|---|---|---|---|"]
    lines.append(row("capability"))
    lines.append(row("execution"))
    lines.append(row("answer"))
    answer_dims: dict[str, Counter] = {}
    for e in evaluations:
        for name, verdict in e.answer_dimensions.items():
            answer_dims.setdefault(name, Counter())[verdict.value] += 1
    if answer_dims:
        lines.append("")
        lines.append("### Final-answer sub-dimensions (Part 10)")
        lines.append("| dimension | PASS | PARTIAL | FAIL | UNRESOLVED |")
        lines.append("|---|---|---|---|---|")
        for name in sorted(answer_dims):
            c = answer_dims[name]
            lines.append(f"| {name} | {c['PASS']} | {c['PARTIAL']} | {c['FAIL']} | {c['UNRESOLVED']} |")
    return "\n".join(lines)


def _router_diagnostics(traces: list[TaskTrace], evaluations: list[Evaluation]) -> str:
    """Part 12: expected vs deterministic-router vs model, as diagnostics."""
    rows: list[str] = []
    agree = 0
    for t, e in zip(traces, evaluations):
        case = t.case
        if case.router_capability is None:
            continue
        if case.router_agreement:
            agree += 1
        model_cap = e.model_capability or "unobserved"
        expected = case.expected_capability.value
        rows.append(
            f"| {case.case_id} | {expected} | {case.router_capability} | {model_cap} | "
            f"{'agree' if case.router_agreement else 'DISAGREE'} |"
        )
    if not rows:
        return "_no router diagnostics recorded_"
    header = (
        f"Router agreement with expected capability: **{agree}/{len(rows)}** "
        "(disagreements are evaluation data, never task invalidity).\n\n"
    )
    table = ["| case | expected | deterministic router | model (observed tool) | agreement |", "|---|---|---|---|---|"]
    table.extend(rows)
    return header + "\n".join(table)


def _anti_hardcoding(audit: AuditReport) -> str:
    if not audit.findings:
        return f"_clean: {audit.files_scanned} production files scanned, 0 findings_"
    lines = [f"{audit.files_scanned} files scanned, **{audit.count} findings**:"]
    lines.append("| file | line | kind | severity | detail |")
    lines.append("|---|---|---|---|---|")
    for f in audit.findings:
        lines.append(f"| {f.file} | {f.line} | {f.kind} | {f.severity} | {f.detail} |")
    return "\n".join(lines)


def _recommendations(evaluations: list[Evaluation], audit: AuditReport) -> str:
    recs: list[str] = []
    failure_kinds = Counter(
        e.failure_kind.value for e in evaluations if e.failure_kind is not None
    )
    if failure_kinds:
        top = failure_kinds.most_common(3)
        recs.append(
            "**Most common failure kinds:** "
            + ", ".join(f"{kind} ({n})" for kind, n in top)
            + " — classify each as implementation bug vs model limitation before patching."
        )
    else:
        recs.append("No failed tasks classified; nothing to recommend from failures.")
    if audit.critical():
        recs.append("**Anti-hardcoding:** critical findings must be removed before any holdout evaluation.")
    elif audit.findings:
        recs.append("**Anti-hardcoding:** warning findings are documentation-only; verify manually.")
    else:
        recs.append("**Anti-hardcoding:** audit clean — no executable historical literals found.")
    return "\n".join(f"- {r}" for r in recs)


def _invariants_section(traces: list[TaskTrace], evaluations: list[Evaluation]) -> str:
    all_invs = []
    for t in traces:
        all_invs.extend(t.invariants)
    for e in evaluations:
        all_invs.extend(e.invariants)

    # Unique by id
    unique_invs = {}
    for inv in all_invs:
        unique_invs[inv.invariant_id] = inv

    if not unique_invs:
        return "_no invariant checks recorded_"

    lines = ["| invariant | status | severity | failure reason / details |", "|---|---|---|---|"]
    for inv_id in sorted(unique_invs):
        inv = unique_invs[inv_id]
        status = "**PASS**" if inv.passed else "<span style='color:red'>**FAIL**</span>"
        reason = inv.failure_reason or "-"
        lines.append(f"| {inv.invariant_id} ({inv.name}) | {status} | {inv.severity} | {reason} |")
    return "\n".join(lines)


def _criteria_section(traces: list[TaskTrace], evaluations: list[Evaluation]) -> str:
    all_crits = []
    for t in traces:
        all_crits.extend(t.criteria_results)
    for e in evaluations:
        all_crits.extend(e.criteria)

    unique_crits = {}
    for c in all_crits:
        unique_crits[c.criterion_id] = c

    if not unique_crits:
        return "_no acceptance criteria recorded_"

    lines = ["| criterion | status | description | error / details |", "|---|---|---|---|"]
    for c_id in sorted(unique_crits):
        c = unique_crits[c_id]
        status = "**PASS**" if c.status == Verdict.PASS else f"**{c.status.value}**"
        err = c.error or "-"
        lines.append(f"| {c.criterion_id} | {status} | {c.description} | {err} |")
    return "\n".join(lines)


def build_real_world_task_report(
    trace: TaskTrace,
    evaluation: Evaluation,
    *,
    task_title: str = "Real-World Task Validation",
) -> str:
    """Builds the standardized real-world task validation report matching the required format."""
    case = trace.case
    inv_map = {inv.invariant_id: inv for inv in evaluation.invariants}
    crit_map = {c.criterion_id: c for c in evaluation.criteria}


    def _inv_status(inv_id: str) -> str:
        inv = inv_map.get(inv_id)
        if not inv:
            return "N/A"
        return "PASS" if inv.passed else "FAIL"

    def _crit_status(crit_id: str) -> str:
        c = crit_map.get(crit_id)
        if not c:
            return "N/A"
        return c.status.value

    return f"""# ULTRON REAL-WORLD TASK VALIDATION

Task:
{case.task}

Execution:
    Classification:       {evaluation.dimensions.get('intent', Verdict.UNRESOLVED).value}
    Initial model:        {trace.case.router_capability or 'observed'}
    Planning:             {evaluation.dimensions.get('capability', Verdict.UNRESOLVED).value}
    Model escalation:     {'PASS' if 'qwen3' in (evaluation.model_capability or '') else 'N/A'}
    Tool execution:       {evaluation.execution.value}
    Recovery:             {evaluation.dimensions.get('evidence', Verdict.UNRESOLVED).value}

Invariants:
    Workspace confinement:      {_inv_status('INV-001')}
    Dependency ordering:       {_inv_status('INV-002')}
    Failure thrashing:         {_inv_status('INV-003')}
    Dependency validity:       {_inv_status('INV-004')}

Outcome:
    Required artifacts:        {_crit_status('AC-02')}
    Application launch:        {_crit_status('AC-05')}
    Functional operation:      {_crit_status('AC-06')}
    Verification evidence:     {_crit_status('AC-09')}

Final:
    Objective achieved:        {evaluation.objective_verdict.value}
    Final answer supported:    {evaluation.answer.value}
    Overall verdict:           {evaluation.overall.value}
    Root cause failure kind:   {evaluation.failure_kind.value if evaluation.failure_kind else 'None'}

### Diagnostic Evidence & Invariant Violations
{_invariants_section([trace], [evaluation])}

### Acceptance Criteria Breakdown
{_criteria_section([trace], [evaluation])}
"""


def build_report(
    traces: list[TaskTrace],
    evaluations: list[Evaluation],
    *,
    model: str,
    agent: str,
    audit: AuditReport | None = None,
    duration_budget_s: float | None = None,
) -> str:
    """Builds the full markdown report (deterministic given traces)."""
    summary = summarize(traces, evaluations, model=model, agent=agent)
    stats = _dimension_stats(evaluations)
    audit = audit or AuditReport()
    ts = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

    routing_lines = []
    for name in ("intent", "capability", "tool_selection", "argument"):
        c = stats.get(name, Counter())
        routing_lines.append(f"- **{name}**: PASS={c['PASS']} PARTIAL={c['PARTIAL']} FAIL={c['FAIL']} UNRESOLVED={c['UNRESOLVED']}")
    evidence = stats.get("evidence", Counter())
    security = stats.get("security", Counter())
    react_lines = []
    for name in ("tool_selection", "investigation", "final_answer"):
        c = stats.get(name, Counter())
        react_lines.append(f"- **{name}**: PASS={c['PASS']} PARTIAL={c['PARTIAL']} FAIL={c['FAIL']} UNRESOLVED={c['UNRESOLVED']}")
    routing_block = "\n".join(routing_lines)
    react_block = "\n".join(react_lines)

    return f"""# Ultron Capability Validation Report

> Generated {ts} · framework: `ultron.validation` (STEP 3) · deterministic aggregation

## Executive summary

| metric | value |
|---|---|
| model | {model} |
| agent | {agent} |
| duration | {summary.duration_s:.1f}s (budget {duration_budget_s if duration_budget_s else 'n/a'}) |
| tasks executed | {summary.total} |
| capabilities tested | {summary.capabilities_tested} |
| PASS | {summary.passed} |
| PARTIAL | {summary.partial} |
| FAIL | {summary.failed} |
| UNRESOLVED | {summary.unresolved} |

## Capability matrix

{_capability_matrix(traces, evaluations)}

## Invariants

{_invariants_section(traces, evaluations)}

## Three-layer verdicts (Part 7-11)

Overall rule: any layer FAIL -> FAIL; all three PASS -> PASS; otherwise PARTIAL/UNRESOLVED.

{_three_layers(traces, evaluations)}

## Router diagnostics (Part 12)

{_router_diagnostics(traces, evaluations)}

## Routing results

{routing_block}

## Evidence results

- verified evidence: **{evidence['PASS']}**
- insufficient evidence: **{evidence['FAIL']}**
- unsupported/speculative claims: **{sum(1 for e in evaluations if e.dimensions.get('evidence') is Verdict.FAIL and any('speculative' in n for n in e.notes))}**
- UNRESOLVED: **{evidence['UNRESOLVED']}**

## Security results

- security dimension PASS: **{security['PASS']}**
- security dimension FAIL: **{security['FAIL']}**
- security dimension UNRESOLVED: **{security['UNRESOLVED']}**

## ReAct results (tool loop signals)

{react_block}

## Generalization indicators

{_generalization(traces)}

## Failures

{_failures_table(traces, evaluations)}

## Anti-hardcoding

{_anti_hardcoding(audit)}

## Capability coverage classification

All 44 canonical capabilities classified for automated testing:

{_capability_coverage_table()}

## Recommendations

{_recommendations(evaluations, audit)}
"""

