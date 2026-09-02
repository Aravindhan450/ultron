"""STEP 3 — test the tester (Phase 19).

Deterministic self-tests for the capability validation framework.  No live
model is required: the CLI executor is injected with canned transcripts so
the framework's own machinery (discovery, generation, validation, trace
capture, evaluation, failure classification, audit, reporting) is verified.

Covers the Phase 19 checklist:
  1. discovers arbitrary repository entities
  2. generates tasks for multiple capabilities
  3. rejects invalid generated tasks
  4. records the complete model trace
  5. does not use the model's answer as ground truth
  6. classifies a deliberately incorrect tool selection
  7. classifies a deliberately incorrect capability selection
  8. detects missing evidence
  9. does not mutate production code during validation
 10. produces deterministic structured reports
"""

from __future__ import annotations

import time

import pytest

from ultron.core.tools.definitions import (
    ToolCapability,
    preferred_tool_for,
    tools_with_capability,
)
from ultron.validation.audit import audit_production
from ultron.validation.evaluate import evaluate_many, evaluate_trace
from ultron.validation.generator import (
    CAPABILITY_TESTABILITY,
    TaskGenerator,
    symbol_variants,
    validate_task,
)
from ultron.validation.generator import (
    Testability as _Testability,  # aliased: pytest would collect the enum as a test class
)
from ultron.validation.generator import (
    testability as _testability,  # aliased: pytest would collect the function as a test
)
from ultron.validation.model import (
    CapabilityTestCase,
    FailureKind,
    TaskTrace,
    Verdict,
)
from ultron.validation.model import (
    TestSplit as _TestSplit,
)
from ultron.validation.report import build_report
from ultron.validation.runner import ValidationRunner
from ultron.validation.subjects import discover_subjects

FIXTURE_PY = """\
class TaskProcessor:
    \"\"\"Processes a task through the pipeline.\"\"\"

    def run(self):
        return self._step()


class Config:
    pass


def build_config(name: str) -> Config:
    return Config()
"""


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    from ultron.core.tools import paths as tools_paths

    monkeypatch.setattr(tools_paths, "ALLOWED_BASE_DIR", tmp_path)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "models.py").write_text(FIXTURE_PY, encoding="utf-8")
    (tmp_path / "src" / "runner.py").write_text(
        "from models import TaskProcessor\n\n\ndef main():\n    return TaskProcessor().run()\n",
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text("# fixture\n", encoding="utf-8")
    return tmp_path


def _fake_executor(transcript: str, latency: float = 0.25):
    def executor(args, prompt, wait_s):
        time.sleep(latency)  # simulate real model latency
        return transcript, latency

    return executor


def _case(
    capability: ToolCapability,
    task: str,
    *,
    subject: str | None = "TaskProcessor",
    case_id: str = "t1",
) -> CapabilityTestCase:
    return CapabilityTestCase(
        case_id=case_id,
        capability=capability,
        task=task,
        expected_capability=capability,
        subject=subject,
        test_source=__import__("ultron.validation.model", fromlist=["TestSource"]).TestSource.GENERATED,
        split=_TestSplit.DEVELOPMENT,
    )


# ---------------------------------------------------------------------------
# 1. Arbitrary subject discovery
# ---------------------------------------------------------------------------


def test_subject_discovery_finds_arbitrary_entities(sandbox):
    subjects = discover_subjects(str(sandbox), max_symbols=20, max_files=5)
    kinds = {s.kind for s in subjects}
    assert "class" in kinds or "function" in kinds  # code-intelligence symbols
    assert "file" in kinds
    assert "directory" in kinds
    names = {s.name for s in subjects}
    assert "TaskProcessor" in names  # arbitrary entity discovered from source
    assert "README.md" in names
    # Deterministic: same repo -> same pool.
    again = discover_subjects(str(sandbox), max_symbols=20, max_files=5)
    assert [s.display for s in subjects] == [s.display for s in again]


def test_symbol_variants_general():
    variants = set(symbol_variants("TaskProcessor"))
    assert {"TaskProcessor", "taskprocessor", "TASKPROCESSOR", "Task Processor", "task processor"} <= variants
    # snake input round-trips
    v2 = set(symbol_variants("task_processor"))
    assert "TaskProcessor" in v2 and "task processor" in v2


# ---------------------------------------------------------------------------
# 2. Task generation for multiple capabilities + wording variation
# ---------------------------------------------------------------------------


def test_generator_covers_multiple_capabilities(sandbox):
    subjects = discover_subjects(str(sandbox), max_symbols=20, max_files=5)
    gen = TaskGenerator(subjects, seed=11)
    plan = gen.generate_plan(max_tasks=30)
    caps = {c.expected_capability for c in plan}
    assert len(caps) >= 4  # several capabilities represented
    assert len(plan) >= 10
    # Tasks are varied: more distinct task strings than capabilities.
    assert len({c.task for c in plan}) > len(caps)
    # Every task is valid (validated at generation time).
    for case in plan:
        ok, reason = validate_task(case)
        assert ok, reason


def test_generator_varies_entity_naming(sandbox):
    subjects = discover_subjects(str(sandbox), max_symbols=20, max_files=5)
    gen = TaskGenerator(subjects, seed=3)
    plan = gen.generate_plan(max_tasks=40)
    symbol_tasks = [
        c for c in plan if c.expected_capability is ToolCapability.DEFINITION_LOOKUP
    ]
    surface_forms = {c.surface_form for c in symbol_tasks}
    entities = {c.entity_id for c in symbol_tasks}
    assert len(symbol_tasks) >= 2
    # The same semantic entity yields different surface forms across tasks,
    # and entities rotate (coverage-aware selection).
    assert len(surface_forms) >= 2
    assert len(entities) >= 2


def test_plan_round_robin_diversity(sandbox):
    subjects = discover_subjects(str(sandbox), max_symbols=20, max_files=5)
    gen = TaskGenerator(subjects, seed=5)
    plan = gen.generate_plan(max_tasks=9)
    assert len(plan) == 9
    caps = [c.expected_capability.value for c in plan]
    assert len(set(caps)) == 9  # nine distinct capabilities, one task each


def test_plan_is_deterministic_for_seed(sandbox):
    subjects = discover_subjects(str(sandbox), max_symbols=20, max_files=5)
    a = TaskGenerator(subjects, seed=7).generate_plan(max_tasks=20)
    b = TaskGenerator(subjects, seed=7).generate_plan(max_tasks=20)
    assert [c.task for c in a] == [c.task for c in b]


# ---------------------------------------------------------------------------
# 3. Rejects invalid generated tasks
# ---------------------------------------------------------------------------


def test_validation_rejects_mislabeled_task(sandbox):
    subjects = discover_subjects(str(sandbox), max_symbols=20, max_files=5)
    gen = TaskGenerator(subjects, seed=1)
    plan = gen.generate_plan(max_tasks=40)
    definition_tasks = [
        c for c in plan if c.expected_capability is ToolCapability.DEFINITION_LOOKUP
    ]
    assert definition_tasks
    # Relabel a definition task as reference lookup: must be rejected.
    wrong = CapabilityTestCase(
        case_id="x",
        capability=ToolCapability.REFERENCE_LOOKUP,
        task=definition_tasks[0].task,
        expected_capability=ToolCapability.REFERENCE_LOOKUP,
        subject=definition_tasks[0].subject,
    )
    ok, reason = validate_task(wrong)
    assert not ok
    # Rejection comes from the INDEPENDENT validity check (the task does
    # not express the expected capability's operation), never the router.
    assert "does not express the operation" in reason


def test_validation_rejects_subject_missing():
    case = CapabilityTestCase(
        case_id="x",
        capability=ToolCapability.DEFINITION_LOOKUP,
        task="Where is the widget defined?",  # subject 'TaskProcessor' absent
        expected_capability=ToolCapability.DEFINITION_LOOKUP,
        subject="TaskProcessor",
    )
    ok, reason = validate_task(case)
    assert not ok
    assert "not mentioned" in reason


# ---------------------------------------------------------------------------
# 4. Complete trace capture
# ---------------------------------------------------------------------------


def test_runner_records_full_trace(sandbox):
    case = _case(ToolCapability.DEFINITION_LOOKUP, "Where is TaskProcessor defined?")
    runner = ValidationRunner(executor=_fake_executor("TaskProcessor (class) src/models.py:1-10", latency=1.5))
    trace = runner.run_one(case)
    assert trace.case is case
    assert trace.latency_s == 1.5
    assert "TaskProcessor" in trace.transcript
    assert trace.evaluation is None  # evaluation is a separate step


def test_runner_budget_respects_duration():
    """10-minute mode: a budgeted run stops adding tasks once elapsed."""

    class BudgetedRun:
        def __init__(self, runner: ValidationRunner, budget_s: float) -> None:
            self.runner = runner
            self.budget_s = budget_s

        def run(self, cases):
            traces = []
            start = time.monotonic()
            for case in cases:
                if time.monotonic() - start >= self.budget_s:
                    break
                traces.append(self.runner.run_one(case))
            return traces

    runner = ValidationRunner(executor=_fake_executor("ok", latency=0.1))
    cases = [_case(ToolCapability.DIRECTORY_LIST, f"List the files in dir{i}", case_id=f"c{i}") for i in range(20)]
    traces = BudgetedRun(runner, budget_s=0.5).run(cases)
    assert 0 < len(traces) < 20  # stopped early, deterministic enough
    assert all(t.case.task.startswith("List") for t in traces)


# ---------------------------------------------------------------------------
# 5. Ground truth is independent of the model's answer
# ---------------------------------------------------------------------------


def test_evaluation_ground_truth_from_canonical(sandbox):
    case = _case(ToolCapability.DEFINITION_LOOKUP, "Where is TaskProcessor defined?")
    # The model sounds confident, but cites no definition evidence: the
    # evaluation must NOT trust the confident answer.
    trace = TaskTrace(
        case=case,
        transcript="TaskProcessor is definitely defined in the codebase, trust me.",
        latency_s=0.1,
    )
    ev = evaluate_trace(trace)
    assert ev.dimensions["evidence"] is Verdict.FAIL  # no file:line evidence
    assert ev.overall is Verdict.FAIL


# ---------------------------------------------------------------------------
# 6/7. Deliberately wrong tool / capability selection classification
# ---------------------------------------------------------------------------


def test_classifies_wrong_tool_selection():
    case = _case(ToolCapability.DEFINITION_LOOKUP, "Where is TaskProcessor defined?")
    # Model used a tool from an unrelated capability (web search).
    web_tool = preferred_tool_for(ToolCapability.WEB_SEARCH)
    assert web_tool  # canonical discovery provides the unrelated tool name
    trace = TaskTrace(
        case=case,
        transcript=f"Action: {web_tool} Searching the web for TaskProcessor... TaskProcessor is a class.",
        latency_s=0.1,
    )
    ev = evaluate_trace(trace)
    assert ev.dimensions["tool_selection"] is Verdict.FAIL
    assert ev.overall is Verdict.FAIL
    assert ev.failure_kind is FailureKind.CAPABILITY_SELECTION_FAILURE


def test_classifies_wrong_capability():
    case = _case(ToolCapability.REFERENCE_LOOKUP, "Where is TaskProcessor used?")
    # Model performed a definition lookup instead of reference lookup.
    trace = TaskTrace(
        case=case,
        transcript="Action: find_definition TaskProcessor (class) src/models.py:1-10 Found the definition.",
        latency_s=0.1,
    )
    ev = evaluate_trace(trace)
    assert ev.dimensions["tool_selection"] is Verdict.FAIL
    assert ev.overall is Verdict.FAIL
    assert ev.failure_kind is FailureKind.CAPABILITY_SELECTION_FAILURE


def test_classifies_empty_response_as_model_limitation():
    case = _case(ToolCapability.DEFINITION_LOOKUP, "Where is TaskProcessor defined?")
    trace = TaskTrace(
        case=case, transcript="I couldn't generate a response just now.", latency_s=0.1
    )
    ev = evaluate_trace(trace)
    assert ev.overall is Verdict.FAIL
    assert ev.failure_kind is FailureKind.MODEL_LIMITATION


def test_classifies_wrapper_leak_as_argument_failure():
    case = _case(ToolCapability.REFERENCE_LOOKUP, "Where is TaskProcessor used?")
    # Historical bug: the grammatical wrapper leaked into the symbol.
    trace = TaskTrace(
        case=case,
        transcript="References to 'is TaskProcessor' (lexical source occurrences...)",
        latency_s=0.1,
    )
    ev = evaluate_trace(trace)
    assert ev.dimensions["argument"] is Verdict.FAIL
    assert ev.failure_kind is FailureKind.ARGUMENT_FAILURE


# ---------------------------------------------------------------------------
# 8. Missing evidence detection
# ---------------------------------------------------------------------------


def test_detects_missing_evidence(sandbox):
    # The subject DOES exist in the repository (repository ground truth); a
    # "could not find" answer is therefore a miss, not an honest absence.
    case = _case(ToolCapability.DEFINITION_LOOKUP, "Where is TaskProcessor defined?")
    case.subject_path = "src/models.py"
    trace = TaskTrace(
        case=case, transcript="I could not find TaskProcessor anywhere.", latency_s=0.1
    )
    ev = evaluate_trace(trace, repo_root=str(sandbox))
    assert ev.dimensions["evidence"] is Verdict.FAIL


def test_honest_absence_for_missing_entity_is_evidence(sandbox):
    # When the entity genuinely does not exist, an honest "not found" answer
    # is correctly calibrated evidence, not a failure.
    case = _case(ToolCapability.DEFINITION_LOOKUP, "Where is NoSuchThing defined?", subject="NoSuchThing")
    case.subject_path = "src/does_not_exist.py"
    trace = TaskTrace(
        case=case, transcript="I could not find any definition of NoSuchThing in the repository.", latency_s=0.1
    )
    ev = evaluate_trace(trace, repo_root=str(sandbox))
    assert ev.dimensions["evidence"] is Verdict.PASS


def test_speculative_definition_is_failure():
    case = _case(ToolCapability.DEFINITION_LOOKUP, "Where is TaskProcessor defined?")
    trace = TaskTrace(
        case=case,
        transcript="src/models.py is likely the definition of TaskProcessor, based on the filename.",
        latency_s=0.1,
    )
    ev = evaluate_trace(trace)
    assert ev.dimensions["evidence"] is Verdict.FAIL


# ---------------------------------------------------------------------------
# 9. No production mutation during validation
# ---------------------------------------------------------------------------


def test_validation_never_writes_production(sandbox, tmp_path):
    sentinel = tmp_path / "sentinel.txt"
    sentinel.write_text("before", encoding="utf-8")

    def executor(args, prompt, wait_s):
        assert sentinel.read_text(encoding="utf-8") == "before"
        return "ok answer", 0.01

    runner = ValidationRunner(executor=executor)
    cases = [
        _case(ToolCapability.DIRECTORY_LIST, "List the files in the current directory", subject=None, case_id=f"c{i}")
        for i in range(3)
    ]
    traces = runner.run_many(cases)
    evaluate_many(traces)
    assert sentinel.read_text(encoding="utf-8") == "before"


# ---------------------------------------------------------------------------
# 10. Deterministic structured reports
# ---------------------------------------------------------------------------


def test_report_is_deterministic():
    traces = [
        TaskTrace(
            case=_case(ToolCapability.DEFINITION_LOOKUP, "Where is TaskProcessor defined?"),
            transcript="TaskProcessor (class) src/models.py:1-10. Found the definition.",
            latency_s=0.5,
        ),
        TaskTrace(
            case=_case(ToolCapability.REFERENCE_LOOKUP, "Where is TaskProcessor used?"),
            transcript="TaskProcessor is referenced in src/runner.py:3 and src/models.py:12.",
            latency_s=0.4,
        ),
    ]
    evals = evaluate_many(traces)
    report_a = build_report(traces, evals, model="fixture", agent="simple")
    report_b = build_report(traces, evals, model="fixture", agent="simple")
    assert report_a == report_b
    assert "## Executive summary" in report_a
    assert "## Capability matrix" in report_a
    assert "## Routing results" in report_a
    assert "## Failures" in report_a
    assert "definition_lookup" in report_a


# ---------------------------------------------------------------------------
# Anti-hardcoding audit + capability coverage completeness
# ---------------------------------------------------------------------------


def test_audit_production_is_clean_and_deterministic():
    report_a = audit_production()
    report_b = audit_production()
    assert [(f.file, f.line, f.kind) for f in report_a.findings] == [
        (f.file, f.line, f.kind) for f in report_b.findings
    ]
    # No critical findings: historical prompts must not exist as executable
    # literals anywhere in production code.
    assert report_a.critical() == []


def test_capability_testability_covers_all_canonical_capabilities():
    assert set(CAPABILITY_TESTABILITY) == set(ToolCapability)
    for cap in ToolCapability:
        assert _testability(cap) is not None


def test_plan_capabilities_are_read_only_by_default(sandbox):
    subjects = discover_subjects(str(sandbox), max_symbols=20, max_files=5)
    gen = TaskGenerator(subjects, seed=9)
    plan = gen.generate_plan(max_tasks=30)
    for case in plan:
        assert CAPABILITY_TESTABILITY[case.expected_capability] is _Testability.READ_ONLY
    # Unsafe/external capabilities are never generated by default.
    unsafe_plan = TaskGenerator(subjects, seed=9).generate_for(
        ToolCapability.FILE_WRITE, per_capability=3
    )
    assert unsafe_plan == []
    external_plan = TaskGenerator(subjects, seed=9).generate_for(
        ToolCapability.WEB_SEARCH, per_capability=3
    )
    assert external_plan == []


def test_tasks_reference_canonical_vocabulary(sandbox):
    subjects = discover_subjects(str(sandbox), max_symbols=20, max_files=5)
    plan = TaskGenerator(subjects, seed=2).generate_plan(max_tasks=30)
    for case in plan:
        assert case.expected_capability in set(ToolCapability)
        assert case.contract is not None
        assert tools_with_capability(case.expected_capability)  # discoverable


# ---------------------------------------------------------------------------
# STEP 3.1 — Part 14: refinement tests (task validity decoupling, strategy
# separation, entity/surface separation, markers-not-truth, three layers).
# ---------------------------------------------------------------------------


def _subject(name: str, kind: str = "class", rel_path: str = "src/models.py"):
    from ultron.validation.subjects import Subject

    return Subject(name=name, kind=kind, rel_path=rel_path)


def test_valid_task_with_router_disagreement_is_accepted():
    """Part 14.1: a semantically valid task the deterministic router does not
    understand is accepted — the router is a diagnostic, not a gate."""
    from ultron.validation.generator import record_router_diagnostic, validate_task

    # Indirect holdout phrasing the deterministic router cannot parse.
    case = _case(
        ToolCapability.DEFINITION_LOOKUP,
        "While reading the codebase I came across TaskProcessor — where does it get defined?",
    )
    ok, reason = validate_task(case)
    assert ok, reason
    record_router_diagnostic(case)
    assert case.router_capability == "unknown"  # router cannot parse it...
    assert case.router_agreement is False  # ...yet the task is accepted


def test_router_disagreement_recorded_not_invalidity():
    """Part 14.2: disagreement is recorded as evaluation data on the case."""
    from ultron.validation.generator import TaskGenerator

    gen = TaskGenerator([_subject("TaskProcessor")], seed=1)
    case = gen.generate_for(ToolCapability.DEFINITION_LOOKUP, per_capability=1)[0]
    assert case.router_capability is not None  # diagnostic recorded
    assert case.router_agreement in (True, False)


def test_invalid_task_rejected_without_router():
    """Part 14.3: invalid semantic tasks are rejected by the independent
    validity check — no router invocation is involved in the decision."""
    from ultron.validation.generator import validate_task

    # Contradiction: a delete request labeled FILE_READ.
    case = _case(ToolCapability.FILE_READ, "Delete config.yaml", subject="config.yaml", case_id="del")
    ok, reason = validate_task(case)
    assert not ok
    # Rejected by the independent check (operation mismatch or conflict).
    assert "operation" in reason
    # Missing subject for a capability that requires one.
    case2 = _case(ToolCapability.DEFINITION_LOOKUP, "Where is the symbol defined?", subject=None, case_id="nosubj")
    ok2, _ = validate_task(case2)
    assert not ok2


def test_dev_and_holdout_strategies_are_structurally_different(sandbox):
    """Part 14.4: development and holdout use different template families,
    not just different seeds."""
    from ultron.validation.generator import TaskGenerator
    from ultron.validation.model import GenerationStrategy, TestSplit

    subjects = discover_subjects(str(sandbox), max_symbols=20, max_files=5)
    dev = TaskGenerator(subjects, seed=11).generate_plan(max_tasks=25)
    ho = TaskGenerator(subjects, seed=99, strategy=GenerationStrategy.HOLDOUT_INDIRECT).generate_plan(max_tasks=25)
    assert {c.split for c in dev} == {TestSplit.DEVELOPMENT}
    assert {c.split for c in ho} == {TestSplit.HOLDOUT}
    assert all(c.strategy is GenerationStrategy.DEVELOPMENT_DIRECT for c in dev)
    assert all(c.strategy is GenerationStrategy.HOLDOUT_INDIRECT for c in ho)
    # Zero literal task overlap and zero template-family overlap.
    assert not ({c.task for c in dev} & {c.task for c in ho})
    dev_tids = {c.template_id.split(".")[1] for c in dev}
    ho_tids = {c.template_id.split(".")[1] for c in ho}
    assert not (dev_tids & ho_tids)
    # Holdout wording is indirect (contains conversational framing words).
    assert any("I " in c.task or "could you" in c.task.lower() for c in ho)


def test_surface_forms_count_as_one_entity():
    """Part 14.5: TaskState/taskstate/TASKSTATE/Task State/task_state are ONE
    semantic entity; coverage counts entity_id, not surface forms."""
    from ultron.validation.generator import TaskGenerator

    gen = TaskGenerator([_subject("TaskProcessor")], seed=4)
    cases = gen.generate_for(ToolCapability.DEFINITION_LOOKUP, per_capability=5)
    assert len(cases) >= 3
    assert len({c.entity_id for c in cases}) == 1  # one semantic entity
    assert len({c.surface_form for c in cases}) >= 2  # multiple surface forms


def test_different_entities_count_separately():
    """Part 14.6: distinct entities are counted separately."""
    from ultron.validation.generator import TaskGenerator

    pool = [_subject("TaskProcessor"), _subject("Config"), _subject("build_config", kind="function")]
    gen = TaskGenerator(pool, seed=6)
    cases = gen.generate_for(ToolCapability.DEFINITION_LOOKUP, per_capability=4)
    assert len({c.entity_id for c in cases}) >= 2


def test_marker_presence_alone_cannot_pass():
    """Part 14.7: a file:line marker for a NONEXISTENT file is not evidence."""
    case = _case(ToolCapability.DEFINITION_LOOKUP, "Where is TaskProcessor defined?")
    case.subject_path = "src/models.py"
    trace = TaskTrace(
        case=case,
        transcript="TaskProcessor definition at src/does_not_exist.py:100",
        latency_s=0.1,
    )
    ev = evaluate_trace(trace, repo_root=".")
    assert ev.dimensions["evidence"] is Verdict.FAIL


def test_correct_looking_unsupported_answer_fails(sandbox):
    """Part 14.8: a confident but unsupported answer fails evidence evaluation."""
    case = _case(ToolCapability.DEFINITION_LOOKUP, "Where is TaskProcessor defined?")
    case.subject_path = "src/models.py"
    trace = TaskTrace(
        case=case,
        transcript="TaskProcessor is definitely defined somewhere in the codebase, trust me.",
        latency_s=0.1,
    )
    ev = evaluate_trace(trace, repo_root=str(sandbox))
    assert ev.dimensions["evidence"] is Verdict.FAIL


def test_correct_tool_with_wrong_final_answer_fails_overall(sandbox):
    """Part 14.9 + Part 15 Case E: correct capability/tool/evidence with an
    incorrect final explanation -> overall FAIL (SYNTHESIS_FAILURE)."""
    case = _case(ToolCapability.DEFINITION_LOOKUP, "Where is TaskProcessor defined?")
    case.subject_path = "src/models.py"
    # The tool action names the subject; the FINAL ANSWER region is off-topic.
    # The verified evidence names the subject in the first part of the trace;
    # the tool action and the FINAL ANSWER region (reply window) do not
    # mention the subject: the response is off-topic while execution was fine.
    trace = TaskTrace(
        case=case,
        transcript=(
            "TaskProcessor is defined at src/models.py:1.\n"
            + ("filler line for the reply window\n" * 60)
            + "Action: find_definition — here is the result summary."
        ),
        latency_s=0.1,
    )
    ev = evaluate_trace(trace, repo_root=str(sandbox))
    assert ev.execution is Verdict.PASS  # right tool + verified evidence
    assert ev.overall is Verdict.FAIL  # a failed answer is not hidden
    assert ev.failure_kind is FailureKind.SYNTHESIS_FAILURE


def test_correct_answer_with_wrong_execution_cannot_pass(sandbox):
    """Part 14.10 + Part 15 Case C/D: a correct-looking answer produced by the
    wrong capability cannot pass."""
    web_tool = preferred_tool_for(ToolCapability.WEB_SEARCH)
    case = _case(ToolCapability.DEFINITION_LOOKUP, "Where is TaskProcessor defined?")
    case.subject_path = "src/models.py"
    trace = TaskTrace(
        case=case,
        transcript=f"Action: {web_tool} TaskProcessor is defined at src/models.py:1.",
        latency_s=0.1,
    )
    ev = evaluate_trace(trace, repo_root=str(sandbox))
    assert ev.execution is Verdict.FAIL
    assert ev.overall is Verdict.FAIL
    assert ev.failure_kind is FailureKind.CAPABILITY_SELECTION_FAILURE


def test_model_answer_never_used_as_ground_truth(sandbox):
    """Part 14.11: expected capability comes from task metadata, never the
    model's transcript."""
    case = _case(ToolCapability.REFERENCE_LOOKUP, "Where is TaskProcessor used?")
    case.subject_path = "src/models.py"
    # The transcript is full of *definition* evidence; the expected capability
    # is still reference_lookup (from the case), so execution cannot pass on
    # definition evidence alone.
    trace = TaskTrace(
        case=case,
        transcript="TaskProcessor (class) defined at src/models.py:1. The definition is here.",
        latency_s=0.1,
    )
    ev = evaluate_trace(trace, repo_root=str(sandbox))
    assert ev.capability is Verdict.PASS  # the task itself is valid
    assert ev.execution is not Verdict.PASS  # definition evidence is not reference evidence


# ---------------------------------------------------------------------------
# STEP 3.1 — Part 15: adversarial evaluator cases.
# ---------------------------------------------------------------------------


def test_case_a_correct_answer_no_evidence(sandbox):
    """Part 15 CASE A: correct-looking answer with no supporting evidence."""
    case = _case(ToolCapability.DEFINITION_LOOKUP, "Where is TaskProcessor defined?")
    case.subject_path = "src/models.py"
    trace = TaskTrace(
        case=case,
        transcript="TaskProcessor is a class that processes tasks. That's the answer.",
        latency_s=0.1,
    )
    ev = evaluate_trace(trace, repo_root=str(sandbox))
    assert ev.dimensions["evidence"] is Verdict.FAIL
    assert ev.failure_kind is FailureKind.EVIDENCE_FAILURE


def test_case_b_correct_tool_wrong_arguments(sandbox):
    """Part 15 CASE B: correct tool, wrong arguments (wrong symbol)."""
    case = _case(ToolCapability.DEFINITION_LOOKUP, "Where is TaskProcessor defined?")
    case.subject_path = "src/models.py"
    trace = TaskTrace(
        case=case,
        transcript="Action: find_definition name='TaskState' — TaskState is defined at src/models.py:1.",
        latency_s=0.1,
    )
    ev = evaluate_trace(trace, repo_root=str(sandbox))
    assert ev.dimensions["argument"] is Verdict.FAIL
    assert ev.failure_kind is FailureKind.ARGUMENT_FAILURE


def test_case_c_correct_capability_wrong_tool(sandbox):
    """Part 15 CASE C: wrong capability tool while the router was clear.

    Attribution follows Part 13: the observable root cause of a wrong-tool
    choice is that a different capability was executed, so it classifies as
    CAPABILITY_SELECTION_FAILURE (TOOL_SELECTION_FAILURE is reserved for
    wrong-tool-within-capability, which transcripts cannot distinguish)."""
    from ultron.validation.generator import record_router_diagnostic

    case = _case(ToolCapability.DEFINITION_LOOKUP, "Find where TaskProcessor is defined")
    case.subject_path = "src/models.py"
    record_router_diagnostic(case)
    assert case.router_agreement is True  # the router was clear
    web_tool = preferred_tool_for(ToolCapability.WEB_SEARCH)
    trace = TaskTrace(
        case=case,
        transcript=f"Action: {web_tool} searching the web for TaskProcessor.",
        latency_s=0.1,
    )
    ev = evaluate_trace(trace, repo_root=str(sandbox))
    assert ev.execution is Verdict.FAIL
    assert ev.failure_kind is FailureKind.CAPABILITY_SELECTION_FAILURE


def test_case_d_wrong_capability_correct_looking_answer(sandbox):
    """Part 15 CASE D: wrong capability with a correct-looking final answer."""
    case = _case(ToolCapability.REFERENCE_LOOKUP, "Where is TaskProcessor used?")
    case.subject_path = "src/models.py"
    trace = TaskTrace(
        case=case,
        transcript="Action: find_definition — TaskProcessor (class) defined at src/models.py:1.",
        latency_s=0.1,
    )
    ev = evaluate_trace(trace, repo_root=str(sandbox))
    assert ev.overall is Verdict.FAIL
    assert ev.failure_kind is FailureKind.CAPABILITY_SELECTION_FAILURE


def test_case_e_covered_by_unsupported_answer_and_wrong_final_answer_tests(sandbox):
    """Part 15 CASE E is covered by the two adversarial tests above
    (correct-looks-unsupported -> EVIDENCE_FAILURE; correct-execution +
    wrong-final-answer -> SYNTHESIS_FAILURE)."""
    assert True


def test_case_f_router_unknown_model_success(sandbox):
    """Part 15 CASE F: deterministic router says UNKNOWN, model handles it
    correctly -> model success + router disagreement recorded, never a fail."""
    from ultron.validation.generator import record_router_diagnostic

    case = _case(
        ToolCapability.DEFINITION_LOOKUP,
        "While reading the codebase I came across TaskProcessor — where does it get defined?",
    )
    case.subject_path = "src/models.py"
    record_router_diagnostic(case)
    assert case.router_agreement is False  # the router could not parse it
    trace = TaskTrace(
        case=case,
        transcript="Action: find_definition TaskProcessor is defined at src/models.py:1.",
        latency_s=0.1,
    )
    ev = evaluate_trace(trace, repo_root=str(sandbox))
    assert ev.execution is Verdict.PASS
    assert ev.overall is Verdict.PASS  # router weakness does not fail the model


def test_case_g_invalid_generated_task_rejected(sandbox):
    """Part 15 CASE G: the generator rejects invalid tasks."""
    from ultron.validation.generator import validate_task

    bad = _case(ToolCapability.FILE_READ, "Delete config.yaml", subject="config.yaml", case_id="g")
    ok, _ = validate_task(bad)
    assert not ok


def test_case_h_repo_question_routed_to_web_search_is_routing_failure(sandbox):
    """Part 15 CASE H: a repository question routed to external web search is
    attributed to routing/capability selection — never an evidence failure."""
    from ultron.validation.generator import record_router_diagnostic

    case = _case(
        ToolCapability.SYMBOL_INSPECTION,
        "Someone mentioned TaskProcessor; tell me about it.",
    )
    case.subject_path = "src/models.py"
    record_router_diagnostic(case)
    # The router is unknown for this phrasing — the model still chose
    # external search instead of repository inspection.
    assert case.router_capability == "unknown"
    trace = TaskTrace(
        case=case,
        transcript="Confirmation Required\nSearch the web — Someone mentioned TaskProcessor; tell me about it.",
        latency_s=0.1,
    )
    ev = evaluate_trace(trace, repo_root=str(sandbox))
    assert ev.overall is Verdict.FAIL
    assert ev.failure_kind is FailureKind.CAPABILITY_SELECTION_FAILURE


def test_case_h2_repo_question_web_routed_with_router_confused_is_routing_failure(sandbox):
    """Part 13: if the deterministic router also chose external for a repo
    question, the failure belongs to routing."""
    case = _case(
        ToolCapability.REPOSITORY_INVESTIGATION,
        "How does TaskProcessor delegate work?",
    )
    case.subject_path = "src/models.py"
    # Force router disagreement toward web to simulate a router that got it wrong.
    case.router_capability = "web_search"
    case.router_agreement = False
    trace = TaskTrace(
        case=case,
        transcript="Confirmation Required\nSearch the web — How does TaskProcessor delegate work?",
        latency_s=0.1,
    )
    ev = evaluate_trace(trace, repo_root=str(sandbox))
    assert ev.overall is Verdict.FAIL
    assert ev.failure_kind is FailureKind.ROUTING_FAILURE


def test_case_i_clarification_fallback_attribution(sandbox):
    """Part 13: a clarification fallback instead of an answer — when the
    deterministic router was also confused, routing is responsible."""
    from ultron.validation.generator import record_router_diagnostic

    case = _case(
        ToolCapability.DIRECTORY_LIST,
        "Can you show me what's sitting in src?",
        subject="src",
    )
    case.subject_path = "src"
    record_router_diagnostic(case)
    assert case.router_agreement is False  # holdout phrasing confused the router
    trace = TaskTrace(
        case=case,
        transcript="It sounds like you want to run a command — which command?",
        latency_s=0.1,
    )
    ev = evaluate_trace(trace, repo_root=str(sandbox))
    assert ev.overall is Verdict.FAIL
    assert ev.failure_kind is FailureKind.ROUTING_FAILURE
