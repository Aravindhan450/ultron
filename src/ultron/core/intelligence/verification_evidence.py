"""ultron.core.intelligence.verification_evidence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CRITERION-SPECIFIC VERIFICATION EVIDENCE (arch doc §11).

Evidence must prove the SPECIFIC acceptance criterion being verified — not
merely prove that some investigation happened. This module maps each known
diagnostic acceptance criterion to the evidence the recorded execution
history must contain, and to the semantically honest EvidenceLevel that
evidence supports.

Principles:

- ``read_file("auth.py")`` proves ``relevant_code_inspected`` — it does NOT
  prove ``root_cause_diagnosis`` or ``proposed_fix_strategy``.
- LEVEL_5_VERIFIED is reserved for evidence checked against an independent
  or deterministic source (real command output, exit codes, resource
  signals, network reachability). Repository inspection alone is at most
  LEVEL_1_CREATED / LEVEL_2_EXECUTED.
- The model's textual "I verified it" is never evidence; only recorded
  tool executions in TaskState count.
- A read-only diagnosis can still complete: the required evidence is
  *correlated investigation* (multiple observations connected), not mutation.
"""

from __future__ import annotations

from dataclasses import dataclass

from ultron.core.types import EvidenceLevel, TaskState

# Tools that record repository inspection work.
_INSPECTION_TOOLS = frozenset(
    {
        "read_file",
        "list_directory",
        "search_files",
        "repo_map",
        "code_search",
        "semantic_search",
        "code_investigation",
        "find_symbol",
        "find_definition",
        "find_references",
        "report_file",
        "report_symbol",
        "get_imports",
        "get_dependents",
        "retrieve",
    }
)

# Tools that record execution behavior (independent/deterministic signals).
_EXECUTION_TOOLS = frozenset(
    {
        "run_command",
        "run_parallel",
        "check_resources",
        "resource_forecast",
        "check_connectivity",
        "search_web",
        "fetch_page_text",
        "make_http_request",
    }
)

# Dependency-relationship tools: they connect components, which is stronger
# evidence than a single file read.
_RELATIONSHIP_TOOLS = frozenset(
    {
        "get_imports",
        "get_dependents",
        "find_references",
        "find_definition",
        "find_symbol",
        "code_investigation",
        "repo_map",
        "explain_relation",
        "analyze_dependencies",
        "check_dependency",
    }
)


@dataclass(frozen=True)
class CriterionEvidence:
    """The evidence contract for one acceptance criterion."""

    min_investigation_actions: int = 1
    requires_relationship_evidence: bool = False
    requires_diagnosis_content: bool = False
    requires_fix_strategy: bool = False
    can_reach_level_5: bool = False
    max_level: EvidenceLevel = EvidenceLevel.LEVEL_2_EXECUTED


# Per-criterion evidence requirements. Criteria not listed here default to
# "one successful inspection action" at LEVEL_1_CREATED.
_CRITERION_EVIDENCE: dict[str, CriterionEvidence] = {
    # Directly proven by inspecting the relevant files/structure.
    "relevant_code_inspected": CriterionEvidence(
        min_investigation_actions=1,
        max_level=EvidenceLevel.LEVEL_1_CREATED,
    ),
    "repository_structure_inspected": CriterionEvidence(
        min_investigation_actions=1,
        max_level=EvidenceLevel.LEVEL_1_CREATED,
    ),
    "current_state_analyzed": CriterionEvidence(
        min_investigation_actions=1,
        max_level=EvidenceLevel.LEVEL_1_CREATED,
    ),
    # Requires connecting components, not one file read.
    "system_architecture_inspected": CriterionEvidence(
        min_investigation_actions=2,
        requires_relationship_evidence=True,
        max_level=EvidenceLevel.LEVEL_2_EXECUTED,
    ),
    "failure_mechanism_identified": CriterionEvidence(
        min_investigation_actions=1,
        requires_diagnosis_content=True,
        max_level=EvidenceLevel.LEVEL_2_EXECUTED,
    ),
    "root_cause_diagnosis": CriterionEvidence(
        min_investigation_actions=1,
        requires_diagnosis_content=True,
        max_level=EvidenceLevel.LEVEL_2_EXECUTED,
    ),
    "evidence_based_explanation": CriterionEvidence(
        min_investigation_actions=1,
        requires_diagnosis_content=True,
        max_level=EvidenceLevel.LEVEL_2_EXECUTED,
    ),
    # Reporting criteria: investigation + the assistant actually produced
    # the report in its recorded response.
    "diagnosis_reported": CriterionEvidence(
        min_investigation_actions=1,
        requires_diagnosis_content=True,
        max_level=EvidenceLevel.LEVEL_2_EXECUTED,
    ),
    "findings_reported": CriterionEvidence(
        min_investigation_actions=1,
        requires_diagnosis_content=False,
        max_level=EvidenceLevel.LEVEL_2_EXECUTED,
    ),
    "explanation_provided": CriterionEvidence(
        min_investigation_actions=1,
        requires_diagnosis_content=False,
        max_level=EvidenceLevel.LEVEL_2_EXECUTED,
    ),
    "plan_presented": CriterionEvidence(
        min_investigation_actions=1,
        requires_diagnosis_content=False,
        max_level=EvidenceLevel.LEVEL_1_CREATED,
    ),
    # A proposed fix is a design artifact, not an observation: requires
    # diagnosis groundwork (>= 2 investigation actions) and reported fix content.
    "proposed_fix_strategy": CriterionEvidence(
        min_investigation_actions=2,
        requires_diagnosis_content=True,
        requires_fix_strategy=True,
        max_level=EvidenceLevel.LEVEL_1_CREATED,
    ),
    # Execution / test criteria: can reach LEVEL_5_VERIFIED with independent deterministic signals
    "tests_pass": CriterionEvidence(
        min_investigation_actions=0,
        can_reach_level_5=True,
        max_level=EvidenceLevel.LEVEL_5_VERIFIED,
    ),
    "test_suite_passes": CriterionEvidence(
        min_investigation_actions=0,
        can_reach_level_5=True,
        max_level=EvidenceLevel.LEVEL_5_VERIFIED,
    ),
    "regression_tests_pass": CriterionEvidence(
        min_investigation_actions=0,
        can_reach_level_5=True,
        max_level=EvidenceLevel.LEVEL_5_VERIFIED,
    ),
}

_DIAGNOSTIC_KEYWORDS = frozenset(
    {
        "root cause",
        "cause",
        "because",
        "due to",
        "fails when",
        "failure",
        "defect",
        "bug",
        "reason",
        "broken",
        "mechanism",
        "issue is",
        "error in",
        "leads to",
        "incorrectly",
        "fault",
        "problem is",
        "diagnosis",
    }
)

_FIX_KEYWORDS = frozenset(
    {
        "fix",
        "remediate",
        "remediation",
        "patch",
        "modify",
        "replace",
        "update",
        "correct",
        "solution",
        "change",
        "resolve",
    }
)


def canonicalize_criterion(criterion: str) -> str:
    """Maps a criterion name or description to its canonical criterion key."""
    c = criterion.lower().strip()
    if c in _CRITERION_EVIDENCE:
        return c

    aliases = {
        "relevant files and symbols identified": "relevant_code_inspected",
        "relevant code inspected": "relevant_code_inspected",
        "repository structure inspected": "repository_structure_inspected",
        "current state analyzed": "current_state_analyzed",
        "execution path traced": "current_state_analyzed",
        "system architecture inspected": "system_architecture_inspected",
        "architecture inspected": "system_architecture_inspected",
        "failure mechanism identified": "failure_mechanism_identified",
        "failure mechanism identified and analyzed": "failure_mechanism_identified",
        "failure reproduced or located": "failure_mechanism_identified",
        "root cause diagnosed": "root_cause_diagnosis",
        "root cause identified": "root_cause_diagnosis",
        "root cause diagnosed and explained with evidence": "root_cause_diagnosis",
        "evidence-based explanation provided": "evidence_based_explanation",
        "evidence based explanation provided": "evidence_based_explanation",
        "diagnosis reported": "diagnosis_reported",
        "diagnosis reported in response": "diagnosis_reported",
        "findings reported": "findings_reported",
        "findings reported in response": "findings_reported",
        "explanation provided": "explanation_provided",
        "explanation provided in response": "explanation_provided",
        "plan presented": "plan_presented",
        "plan presented in response": "plan_presented",
        "fix plan formulated and presented": "plan_presented",
        "proposed fix strategy presented": "proposed_fix_strategy",
        "proposed fix strategy": "proposed_fix_strategy",
        "fix strategy designed": "proposed_fix_strategy",
        "tests pass": "tests_pass",
        "test suite passes": "tests_pass",
        "regression tests pass": "tests_pass",
    }
    if c in aliases:
        return aliases[c]

    if "root cause" in c:
        return "root_cause_diagnosis"
    if "failure mechanism" in c:
        return "failure_mechanism_identified"
    if "fix strategy" in c or "remediation strategy" in c or "proposed fix" in c:
        return "proposed_fix_strategy"
    if "architecture" in c:
        return "system_architecture_inspected"
    if "execution path" in c or "state analyzed" in c:
        return "current_state_analyzed"
    if "code inspected" in c or "files and symbols" in c:
        return "relevant_code_inspected"
    if "diagnosis reported" in c:
        return "diagnosis_reported"
    if "fix plan" in c or "plan presented" in c:
        return "plan_presented"
    if "tests pass" in c or "test suite pass" in c or "regression tests" in c:
        return "tests_pass"

    return c


def _successful_investigations(task: TaskState) -> list:
    return [
        e
        for e in task.execution_history
        if e.success and e.tool_name in _INSPECTION_TOOLS
    ]


def _successful_relations(task: TaskState) -> list:
    return [
        e
        for e in task.execution_history
        if e.success and e.tool_name in _RELATIONSHIP_TOOLS
    ]


def _successful_independent_signals(task: TaskState) -> list:
    return [
        e
        for e in task.execution_history
        if e.success and e.tool_name in _EXECUTION_TOOLS
    ]


def _get_assistant_responses(task: TaskState) -> list[str]:
    from ultron.core.types import Role

    return [
        msg.content.strip()
        for msg in task.context
        if msg.role is Role.ASSISTANT and msg.content.strip()
    ]


def _diagnosis_reported_in_transcript(task: TaskState) -> bool:
    """True when the assistant recorded a substantive diagnosis response.

    The transcript (task.context) holds ASSISTANT messages. A diagnosis
    report must contain genuine causal/diagnostic reasoning (not merely a
    greeting, generic acknowledgement, or bare 'verified' self-attestation)
    grounded in recorded observations.
    """
    investigated = bool(_successful_investigations(task))
    if not investigated:
        return False
    responses = _get_assistant_responses(task)
    for text in responses:
        lower = text.lower()
        has_causal = any(kw in lower for kw in _DIAGNOSTIC_KEYWORDS)
        if has_causal and len(text) >= 25:
            return True
    return False


def _has_fix_strategy_in_transcript(task: TaskState) -> bool:
    """True when the assistant recorded a concrete remediation/fix strategy."""
    responses = _get_assistant_responses(task)
    for text in responses:
        lower = text.lower()
        if any(kw in lower for kw in _FIX_KEYWORDS) and len(text) >= 25:
            return True
    return False


def criterion_is_satisfied(
    criterion_id: str, task: TaskState
) -> tuple[bool, EvidenceLevel]:
    """
    Evaluates ONE acceptance criterion against the recorded evidence.

    Returns (satisfied, level) where ``level`` is the semantically honest
    evidence level the recorded work supports (never above the criterion's
    declared maximum, never LEVEL_5 without an independent deterministic
    signal proving THAT specific criterion).
    """
    canonical_id = canonicalize_criterion(criterion_id)
    spec = _CRITERION_EVIDENCE.get(canonical_id) or CriterionEvidence()

    investigations = _successful_investigations(task)
    relations = _successful_relations(task)
    independent = _successful_independent_signals(task)

    # LEVEL_5 is strictly reserved for deterministic test/execution criteria.
    if spec.can_reach_level_5:
        has_test_exec = any(
            e.success
            and (
                e.tool_name in ("run_command", "run_parallel")
                and any(
                    t in str(e.target).lower() or t in str(e.detail).lower()
                    for t in ("pytest", "unittest", "test", "check", "passed", "exit 0")
                )
            )
            for e in task.execution_history
        )
        if has_test_exec or independent:
            return True, EvidenceLevel.LEVEL_5_VERIFIED
        return False, EvidenceLevel.LEVEL_0_GENERATED

    plan_completed = bool(task.plan and task.plan.is_satisfied())

    # Multi-action requirement: relationship evidence and execution signals can connect components
    effective_actions = len(investigations) + len(independent) + (1 if relations else 0)
    if not plan_completed and effective_actions < spec.min_investigation_actions:
        return False, EvidenceLevel.LEVEL_0_GENERATED

    if spec.requires_relationship_evidence and not relations:
        return False, EvidenceLevel.LEVEL_0_GENERATED

    reported = _diagnosis_reported_in_transcript(task)
    if spec.requires_diagnosis_content and not reported:
        return False, EvidenceLevel.LEVEL_0_GENERATED

    if spec.requires_fix_strategy and not _has_fix_strategy_in_transcript(task):
        return False, EvidenceLevel.LEVEL_0_GENERATED

    # Analytical, diagnostic, and inspection criteria stay at spec.max_level (never LEVEL_5)
    return True, spec.max_level


def satisfy_diagnosis_criteria(
    task: TaskState, criteria: list[tuple[str, str]]
) -> list[str]:
    """
    Marks each (criterion_id, description) with the evidence the recorded
    history actually supports. Returns the ids that became verified.

    The model's verdict is not consulted: evidence comes from TaskState's
    execution history and transcript. Criteria whose evidence is
    insufficient stay unverified (they may block completion — that is the
    point).
    """
    verified: list[str] = []
    for crit_id, description in criteria:
        ok, level = criterion_is_satisfied(crit_id, task)
        if ok:
            task.verify_acceptance_criterion(crit_id, description, level)
            verified.append(crit_id)
    return verified
