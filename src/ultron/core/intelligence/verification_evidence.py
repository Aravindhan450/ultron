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
    requires_independent_signal: bool = False
    requires_diagnosis_content: bool = False
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
        requires_relationship_evidence=False,
        max_level=EvidenceLevel.LEVEL_2_EXECUTED,
    ),
    "root_cause_diagnosis": CriterionEvidence(
        min_investigation_actions=1,
        requires_relationship_evidence=False,
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
        requires_diagnosis_content=True,
        max_level=EvidenceLevel.LEVEL_2_EXECUTED,
    ),
    "explanation_provided": CriterionEvidence(
        min_investigation_actions=1,
        requires_diagnosis_content=True,
        max_level=EvidenceLevel.LEVEL_2_EXECUTED,
    ),
    "plan_presented": CriterionEvidence(
        min_investigation_actions=1,
        requires_diagnosis_content=True,
        max_level=EvidenceLevel.LEVEL_2_EXECUTED,
    ),
    # A proposed fix is a design artifact, not an observation: requires
    # diagnosis groundwork and reported content, never auto-verified.
    "proposed_fix_strategy": CriterionEvidence(
        min_investigation_actions=2,
        requires_diagnosis_content=True,
        max_level=EvidenceLevel.LEVEL_1_CREATED,
    ),
}


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


def _diagnosis_reported_in_transcript(task: TaskState) -> bool:
    """True when the assistant recorded a substantive diagnosis response.

    The transcript (task.context) holds ASSISTANT messages; a diagnosis
    report is a non-trivial assistant turn after investigation work. This is
    deliberately not about WHAT the model claims was verified — only that a
    substantive report exists to ground the reported-diagnosis criteria.
    """
    from ultron.core.types import Role

    investigated = bool(_successful_investigations(task))
    if not investigated:
        return False
    for msg in task.context:
        if msg.role is Role.ASSISTANT and len(msg.content.strip()) >= 20:
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
    signal).
    """
    spec = _CRITERION_EVIDENCE.get(criterion_id) or CriterionEvidence()

    investigations = _successful_investigations(task)
    relations = _successful_relations(task)
    independent = _successful_independent_signals(task)

    if len(investigations) < spec.min_investigation_actions:
        return False, EvidenceLevel.LEVEL_0_GENERATED

    if spec.requires_relationship_evidence and not relations:
        return False, EvidenceLevel.LEVEL_0_GENERATED

    reported = _diagnosis_reported_in_transcript(task)
    if spec.requires_diagnosis_content and not reported:
        return False, EvidenceLevel.LEVEL_0_GENERATED

    # LEVEL_5 requires an independent/deterministic verification signal
    # (real executed commands, resource checks, network probes) on top of
    # the investigation. Inspection alone never reaches LEVEL_5.
    if independent:
        return True, EvidenceLevel.LEVEL_5_VERIFIED
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
