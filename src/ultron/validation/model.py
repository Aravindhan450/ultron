"""Data model for the capability validation framework.

Pure data — no logic that redefines capabilities or tools.  A
:class:`CapabilityTestCase` references the canonical ``ToolCapability`` and
its ``CapabilityContract``; a :class:`TaskTrace` records everything observed
for one execution; :class:`Evaluation` holds the three-layer verdicts
(capability truth / execution-evidence truth / final-answer truth).

STEP 3.1 refinements:

- task validity is decided by repository/capability ground truth, never by
  the deterministic intent router (router output is recorded as a
  DIAGNOSTIC on the case);
- one semantic entity is tracked separately from its surface forms
  (``entity_id`` vs ``surface_form``);
- development and holdout tasks use structurally different generation
  strategies (``strategy``) with independent splits/seeds;
- evaluation separates capability truth, execution/evidence truth, and
  final-answer truth, and the overall verdict is explicit (Part 11).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ultron.core.capabilities.contracts import CapabilityContract, contract_for
from ultron.core.tools.definitions import ToolCapability


class TestSplit(str, Enum):
    """Where a test case came from.

    Development tasks may be used while fixing implementation; holdout tasks
    are generated independently after implementation is frozen (STEP 4) and
    must never become production routing rules.
    """

    DEVELOPMENT = "development"
    HOLDOUT = "holdout"


class GenerationStrategy(str, Enum):
    """How a task was worded (structurally different per split).

    DEVELOPMENT uses direct, explicit-capability wording; HOLDOUT uses
    indirect, conversational, multi-clause wording with implicit capability.
    The strategies are separate template families, not just different seeds.
    """

    DEVELOPMENT_DIRECT = "development.direct"
    HOLDOUT_INDIRECT = "holdout.indirect"


class TestSource(str, Enum):
    """Whether a test case was generated dynamically or fixed."""

    GENERATED = "generated"
    FIXED = "fixed"


class Difficulty(str, Enum):
    """Relative difficulty of a generated task."""

    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class Verdict(str, Enum):
    """Outcome of one evaluation dimension or a whole trace."""

    PASS = "PASS"
    PARTIAL = "PARTIAL"
    FAIL = "FAIL"
    UNRESOLVED = "UNRESOLVED"


class FailureKind(str, Enum):
    """Root-cause classification of a failed task.

    STEP 3.1 refines attribution (Part 13): a wrong capability selection by
    the model is distinguished from a routing failure and from a synthesis
    failure, using the (expected, deterministic-router, model) triple.
    """

    INTENT_FAILURE = "intent_failure"
    CAPABILITY_SELECTION_FAILURE = "capability_selection_failure"
    TOOL_SELECTION_FAILURE = "tool_selection_failure"
    ARGUMENT_FAILURE = "argument_failure"
    ROUTING_FAILURE = "routing_failure"
    SECURITY_FAILURE = "security_failure"
    EXECUTION_FAILURE = "execution_failure"
    EVIDENCE_FAILURE = "evidence_failure"
    INVESTIGATION_FAILURE = "investigation_failure"
    SYNTHESIS_FAILURE = "synthesis_failure"
    MODEL_LIMITATION = "model_limitation"
    ENVIRONMENT_FAILURE = "environment_failure"
    EVALUATION_FAILURE = "evaluation_failure"
    UNKNOWN = "unknown"


@dataclass
class CapabilityTestCase:
    """One validation task: a natural-language request for a capability.

    References the canonical vocabulary — never redefines capabilities,
    tools, schemas, risk, or aliases.  Mutable so the generator can record
    the deterministic-router diagnostic after construction (diagnostic data
    is never part of the task's semantic identity).
    """

    case_id: str
    capability: ToolCapability
    task: str
    expected_capability: ToolCapability
    # The entity/query the task is about (arbitrary repository subject).
    subject: str | None
    # Canonical entity identity — one entity, many surface forms.
    entity_id: str | None = None
    # The exact surface form rendered into the task (e.g. "taskstate").
    surface_form: str | None = None
    # Kind of the subject ("class"/"function"/"file"/"directory"/None).
    subject_kind: str | None = None
    # Where the subject lives in the repository (rel path), when applicable.
    subject_path: str | None = None
    # Generation strategy (development direct vs holdout indirect).
    strategy: GenerationStrategy = GenerationStrategy.DEVELOPMENT_DIRECT
    # Deterministic expected intent when the request is routable, else None.
    expected_intent: str | None = None
    expected_behavior: str | None = None
    evidence_requirements: tuple[str, ...] = ()
    difficulty: Difficulty = Difficulty.BASIC
    test_source: TestSource = TestSource.GENERATED
    split: TestSplit = TestSplit.DEVELOPMENT
    template_id: str | None = None
    # --- deterministic-router DIAGNOSTIC (never a validity gate) -----------
    router_capability: str | None = None  # resolved capability value or state
    router_agreement: bool | None = None  # router resolved to expected?

    @property
    def contract(self) -> CapabilityContract | None:
        """The behavioral contract for this capability (canonical)."""
        return contract_for(self.capability)


@dataclass
class TaskTrace:
    """Complete observed record of one model execution."""

    case: CapabilityTestCase
    # Raw transcript window from the CLI (ANSI-stripped).
    transcript: str
    latency_s: float
    # Observed signals parsed from the transcript.
    detected_tool_hint: str | None = None
    security_decision: str | None = None
    evaluation: Evaluation | None = None


@dataclass
class Evaluation:
    """Three-layer evaluation of one trace (Part 7-11 of STEP 3.1)."""

    # Layer A — capability truth: what the task actually requires.
    capability: Verdict = Verdict.UNRESOLVED
    # Layer B — execution/evidence truth: appropriate operation + valid evidence.
    execution: Verdict = Verdict.UNRESOLVED
    # Layer C — final-answer truth: relevance, grounding, completeness, calibration.
    answer: Verdict = Verdict.UNRESOLVED
    # Final-answer sub-dimensions (Part 10).
    answer_dimensions: dict[str, Verdict] = field(default_factory=dict)
    # Overall verdict (explicit aggregation rule, Part 11).
    overall: Verdict = Verdict.UNRESOLVED
    # Fine-grained detail dimensions (kept as evidence, not the verdict).
    dimensions: dict[str, Verdict] = field(default_factory=dict)
    # The model's capability as inferred from its observed tool (diagnostic).
    model_capability: str | None = None
    # Root-cause classification when the task failed.
    failure_kind: FailureKind | None = None
    # Human-readable notes (evidence observed, reasons).
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"{self.overall.value} "
            f"(capability={self.capability.value} execution={self.execution.value} "
            f"answer={self.answer.value})"
        )
