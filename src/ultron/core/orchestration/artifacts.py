"""
ultron.core.orchestration.artifacts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Structured agent results + artifacts (Fix #7, section 7.3).

Agents communicate through *structured artifacts*, never by dumping their
internal conversation or trajectory into the next component. This module
defines the artifact types:

- :class:`ResearchFinding` — what a research agent learned about a codebase
- :class:`ImplementationResult` — what a coding agent changed
- :class:`TestResult` — what a test/QA run produced (reuses Fix #5's
  :class:`~ultron.core.coding.command.CommandResult` and
  :class:`~ultron.core.coding.executor.FailureAnalysis` — no second
  test-result system)
- :class:`ReviewResult` — a review's findings and approval decision
- :class:`SecurityFinding` — a security analysis finding (links to the
  security layer's :class:`~ultron.security.models.GuardrailFinding`)

Every artifact carries provenance: ``task_id`` + ``agent_id`` (ownership),
``timestamp``, ``summary``, ``evidence``, ``source``, ``confidence``
(reusing Fix #6's :class:`~ultron.core.memory.models.MemoryConfidence` —
LLM guesses never look like direct observation), ``related_files`` /
``related_symbols``, and free-form ``metadata``.

Serialization is lossless pydantic JSON (:func:`artifact_to_json` /
:func:`artifact_from_json`) with strict artifact-type dispatch — an unknown
or malformed artifact fails loudly, never silently. :class:`ArtifactStore`
persists artifacts as JSONL (one record per line) inside a directory, so
artifacts survive a process restart without persisting any internal agent
trajectory.

:meth:`AgentArtifact.to_agent_result` folds an artifact into the section-7.1
:class:`~ultron.core.orchestration.models.AgentResult` envelope (summary,
evidence, changed files, tests, recommendations/blockers) — the structured
communication protocol between agents.

TaskState association: TaskState has no id yet (the supervisor section will
introduce real task ids); until then :func:`task_key` derives a stable,
deterministic key from the task goal so artifacts can be grouped per task.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from ultron.core.coding.command import CommandResult
from ultron.core.coding.executor import FailureAnalysis
from ultron.core.memory.models import MemoryConfidence
from ultron.security.models import GuardrailFinding

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ArtifactType(str, Enum):
    """The five standardized artifact kinds agents can produce."""

    RESEARCH_FINDING = "research_finding"
    IMPLEMENTATION_RESULT = "implementation_result"
    TEST_RESULT = "test_result"
    REVIEW_RESULT = "review_result"
    SECURITY_FINDING = "security_finding"


class Severity(str, Enum):
    """Coarse severity labels shared by review and security findings."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ApprovalStatus(str, Enum):
    """Approval decision attached to a review result."""

    NOT_REVIEWED = "not_reviewed"
    APPROVED = "approved"
    CHANGES_REQUESTED = "changes_requested"
    REJECTED = "rejected"


# ---------------------------------------------------------------------------
# Base artifact
# ---------------------------------------------------------------------------


class AgentArtifact(BaseModel):
    """
    Base class for every structured agent artifact.

    All artifacts carry the shared provenance/context fields; specialized
    artifacts add their own fields. ``task_id`` + ``agent_id`` are required
    (an artifact is always owned by one agent run inside one task) and
    ``artifact_id`` defaults to a deterministic ``task:agent:type`` key.

    ``confidence`` reuses Fix #6's :class:`MemoryConfidence` so that a fact
    observed directly from source code is never mistaken for an LLM guess.
    """

    artifact_type: ArtifactType
    task_id: str
    agent_id: str
    artifact_id: str = ""
    summary: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    evidence: list[str] = Field(default_factory=list)
    source: str = ""  # provenance: tool name / file / command / "llm_inference"
    confidence: MemoryConfidence | None = None
    related_files: list[str] = Field(default_factory=list)
    related_symbols: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, context: Any) -> None:
        if not self.artifact_id:
            self.artifact_id = (
                f"{self.task_id}:{self.agent_id}:{self.artifact_type.value}"
            )

    # -- ownership -----------------------------------------------------------

    def owned_by(self, task_id: str | None = None, agent_id: str | None = None) -> bool:
        """True when this artifact belongs to the given task and/or agent."""
        return (task_id is None or self.task_id == task_id) and (
            agent_id is None or self.agent_id == agent_id
        )

    # -- envelope conversion --------------------------------------------------

    def _result_extras(self) -> dict[str, Any]:
        """Subclass hook: fields to fold into the AgentResult envelope."""
        return {}

    def to_agent_result(self, status: str | None = None) -> Any:
        """
        Folds this artifact into a section-7.1 :class:`AgentResult`.

        The envelope carries the compact protocol (summary, evidence,
        changed files, tests, recommendations/blockers) while ``result.artifact``
        retains the full structured artifact. The agent's internal trajectory
        is never included — only what the artifact records.
        """
        from ultron.core.orchestration.models import AgentResult, AgentResultStatus

        result_status = (
            AgentResultStatus(status) if status else AgentResultStatus.SUCCESS
        )
        return AgentResult(
            status=result_status,
            summary=self.summary or (
                f"{self.artifact_type.value} produced by agent {self.agent_id}"
            ),
            artifacts=[self.artifact_id],
            evidence=list(self.evidence),
            **self._result_extras(),
            metadata={
                "artifact_type": self.artifact_type.value,
                "task_id": self.task_id,
                "agent_id": self.agent_id,
                "source": self.source,
                "confidence": self.confidence.value if self.confidence else None,
            },
            artifact=self,
        )


# ---------------------------------------------------------------------------
# Research finding
# ---------------------------------------------------------------------------


class ResearchFinding(AgentArtifact):
    """What a research agent learned about a repository or task."""

    artifact_type: Literal[ArtifactType.RESEARCH_FINDING] = ArtifactType.RESEARCH_FINDING
    architecture_findings: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)

    def _result_extras(self) -> dict[str, Any]:
        return {"recommendations": list(self.recommendations)}


# ---------------------------------------------------------------------------
# Implementation result
# ---------------------------------------------------------------------------


class ImplementationResult(AgentArtifact):
    """What a coding agent changed and validated."""

    artifact_type: Literal[ArtifactType.IMPLEMENTATION_RESULT] = (
        ArtifactType.IMPLEMENTATION_RESULT
    )
    changed_files: list[str] = Field(default_factory=list)
    changed_symbols: list[str] = Field(default_factory=list)
    tests_added: list[str] = Field(default_factory=list)
    tests_run: list[str] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)

    def _result_extras(self) -> dict[str, Any]:
        # dedupe while preserving order
        tests: list[str] = []
        for t in [*self.tests_added, *self.tests_run]:
            if t not in tests:
                tests.append(t)
        return {
            "changed_files": list(self.changed_files),
            "tests": tests,
            "blockers": list(self.blockers),
        }


# ---------------------------------------------------------------------------
# Test result — reuses Fix #5, never duplicates it
# ---------------------------------------------------------------------------


class TestResult(AgentArtifact):
    """
    Outcome of one test run.

    Reuses Fix #5's models directly: :class:`CommandResult` for the raw
    execution and :class:`FailureAnalysis` (category, file/line, test name,
    repair hint) for each failure. ``passed`` / ``failed`` / ``skipped`` are
    the deterministic counters; :attr:`passed_all` is the pass predicate.
    """

    artifact_type: Literal[ArtifactType.TEST_RESULT] = ArtifactType.TEST_RESULT
    command: str = ""
    command_result: CommandResult | None = None
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    failures: list[FailureAnalysis] = Field(default_factory=list)
    affected_tests: list[str] = Field(default_factory=list)
    timed_out: bool = False

    @property
    def passed_all(self) -> bool:
        return self.failed == 0 and not self.timed_out

    def _result_extras(self) -> dict[str, Any]:
        return {
            "tests": [self.command] if self.command else [],
            "blockers": [f.summary for f in self.failures[:5]],
        }


# ---------------------------------------------------------------------------
# Review result
# ---------------------------------------------------------------------------


class ReviewFinding(BaseModel):
    """One item in a code review."""

    severity: Severity = Severity.INFO
    affected_file: str = ""
    description: str = ""
    evidence: str = ""
    recommendation: str = ""


class ReviewResult(AgentArtifact):
    """A review agent's findings and approval decision."""

    artifact_type: Literal[ArtifactType.REVIEW_RESULT] = ArtifactType.REVIEW_RESULT
    findings: list[ReviewFinding] = Field(default_factory=list)
    overall_severity: Severity = Severity.INFO
    required_changes: list[str] = Field(default_factory=list)
    approval: ApprovalStatus = ApprovalStatus.NOT_REVIEWED

    def _result_extras(self) -> dict[str, Any]:
        extras: dict[str, Any] = {
            "recommendations": list(self.required_changes)
        }
        if self.approval is ApprovalStatus.REJECTED:
            extras["blockers"] = list(self.required_changes)
        return extras


# ---------------------------------------------------------------------------
# Security finding
# ---------------------------------------------------------------------------


class SecurityFinding(AgentArtifact):
    """A security analysis finding (optionally backed by guardrail hits)."""

    artifact_type: Literal[ArtifactType.SECURITY_FINDING] = ArtifactType.SECURITY_FINDING
    severity: Severity = Severity.INFO
    issue: str = ""
    affected_component: str = ""
    recommendation: str = ""
    blocking: bool = False
    guardrail_findings: list[GuardrailFinding] = Field(default_factory=list)

    def _result_extras(self) -> dict[str, Any]:
        extras: dict[str, Any] = {
            "recommendations": [self.recommendation] if self.recommendation else []
        }
        if self.blocking:
            extras["blockers"] = [self.issue]
        return extras


# ---------------------------------------------------------------------------
# Discriminated union for polymorphic deserialization
# ---------------------------------------------------------------------------


AgentArtifactUnion = Annotated[
    ResearchFinding | ImplementationResult | TestResult | ReviewResult | SecurityFinding,
    Field(discriminator="artifact_type"),
]
"""
Polymorphic artifact type: deserializes to the concrete subclass based on
``artifact_type``. Used by :class:`AgentResult.artifact` so an envelope
restored from JSON yields the original artifact type, not the base class.
"""


# ---------------------------------------------------------------------------
# Serialization — strict, type-dispatched, lossless
# ---------------------------------------------------------------------------

# Dispatch table keyed by ArtifactType value. The values are referenced via
# the enum (not the pydantic field default, which is not a class attribute).
_ARTIFACT_CLASSES: dict[str, type[AgentArtifact]] = {
    ArtifactType.RESEARCH_FINDING.value: ResearchFinding,
    ArtifactType.IMPLEMENTATION_RESULT.value: ImplementationResult,
    ArtifactType.TEST_RESULT.value: TestResult,
    ArtifactType.REVIEW_RESULT.value: ReviewResult,
    ArtifactType.SECURITY_FINDING.value: SecurityFinding,
}


def artifact_to_json(artifact: AgentArtifact) -> str:
    """Lossless JSON serialization (pydantic round-trip preserves all fields)."""
    return artifact.model_dump_json()


def artifact_from_dict(data: dict[str, Any]) -> AgentArtifact:
    """
    Deserializes an artifact from a plain dict.

    Raises :class:`ValueError` for an unknown ``artifact_type`` and lets
    pydantic's :class:`ValidationError` propagate for malformed payloads
    (missing required fields, wrong types). Never silently mis-types.
    """
    artifact_type = data.get("artifact_type")
    cls = _ARTIFACT_CLASSES.get(artifact_type)
    if cls is None:
        raise ValueError(f"unknown artifact_type: {artifact_type!r}")
    return cls.model_validate(data)


def artifact_from_json(text: str) -> AgentArtifact:
    """Deserializes an artifact from JSON; malformed JSON raises ValueError."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed artifact JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise TypeError("artifact JSON must decode to an object")
    return artifact_from_dict(data)


# ---------------------------------------------------------------------------
# Task association (pre-supervisor deterministic key)
# ---------------------------------------------------------------------------


def task_key(task_state: Any) -> str:
    """
    Stable, deterministic task id derived from a TaskState's goal.

    TaskState has no id field yet — the supervisor section will introduce
    real task ids. Until then the goal is the stable identity, so artifacts
    produced for the same task can be grouped reliably (and reproduced).
    """
    goal = getattr(task_state, "goal", "")
    return hashlib.sha256(str(goal).encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Artifact store — JSONL persistence
# ---------------------------------------------------------------------------


class ArtifactStore:
    """
    Filesystem persistence for structured artifacts.

    One JSON record per line in ``artifacts.jsonl`` under ``directory``.
    :meth:`save` appends; reads build an in-memory index lazily and skip
    corrupted lines (a broken record — invalid JSON, a non-object payload,
    an unknown type, a validation failure — never fails the store). A fresh
    store instance over the same directory sees everything previously saved:
    artifacts survive a process restart, while internal agent trajectories
    are never persisted.

    Uniqueness: ``artifact_id`` is the identity. Saving an id that is
    already stored raises ``ValueError`` rather than silently dropping the
    earlier artifact — callers must supply an explicit unique id when an
    agent produces multiple artifacts of the same type. The store is
    thread-safe.
    """

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._file = self.directory / "artifacts.jsonl"
        self._index: dict[str, AgentArtifact] | None = None
        self._lock = threading.Lock()

    def _ensure_index(self) -> None:
        if self._index is not None:
            return
        self._index = {}
        if self._file.exists():
            for line in self._file.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    artifact = artifact_from_json(line)
                except (ValueError, TypeError) as exc:  # corrupted record — skip
                    logger.warning("skipping corrupted artifact record: %s", exc)
                    continue
                self._index[artifact.artifact_id] = artifact

    def save(self, artifact: AgentArtifact) -> AgentArtifact:
        """Persists (and indexes) one artifact; returns it for chaining.

        Raises ``ValueError`` when ``artifact.artifact_id`` is already
        stored — ids are unique identities, never silently merged.
        """
        with self._lock:
            self._ensure_index()
            if artifact.artifact_id in self._index:
                raise ValueError(
                    f"artifact_id {artifact.artifact_id!r} is already stored; "
                    "pass an explicit unique artifact_id"
                )
            with self._file.open("a", encoding="utf-8") as fh:
                fh.write(artifact_to_json(artifact) + "\n")
            self._index[artifact.artifact_id] = artifact
            return artifact

    def get(self, artifact_id: str) -> AgentArtifact | None:
        with self._lock:
            self._ensure_index()
            return self._index.get(artifact_id)

    def load_for_task(self, task_id: str) -> list[AgentArtifact]:
        """All artifacts owned by one task (in save order)."""
        with self._lock:
            self._ensure_index()
            return [a for a in self._index.values() if a.task_id == task_id]

    def load_for_agent(self, agent_id: str) -> list[AgentArtifact]:
        """All artifacts produced by one agent run (in save order)."""
        with self._lock:
            self._ensure_index()
            return [a for a in self._index.values() if a.agent_id == agent_id]

    def all(self) -> list[AgentArtifact]:
        with self._lock:
            self._ensure_index()
            return list(self._index.values())
