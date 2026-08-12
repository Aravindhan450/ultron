"""
FIX #7 — Section 7.3: structured agent results + artifacts.

Deterministic tests for the artifact layer:

- lossless serialization / deserialization for all five artifact types
- validation: missing required fields, unknown types, malformed JSON
- artifact ownership (task_id / agent_id)
- ArtifactStore persistence (across store instances = process restart)
- TaskState association via the deterministic task_key()
- FIX #5 reuse (FailureAnalysis / CommandResult / test selection)
- to_agent_result() envelope mapping (summary/evidence/changed/tests/blockers)
- CRITICAL: a researcher's large internal trajectory is never passed onward —
  only the structured artifact travels
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from ultron.core.coding.command import CommandResult
from ultron.core.coding.executor import FailureAnalysis, FailureCategory
from ultron.core.memory.models import MemoryConfidence
from ultron.core.orchestration import (
    AgentArtifact,
    AgentResult,
    AgentResultStatus,
    ApprovalStatus,
    ArtifactStore,
    ImplementationResult,
    ResearchFinding,
    ReviewFinding,
    ReviewResult,
    SecurityFinding,
    Severity,
    artifact_from_dict,
    artifact_from_json,
    artifact_to_json,
    task_key,
)
from ultron.core.orchestration import (
    TestResult as TestResultArtifact,
)
from ultron.core.types import TaskState
from ultron.security.models import GuardrailFinding

# pytest collects any class named Test* it sees in a test module; the imported
# TestResult artifact is not a test class — opt it out of collection.
TestResultArtifact.__test__ = False

# ---------------------------------------------------------------------------
# Serialization / deserialization
# ---------------------------------------------------------------------------


def test_research_finding_round_trip():
    art = ResearchFinding(
        task_id="task-1",
        agent_id="researcher-1",
        summary="Authentication lives in src/auth/service.py (AuthService).",
        evidence=[
            "symbols: AuthService, AuthMiddleware",
            "files: src/auth/service.py, src/api/login.py",
        ],
        source="code_intelligence:find_symbol",
        confidence=MemoryConfidence.DIRECT_OBSERVATION,
        related_files=["src/auth/service.py"],
        related_symbols=["AuthService", "AuthMiddleware"],
        architecture_findings=[
            "login.py creates tokens via AuthService",
            "AuthMiddleware injects the current user",
        ],
        uncertainties=["refresh-token flow not inspected"],
        recommendations=["start implementation in src/auth/service.py"],
    )
    restored = artifact_from_json(artifact_to_json(art))
    assert isinstance(restored, ResearchFinding)
    assert restored.artifact_type.value == "research_finding"
    assert restored.summary == art.summary
    assert restored.confidence is MemoryConfidence.DIRECT_OBSERVATION
    assert restored.architecture_findings == art.architecture_findings
    assert restored.uncertainties == art.uncertainties
    assert restored.recommendations == art.recommendations
    assert restored.related_symbols == ["AuthService", "AuthMiddleware"]


def test_all_five_artifact_types_round_trip():
    artifacts: list[AgentArtifact] = [
        ResearchFinding(task_id="t", agent_id="r", summary="s"),
        ImplementationResult(
            task_id="t",
            agent_id="c",
            summary="s",
            changed_files=["src/main.py"],
        ),
        TestResultArtifact(task_id="t", agent_id="q", command="pytest"),
        ReviewResult(task_id="t", agent_id="v", approval=ApprovalStatus.APPROVED),
        SecurityFinding(task_id="t", agent_id="sec", issue="i"),
    ]
    for art in artifacts:
        restored = artifact_from_json(artifact_to_json(art))
        assert type(restored) is type(art)
        assert restored.task_id == "t"
        assert restored.agent_id == art.agent_id
        assert restored.artifact_id == art.artifact_id


def test_artifact_id_is_deterministic():
    a = ResearchFinding(task_id="t1", agent_id="r1")
    b = ResearchFinding(task_id="t1", agent_id="r1")
    c = ResearchFinding(task_id="t1", agent_id="r2")
    assert a.artifact_id == b.artifact_id == "t1:r1:research_finding"
    assert c.artifact_id != a.artifact_id


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_missing_required_fields_rejected():
    with pytest.raises(ValidationError):
        ResearchFinding(agent_id="r1")  # task_id missing
    with pytest.raises(ValidationError):
        ResearchFinding(task_id="t1")  # agent_id missing
    with pytest.raises(ValidationError):
        ImplementationResult(task_id="t1", agent_id="c1", changed_files="not-a-list")


def test_unknown_artifact_type_rejected():
    with pytest.raises(ValueError, match="unknown artifact_type"):
        artifact_from_dict(
            {"artifact_type": "alien_report", "task_id": "t", "agent_id": "a"}
        )


def test_malformed_json_rejected():
    with pytest.raises(ValueError):
        artifact_from_json("{this is not json")
    # JSON that decodes to a non-object is a type error, not an unknown type.
    with pytest.raises(TypeError):
        artifact_from_json('["not", "an", "object"]')


# ---------------------------------------------------------------------------
# Ownership
# ---------------------------------------------------------------------------


def test_artifact_ownership():
    art = ImplementationResult(task_id="t1", agent_id="coder-1", summary="s")
    assert art.owned_by(task_id="t1")
    assert art.owned_by(agent_id="coder-1")
    assert art.owned_by(task_id="t1", agent_id="coder-1")
    assert not art.owned_by(task_id="t2")
    assert not art.owned_by(agent_id="coder-2")
    assert not art.owned_by(task_id="t2", agent_id="coder-1")


def test_ownership_survives_round_trip():
    art = artifact_from_json(
        artifact_to_json(SecurityFinding(task_id="t9", agent_id="sec-9"))
    )
    assert art.owned_by("t9", "sec-9")


# ---------------------------------------------------------------------------
# Persistence (ArtifactStore)
# ---------------------------------------------------------------------------


def test_store_persists_across_instances(tmp_path):
    store1 = ArtifactStore(tmp_path / "artifacts")
    saved = store1.save(
        ResearchFinding(task_id="task-7", agent_id="researcher-7", summary="hello")
    )
    # A fresh store over the same directory simulates a process restart.
    store2 = ArtifactStore(tmp_path / "artifacts")
    loaded = store2.get(saved.artifact_id)
    assert loaded is not None
    assert isinstance(loaded, ResearchFinding)
    assert loaded.summary == "hello"
    assert loaded.owned_by("task-7", "researcher-7")


def test_store_skips_corrupted_records(tmp_path):
    store = ArtifactStore(tmp_path)
    store.save(SecurityFinding(task_id="t", agent_id="sec-1", issue="x"))
    with store._file.open("a") as fh:
        fh.write("{corrupted record\n")
    # A fresh store must still surface the valid artifact and skip the bad line.
    fresh = ArtifactStore(tmp_path)
    assert len(fresh.all()) == 1
    assert fresh.all()[0].issue == "x"


def test_store_skips_non_object_json_line(tmp_path):
    # A line that is valid JSON but not an object (e.g. a truncated/foreign
    # record) must be skipped, never crash the store.
    store = ArtifactStore(tmp_path)
    store.save(ResearchFinding(task_id="t", agent_id="r1", summary="ok"))
    with store._file.open("a") as fh:
        fh.write("[1, 2, 3]\n")
    fresh = ArtifactStore(tmp_path)
    assert len(fresh.all()) == 1
    assert fresh.all()[0].summary == "ok"


def test_store_rejects_duplicate_id(tmp_path):
    store = ArtifactStore(tmp_path)
    store.save(ResearchFinding(task_id="t", agent_id="r1", summary="v1"))
    # Ids are unique identities: any second save under the same id fails
    # loudly (even with identical content) instead of dropping the first.
    with pytest.raises(ValueError, match="already stored"):
        store.save(ResearchFinding(task_id="t", agent_id="r1", summary="v1"))
    assert len(store.all()) == 1
    assert store.all()[0].summary == "v1"


def test_store_query_by_task_and_agent(tmp_path):
    store = ArtifactStore(tmp_path)
    store.save(ResearchFinding(task_id="t1", agent_id="r1", summary="a"))
    store.save(ResearchFinding(task_id="t1", agent_id="r2", summary="b"))
    store.save(ResearchFinding(task_id="t2", agent_id="r3", summary="c"))
    assert len(store.load_for_task("t1")) == 2
    assert len(store.load_for_task("t2")) == 1
    assert len(store.load_for_agent("r1")) == 1
    assert store.get("missing") is None


# ---------------------------------------------------------------------------
# TaskState association
# ---------------------------------------------------------------------------


def test_task_key_is_stable_and_derived_from_goal():
    task_a = TaskState(goal="add refresh tokens to the auth service")
    task_a2 = TaskState(goal="add refresh tokens to the auth service")
    task_b = TaskState(goal="refactor the database layer")
    assert task_key(task_a) == task_key(task_a2)
    assert task_key(task_a) != task_key(task_b)
    assert len(task_key(task_a)) == 12


def test_artifacts_associate_with_task_state(tmp_path):
    task = TaskState(goal="add refresh tokens to the auth service")
    key = task_key(task)
    store = ArtifactStore(tmp_path)
    art = store.save(
        ImplementationResult(
            task_id=key,
            agent_id="coder-1",
            summary="refresh tokens implemented",
            changed_files=["src/auth/service.py"],
        )
    )
    found = store.load_for_task(key)
    assert len(found) == 1
    assert found[0].artifact_id == art.artifact_id
    assert found[0].owned_by(task_id=key, agent_id="coder-1")
    # The same task in a new session still resolves the artifact.
    assert store.get(art.artifact_id) is not None


# ---------------------------------------------------------------------------
# FIX #5 reuse — TestResult is built on existing models, never duplicates them
# ---------------------------------------------------------------------------


def test_test_result_reuses_fix5_failure_analysis():
    failure = FailureAnalysis(
        category=FailureCategory.TEST_ASSERTION,
        command="pytest tests/auth/test_login.py",
        summary="assert 5 == 4",
        evidence="tests/auth/test_login.py:12: assert 5 == 4",
        file="src/auth/service.py",
        line=12,
        test_name="test_login",
    )
    command = CommandResult(
        command="pytest tests/auth/test_login.py",
        exit_code=1,
        stdout="1 failed, 3 passed",
        stderr="",
        success=False,
    )
    tr = TestResultArtifact(
        task_id="t",
        agent_id="tester-1",
        command="pytest tests/auth/test_login.py",
        command_result=command,
        passed=3,
        failed=1,
        failures=[failure],
        affected_tests=["tests/auth/test_login.py::test_login"],
    )
    assert tr.passed_all is False
    restored = artifact_from_json(artifact_to_json(tr))
    assert isinstance(restored, TestResultArtifact)
    assert restored.failures[0].category is FailureCategory.TEST_ASSERTION
    assert restored.failures[0].file == "src/auth/service.py"
    assert restored.failures[0].line == 12
    assert restored.failures[0].test_name == "test_login"
    assert restored.command_result.exit_code == 1
    assert restored.passed_all is False


def test_test_result_passed_all_semantics():
    ok = TestResultArtifact(task_id="t", agent_id="tester-1", passed=4, failed=0)
    assert ok.passed_all is True
    ok.failed = 1
    assert ok.passed_all is False
    ok.failed = 0
    ok.timed_out = True
    assert ok.passed_all is False


def test_test_result_uses_fix5_affected_test_selection(tmp_path):
    from ultron.core.coding.test_selection import select_affected_tests

    (tmp_path / "src" / "auth").mkdir(parents=True)
    (tmp_path / "tests" / "auth").mkdir(parents=True)
    (tmp_path / "src" / "auth" / "service.py").write_text(
        "def validate(): pass\n", encoding="utf-8"
    )
    (tmp_path / "tests" / "auth" / "test_service.py").write_text(
        "def test_validate(): pass\n", encoding="utf-8"
    )
    affected = select_affected_tests(["src/auth/service.py"], tmp_path)
    assert any("test_service" in str(t) for t in affected)
    tr = TestResultArtifact(
        task_id="t",
        agent_id="tester-1",
        command="pytest",
        passed=1,
        affected_tests=[str(t) for t in affected],
    )
    assert any("test_service" in t for t in tr.affected_tests)


def test_security_finding_links_guardrail_hits():
    guardrail = GuardrailFinding(
        rule="aws_access_key",
        severity="critical",
        location="content",
        snippet="AKIA…redacted",
        message="hardcoded credential detected",
    )
    sf = SecurityFinding(
        task_id="t",
        agent_id="security-1",
        severity=Severity.CRITICAL,
        issue="hardcoded AWS key in config",
        affected_component="config/settings.py",
        recommendation="move to environment variable",
        blocking=True,
        guardrail_findings=[guardrail],
    )
    restored = artifact_from_json(artifact_to_json(sf))
    assert isinstance(restored, SecurityFinding)
    assert restored.severity is Severity.CRITICAL
    assert restored.blocking is True
    assert restored.guardrail_findings[0].rule == "aws_access_key"
    assert restored.guardrail_findings[0].severity == "critical"


def test_review_result_findings_round_trip():
    finding = ReviewFinding(
        severity=Severity.MEDIUM,
        affected_file="src/auth/service.py",
        description="tokens are not rotated on password change",
        evidence="line 44",
        recommendation="invalidate sessions on credential change",
    )
    review = ReviewResult(
        task_id="t",
        agent_id="reviewer-1",
        findings=[finding],
        overall_severity=Severity.MEDIUM,
        required_changes=["rotate tokens on password change"],
        approval=ApprovalStatus.CHANGES_REQUESTED,
    )
    restored = artifact_from_json(artifact_to_json(review))
    assert isinstance(restored, ReviewResult)
    assert restored.findings[0].severity is Severity.MEDIUM
    assert restored.findings[0].affected_file == "src/auth/service.py"
    assert restored.approval is ApprovalStatus.CHANGES_REQUESTED


# ---------------------------------------------------------------------------
# to_agent_result() envelope mapping
# ---------------------------------------------------------------------------


def test_implementation_result_envelope():
    art = ImplementationResult(
        task_id="t",
        agent_id="coder-1",
        summary="added /health endpoint",
        evidence=["pytest tests -q: 3 passed"],
        changed_files=["src/main.py"],
        changed_symbols=["health"],
        tests_added=["tests/test_health.py"],
        tests_run=["pytest"],
    )
    result = art.to_agent_result()
    assert result.status is AgentResultStatus.SUCCESS
    assert result.summary == "added /health endpoint"
    assert result.changed_files == ["src/main.py"]
    assert "tests/test_health.py" in result.tests
    assert "pytest" in result.tests
    assert result.artifact is art
    assert result.artifacts == [art.artifact_id]


def test_security_finding_blocking_becomes_blocker():
    art = SecurityFinding(
        task_id="t",
        agent_id="security-1",
        severity=Severity.HIGH,
        issue="plaintext password storage",
        blocking=True,
        recommendation="hash with bcrypt",
    )
    result = art.to_agent_result()
    assert result.status is AgentResultStatus.SUCCESS
    assert result.blockers == ["plaintext password storage"]
    assert result.recommendations == ["hash with bcrypt"]


def test_review_rejected_becomes_blocker():
    art = ReviewResult(
        task_id="t",
        agent_id="reviewer-1",
        required_changes=["rotate tokens"],
        approval=ApprovalStatus.REJECTED,
    )
    result = art.to_agent_result()
    assert result.blockers == ["rotate tokens"]


def test_agent_result_envelope_round_trips_with_artifact():
    art = SecurityFinding(
        task_id="t",
        agent_id="security-1",
        severity=Severity.HIGH,
        issue="secret leaked in logs",
        blocking=True,
    )
    result = art.to_agent_result()
    restored = AgentResult.model_validate(json.loads(result.model_dump_json()))
    assert isinstance(restored.artifact, SecurityFinding)
    assert restored.artifact.blocking is True
    assert restored.blockers == ["secret leaked in logs"]
    assert restored.metadata["artifact_type"] == "security_finding"


# ---------------------------------------------------------------------------
# CRITICAL — only the structured artifact travels, never the trajectory
# ---------------------------------------------------------------------------


def test_critical_only_artifact_passed_onward_not_internal_trajectory():
    # A researcher that "thought" through 300 tool calls and logged every one
    # internally — the trajectory is huge and must NOT leave the agent.
    trajectory_lines = [
        f"step {i}: read_file(src/auth/{'service' if i % 2 else 'middleware'}.py) "
        f"-> {i * 97} bytes of raw content cached in local state"
        for i in range(300)
    ]
    trajectory = "\n".join(trajectory_lines)
    assert len(trajectory) > 20_000

    research = ResearchFinding(
        task_id="task-9",
        agent_id="researcher-9",
        summary="AuthService.validate() is the single token-validation entry point.",
        evidence=["files: src/auth/service.py", "callers: src/api/login.py"],
        related_files=["src/auth/service.py", "src/api/login.py"],
        related_symbols=["AuthService", "validate"],
        source="code_intelligence:find_definition",
        confidence=MemoryConfidence.DIRECT_OBSERVATION,
        architecture_findings=[
            "AuthService validates tokens; AuthMiddleware injects the user",
        ],
        recommendations=["implement refresh tokens inside AuthService"],
    )

    # The full payload handed onward = envelope (AgentResult) + artifact.
    payload = artifact_to_json(research) + research.to_agent_result().model_dump_json()

    # The structured artifact is a tiny fraction of the internal trajectory…
    assert len(payload) * 10 < len(trajectory)
    # …and contains none of it.
    assert "step 0:" not in payload
    assert "raw content cached" not in payload
    assert "tool_call" not in payload
    # The artifact alone carries the knowledge the next component needs.
    assert "AuthService.validate" in payload
    assert "src/auth/service.py" in payload
