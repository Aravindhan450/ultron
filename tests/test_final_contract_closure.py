"""tests.test_final_contract_closure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Deterministic regression test suite for the Final Policy & Verification Contract Closure:
- Test Group A: Read-Only Criterion Specificity (A1 - A9)
- Test Group B: Criterion-Specific Level 5 (B1 - B4)
- Test Group C: Batch Confirmation Lifecycle (C1 - C9)
"""

import json

import pytest

from ultron.core.intelligence.parallel_tools import execute_batch, run_tool_batch
from ultron.core.intelligence.verification_evidence import (
    criterion_is_satisfied,
)
from ultron.core.tools.registry import coerce_tool_arguments, get_tool
from ultron.core.types import (
    AuthorityLevel,
    Capability,
    ChatMessage,
    EvidenceLevel,
    ExecutionPolicy,
    Role,
    TaskState,
    TaskType,
)
from ultron.main import execute_pending_action


class FakeEngine:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.call_count = 0

    async def generate(self, messages: list[dict], **kwargs) -> str:
        self.call_count += 1
        if not self._responses:
            return ""
        return self._responses.pop(0)


# ===========================================================================
# TEST GROUP A — READ-ONLY CRITERION SPECIFICITY (A1 - A9)
# ===========================================================================


def test_a1_read_file_verifies_relevant_code_inspected():
    """A1: Execute read_file('auth.py') -> relevant_code_inspected is VERIFIED."""
    task = TaskState(goal="Investigate login reliability", task_type=TaskType.DEBUGGING)
    task.record_tool_execution(
        "read_file", target="auth.py", success=True, detail="def login(): return False"
    )

    ok, level = criterion_is_satisfied("relevant_code_inspected", task)
    assert ok is True
    assert level == EvidenceLevel.LEVEL_1_CREATED


def test_a2_read_file_only_does_not_verify_root_cause_diagnosis():
    """A2: Execute only read_file('auth.py') -> root_cause_diagnosis is NOT VERIFIED."""
    task = TaskState(goal="Investigate login reliability", task_type=TaskType.DEBUGGING)
    task.record_tool_execution(
        "read_file", target="auth.py", success=True, detail="def login(): return False"
    )

    ok, level = criterion_is_satisfied("root_cause_diagnosis", task)
    assert ok is False
    assert level == EvidenceLevel.LEVEL_0_GENERATED


def test_a3_read_file_only_does_not_verify_failure_mechanism_identified():
    """A3: Execute only read_file('auth.py') -> failure_mechanism_identified is NOT VERIFIED."""
    task = TaskState(goal="Investigate login reliability", task_type=TaskType.DEBUGGING)
    task.record_tool_execution(
        "read_file", target="auth.py", success=True, detail="def login(): return False"
    )

    ok, level = criterion_is_satisfied("failure_mechanism_identified", task)
    assert ok is False
    assert level == EvidenceLevel.LEVEL_0_GENERATED


def test_a4_model_saying_verified_without_execution_evidence_not_verified():
    """A4: Model says 'The root cause has been verified.' with insufficient evidence -> NOT VERIFIED."""
    task = TaskState(goal="Investigate login reliability", task_type=TaskType.DEBUGGING)
    task.context.append(
        ChatMessage(
            role=Role.ASSISTANT, content="The root cause has been verified."
        )
    )

    ok, level = criterion_is_satisfied("root_cause_diagnosis", task)
    assert ok is False
    assert level == EvidenceLevel.LEVEL_0_GENERATED


def test_a5_multiple_observations_with_causal_relationship_verifies_root_cause():
    """A5: Multiple relevant observations + causal relationship -> root_cause_diagnosis is VERIFIED."""
    task = TaskState(goal="Investigate login reliability", task_type=TaskType.DEBUGGING)
    task.record_tool_execution(
        "read_file",
        target="auth.py",
        success=True,
        detail="def login(): return check_password()",
    )
    task.record_tool_execution(
        "read_file",
        target="crypto.py",
        success=True,
        detail="def check_password(): return hash(a) == hash(b)",
    )
    task.context.append(
        ChatMessage(
            role=Role.ASSISTANT,
            content=(
                "The root cause in auth.py is that login fails because check_password in "
                "crypto.py uses mismatched salt hashing."
            ),
        )
    )

    ok, level = criterion_is_satisfied("root_cause_diagnosis", task)
    assert ok is True
    assert level == EvidenceLevel.LEVEL_2_EXECUTED


def test_a6_investigation_evidence_without_reported_diagnosis_not_verified():
    """A6: Investigation evidence exists but no diagnosis in assistant response -> diagnosis_reported is NOT VERIFIED."""
    task = TaskState(goal="Investigate login reliability", task_type=TaskType.DEBUGGING)
    task.record_tool_execution(
        "read_file", target="auth.py", success=True, detail="file content"
    )
    task.context.append(
        ChatMessage(
            role=Role.ASSISTANT,
            content="I have read the auth.py file and reviewed its lines.",
        )
    )

    ok, level = criterion_is_satisfied("diagnosis_reported", task)
    assert ok is False
    assert level == EvidenceLevel.LEVEL_0_GENERATED


def test_a7_grounded_diagnosis_based_on_recorded_observations_verified():
    """A7: Grounded diagnosis based on recorded observations -> diagnosis_reported is VERIFIED."""
    task = TaskState(goal="Investigate login reliability", task_type=TaskType.DEBUGGING)
    task.record_tool_execution(
        "read_file", target="auth.py", success=True, detail="def login(): return False"
    )
    task.context.append(
        ChatMessage(
            role=Role.ASSISTANT,
            content=(
                "Diagnosis: In auth.py, the login function fails because the hardcoded "
                "return False bypasses credential validation."
            ),
        )
    )

    ok, level = criterion_is_satisfied("diagnosis_reported", task)
    assert ok is True
    assert level == EvidenceLevel.LEVEL_2_EXECUTED


def test_a8_proposed_fix_without_supporting_diagnosis_not_verified():
    """A8: Proposed fix without supporting diagnosis/evidence -> proposed_fix_strategy is NOT VERIFIED."""
    task = TaskState(goal="Investigate login reliability", task_type=TaskType.DEBUGGING)
    task.context.append(
        ChatMessage(
            role=Role.ASSISTANT,
            content="We should fix this by updating the configuration and replacing the token handler.",
        )
    )

    ok, level = criterion_is_satisfied("proposed_fix_strategy", task)
    assert ok is False
    assert level == EvidenceLevel.LEVEL_0_GENERATED


def test_a9_grounded_diagnosis_followed_by_concrete_fix_strategy_verified():
    """A9: Grounded diagnosis + concrete remediation strategy -> proposed_fix_strategy is VERIFIED."""
    task = TaskState(goal="Investigate login reliability", task_type=TaskType.DEBUGGING)
    task.record_tool_execution(
        "read_file", target="auth.py", success=True, detail="def login(): return False"
    )
    task.record_tool_execution(
        "read_file", target="session.py", success=True, detail="session creation logic"
    )
    task.context.append(
        ChatMessage(
            role=Role.ASSISTANT,
            content=(
                "The root cause in auth.py is that login returns False due to invalid salt. "
                "To fix this, modify auth.py to remediate the hash check and replace the hardcoded "
                "return value with proper password comparison."
            ),
        )
    )

    ok, level = criterion_is_satisfied("proposed_fix_strategy", task)
    assert ok is True
    assert level == EvidenceLevel.LEVEL_1_CREATED


# ===========================================================================
# TEST GROUP B — LEVEL 5 CRITERION-SPECIFICITY (B1 - B4)
# ===========================================================================


def test_b1_read_only_file_inspection_not_level_5():
    """B1: Read-only inspection verifies relevant_code_inspected at Level 1, not Level 5."""
    task = TaskState(goal="Investigate login reliability", task_type=TaskType.DEBUGGING)
    task.record_tool_execution(
        "read_file", target="auth.py", success=True, detail="code content"
    )

    ok, level = criterion_is_satisfied("relevant_code_inspected", task)
    assert ok is True
    assert level == EvidenceLevel.LEVEL_1_CREATED
    assert level != EvidenceLevel.LEVEL_5_VERIFIED

    _, level_diag = criterion_is_satisfied("root_cause_diagnosis", task)
    assert level_diag != EvidenceLevel.LEVEL_5_VERIFIED


def test_b2_deterministic_test_result_allows_level_5_for_test_criteria():
    """B2: Deterministic test execution allows Level 5 for test-result criteria."""
    task = TaskState(goal="Run tests and verify", task_type=TaskType.DEBUGGING)
    task.record_tool_execution(
        "run_command",
        target="pytest tests/",
        success=True,
        detail="10 passed in 0.15s",
    )

    ok, level = criterion_is_satisfied("tests_pass", task)
    assert ok is True
    assert level == EvidenceLevel.LEVEL_5_VERIFIED


def test_b3_test_result_does_not_make_root_cause_diagnosis_level_5():
    """B3: Passing test result does NOT promote root_cause_diagnosis to Level 5."""
    task = TaskState(goal="Fix bug and verify", task_type=TaskType.DEBUGGING)
    task.record_tool_execution(
        "read_file", target="auth.py", success=True, detail="code content"
    )
    task.record_tool_execution(
        "read_file", target="session.py", success=True, detail="code content"
    )
    task.record_tool_execution(
        "run_command", target="pytest tests/", success=True, detail="passed"
    )
    task.context.append(
        ChatMessage(
            role=Role.ASSISTANT,
            content="The root cause in auth.py was traced because password validation failed.",
        )
    )

    ok, level = criterion_is_satisfied("root_cause_diagnosis", task)
    assert ok is True
    # Stays capped at LEVEL_2_EXECUTED, never LEVEL_5_VERIFIED
    assert level == EvidenceLevel.LEVEL_2_EXECUTED
    assert level != EvidenceLevel.LEVEL_5_VERIFIED


def test_b4_model_statement_verified_never_creates_level_5():
    """B4: Model statement 'verified' without deterministic evidence never creates Level 5."""
    task = TaskState(goal="Verify authentication", task_type=TaskType.DEBUGGING)
    task.context.append(
        ChatMessage(
            role=Role.ASSISTANT,
            content="All tests pass! Verified with Level 5 empirical evidence.",
        )
    )

    ok, level = criterion_is_satisfied("tests_pass", task)
    assert ok is False
    assert level == EvidenceLevel.LEVEL_0_GENERATED


# ===========================================================================
# TEST GROUP C — BATCH CONFIRMATION LIFECYCLE (C1 - C9)
# ===========================================================================


@pytest.fixture(autouse=True)
def _sandbox(tmp_path, monkeypatch):
    """Ensure tmp_path is allowed by SecurityBoundary for tool execution."""
    from ultron.core.tools import paths as tools_paths

    monkeypatch.setattr(tools_paths, "ALLOWED_BASE_DIR", tmp_path)
    return tmp_path


def test_c1_read_only_policy_allows_reads_blocks_writes(tmp_path):
    """C1: Under READ_ONLY policy, batch executes read_file and blocks write_file."""
    f = tmp_path / "test.txt"
    f.write_text("initial")

    policy = ExecutionPolicy(
        is_read_only=True,
        allowed_capabilities=frozenset({Capability.READ_FILES}),
        denied_capabilities=frozenset({Capability.WRITE_FILES}),
        mutation_authority=AuthorityLevel.FORBIDDEN,
    )
    calls = [
        {"tool": "read_file", "arguments": {"file_path": str(f)}},
        {
            "tool": "write_file",
            "arguments": {"file_path": str(f), "content": "modified"},
        },
    ]

    gated, _ = execute_batch(calls, policy=policy)
    assert gated[0]["status"] == "ok"
    assert "initial" in gated[0]["result"]
    assert gated[1]["status"] == "blocked"
    assert "Policy violation" in gated[1]["result"]
    assert f.read_text() == "initial"


def test_c2_read_only_policy_blocks_run_command():
    """C2: Under READ_ONLY policy, batch blocks run_command."""
    policy = ExecutionPolicy(
        is_read_only=True,
        allowed_capabilities=frozenset({Capability.READ_FILES}),
        denied_capabilities=frozenset({Capability.RUN_COMMANDS}),
        execution_authority=AuthorityLevel.FORBIDDEN,
    )
    calls = [{"tool": "run_command", "arguments": {"command": "echo test"}}]

    gated, _ = execute_batch(calls, policy=policy)
    assert gated[0]["status"] == "blocked"
    assert "Policy violation" in gated[0]["result"]


def test_c3_to_c7_requires_confirmation_batch_lifecycle(tmp_path):
    """C3 - C7: Full PendingAction lifecycle for batch write_file."""
    target_file = tmp_path / "config.json"
    target_file.write_text('{"debug": false}')

    policy = ExecutionPolicy(
        is_read_only=False,
        allowed_capabilities=frozenset({Capability.WRITE_FILES}),
        mutation_authority=AuthorityLevel.REQUIRES_CONFIRMATION,
    )

    calls = [
        {
            "tool": "write_file",
            "arguments": {
                "file_path": str(target_file),
                "content": '{"debug": true}',
            },
        }
    ]

    # C3: Batch returns ChatMessage with PendingAction, action does not execute
    res = run_tool_batch(json.dumps(calls), policy=policy)
    assert isinstance(res, ChatMessage)
    assert res.pending_action is not None
    assert res.pending_action.action_type in ("overwrite_file", "write_file", "run_tool_batch")

    # C4: Before confirmation, execution count = 0 (file unchanged)
    assert target_file.read_text() == '{"debug": false}'

    # C5: After approval, execute_pending_action executes exactly once
    import anyio

    action_result = anyio.run(execute_pending_action, res.pending_action)
    assert "Error" not in action_result
    assert target_file.read_text() == '{"debug": true}'

    # C6: After task resume, execution count remains 1
    assert target_file.read_text() == '{"debug": true}'


def test_c7_denial_leaves_file_unchanged(tmp_path):
    """C7: User denies confirmation -> execution count = 0, file unchanged."""
    target_file = tmp_path / "config.json"
    target_file.write_text('{"debug": false}')

    policy = ExecutionPolicy(
        is_read_only=False,
        allowed_capabilities=frozenset({Capability.WRITE_FILES}),
        mutation_authority=AuthorityLevel.REQUIRES_CONFIRMATION,
    )
    calls = [
        {
            "tool": "write_file",
            "arguments": {
                "file_path": str(target_file),
                "content": '{"debug": true}',
            },
        }
    ]

    res = run_tool_batch(json.dumps(calls), policy=policy)
    assert isinstance(res, ChatMessage)
    # User denies -> execute_pending_action is NOT called
    assert target_file.read_text() == '{"debug": false}'


def test_c8_mixed_batch_no_inherited_authorization(tmp_path):
    """C8: Mixed batch: read_file executes, write_file awaits confirmation, run_command blocked."""
    f = tmp_path / "data.txt"
    f.write_text("original")

    policy = ExecutionPolicy(
        is_read_only=False,
        allowed_capabilities=frozenset({Capability.READ_FILES, Capability.WRITE_FILES}),
        denied_capabilities=frozenset({Capability.RUN_COMMANDS}),
        mutation_authority=AuthorityLevel.REQUIRES_CONFIRMATION,
        execution_authority=AuthorityLevel.FORBIDDEN,
    )

    calls = [
        {"tool": "read_file", "arguments": {"file_path": str(f)}},
        {
            "tool": "write_file",
            "arguments": {"file_path": str(f), "content": "updated"},
        },
        {"tool": "run_command", "arguments": {"command": "rm -rf /"}},
    ]

    gated, _ = execute_batch(calls, policy=policy)
    # read_file ran
    assert gated[0]["status"] == "ok"
    assert "original" in gated[0]["result"]
    # write_file needs confirmation
    assert gated[1]["status"] == "confirm"
    # run_command blocked
    assert gated[2]["status"] == "blocked"

    # Full batch run returns ChatMessage with PendingAction for the confirm call
    res = run_tool_batch(json.dumps(calls), policy=policy)
    assert isinstance(res, ChatMessage)
    assert res.pending_action is not None
    # File is still unchanged before approval
    assert f.read_text() == "original"


def test_c9_batch_arguments_canonical_coercion(tmp_path):
    """C9: Batch arguments pass through same canonical coercion as direct invocation."""
    f = tmp_path / "coercion_test.txt"
    f.write_text("hello world")

    # write_file takes overwrite: bool = False. Passing string "true" is coerced to True.
    func = get_tool("write_file")
    coerced = coerce_tool_arguments(
        func, {"file_path": str(f), "content": "replaced", "overwrite": "true"}
    )
    assert coerced["overwrite"] is True

    # _safe_run (used for batch execution) invokes coerce_tool_arguments identically
    from ultron.core.intelligence.parallel_tools import _safe_run

    result = _safe_run(
        "write_file",
        {"file_path": str(f), "content": "replaced", "overwrite": "true"},
    )
    assert "Error" not in result
    assert f.read_text() == "replaced"
