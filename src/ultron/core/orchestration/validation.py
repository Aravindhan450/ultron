"""
ultron.core.orchestration.validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Deterministic orchestration validation layer (Fix #7, validation).

Agent claims are NOT proof. This module checks — deterministically, with no
LLM, no execution, and no side effects — whether an agent operation actually
obeyed the orchestration contract:

    Agent -> Execution -> AgentResult -> Validation -> ValidationResult
                                                       -> Supervisor / TaskState

:class:`OrchestrationValidator` runs a set of modular checks over a
:class:`ValidationContext` (the *record* of one execution: agent state,
result, delegation, task state, artifacts, recorded tool uses, workspace
and scope):

- lifecycle        — every recorded transition is legal; result status
                     matches the lifecycle status (no corruption)
- delegation       — the delegation is well-formed and registered
- permissions      — every recorded tool use was whitelisted and its
                     category level allowed (deterministic; the security
                     boundary's verdicts are inspected, never re-run)
- budget           — used steps/tool-calls within the configured limits
- timeout          — elapsed execution time within the configured limit
- result           — the AgentResult satisfies its schema per status
- artifacts        — schema, provenance, agent ownership, task association
- workspace scope  — changed files stay inside the workspace AND the
                     agent's allowed scope (never reverted — only reported)
- task state       — the result does not contradict the authoritative
                     TaskState (e.g. a blocked/cancelled task "succeeding")
- completion claim — a SUCCESS claim is only justified when required
                     evidence exists and the task's requirements/plan are
                     satisfied (FALSE_COMPLETION_CLAIM otherwise)
- ownership        — delegation belongs to the task; run belongs to the
                     delegation; no cross-task contamination

Each failed check becomes a :class:`ValidationViolation` (stable
:class:`ViolationCode` + severity + agent/task/delegation ids + evidence)
or a :class:`ValidationWarning` for non-fatal findings. The aggregate
:class:`ValidationResult` reports PASS / WARNING / FAIL / BLOCKED and never
mutates anything — validation is pure and idempotent. The Supervisor
decides what to do with failures; this module only reports them.

No existing VerificationResult/ValidationResult model existed in the
repository at the time of writing, so these models are new; they reuse the
artifact layer's :class:`~ultron.core.orchestration.artifacts.Severity` and
the standard :class:`~ultron.core.types.TaskState` / agent lifecycle models
rather than duplicating them.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ultron.core.logging import get_logger
from ultron.core.orchestration.artifacts import (
    AgentArtifact,
    AgentArtifactUnion,
    Severity,
    TestResult,
)
from ultron.core.orchestration.delegation import DelegationRequest, task_key
from ultron.core.orchestration.lifecycle import AgentStatus, can_transition
from ultron.core.orchestration.models import (
    AgentResult,
    AgentResultStatus,
    AgentState,
)
from ultron.core.orchestration.permissions import classify_tool
from ultron.core.orchestration.registry import DEFAULT_REGISTRY, AgentRegistry
from ultron.core.types import TaskState, TaskStatus, ToolExecution
from ultron.security import Decision

logger = get_logger("ultron.orchestration.validation")

# ---------------------------------------------------------------------------
# Violation codes — stable, machine-readable
# ---------------------------------------------------------------------------


class ViolationCode(str, Enum):
    """Stable machine-readable codes for validation violations."""

    INVALID_LIFECYCLE_TRANSITION = "invalid_lifecycle_transition"
    UNAUTHORIZED_TOOL = "unauthorized_tool"
    UNAUTHORIZED_FILE_ACCESS = "unauthorized_file_access"
    BUDGET_EXCEEDED = "budget_exceeded"
    TIMEOUT_EXCEEDED = "timeout_exceeded"
    INVALID_AGENT_RESULT = "invalid_agent_result"
    INVALID_ARTIFACT = "invalid_artifact"
    ARTIFACT_OWNERSHIP_VIOLATION = "artifact_ownership_violation"
    ARTIFACT_TASK_MISMATCH = "artifact_task_mismatch"
    TASK_STATE_CONFLICT = "task_state_conflict"
    WORKSPACE_SCOPE_VIOLATION = "workspace_scope_violation"
    FALSE_COMPLETION_CLAIM = "false_completion_claim"
    TEST_CLAIM_CONTRADICTION = "test_claim_contradiction"


class ValidationStatus(str, Enum):
    """Aggregate outcome of a validation run."""

    PASS = "pass"
    WARNING = "warning"  # non-fatal findings only
    FAIL = "fail"  # one or more violations
    BLOCKED = "blocked"  # critical violations (state corruption / security)


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


class ValidationCheck(BaseModel):
    """One deterministic invariant check."""

    name: str
    passed: bool
    severity: Severity = Severity.INFO
    message: str = ""
    evidence: list[str] = Field(default_factory=list)
    code: ViolationCode | None = None  # set on failed HIGH/CRITICAL checks
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_prompt_line(self, max_len: int = 200) -> str:
        mark = "PASS" if self.passed else "FAIL"
        head = f"[{mark}] {self.name}"
        if self.code is not None:
            head = f"{head} ({self.code.value})"
        if self.message:
            return f"{head}: {self.message[: max_len - len(head) - 2]}"
        return head[:max_len]


class ValidationViolation(BaseModel):
    """A diagnosed contract violation — enough to act on."""

    code: ViolationCode
    severity: Severity
    message: str
    agent_id: str = ""
    task_id: str = ""
    delegation_id: str = ""
    evidence: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ValidationWarning(BaseModel):
    """A non-fatal finding (missing provenance, at-budget-limit, ...)."""

    name: str
    severity: Severity = Severity.LOW
    message: str = ""
    evidence: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ValidationResult(BaseModel):
    """Aggregate outcome of a validation run (read-only report)."""

    valid: bool
    status: ValidationStatus
    checks: list[ValidationCheck] = Field(default_factory=list)
    violations: list[ValidationViolation] = Field(default_factory=list)
    warnings: list[ValidationWarning] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    summary: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def passed_checks(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    def to_prompt_line(self, max_len: int = 240) -> str:
        head = f"[{self.status.value}] {self.summary}"
        return head[:max_len]


# ---------------------------------------------------------------------------
# Validation context — the read-only record of one execution
# ---------------------------------------------------------------------------


class ValidationContext(BaseModel):
    """
    The record of one agent execution that validation inspects.

    All fields are optional; validation only checks what is provided and
    reports what it cannot verify as warnings. Nothing here is mutated by
    validation (pure, idempotent).

    - ``agent_state`` — the AgentState of the run (lifecycle + metadata)
    - ``result`` — the AgentResult produced by the run
    - ``delegation`` — the DelegationRequest that dispatched the run
    - ``task_state`` — the authoritative TaskState
    - ``artifacts`` — structured artifacts produced/submitted
    - ``tool_uses`` — recorded tool executions (``ToolExecution`` records)
    - ``workspace`` — the workspace root path (for scope checks)
    - ``allowed_scope`` — path prefixes the agent was allowed to touch
    - ``required_evidence`` — evidence categories a SUCCESS claim must show
      (e.g. ``changed_files``, ``tests``, ``verification``)
    - ``test_results`` — Fix #5/7.3 TestResult artifacts (for contradiction
      detection)
    - ``claims_completion`` — set when the result explicitly claims task
      completion (also auto-detected from common completion phrasing)
    - ``now`` — clock override for deterministic timeout checks
    """

    agent_state: AgentState | None = None
    result: AgentResult | None = None
    delegation: DelegationRequest | None = None
    task_state: TaskState | None = None
    artifacts: list[AgentArtifactUnion] = Field(default_factory=list)
    tool_uses: list[ToolExecution] = Field(default_factory=list)
    workspace: str = ""
    allowed_scope: list[str] = Field(default_factory=list)
    required_evidence: list[str] = Field(default_factory=list)
    test_results: list[TestResult] = Field(default_factory=list)
    claims_completion: bool = False
    now: datetime | None = None


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

#: lifecycle status -> the only AgentResultStatus it may carry
_STATUS_TO_RESULT: dict[AgentStatus, AgentResultStatus] = {
    AgentStatus.COMPLETED: AgentResultStatus.SUCCESS,
    AgentStatus.FAILED: AgentResultStatus.FAILED,
    AgentStatus.BLOCKED: AgentResultStatus.BLOCKED,
    AgentStatus.CANCELLED: AgentResultStatus.CANCELLED,
    AgentStatus.WAITING: AgentResultStatus.NEEDS_INPUT,
}

#: SUCCESS-blocking task statuses — a task in these states must never be
#: reported as successfully completed by an agent.
_NON_SUCCESS_TASK_STATUSES = frozenset(
    {TaskStatus.TASK_FAILED, TaskStatus.TASK_BLOCKED}
)

#: conservative completion phrasing; a SUCCESS summary matching this counts
#: as a completion claim (the explicit context flag is authoritative).
#: Word boundaries prevent substring false positives ("completely",
#: "incomplete", "undone"); negated claims ("not all done", "not
#: everything is done") are excluded; the comma in "done, everything"
#: disambiguates the completion form from "done everything I can".
_COMPLETION_CLAIM_RE = re.compile(
    r"(?:task\s+(?:is\s+)?\b(?:complete(?:d)?|done)\b|"
    r"(?<!not )(?<!no )(?<!never )everything\s+(?:\bworks\b|is\s+done\b)|"
    r"(?<!not )(?<!no )(?<!never )all\s+done\b|"
    r"\bdone,\s+everything\b)",
    re.IGNORECASE,
)


class OrchestrationValidator:
    """
    Deterministic, read-only validation of one agent execution record.

    Individual ``validate_*`` methods run one family of checks and return
    ``list[ValidationCheck]``; :meth:`validate` runs them all and aggregates
    into a :class:`ValidationResult`. No execution, no mutation, no security
    re-negotiation — security verdicts are inspected (recorded tool uses
    and the frozen permission profile), never re-issued.
    """

    def __init__(self, registry: AgentRegistry | None = None) -> None:
        self.registry = registry or DEFAULT_REGISTRY

    # -- public entry point ----------------------------------------------------

    def validate(self, ctx: ValidationContext) -> ValidationResult:
        """Runs every check family and aggregates the result (pure)."""
        checks: list[ValidationCheck] = []
        checks.extend(self.validate_lifecycle(ctx))
        checks.extend(self.validate_delegation(ctx))
        checks.extend(self.validate_permissions(ctx))
        checks.extend(self.validate_budget(ctx))
        checks.extend(self.validate_timeout(ctx))
        checks.extend(self.validate_result(ctx))
        checks.extend(self.validate_artifacts(ctx))
        checks.extend(self.validate_workspace_scope(ctx))
        checks.extend(self.validate_task_state(ctx))
        checks.extend(self.validate_completion_claim(ctx))
        checks.extend(self.validate_ownership(ctx))
        return self._aggregate(ctx, checks)

    # -- helpers ---------------------------------------------------------------

    @staticmethod
    def _check(
        name: str,
        passed: bool,
        severity: Severity = Severity.INFO,
        message: str = "",
        evidence: list[str] | None = None,
        code: ViolationCode | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ValidationCheck:
        return ValidationCheck(
            name=name,
            passed=passed,
            severity=severity,
            message=message,
            evidence=evidence or [],
            code=code,
            metadata=metadata or {},
        )

    def _aggregate(
        self, ctx: ValidationContext, checks: list[ValidationCheck]
    ) -> ValidationResult:
        violations: list[ValidationViolation] = []
        warnings: list[ValidationWarning] = []
        for check in checks:
            if check.passed:
                continue
            if check.severity in (Severity.HIGH, Severity.CRITICAL):
                code = check.code or ViolationCode.TASK_STATE_CONFLICT
                violations.append(
                    ValidationViolation(
                        code=code,
                        severity=check.severity,
                        message=check.message or check.name,
                        agent_id=(
                            ctx.agent_state.agent_id
                            if ctx.agent_state is not None
                            else ""
                        ),
                        task_id=ctx.delegation.task_id if ctx.delegation else "",
                        delegation_id=(
                            ctx.delegation.delegation_id if ctx.delegation else ""
                        ),
                        evidence=list(check.evidence),
                        metadata=dict(check.metadata),
                    )
                )
            else:
                warnings.append(
                    ValidationWarning(
                        name=check.name,
                        severity=check.severity,
                        message=check.message,
                        evidence=list(check.evidence),
                        metadata=dict(check.metadata),
                    )
                )

        if any(v.severity is Severity.CRITICAL for v in violations):
            status = ValidationStatus.BLOCKED
        elif violations:
            status = ValidationStatus.FAIL
        elif warnings:
            status = ValidationStatus.WARNING
        else:
            status = ValidationStatus.PASS

        evidence: list[str] = []
        for check in checks:
            for item in check.evidence:
                if item and item not in evidence:
                    evidence.append(item)

        summary = (
            f"{len(checks)} checks, {len(violations)} violation(s), "
            f"{len(warnings)} warning(s)"
        )
        if violations:
            summary += ": " + "; ".join(
                f"{v.code.value} ({v.severity.value})" for v in violations[:5]
            )

        result = ValidationResult(
            valid=not violations,
            status=status,
            checks=checks,
            violations=violations,
            warnings=warnings,
            evidence=evidence,
            summary=summary,
            metadata={"validated_at": datetime.now(UTC).isoformat()},
        )
        if ctx.delegation is not None:
            task_label = ctx.delegation.task_id
        elif ctx.task_state is not None:
            task_label = ctx.task_state.goal
        else:
            task_label = "-"
        agent_label = ctx.agent_state.agent_id if ctx.agent_state is not None else "-"
        logger.info("validation: %s (task=%s agent=%s)", result.summary, task_label, agent_label)
        for violation in violations:
            logger.warning(
                "validation violation: %s %s agent=%s task=%s delegation=%s",
                violation.code.value,
                violation.message,
                violation.agent_id,
                violation.task_id,
                violation.delegation_id,
            )
        return result

    # ------------------------------------------------------------------
    # 1. Lifecycle validation
    # ------------------------------------------------------------------

    def validate_lifecycle(self, ctx: ValidationContext) -> list[ValidationCheck]:
        checks: list[ValidationCheck] = []
        state = ctx.agent_state
        result = ctx.result

        if state is not None:
            # Every recorded transition must be legal per the state machine.
            illegal = [
                change
                for change in state.metadata.status_history
                if not can_transition(change.from_status, change.to_status)
            ]
            checks.append(
                self._check(
                    "lifecycle_history_transitions",
                    passed=not illegal,
                    severity=Severity.HIGH,
                    message=(
                        "illegal lifecycle transition(s) recorded: "
                        + "; ".join(
                            f"{c.from_status.value}->{c.to_status.value}"
                            for c in illegal[:3]
                        )
                    )
                    if illegal
                    else "all recorded lifecycle transitions are legal",
                    code=ViolationCode.INVALID_LIFECYCLE_TRANSITION,
                    evidence=[c.reason for c in illegal[:5]],
                )
            )

            # A finished run (result present) must be in a terminal state.
            if result is not None and not state.is_terminal:
                checks.append(
                    self._check(
                        "lifecycle_state_terminal",
                        passed=False,
                        severity=Severity.HIGH,
                        message=(
                            f"agent run is '{state.status.value}' but produced a "
                            "result — expected a terminal state"
                        ),
                        code=ViolationCode.INVALID_LIFECYCLE_TRANSITION,
                    )
                )
            else:
                checks.append(
                    self._check("lifecycle_state_terminal", passed=True)
                )

            # Result status must match the lifecycle status (no corruption).
            expected = _STATUS_TO_RESULT.get(state.status)
            if result is not None and expected is not None and result.status is not expected:
                checks.append(
                    self._check(
                        "lifecycle_result_status_match",
                        passed=False,
                        severity=Severity.HIGH,
                        message=(
                            f"lifecycle '{state.status.value}' conflicts with "
                            f"result '{result.status.value}' (expected "
                            f"'{expected.value}')"
                        ),
                        code=ViolationCode.INVALID_LIFECYCLE_TRANSITION,
                    )
                )
            else:
                checks.append(
                    self._check("lifecycle_result_status_match", passed=True)
                )

        # Delegation lifecycle must agree with the result as well.
        delegation = ctx.delegation
        if delegation is not None:
            expected = _STATUS_TO_RESULT.get(delegation.status)
            if (
                result is not None
                and expected is not None
                and result.status is not expected
            ):
                checks.append(
                    self._check(
                        "delegation_result_status_match",
                        passed=False,
                        severity=Severity.HIGH,
                        message=(
                            f"delegation '{delegation.status.value}' conflicts "
                            f"with result '{result.status.value}' (expected "
                            f"'{expected.value}')"
                        ),
                        code=ViolationCode.INVALID_LIFECYCLE_TRANSITION,
                    )
                )
            else:
                checks.append(
                    self._check("delegation_result_status_match", passed=True)
                )

        if delegation is None and state is None:
            checks.append(self._check("lifecycle_scope", passed=True, message="no lifecycle record to validate"))
        return checks

    # ------------------------------------------------------------------
    # 2. Delegation validity
    # ------------------------------------------------------------------

    def validate_delegation(self, ctx: ValidationContext) -> list[ValidationCheck]:
        checks: list[ValidationCheck] = []
        delegation = ctx.delegation
        if delegation is None:
            checks.append(
                self._check("delegation_present", passed=True, message="no delegation to validate")
            )
            return checks

        missing = [
            field
            for field in ("delegation_id", "task_id", "objective")
            if not getattr(delegation, field)
        ]
        checks.append(
            self._check(
                "delegation_fields_present",
                passed=not missing,
                severity=Severity.HIGH,
                message=(
                    f"delegation missing required fields: {', '.join(missing)}"
                    if missing
                    else "delegation fields present"
                ),
                code=ViolationCode.TASK_STATE_CONFLICT,
            )
        )

        registered = True
        try:
            self.registry.get(delegation.agent_type)
        except KeyError:
            registered = False
        checks.append(
            self._check(
                "delegation_agent_registered",
                passed=registered,
                severity=Severity.HIGH,
                message=(
                    f"delegation references unregistered agent type "
                    f"'{delegation.agent_type.value}'"
                    if not registered
                    else f"agent type '{delegation.agent_type.value}' registered"
                ),
                code=ViolationCode.TASK_STATE_CONFLICT,
            )
        )
        return checks

    # ------------------------------------------------------------------
    # 3. Permissions + tool authorization
    # ------------------------------------------------------------------

    def _profile_for(self, ctx: ValidationContext):
        """The frozen permission profile governing this run, if any."""
        delegation = ctx.delegation
        if delegation is not None and delegation.permissions is not None:
            return delegation.permissions
        state = ctx.agent_state
        if state is not None and state.context is not None:
            return state.context.agent_permissions
        return None

    def validate_permissions(self, ctx: ValidationContext) -> list[ValidationCheck]:
        checks: list[ValidationCheck] = []
        profile = self._profile_for(ctx)

        if profile is None:
            checks.append(
                self._check(
                    "permission_profile_present",
                    passed=not ctx.tool_uses,
                    severity=Severity.MEDIUM,
                    message=(
                        "no permission profile to validate recorded tool uses"
                        if ctx.tool_uses
                        else "no permission profile (nothing to validate)"
                    ),
                )
            )
            return checks
        checks.append(
            self._check("permission_profile_present", passed=True, message="frozen profile present")
        )

        for use in ctx.tool_uses:
            tool = use.tool_name
            # 1. Whitelist: a tool not listed is denied by default.
            if tool not in profile.allowed_tools:
                checks.append(
                    self._check(
                        f"tool_allowed:{tool}",
                        passed=False,
                        severity=Severity.HIGH,
                        message=f"tool '{tool}' is not in the agent's allowed_tools",
                        code=ViolationCode.UNAUTHORIZED_TOOL,
                        evidence=[use.target or tool],
                    )
                )
                continue
            # 2. Category level (deterministic; boundary verdicts are not
            #    re-issued here — the profile level decides ALLOW/DENY).
            category = classify_tool(tool, use.target)
            level = profile.level_for(category)
            if level is Decision.DENY:
                checks.append(
                    self._check(
                        f"tool_category_denied:{tool}",
                        passed=False,
                        severity=Severity.HIGH,
                        message=(
                            f"tool '{tool}' classifies as {category.value} which "
                            f"is DENY for this agent"
                        ),
                        code=ViolationCode.UNAUTHORIZED_TOOL,
                        evidence=[use.target or tool],
                    )
                )
            else:
                note = (
                    "boundary-controlled (CONFIRM)"
                    if level is Decision.CONFIRM
                    else f"allowed ({category.value}=ALLOW)"
                )
                checks.append(
                    self._check(
                        f"tool_allowed:{tool}",
                        passed=True,
                        message=f"tool '{tool}' {note}",
                        evidence=[use.target or tool],
                    )
                )

        # The execution context's whitelist must match the profile.
        state = ctx.agent_state
        if state is not None and state.context is not None:
            extra = [
                t
                for t in state.context.allowed_tools
                if t not in profile.allowed_tools
            ]
            checks.append(
                self._check(
                    "context_scope_matches_profile",
                    passed=not extra,
                    severity=Severity.HIGH,
                    message=(
                        f"execution context allows tools outside the profile: "
                        f"{', '.join(extra)}"
                        if extra
                        else "execution context whitelist matches the profile"
                    ),
                    code=ViolationCode.UNAUTHORIZED_TOOL,
                )
            )
        return checks

    # ------------------------------------------------------------------
    # 4. Budget validation
    # ------------------------------------------------------------------

    def _budget_for(self, ctx: ValidationContext):
        if ctx.delegation is not None:
            return ctx.delegation.budget
        if ctx.agent_state is not None and ctx.agent_state.context is not None:
            return ctx.agent_state.context.budget
        return None

    def validate_budget(self, ctx: ValidationContext) -> list[ValidationCheck]:
        checks: list[ValidationCheck] = []
        budget = self._budget_for(ctx)
        if budget is None:
            checks.append(
                self._check("budget_configured", passed=True, message="no budget to validate")
            )
            return checks

        exceeded = (
            budget.steps_used > budget.max_steps
            or budget.tool_calls_used > budget.max_tool_calls
        )
        checks.append(
            self._check(
                "budget_within_limits",
                passed=not exceeded,
                severity=Severity.HIGH,
                message=(
                    f"budget exceeded: {budget.summary()}"
                    if exceeded
                    else f"budget within limits: {budget.summary()}"
                ),
                code=ViolationCode.BUDGET_EXCEEDED,
                metadata={"budget": budget.summary()},
            )
        )
        # Exact boundary: at the limit is a warning, over it is a violation.
        at_limit = (
            budget.steps_used == budget.max_steps
            or budget.tool_calls_used == budget.max_tool_calls
        )
        checks.append(
            self._check(
                "budget_at_limit",
                passed=not at_limit,
                severity=Severity.MEDIUM,
                message=(
                    f"budget exactly at the limit: {budget.summary()}"
                    if at_limit
                    else "budget not at the limit"
                ),
            )
        )
        return checks

    # ------------------------------------------------------------------
    # 5. Timeout validation
    # ------------------------------------------------------------------

    def validate_timeout(self, ctx: ValidationContext) -> list[ValidationCheck]:
        checks: list[ValidationCheck] = []
        budget = self._budget_for(ctx)
        timeout = budget.timeout_seconds if budget is not None else None
        state = ctx.agent_state

        if timeout is None:
            checks.append(
                self._check("timeout_configured", passed=True, message="no timeout configured")
            )
            return checks

        if state is None or state.metadata.started_at is None:
            checks.append(
                self._check(
                    "timing_data_present",
                    passed=False,
                    severity=Severity.MEDIUM,
                    message="missing timing data — cannot verify the timeout",
                )
            )
            return checks

        end = state.metadata.completed_at or ctx.now or datetime.now(UTC)
        elapsed = (end - state.metadata.started_at).total_seconds()
        over = elapsed > timeout
        # The execution layer may already have flagged the timeout; either
        # signal means the run violated its timeout contract.
        flagged = bool(ctx.result is not None and ctx.result.metadata.get("timeout"))
        checks.append(
            self._check(
                "timeout_respected",
                passed=not over and not flagged,
                severity=Severity.HIGH,
                message=(
                    f"execution took {elapsed:.1f}s against a {timeout}s timeout"
                    if over
                    else (
                        f"execution flagged as timed out (limit {timeout}s)"
                        if flagged
                        else f"execution within timeout ({elapsed:.1f}s <= {timeout}s)"
                    )
                ),
                code=ViolationCode.TIMEOUT_EXCEEDED,
                metadata={"elapsed_seconds": round(elapsed, 3), "timeout_seconds": timeout},
            )
        )
        return checks

    # ------------------------------------------------------------------
    # 6. Agent result validation
    # ------------------------------------------------------------------

    def validate_result(self, ctx: ValidationContext) -> list[ValidationCheck]:
        checks: list[ValidationCheck] = []
        result = ctx.result
        if result is None:
            checks.append(
                self._check(
                    "result_present",
                    passed=False,
                    severity=Severity.CRITICAL,
                    message="no AgentResult to validate",
                    code=ViolationCode.INVALID_AGENT_RESULT,
                )
            )
            return checks

        checks.append(
            self._check(
                "result_present",
                passed=True,
                message=f"result status '{result.status.value}'",
            )
        )

        has_content = bool(
            result.summary or result.evidence or result.artifact is not None or result.changed_files
        )
        if result.status is AgentResultStatus.SUCCESS:
            checks.append(
                self._check(
                    "result_success_evidence",
                    passed=has_content,
                    severity=Severity.HIGH,
                    message=(
                        "SUCCESS result carries no evidence (no summary, evidence, "
                        "artifact or changed files)"
                        if not has_content
                        else "SUCCESS result carries evidence"
                    ),
                    code=ViolationCode.INVALID_AGENT_RESULT,
                )
            )
        elif result.status is AgentResultStatus.FAILED:
            has_failure = bool(result.summary or result.metadata.get("exception"))
            checks.append(
                self._check(
                    "result_failure_info",
                    passed=has_failure,
                    severity=Severity.HIGH,
                    message=(
                        "FAILED result carries no failure information"
                        if not has_failure
                        else "FAILED result carries failure information"
                    ),
                    code=ViolationCode.INVALID_AGENT_RESULT,
                )
            )
        elif result.status is AgentResultStatus.BLOCKED:
            has_blocker = bool(result.blockers or result.summary)
            checks.append(
                self._check(
                    "result_blocked_info",
                    passed=has_blocker,
                    severity=Severity.HIGH,
                    message=(
                        "BLOCKED result carries no blocker information"
                        if not has_blocker
                        else "BLOCKED result carries blocker information"
                    ),
                    code=ViolationCode.INVALID_AGENT_RESULT,
                )
            )
        elif result.status is AgentResultStatus.NEEDS_INPUT:
            has_need = bool(result.summary or result.blockers)
            checks.append(
                self._check(
                    "result_needs_input_info",
                    passed=has_need,
                    severity=Severity.HIGH,
                    message=(
                        "NEEDS_INPUT result does not describe the required input"
                        if not has_need
                        else "NEEDS_INPUT result describes the required input"
                    ),
                    code=ViolationCode.INVALID_AGENT_RESULT,
                )
            )
        return checks

    # ------------------------------------------------------------------
    # 7. Artifact validation
    # ------------------------------------------------------------------

    def validate_artifacts(self, ctx: ValidationContext) -> list[ValidationCheck]:
        checks: list[ValidationCheck] = []
        artifacts: list[AgentArtifact] = list(ctx.artifacts)
        if ctx.result is not None and ctx.result.artifact is not None:
            artifacts.append(ctx.result.artifact)
        # The same artifact may appear both in ctx.artifacts and on the result.
        seen_ids: set[str] = set()
        unique: list[AgentArtifact] = []
        for artifact in artifacts:
            if artifact.artifact_id in seen_ids:
                continue
            seen_ids.add(artifact.artifact_id)
            unique.append(artifact)
        artifacts = unique

        expected_task = (
            ctx.delegation.task_id if ctx.delegation is not None else None
        ) or (task_key(ctx.task_state) if ctx.task_state is not None else None)
        expected_agent = (
            ctx.agent_state.agent_id if ctx.agent_state is not None else None
        )

        if not artifacts:
            checks.append(
                self._check("artifacts_present", passed=True, message="no artifacts to validate")
            )
            return checks

        for artifact in artifacts:
            aid = artifact.artifact_id or artifact.artifact_type.value
            malformed = not artifact.task_id or not artifact.agent_id
            checks.append(
                self._check(
                    f"artifact_schema:{aid}",
                    passed=not malformed,
                    severity=Severity.HIGH,
                    message=(
                        f"artifact '{aid}' missing required task_id/agent_id"
                        if malformed
                        else f"artifact '{aid}' schema valid"
                    ),
                    code=ViolationCode.INVALID_ARTIFACT,
                )
            )
            has_provenance = bool(artifact.source or artifact.confidence is not None)
            checks.append(
                self._check(
                    f"artifact_provenance:{aid}",
                    passed=has_provenance,
                    severity=Severity.LOW,
                    message=(
                        f"artifact '{aid}' has no provenance (no source, no confidence)"
                        if not has_provenance
                        else f"artifact '{aid}' has provenance"
                    ),
                )
            )
            if expected_agent is not None:
                owned = artifact.agent_id == expected_agent
                checks.append(
                    self._check(
                        f"artifact_agent_owned:{aid}",
                        passed=owned,
                        severity=Severity.HIGH,
                        message=(
                            f"artifact '{aid}' claims agent '{artifact.agent_id}' "
                            f"but the run belongs to agent '{expected_agent}'"
                            if not owned
                            else f"artifact '{aid}' owned by agent '{expected_agent}'"
                        ),
                        code=ViolationCode.ARTIFACT_OWNERSHIP_VIOLATION,
                    )
                )
            if expected_task is not None:
                matched = artifact.task_id == expected_task
                checks.append(
                    self._check(
                        f"artifact_task_matched:{aid}",
                        passed=matched,
                        severity=Severity.HIGH,
                        message=(
                            f"artifact '{aid}' belongs to task "
                            f"'{artifact.task_id}' but the run belongs to task "
                            f"'{expected_task}' (cross-task contamination)"
                            if not matched
                            else f"artifact '{aid}' associated with task '{expected_task}'"
                        ),
                        code=ViolationCode.ARTIFACT_TASK_MISMATCH,
                    )
                )
        return checks

    # ------------------------------------------------------------------
    # 8. Workspace scope validation
    # ------------------------------------------------------------------

    @staticmethod
    def _changed_files(ctx: ValidationContext) -> list[str]:
        changed: list[str] = []
        if ctx.result is not None:
            changed.extend(ctx.result.changed_files)
        for artifact in ctx.artifacts:
            if artifact.artifact_type.value == "implementation_result":
                changed.extend(getattr(artifact, "changed_files", []))
        # Deduplicate while preserving order.
        seen: set[str] = set()
        unique: list[str] = []
        for path in changed:
            if path and path not in seen:
                seen.add(path)
                unique.append(path)
        return unique

    def validate_workspace_scope(self, ctx: ValidationContext) -> list[ValidationCheck]:
        checks: list[ValidationCheck] = []
        changed = self._changed_files(ctx)
        if not changed:
            checks.append(
                self._check("scope_clean", passed=True, message="no changed files to check")
            )
            return checks

        if not ctx.workspace:
            checks.append(
                self._check(
                    "workspace_defined",
                    passed=False,
                    severity=Severity.LOW,
                    message=(
                        "changed files recorded but no workspace configured — "
                        "cannot verify file scope"
                    ),
                )
            )
            return checks

        root = Path(ctx.workspace).resolve()
        # Scope prefixes are usually expressed relative to the workspace.
        scope_prefixes = [
            (Path(p) if Path(p).is_absolute() else root / p).resolve()
            for p in ctx.allowed_scope
        ]

        out_of_workspace: list[str] = []
        out_of_scope: list[str] = []
        for raw in changed:
            candidate = Path(raw)
            candidate = candidate if candidate.is_absolute() else root / candidate
            candidate = candidate.resolve()
            try:
                candidate.relative_to(root)
            except ValueError:
                out_of_workspace.append(str(candidate))
                continue
            if scope_prefixes and not any(
                _is_within(candidate, prefix) for prefix in scope_prefixes
            ):
                out_of_scope.append(str(candidate))

        for path in out_of_workspace:
            checks.append(
                self._check(
                    f"file_in_workspace:{path}",
                    passed=False,
                    severity=Severity.CRITICAL,
                    message=f"changed file '{path}' escapes the workspace '{root}'",
                    code=ViolationCode.UNAUTHORIZED_FILE_ACCESS,
                    evidence=[path],
                )
            )
        for path in out_of_scope:
            checks.append(
                self._check(
                    f"file_in_scope:{path}",
                    passed=False,
                    severity=Severity.HIGH,
                    message=(
                        f"changed file '{path}' is outside the allowed scope "
                        f"{[str(p) for p in scope_prefixes]}"
                    ),
                    code=ViolationCode.WORKSPACE_SCOPE_VIOLATION,
                    evidence=[path],
                )
            )
        checks.append(
            self._check(
                "scope_clean",
                passed=not out_of_workspace and not out_of_scope,
                severity=Severity.HIGH,
                message=(
                    f"{len(out_of_workspace)} file(s) outside workspace, "
                    f"{len(out_of_scope)} file(s) outside allowed scope"
                    if out_of_workspace or out_of_scope
                    else "all changed files are inside the allowed scope"
                ),
                code=(
                    ViolationCode.UNAUTHORIZED_FILE_ACCESS
                    if out_of_workspace
                    else ViolationCode.WORKSPACE_SCOPE_VIOLATION
                ),
            )
        )
        return checks

    # ------------------------------------------------------------------
    # 9. TaskState consistency
    # ------------------------------------------------------------------

    def validate_task_state(self, ctx: ValidationContext) -> list[ValidationCheck]:
        checks: list[ValidationCheck] = []
        task = ctx.task_state
        result = ctx.result
        if task is None:
            checks.append(
                self._check("task_state_present", passed=True, message="no TaskState to validate")
            )
            return checks

        checks.append(
            self._check("task_state_present", passed=True, message=f"task status '{task.status.value}'")
        )

        if result is not None and result.status is AgentResultStatus.SUCCESS:
            conflict = (
                task.status in _NON_SUCCESS_TASK_STATUSES or task.is_blocked
            )
            checks.append(
                self._check(
                    "task_not_blocked_on_success",
                    passed=not conflict,
                    severity=Severity.HIGH,
                    message=(
                        f"task is '{task.status.value}' but the agent reports "
                        "SUCCESS — a blocked/failed task must never succeed"
                        if conflict
                        else "task state is consistent with the SUCCESS result"
                    ),
                    code=ViolationCode.TASK_STATE_CONFLICT,
                )
            )
        else:
            checks.append(
                self._check("task_not_blocked_on_success", passed=True, message="no success claim to check")
            )

        if result is not None and task.is_complete():
            consistent = result.status is AgentResultStatus.SUCCESS
            checks.append(
                self._check(
                    "task_completion_consistent",
                    passed=consistent,
                    severity=Severity.HIGH,
                    message=(
                        f"task is complete but the result is "
                        f"'{result.status.value}'"
                        if not consistent
                        else "completed task carries a SUCCESS result"
                    ),
                    code=ViolationCode.TASK_STATE_CONFLICT,
                )
            )
        else:
            checks.append(
                self._check("task_completion_consistent", passed=True, message="no completed-task conflict")
            )

        # A task marked complete must actually have no unresolved steps — a
        # TASK_COMPLETED status with incomplete requirements is corruption.
        if task.status is TaskStatus.TASK_COMPLETED:
            unresolved = task.remaining_requirements()
            checks.append(
                self._check(
                    "task_completion_state_consistent",
                    passed=not unresolved,
                    severity=Severity.HIGH,
                    message=(
                        "task is marked complete but requirements remain: "
                        + "; ".join(f"'{r.description}'" for r in unresolved[:3])
                        if unresolved
                        else "completed task has no unresolved requirements"
                    ),
                    code=ViolationCode.TASK_STATE_CONFLICT,
                )
            )
        else:
            checks.append(
                self._check(
                    "task_completion_state_consistent",
                    passed=True,
                    message="task not marked complete",
                )
            )
        return checks

    # ------------------------------------------------------------------
    # 10. Completion claim validation
    # ------------------------------------------------------------------

    @staticmethod
    def _claims_completion(ctx: ValidationContext) -> bool:
        if ctx.claims_completion:
            return True
        if ctx.result is not None and ctx.result.summary:
            return bool(_COMPLETION_CLAIM_RE.search(ctx.result.summary))
        return False

    def _evidence_present(self, ctx: ValidationContext, category: str) -> bool:
        if category == "changed_files":
            return bool(ctx.result is not None and ctx.result.changed_files)
        if category == "tests":
            return bool(
                (ctx.result is not None and ctx.result.tests)
                or any(t.artifact_type.value == "test_result" for t in ctx.artifacts)
                or bool(ctx.test_results)
            )
        if category == "verification":
            return bool(
                ctx.result is not None
                and (ctx.result.metadata.get("verified") is True or ctx.result.metadata.get("verification") is True)
            )
        if category == "artifact":
            return bool(ctx.result is not None and ctx.result.artifact is not None)
        # Unknown category: treat as satisfied only if named in evidence.
        return bool(ctx.result is not None and any(category in e for e in ctx.result.evidence))

    def validate_completion_claim(self, ctx: ValidationContext) -> list[ValidationCheck]:
        checks: list[ValidationCheck] = []
        result = ctx.result
        if result is None or result.status is not AgentResultStatus.SUCCESS:
            checks.append(
                self._check("completion_claim", passed=True, message="no success claim to validate")
            )
            return checks

        claims = self._claims_completion(ctx)
        checks.append(
            self._check(
                "completion_claim",
                passed=True,
                message=(
                    "result claims task completion" if claims else "result does not claim task completion"
                ),
                metadata={"claims_completion": claims},
            )
        )

        # Test-claim contradiction: SUCCESS with tests while Test Intelligence
        # reports failures.
        failed_tests = [t for t in ctx.test_results if t.failed > 0 or t.timed_out]
        test_artifacts = [t for t in ctx.artifacts if isinstance(t, TestResult)]
        failed_tests.extend(t for t in test_artifacts if t.failed > 0 or t.timed_out)
        if failed_tests and result.tests:
            checks.append(
                self._check(
                    "test_claim_consistent",
                    passed=False,
                    severity=Severity.HIGH,
                    message=(
                        f"agent reports tests but {len(failed_tests)} recorded "
                        "test run(s) contain failures — the success claim "
                        "contradicts test intelligence"
                    ),
                    code=ViolationCode.TEST_CLAIM_CONTRADICTION,
                    evidence=[t.command for t in failed_tests if t.command],
                )
            )
        else:
            checks.append(
                self._check(
                    "test_claim_consistent",
                    passed=True,
                    message="no test-claim contradiction",
                )
            )

        if not claims:
            return checks

        # A completion claim must be backed by the task's completion criteria.
        if ctx.task_state is not None:
            remaining = ctx.task_state.remaining_requirements()
            checks.append(
                self._check(
                    "completion_claim_requirements",
                    passed=not remaining,
                    severity=Severity.HIGH,
                    message=(
                        "completion claimed while requirements remain: "
                        + "; ".join(f"'{r.description}'" for r in remaining[:3])
                        if remaining
                        else "all task requirements satisfied"
                    ),
                    code=ViolationCode.FALSE_COMPLETION_CLAIM,
                )
            )
            remaining_steps = ctx.task_state.remaining_steps()
            checks.append(
                self._check(
                    "completion_claim_plan",
                    passed=not remaining_steps,
                    severity=Severity.HIGH,
                    message=(
                        "completion claimed while plan steps remain: "
                        + ", ".join(f"step {s.id}" for s in remaining_steps[:5])
                        if remaining_steps
                        else "no unfinished plan steps"
                    ),
                    code=ViolationCode.FALSE_COMPLETION_CLAIM,
                )
            )

        # Required evidence must exist for a completion claim.
        for category in ctx.required_evidence:
            present = self._evidence_present(ctx, category)
            checks.append(
                self._check(
                    f"completion_evidence:{category}",
                    passed=present,
                    severity=Severity.HIGH,
                    message=(
                        f"completion claimed but required evidence "
                        f"'{category}' is missing"
                        if not present
                        else f"required evidence '{category}' present"
                    ),
                    code=ViolationCode.FALSE_COMPLETION_CLAIM,
                )
            )
        return checks

    # ------------------------------------------------------------------
    # 11. Task/delegation/agent ownership
    # ------------------------------------------------------------------

    def validate_ownership(self, ctx: ValidationContext) -> list[ValidationCheck]:
        checks: list[ValidationCheck] = []
        delegation = ctx.delegation

        if delegation is not None and ctx.task_state is not None:
            expected = task_key(ctx.task_state)
            bound = delegation.task_id == expected
            checks.append(
                self._check(
                    "delegation_task_bound",
                    passed=bound,
                    severity=Severity.HIGH,
                    message=(
                        f"delegation '{delegation.delegation_id}' is bound to "
                        f"task '{delegation.task_id}' but the TaskState is task "
                        f"'{expected}' — cross-task delegation"
                        if not bound
                        else "delegation is bound to its TaskState's task"
                    ),
                    code=ViolationCode.TASK_STATE_CONFLICT,
                )
            )
        else:
            checks.append(
                self._check("delegation_task_bound", passed=True, message="no delegation/task pair to check")
            )

        if delegation is not None and ctx.agent_state is not None:
            bound = ctx.agent_state.task_id == delegation.task_id
            checks.append(
                self._check(
                    "run_task_bound",
                    passed=bound,
                    severity=Severity.HIGH,
                    message=(
                        f"agent run is bound to task '{ctx.agent_state.task_id}' "
                        f"but the delegation belongs to task '{delegation.task_id}'"
                        if not bound
                        else "agent run is bound to the delegation's task"
                    ),
                    code=ViolationCode.TASK_STATE_CONFLICT,
                )
            )
        else:
            checks.append(
                self._check("run_task_bound", passed=True, message="no run/delegation pair to check")
            )
        return checks


def _is_within(candidate: Path, prefix: Path) -> bool:
    """True when ``candidate`` is inside ``prefix`` (prefix itself counts)."""
    try:
        candidate.relative_to(prefix)
        return True
    except ValueError:
        return False
