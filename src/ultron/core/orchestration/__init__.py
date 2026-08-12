"""
ultron.core.orchestration
~~~~~~~~~~~~~~~~~~~~~~~~~~

Multi-agent orchestration layer (Fix #7).

Section 7.1 — agent contract + lifecycle (implemented):

- lifecycle: :class:`AgentStatus`, transition table + validation
- identity: :class:`AgentType`, :class:`AgentIdentity`
- result: :class:`AgentResultStatus`, :class:`AgentResult`
- budget: :class:`ExecutionBudget`
- scoped context: :class:`ExecutionContext`
- runtime record: :class:`AgentState` (lifecycle methods + audit metadata)
- contract: :class:`Agent` (abstract base every orchestrated agent
  implements)

Section 7.2 — agent registry + permissions (implemented):

- permissions: :class:`AgentPermissions` (frozen, runtime-controlled),
  :class:`PermissionCategory`, :func:`classify_tool`
- registry: :class:`AgentSpec`, :class:`AgentRegistry`, and
  :data:`DEFAULT_REGISTRY` with the six baseline agent types
  (supervisor / researcher / coder / tester / reviewer / security)

Section 7.3 — structured agent results + artifacts (implemented):

- artifacts: :class:`AgentArtifact` base + the five artifact types
  (:class:`ResearchFinding`, :class:`ImplementationResult`,
  :class:`TestResult`, :class:`ReviewResult`, :class:`SecurityFinding`),
  lossless serialization, deterministic :func:`task_key` association, and
  :class:`ArtifactStore` JSONL persistence — agents communicate through
  artifacts, never raw trajectories

Section 7.4 — supervisor + specialist delegation (implemented):

- delegation: :class:`DelegationRequest` (task/parent ids, objective,
  input artifacts, constraints, frozen permissions, expected output,
  budget/timeout) with a lifecycle that reuses :class:`AgentStatus`
- supervisor: :class:`Supervisor` — selects the specialist from the
  registry, dispatches inside a scoped ExecutionContext, enforces
  timeout/cancellation, records the result + artifact on the TaskState,
  and decides whether to continue (sequential only)
- isolation: :func:`task_brief` — the string-only TaskState subset a
  specialist receives; never the transcript or other agents' reasoning

Validation layer (implemented):

- validation: :class:`OrchestrationValidator` — deterministic, read-only
  checks over a :class:`ValidationContext` (lifecycle, delegation,
  permissions/tools, budget, timeout, result schema, artifacts,
  workspace scope, TaskState consistency, completion claims, ownership)
  aggregated into a :class:`ValidationResult` with stable
  :class:`ViolationCode`s. Pure and idempotent — it reports, the
  supervisor acts.

Not implemented (later sections): parallel execution, workflows, locks,
dynamic routing, agent-to-agent messaging.
"""

from ultron.core.orchestration.artifacts import (
    AgentArtifact,
    AgentArtifactUnion,
    ApprovalStatus,
    ArtifactStore,
    ArtifactType,
    ImplementationResult,
    ResearchFinding,
    ReviewFinding,
    ReviewResult,
    SecurityFinding,
    Severity,
    TestResult,
    artifact_from_dict,
    artifact_from_json,
    artifact_to_json,
    task_key,
)
from ultron.core.orchestration.contract import Agent
from ultron.core.orchestration.delegation import (
    AgentFactory,
    DelegationRequest,
    Supervisor,
    SupervisorDecision,
    task_brief,
)
from ultron.core.orchestration.lifecycle import (
    ACTIVE_STATUSES,
    TERMINAL_STATUSES,
    TRANSITIONS,
    AgentStatus,
    assert_transition,
    can_transition,
    transitions_from,
)
from ultron.core.orchestration.models import (
    AGENT_TYPES,
    AgentIdentity,
    AgentMetadata,
    AgentResult,
    AgentResultStatus,
    AgentState,
    AgentStatusChange,
    AgentType,
    ExecutionBudget,
    ExecutionContext,
)
from ultron.core.orchestration.permissions import (
    AgentPermissions,
    PermissionCategory,
    PermissionCheck,
    classify_tool,
)
from ultron.core.orchestration.registry import (
    DEFAULT_REGISTRY,
    AgentRegistry,
    AgentSpec,
)
from ultron.core.orchestration.validation import (
    OrchestrationValidator,
    ValidationCheck,
    ValidationContext,
    ValidationResult,
    ValidationStatus,
    ValidationViolation,
    ValidationWarning,
    ViolationCode,
)
from ultron.core.orchestration.workflow import (
    STEP_TERMINAL_STATUSES,
    STEP_TRANSITIONS,
    WORKFLOW_TERMINAL_STATUSES,
    WORKFLOW_TRANSITIONS,
    Workflow,
    WorkflowEngine,
    WorkflowEvent,
    WorkflowStatus,
    WorkflowStep,
    WorkflowStepStatus,
)

__all__ = [
    "ACTIVE_STATUSES",
    "AGENT_TYPES",
    "DEFAULT_REGISTRY",
    "STEP_TERMINAL_STATUSES",
    "STEP_TRANSITIONS",
    "TERMINAL_STATUSES",
    "TRANSITIONS",
    "WORKFLOW_TERMINAL_STATUSES",
    "WORKFLOW_TRANSITIONS",
    "Agent",
    "AgentArtifact",
    "AgentArtifactUnion",
    "AgentFactory",
    "AgentIdentity",
    "AgentMetadata",
    "AgentPermissions",
    "AgentRegistry",
    "AgentResult",
    "AgentResultStatus",
    "AgentSpec",
    "AgentState",
    "AgentStatus",
    "AgentStatusChange",
    "AgentType",
    "ApprovalStatus",
    "ArtifactStore",
    "ArtifactType",
    "DelegationRequest",
    "ExecutionBudget",
    "ExecutionContext",
    "ImplementationResult",
    "OrchestrationValidator",
    "PermissionCategory",
    "PermissionCheck",
    "ResearchFinding",
    "ReviewFinding",
    "ReviewResult",
    "SecurityFinding",
    "Severity",
    "Supervisor",
    "SupervisorDecision",
    "TestResult",
    "ValidationCheck",
    "ValidationContext",
    "ValidationResult",
    "ValidationStatus",
    "ValidationViolation",
    "ValidationWarning",
    "ViolationCode",
    "Workflow",
    "WorkflowEngine",
    "WorkflowEvent",
    "WorkflowStatus",
    "WorkflowStep",
    "WorkflowStepStatus",
    "artifact_from_dict",
    "artifact_from_json",
    "artifact_to_json",
    "assert_transition",
    "can_transition",
    "classify_tool",
    "task_brief",
    "task_key",
    "transitions_from",
]
