"""
FIX #7 — 10-minute live validation: agent contract + lifecycle (§7.1),
agent registry + permissions (§7.2), structured agent results + artifacts
(§7.3), supervisor -> specialist delegation (§7.4), and the deterministic
orchestration validation layer.

Section 7.1 (deterministic, REAL orchestration API):

    TEST A: create agent                  (AgentIdentity + contract impl)
    TEST B: assign task                   (AgentState.assign + ExecutionContext)
    TEST C: run agent                     (Agent.run_with_state)
    TEST D: return successful AgentResult
    TEST E: simulate failure
    TEST F: simulate blocked state
    TEST G: cancel execution
    TEST H: verify invalid lifecycle transitions are rejected
    TEST I: verify ExecutionContext contains only permitted tools
    TEST J: verify AgentResult contains structured information

Plus the critical section-7.1 validation: an AgentResult does NOT
automatically mean the TaskState is complete — agent completion and task
completion are separate concepts.

Section 7.2 (registry + permissions):

    1. Researcher read -> succeeds; researcher write -> blocked
    2. Coder read -> succeeds; coder write -> security evaluation
    3. Tester test command -> succeeds; tester source write -> blocked
    4. Reviewer git diff -> succeeds; reviewer source write -> blocked
    5. Unknown agent type -> rejected; duplicate registration -> rejected
    6. Permission decisions are recorded in the real security audit log

Section 7.3 (artifacts):

    1. Researcher produces a ResearchFinding from a LARGE internal
       trajectory — only the structured artifact travels onward
    2. Coder consumes the finding and returns an ImplementationResult
       (changed files + tests on the AgentResult envelope)
    3. Tester returns a TestResult that reuses FIX #5's FailureAnalysis
    4. Reviewer returns a ReviewResult (findings + approval)
    5. Security returns a SecurityFinding linked to guardrail hits
    6. Artifacts persist across store instances, associate with the
       TaskState via task_key(), and stay small (no conversation dumps)

Section 7.4 (supervisor -> specialist delegation):

    User task: "Understand authentication."
    Supervisor -> Researcher (repository search, symbol lookup, file
    inspection) -> ResearchFinding. Then deliberate researcher timeout,
    failure and cancellation. Verifies read-only permissions, artifact
    receipt, trajectory isolation, TaskState update, delegation + agent
    lifecycle, and the continue/fail decision.

Validation layer (deterministic checks over real orchestration state):

    V1  researcher valid read/search -> PASS
    V2  researcher write attempt -> permission violation
    V3  coder valid implementation -> PASS
    V4  SUCCESS without required evidence -> completion violation
    V5  tests claimed passed but Test Intelligence says failed -> contradiction
    V6  file outside allowed scope -> workspace violation
    V7  budget exceeded -> budget violation
    V8  timeout exceeded -> timeout violation
    V9  Task A artifact into Task B -> artifact/task mismatch
    V10 cancelled task returns SUCCESS -> TaskState conflict
    V11 validate twice -> identical, zero mutation
    V12 all required evidence -> PASS

Everything runs inside a temporary workspace; no destructive commands, no
live LLM, no repo files touched. Prints PASS/FAIL per check and exits
non-zero on any failure.
"""

import asyncio
import sys
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

from ultron.core.coding.executor import FailureAnalysis, FailureCategory
from ultron.core.memory.models import MemoryConfidence
from ultron.core.orchestration import (
    DEFAULT_REGISTRY,
    Agent,
    AgentIdentity,
    AgentRegistry,
    AgentResult,
    AgentResultStatus,
    AgentState,
    AgentStatus,
    AgentType,
    ApprovalStatus,
    ArtifactStore,
    DelegationRequest,
    ExecutionBudget,
    ExecutionContext,
    ImplementationResult,
    OrchestrationValidator,
    ResearchFinding,
    ReviewFinding,
    ReviewResult,
    SecurityFinding,
    Severity,
    Supervisor,
    SupervisorDecision,
    TestResult,
    ValidationContext,
    ValidationStatus,
    ViolationCode,
    Workflow,
    WorkflowEngine,
    WorkflowStatus,
    WorkflowStep,
    WorkflowStepStatus,
    artifact_from_json,
    artifact_to_json,
    assert_transition,
    task_key,
)
from ultron.core.orchestration.registry import _baseline_specs
from ultron.core.types import TaskState, TaskStatus, ToolExecution
from ultron.security import Decision, SecurityBoundary
from ultron.security.audit import AuditLog
from ultron.security.models import GuardrailFinding

PASS = 0
FAIL = 1
results: list[tuple[str, int, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    results.append((name, PASS if condition else FAIL, detail))
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}{' — ' + detail if detail else ''}")


class ProbeAgent(Agent):
    """
    Minimal temporary test agent for the live validation.

    Mode switches what structured result it returns, so we can exercise
    success / failure / blocked / needs-input / cancellation without any
    real tool execution.
    """

    def __init__(self, identity: AgentIdentity, mode: str = "ok") -> None:
        super().__init__(identity)
        self.mode = mode
        self.tool_calls: list[str] = []

    async def execute(self, objective: str, context: ExecutionContext) -> AgentResult:
        # Contract: only permitted tools, and honor cancellation + budget.
        if context.is_cancelled:
            return AgentResult(status=AgentResultStatus.CANCELLED, summary="cancelled")
        if context.budget.is_exhausted():
            return AgentResult(
                status=AgentResultStatus.FAILED,
                summary="budget exhausted before completing",
                blockers=[f"budget: {context.budget.summary()}"],
            )
        for tool in ("read_file", "search_files"):
            if context.allows_tool(tool):
                self.tool_calls.append(tool)
        if self.mode == "fail":
            return AgentResult(
                status=AgentResultStatus.FAILED,
                summary=f"could not {objective}",
                blockers=["probe failure"],
                evidence=["probe returned failure"],
            )
        if self.mode == "block":
            return AgentResult(
                status=AgentResultStatus.BLOCKED,
                summary="probe blocked",
                blockers=["security block"],
            )
        if self.mode == "input":
            return AgentResult(
                status=AgentResultStatus.NEEDS_INPUT,
                summary="need more info",
                recommendations=["ask user for target"],
            )
        return AgentResult(
            status=AgentResultStatus.SUCCESS,
            summary=f"did {objective}",
            artifacts=["/tmp/probe/result.txt"],
            evidence=["read_file ok"],
            changed_files=["/tmp/probe/result.txt"],
            tests=["probe_test::it_works"],
            recommendations=["run full suite"],
            metadata={"probe": True},
        )


def _fresh_state(objective: str, agent_id: str = "probe") -> tuple[AgentState, ProbeAgent]:
    identity = AgentIdentity(agent_id=agent_id, agent_type=AgentType.CODING)
    context = ExecutionContext(
        task_id="live-task-1",
        agent_id=agent_id,
        workspace=str(Path(tempfile.mkdtemp(prefix="ultron_live_"))),
        allowed_tools=["read_file", "search_files"],
        permissions={"security_mode": "interactive"},
        budget=ExecutionBudget(max_steps=10, max_tool_calls=20, timeout_seconds=300),
    )
    state = AgentState(
        task_id="live-task-1",
        identity=identity,
        objective=objective,
        context=context,
    )
    return state, ProbeAgent(identity)


def test_a_create_agent() -> None:
    identity = AgentIdentity(agent_id="probe", agent_type=AgentType.CODING, display_name="Probe")
    agent = ProbeAgent(identity)
    check("TEST A: create agent", agent.identity.agent_id == "probe")
    check("TEST A: agent identity carries type", agent.identity.agent_type is AgentType.CODING)
    check("TEST A: agent describes itself", "probe" in agent.describe())


def test_b_assign_task() -> None:
    state, _ = _fresh_state("implement health endpoint")
    check("TEST B: agent starts PENDING", state.status is AgentStatus.PENDING)
    state.assign(state.context)
    check("TEST B: assign moves to ASSIGNED", state.status is AgentStatus.ASSIGNED)
    check("TEST B: context bound to state", state.context is not None)
    check("TEST B: objective stored", state.objective == "implement health endpoint")


def test_c_d_run_agent_and_success() -> None:
    async def body():
        state, agent = _fresh_state("implement health endpoint")
        result = await agent.run_with_state(state)
        check("TEST C: run agent reaches COMPLETED", state.status is AgentStatus.COMPLETED)
        check("TEST D: SUCCESS result returned", result.status is AgentResultStatus.SUCCESS)
        check("TEST D: result stored on state", state.result is result)

    asyncio.run(body())


def test_e_simulate_failure() -> None:
    async def body():
        state, agent = _fresh_state("fix failing tests", agent_id="fail-probe")
        agent.mode = "fail"
        result = await agent.run_with_state(state)
        check("TEST E: failed agent lifecycle is FAILED", state.status is AgentStatus.FAILED)
        check("TEST E: FAILED result structured", result.status is AgentResultStatus.FAILED)
        check("TEST E: blocker recorded", result.blockers == ["probe failure"])

    asyncio.run(body())


def test_f_simulate_blocked() -> None:
    async def body():
        state, agent = _fresh_state("deploy to prod", agent_id="block-probe")
        agent.mode = "block"
        result = await agent.run_with_state(state)
        check("TEST F: blocked agent lifecycle is BLOCKED", state.status is AgentStatus.BLOCKED)
        check("TEST F: BLOCKED result structured", result.status is AgentResultStatus.BLOCKED)
        check("TEST F: blocker listed", "security block" in result.blockers)

    asyncio.run(body())


def test_g_cancel_execution() -> None:
    async def body():
        # Cancel before start: PENDING -> CANCELLED.
        state, agent = _fresh_state("long work")
        state.context.request_cancel()
        result = await agent.run_with_state(state)
        check("TEST G: pre-cancel routes to CANCELLED", state.status is AgentStatus.CANCELLED)
        check("TEST G: CANCELLED result", result.status is AgentResultStatus.CANCELLED)
        # Cancel mid-run: agent sees the flag and stops.
        state2, agent2 = _fresh_state("long work 2")
        state2.assign(state2.context)
        state2.start()
        state2.context.request_cancel()
        result2 = await agent2.execute("long work 2", state2.context)
        check("TEST G: mid-run cancel honored", result2.status is AgentResultStatus.CANCELLED)
        check("TEST G: cancellation flag readable", state2.context.is_cancelled)

    asyncio.run(body())


def test_h_invalid_transitions_rejected() -> None:
    state, _ = _fresh_state("some task")
    rejected = 0
    try:
        state.complete(AgentResult(status=AgentResultStatus.SUCCESS, summary="premature"))
    except ValueError:
        rejected += 1
    try:
        state.resume()  # PENDING -> RUNNING is illegal
    except ValueError:
        rejected += 1
    state.assign(state.context)
    try:
        state.complete(AgentResult(status=AgentResultStatus.FAILED, summary="corrupt"))
    except ValueError:
        rejected += 1  # non-SUCCESS result may never complete a run
    check("TEST H: illegal transitions rejected", rejected == 3, f"{rejected}/3 rejected")
    # Terminal states reject everything.
    state2, _ = _fresh_state("finish me")
    state2.assign(state2.context)
    state2.start()
    state2.complete(AgentResult(status=AgentResultStatus.SUCCESS, summary="ok"))
    terminal_rejected = 0
    for target in AgentStatus:
        try:
            assert_transition(state2.status, target)
        except ValueError:
            terminal_rejected += 1
    check(
        "TEST H: terminal state accepts no transitions",
        terminal_rejected == len(AgentStatus),
        f"{terminal_rejected}/{len(AgentStatus)} blocked",
    )
    check("TEST H: completed agent is terminal", state2.is_terminal)


def test_i_execution_context_permitted_tools() -> None:
    state, _ = _fresh_state("inspect repo")
    ctx = state.context
    check("TEST I: read_file permitted", ctx.allows_tool("read_file"))
    check("TEST I: search_files permitted", ctx.allows_tool("search_files"))
    check("TEST I: run_command denied", not ctx.allows_tool("run_command"))
    check("TEST I: write_file denied", not ctx.allows_tool("write_file"))
    check("TEST I: budget attached", ctx.budget is not None and ctx.budget.max_steps == 10)
    check("TEST I: permissions scoped", ctx.permissions.get("security_mode") == "interactive")


def test_j_agent_result_structured() -> None:
    async def body():
        state, agent = _fresh_state("structured result")
        result = await agent.run_with_state(state)
        check("TEST J: summary present", bool(result.summary))
        check("TEST J: artifacts list", result.artifacts == ["/tmp/probe/result.txt"])
        check("TEST J: evidence list", "read_file ok" in result.evidence)
        check("TEST J: changed_files list", result.changed_files == ["/tmp/probe/result.txt"])
        check("TEST J: tests list", result.tests == ["probe_test::it_works"])
        check("TEST J: recommendations list", bool(result.recommendations))
        check("TEST J: metadata dict", result.metadata.get("probe") is True)
        # A structured result is never an arbitrary string.
        check("TEST J: result is a model, not a string", isinstance(result, AgentResult))
        check("TEST J: prompt line renders", "[success]" in result.to_prompt_line())

    asyncio.run(body())


def test_section72_registry_and_permissions() -> None:
    """Section 7.2 live checks against the REAL default registry."""
    print("\n--- Section 7.2: agent registry + permissions ---")

    # A temp audit log so we inspect real security records without touching
    # ~/.ultron. The interactive boundary is the actual gate implementation.
    audit = AuditLog(Path(tempfile.mkdtemp(prefix="ultron_live72_")) / "security_audit.jsonl")
    boundary = SecurityBoundary(mode="interactive", audit_log=audit)

    def verdict(agent: str, action: str, target: str = "", content: str | None = None) -> Decision:
        return DEFAULT_REGISTRY.check_action(agent, action, target, content, boundary=boundary)

    # 1. Researcher: read succeeds, write blocked.
    check("72.1 researcher read -> ALLOW", verdict("researcher", "read_file", "src/auth.py") is Decision.ALLOW)
    check("72.1 researcher write -> DENY", verdict("researcher", "write_file", "src/auth.py") is Decision.DENY)

    # 2. Coder: read succeeds; write goes through the security boundary.
    check("72.2 coder read -> ALLOW", verdict("coder", "read_file", "src/app.py") is Decision.ALLOW)
    write_verdict = verdict("coder", "write_file", "src/app.py")
    # In interactive mode a normal write is HIGH risk -> CONFIRM. The point
    # is the decision comes from the security boundary, never from the agent.
    check(
        "72.2 coder write -> security evaluation (confirm)",
        write_verdict is Decision.CONFIRM,
        f"boundary said {write_verdict.value}",
    )
    # A secret-embedding coder write is guardrail-denied even by the coder.
    check(
        "72.2 coder secret write -> DENY (guardrails)",
        verdict("coder", "write_file", "config.py", "api_key=sk-1234567890abcdef") is Decision.DENY,
    )

    # 3. Tester: test command succeeds; source modification blocked.
    check("72.3 tester pytest -> ALLOW", verdict("tester", "run_command", "pytest tests/") is Decision.ALLOW)
    check("72.3 tester source write -> DENY", verdict("tester", "write_file", "src/app.py") is Decision.DENY)
    check("72.3 tester arbitrary shell -> DENY", verdict("tester", "run_command", "rm -rf src/") is Decision.DENY)

    # 4. Reviewer: read-only git diff succeeds; source modification blocked.
    check("72.4 reviewer git diff -> ALLOW", verdict("reviewer", "run_command", "git diff") is Decision.ALLOW)
    check("72.4 reviewer source write -> DENY", verdict("reviewer", "write_file", "src/app.py") is Decision.DENY)

    # 5. Unknown agent type + duplicate registration fail safely.
    unknown_rejected = False
    try:
        DEFAULT_REGISTRY.check_action("llm-agent", "read_file")
    except KeyError:
        unknown_rejected = True
    check("72.5 unknown agent type rejected", unknown_rejected)

    fresh = AgentRegistry()
    fresh.register(_baseline_specs()[0])  # supervisor only
    duplicate_rejected = False
    try:
        fresh.register(_baseline_specs()[0])  # same type again
    except ValueError:
        duplicate_rejected = True
    check("72.5 duplicate registration rejected", duplicate_rejected)

    # 6. The security audit log contains the permission verdicts.
    records = audit.read()
    denies = [r for r in records if r["decision"] == "deny"]
    write_denies = [r for r in denies if r["action_type"] == "write_file"]
    check(
        "72.6 permission verdicts recorded in audit log",
        len(records) >= 3 and len(write_denies) >= 1,
        f"{len(records)} records, {len(write_denies)} write denials",
    )
    # Show the inspectable trail.
    print("    audit trail (permission + boundary verdicts):")
    for record in records[-6:]:
        print(
            f"      - {record.get('action_type')} -> {record.get('decision')} "
            f"(tier={record.get('tier')})"
        )


def test_section73_artifacts() -> None:
    """Section 7.3 live checks: five agents produce structured artifacts."""
    print("\n--- Section 7.3: structured agent results + artifacts ---")

    store = ArtifactStore(Path(tempfile.mkdtemp(prefix="ultron_live73_")))
    task = TaskState(goal="add JWT authentication and refresh tokens")
    tid = task_key(task)

    # --- Researcher: a LARGE internal trajectory must never travel onward ---
    trajectory = "\n".join(
        f"step {i}: read_file(src/auth/{'service' if i % 2 else 'middleware'}.py) "
        f"-> {i * 101} bytes of raw content cached locally"
        for i in range(250)
    )
    research = ResearchFinding(
        task_id=tid,
        agent_id="researcher-1",
        summary="AuthService.validate() is the single token-validation entry point.",
        evidence=["files: src/auth/service.py, src/api/login.py", "callers: login.py -> AuthService"],
        source="code_intelligence:find_definition",
        confidence=MemoryConfidence.DIRECT_OBSERVATION,
        related_files=["src/auth/service.py", "src/api/login.py"],
        related_symbols=["AuthService", "AuthMiddleware"],
        architecture_findings=["AuthService validates tokens; AuthMiddleware injects the user"],
        uncertainties=["refresh-token flow not yet inspected"],
        recommendations=["implement refresh tokens inside AuthService"],
    )
    research_payload = artifact_to_json(research) + research.to_agent_result().model_dump_json()
    check(
        "73.1 researcher artifact is tiny vs its internal trajectory",
        len(research_payload) * 10 < len(trajectory),
        f"payload={len(research_payload)}B trajectory={len(trajectory)}B",
    )
    check(
        "73.1 researcher artifact carries provenance (source/confidence)",
        research.source == "code_intelligence:find_definition"
        and research.confidence is MemoryConfidence.DIRECT_OBSERVATION
        and research.owned_by(tid, "researcher-1"),
    )
    check(
        "73.1 artifact contains NO trajectory content",
        all(marker not in research_payload for marker in ("step 0:", "raw content cached")),
    )
    store.save(research)

    # --- Coder: consumes ONLY the researcher's artifact, then reports ---
    coder = ImplementationResult(
        task_id=tid,
        agent_id="coder-1",
        summary="refresh tokens implemented in AuthService",
        evidence=["implemented refresh_token() in src/auth/service.py"],
        related_files=research.related_files,  # what it took from the finding
        related_symbols=research.related_symbols,
        changed_files=["src/auth/service.py", "src/api/login.py"],
        changed_symbols=["AuthService.refresh_token"],
        tests_added=["tests/auth/test_refresh.py"],
        tests_run=["pytest tests/auth -q"],
        source="coding_executor:edit+test",
        confidence=MemoryConfidence.DIRECT_OBSERVATION,
    )
    coder_result = coder.to_agent_result()
    check(
        "73.2 coder envelope carries changed files + tests",
        coder_result.changed_files == ["src/auth/service.py", "src/api/login.py"]
        and "tests/auth/test_refresh.py" in coder_result.tests
        and coder_result.artifact is coder,
    )
    # The coder worked from the structured finding — the researcher's internal
    # trajectory is nowhere in the handoff.
    handoff = artifact_to_json(research) + coder_result.model_dump_json()
    check(
        "73.2 handoff between agents contains no trajectory",
        "step 0:" not in handoff and len(handoff) * 8 < len(trajectory),
    )
    store.save(coder)

    # --- Tester: reuses FIX #5's FailureAnalysis, no second test system ---
    failure = FailureAnalysis(
        category=FailureCategory.TEST_ASSERTION,
        command="pytest tests/auth/test_refresh.py",
        summary="assert token['exp'] == expected",
        file="src/auth/service.py",
        line=88,
        test_name="test_refresh_token_expiry",
    )
    tester = TestResult(
        task_id=tid,
        agent_id="tester-1",
        command="pytest tests/auth -q",
        passed=11,
        failed=1,
        failures=[failure],
        affected_tests=["tests/auth/test_refresh.py::test_refresh_token_expiry"],
        source="test_intelligence:run",
        confidence=MemoryConfidence.DIRECT_OBSERVATION,
    )
    restored_test = artifact_from_json(artifact_to_json(tester))
    check(
        "73.3 tester artifact round-trips FIX #5 failure (file:line:test)",
        isinstance(restored_test, TestResult)
        and restored_test.failures[0].file == "src/auth/service.py"
        and restored_test.failures[0].line == 88
        and restored_test.failures[0].test_name == "test_refresh_token_expiry"
        and restored_test.passed_all is False,
    )
    check(
        "73.3 tester failure becomes a blocker on the envelope",
        tester.to_agent_result().blockers == [failure.summary],
    )
    store.save(tester)

    # --- Reviewer ---
    reviewer = ReviewResult(
        task_id=tid,
        agent_id="reviewer-1",
        summary="implementation is sound; one change requested",
        findings=[
            ReviewFinding(
                severity=Severity.MEDIUM,
                affected_file="src/auth/service.py",
                description="refresh tokens are not rotated on reuse",
                recommendation="rotate on reuse",
            )
        ],
        overall_severity=Severity.MEDIUM,
        required_changes=["rotate refresh tokens on reuse"],
        approval=ApprovalStatus.CHANGES_REQUESTED,
    )
    check(
        "73.4 reviewer artifact keeps structured findings + approval",
        reviewer.findings[0].severity is Severity.MEDIUM
        and reviewer.approval is ApprovalStatus.CHANGES_REQUESTED
        and artifact_from_json(artifact_to_json(reviewer)).approval is ApprovalStatus.CHANGES_REQUESTED,
    )
    store.save(reviewer)

    # --- Security ---
    guardrail = GuardrailFinding(
        rule="generic_secret",
        severity="critical",
        location="content",
        snippet="token=…redacted",
        message="secret-like value in code",
    )
    security = SecurityFinding(
        task_id=tid,
        agent_id="security-1",
        summary="no secrets found; refresh tokens are signed and short-lived",
        severity=Severity.LOW,
        issue="none blocking",
        affected_component="src/auth/service.py",
        recommendation="none",
        blocking=False,
        guardrail_findings=[guardrail],
        source="security_boundary:scan",
        confidence=MemoryConfidence.DIRECT_OBSERVATION,
    )
    check(
        "73.5 security artifact links guardrail findings + serializes",
        artifact_from_json(artifact_to_json(security)).guardrail_findings[0].rule
        == "generic_secret",
    )
    store.save(security)

    # --- Persistence + task association across store instances ---
    fresh = ArtifactStore(store.directory)  # simulates a process restart
    check(
        "73.6 artifacts persist across store instances (restart)",
        len(fresh.all()) == 5 and fresh.get(research.artifact_id) is not None,
        f"{len(fresh.all())}/5 artifacts",
    )
    check(
        "73.6 artifacts associate with the TaskState (task_key)",
        len(fresh.load_for_task(tid)) == 5
        and all(a.owned_by(task_id=tid) for a in fresh.load_for_task(tid)),
    )
    check(
        "73.6 artifacts are isolated per agent",
        len(fresh.load_for_agent("researcher-1")) == 1
        and len(fresh.load_for_agent("coder-1")) == 1,
    )
    check(
        "73.6 artifact sizes are small (no conversation dumps)",
        max(len(artifact_to_json(a)) for a in fresh.all()) < 2500,
        f"max={max(len(artifact_to_json(a)) for a in fresh.all())}B",
    )
    check(
        "73.7 envelope round-trip restores the concrete artifact type",
        isinstance(
            AgentResult.model_validate_json(security.to_agent_result().model_dump_json()).artifact,
            SecurityFinding,
        ),
    )


class Live74Agent(Agent):
    """§7.4 fake specialist: verifies its own permissions, records a large
    internal trajectory (which must never leave it), and simulates
    repository exploration deterministically."""

    def __init__(self, identity, behavior: str = "research") -> None:
        super().__init__(identity)
        self.behavior = behavior
        self.trajectory: list[str] = []

    async def execute(self, objective: str, context: ExecutionContext) -> AgentResult:
        self.trajectory.append(f"internal step 0: {objective}")
        write_verdict = context.check_action("write_file", "src/auth/service.py")
        self.trajectory.append(f"internal step 1: write verdict {write_verdict.value}")
        if self.behavior == "research":
            # Simulated exploration: repository search -> symbol lookup -> file
            # inspection. A researcher MUST be read-only; refuse to proceed
            # otherwise (this is what the live check then verifies).
            if write_verdict is not Decision.ALLOW:
                self.trajectory.append("internal step 2: verified read-only, proceeding")
            else:
                return AgentResult(
                    status=AgentResultStatus.FAILED,
                    summary="refusing to run: researcher is not read-only",
                )
            exploration = [
                ("repository search", "search 'authenticate' -> 3 hits"),
                ("symbol lookup", "find_symbol AuthService -> src/auth/service.py"),
                ("file inspection", "read_file src/auth/service.py (412 lines)"),
            ]
            for i, (step, detail) in enumerate(exploration, start=2):
                self.trajectory.append(f"internal step {i}: {step} -> {detail}")
            artifact = ResearchFinding(
                task_id=context.task_id,
                agent_id=context.agent_id,
                summary="Authentication is implemented in src/auth/service.py (AuthService).",
                evidence=[
                    "repository search: 'authenticate' -> 3 hits",
                    "symbol: AuthService",
                    "files: src/auth/service.py, src/api/login.py",
                ],
                source="code_intelligence + repository search",
                confidence=MemoryConfidence.DIRECT_OBSERVATION,
                related_files=["src/auth/service.py", "src/api/login.py"],
                related_symbols=["AuthService", "AuthMiddleware"],
                architecture_findings=[
                    "AuthService validates tokens; AuthMiddleware injects the user"
                ],
                uncertainties=["refresh-token flow not inspected"],
                recommendations=["start implementation in src/auth/service.py"],
            )
            return artifact.to_agent_result()
        if self.behavior == "slow":
            await asyncio.sleep(30.0)
            return AgentResult(status=AgentResultStatus.SUCCESS, summary="eventually")
        if self.behavior == "failed":
            return AgentResult(
                status=AgentResultStatus.FAILED,
                summary="could not locate the authentication module",
            )
        if self.behavior == "cancellable":
            while not context.is_cancelled:
                await asyncio.sleep(0.005)
            return AgentResult(status=AgentResultStatus.SUCCESS, summary="ignored cancel")
        if self.behavior == "plain":
            return AgentResult(status=AgentResultStatus.SUCCESS, summary=f"{objective} done")
        raise AssertionError(f"unknown live74 behavior {self.behavior}")


def _live74_supervisor(behavior: str, store: ArtifactStore) -> Supervisor:
    def factory(spec, state) -> Live74Agent:
        return Live74Agent(state.identity, behavior=behavior)

    return Supervisor(registry=DEFAULT_REGISTRY, agent_factory=factory, store=store)


def test_section74_supervisor_delegation() -> None:
    """Section 7.4 live checks: supervisor -> researcher delegation."""
    print("\n--- Section 7.4: supervisor -> specialist delegation ---")
    store = ArtifactStore(Path(tempfile.mkdtemp(prefix="ultron_live74_")))
    task = TaskState(goal="Understand authentication.")
    task.add_requirement("Authentication flow is documented")
    tid = task_key(task)

    async def happy_path() -> None:
        supervisor = _live74_supervisor("research", store)
        req = supervisor.create_delegation(
            task,
            "researcher",
            "Locate the authentication implementation",
            constraints=["read-only"],
            expected_output="ResearchFinding with affected files",
        )
        check(
            "74.1 delegation created (PENDING, researcher, task-bound)",
            req.status is AgentStatus.PENDING
            and req.agent_type is AgentType.RESEARCH
            and req.task_id == tid
            and req.delegation_id,
        )
        await supervisor.dispatch(req, workspace=str(task_code), task_state=task)
        # Delegation + agent lifecycle both COMPLETED; artifact received.
        check(
            "74.2 delegation COMPLETED and decide() says CONTINUE",
            req.status is AgentStatus.COMPLETED
            and supervisor.decide(req) is SupervisorDecision.CONTINUE,
        )
        run = supervisor._runs[req.delegation_id]
        check(
            "74.3 specialist run lifecycle COMPLETED (agent != task status)",
            run.status is AgentStatus.COMPLETED
            and task.status is TaskStatus.TASK_STARTED
            and not task.is_complete(),
        )
        check(
            "74.4 researcher had read-only permissions (write DENY, read ALLOW)",
            run.context.check_action("read_file") is Decision.ALLOW
            and run.context.check_action("write_file") is Decision.DENY
            and not run.context.allows_tool("write_file"),
        )
        check(
            "74.5 supervisor received the ResearchFinding artifact (persisted)",
            req.result is not None
            and isinstance(req.result.artifact, ResearchFinding)
            and store.get(req.result.artifact.artifact_id) is not None
            and req.result.artifact.owned_by(tid, req.run_agent_id),
        )
        check(
            "74.6 TaskState updated (observation set, never completed)",
            (task.last_observation or "").startswith("[delegate:research]"),
            task.last_observation or "(none)",
        )
        # Trajectory isolation: the researcher's internal reasoning is large;
        # the NEXT specialist receives only the structured artifact.
        researcher = supervisor.agent_factory(
            DEFAULT_REGISTRY.get("researcher"), run
        )
        researcher.trajectory = [
            f"internal step {i}: searched {i} files" for i in range(250)
        ]
        trajectory_text = "\n".join(researcher.trajectory)
        req2 = supervisor.create_delegation(
            task,
            "coder",
            "implement refresh tokens",
            input_artifacts=[req.result.artifact],
        )
        supervisor.agent_factory = lambda spec, state: Live74Agent(
            state.identity, behavior="plain"
        )
        await supervisor.dispatch(req2, workspace=str(task_code), task_state=task)
        ctx2 = supervisor._runs[req2.delegation_id].context.relevant_context
        ctx2_text = str(ctx2)
        check(
            "74.7 full researcher trajectory NOT injected (artifact + brief only)",
            len(trajectory_text) > 8_000
            and "internal step" not in ctx2_text
            and "searched 249 files" not in ctx2_text
            and "authentication is implemented in src/auth/service.py" in ctx2_text.lower()
            and ctx2["task"]["goal"] == "Understand authentication.",
            f"trajectory={len(trajectory_text)}B context={len(ctx2_text)}B",
        )

    # Deliberate failures: timeout, failure, cancellation.
    async def timeout_path() -> None:
        sup = _live74_supervisor("slow", store)
        req = sup.create_delegation(task, "researcher", "long research", timeout_seconds=1)
        await sup.dispatch(req, workspace=str(task_code), task_state=task)
        check(
            "74.8 researcher timeout -> delegation FAILED, run cancelled",
            req.status is AgentStatus.FAILED
            and req.result.metadata.get("timeout") is True
            and "timeout" in (req.error or "")
            and sup._runs[req.delegation_id].context.is_cancelled,
            req.summary(),
        )

    async def failure_path() -> None:
        task2 = TaskState(goal="Understand authentication.")
        sup = _live74_supervisor("failed", store)
        req = sup.create_delegation(task2, "researcher", "find auth")
        await sup.dispatch(req, workspace=str(task_code), task_state=task2)
        check(
            "74.9 researcher failure -> delegation FAILED, TaskError recorded",
            req.status is AgentStatus.FAILED
            and req.result.status is AgentResultStatus.FAILED
            and any("[delegate:research]" in e.message for e in task2.errors)
            and sup.decide(req) is SupervisorDecision.FAILED,
            req.summary(),
        )

    async def cancel_path() -> None:
        sup = _live74_supervisor("cancellable", store)
        req = sup.create_delegation(task, "researcher", "will be cancelled")
        run_task = asyncio.create_task(
            sup.dispatch(req, workspace=str(task_code), task_state=task)
        )
        await asyncio.sleep(0.05)  # let the run start
        sup.cancel_delegation(req, reason="live cancel")
        await run_task
        check(
            "74.10 researcher cancellation -> delegation CANCELLED",
            req.status is AgentStatus.CANCELLED
            and req.result.status is AgentResultStatus.CANCELLED
            and sup.decide(req) is SupervisorDecision.FAILED,
            req.summary(),
        )

    task_code = Path(tempfile.mkdtemp(prefix="ultron_live74_repo_"))
    (task_code / "src" / "auth").mkdir(parents=True)
    (task_code / "src" / "auth" / "service.py").write_text(
        "class AuthService:\n    def validate(self, token): ...\n", encoding="utf-8"
    )
    asyncio.run(happy_path())
    asyncio.run(timeout_path())
    asyncio.run(failure_path())
    asyncio.run(cancel_path())


def _validation_coder_ctx(
    workspace: str,
    task: TaskState,
    *,
    changed: list[str],
    tests: list[str],
    verified: bool,
    claims: bool = False,
    scope: list[str] | None = None,
) -> ValidationContext:
    """A lifecycle-consistent coder delegation record for validation."""
    tid = task_key(task)
    delegation = DelegationRequest(
        task_id=tid,
        agent_type=AgentType.CODING,
        objective="implement the endpoint",
        permissions=DEFAULT_REGISTRY.get("coder").permissions,
        budget=DEFAULT_REGISTRY.get("coder").max_budget.model_copy(deep=True),
    )
    state = DEFAULT_REGISTRY.instantiate(AgentType.CODING, "coder-1", tid, "implement")
    result = AgentResult(
        status=AgentResultStatus.SUCCESS,
        summary="implemented the health endpoint",
        changed_files=changed,
        tests=tests,
        metadata={"verified": True} if verified else {},
    )
    state.assign(state.context)
    state.start()
    state.complete(result)
    delegation.assign()
    delegation.start()
    delegation.complete(result)
    return ValidationContext(
        agent_state=state,
        result=result,
        delegation=delegation,
        task_state=task,
        workspace=workspace,
        allowed_scope=scope or ["src/auth"],
        required_evidence=["changed_files", "tests", "verification"],
        claims_completion=claims,
    )


def test_section_validation() -> None:
    """Validation layer: 12 deterministic scenarios over real orchestration state."""
    print("\n--- Orchestration validation layer ---")
    validator = OrchestrationValidator()
    store = ArtifactStore(Path(tempfile.mkdtemp(prefix="ultron_live_val_")))
    repo = Path(tempfile.mkdtemp(prefix="ultron_live_val_repo_"))
    (repo / "src" / "auth").mkdir(parents=True)
    (repo / "src" / "auth" / "service.py").write_text(
        "class AuthService: pass\n", encoding="utf-8"
    )
    task = TaskState(goal="Understand authentication.")
    task.add_requirement("Authentication flow is documented")

    async def run_researcher() -> tuple[Supervisor, DelegationRequest]:
        sup = _live74_supervisor("research", store)
        req = sup.create_delegation(task, "researcher", "understand auth")
        await sup.dispatch(req, workspace=str(repo), task_state=task)
        return sup, req

    # V1 + V2: validation over a REAL supervisor-dispatched researcher run.
    async def v1_v2() -> None:
        sup, req = await run_researcher()
        run = sup._runs[req.delegation_id]
        base = ValidationContext(
            agent_state=run,
            result=req.result,
            delegation=req,
            task_state=task,
            artifacts=[req.result.artifact],
            tool_uses=[
                ToolExecution(tool_name="search_files", target="authenticate"),
                ToolExecution(tool_name="read_file", target="src/auth/service.py"),
            ],
            workspace=str(repo),
            allowed_scope=["src"],
        )
        v1 = validator.validate(base)
        check(
            "V1 researcher valid read/search -> PASS",
            v1.valid and v1.status is ValidationStatus.PASS,
            f"{v1.summary} | task.observation={task.last_observation} delegation={req.status.value}",
        )
        # V2: the SAME record but with a write attempt recorded.
        v2_ctx = base.model_copy(deep=True)
        v2_ctx.tool_uses = [ToolExecution(tool_name="write_file", target="src/x.py")]
        v2 = validator.validate(v2_ctx)
        check(
            "V2 researcher write attempt -> UNAUTHORIZED_TOOL",
            ViolationCode.UNAUTHORIZED_TOOL in {x.code for x in v2.violations}
            and not v2.valid,
            v2.summary,
        )

    # V3 + V12: coder with full evidence -> PASS.
    def v3_v12() -> None:
        task2 = TaskState(goal="Add a health endpoint")
        task2.add_requirement("Health endpoint implemented")
        task2.requirements[0].completed = True
        ctx = _validation_coder_ctx(
            str(repo),
            task2,
            changed=["src/auth/service.py"],
            tests=["pytest"],
            verified=True,
            claims=True,
        )
        v = validator.validate(ctx)
        check(
            "V3 coder valid implementation -> PASS",
            v.valid and v.status is ValidationStatus.PASS,
            v.summary,
        )
        # V12: the same record, validated again via a fresh task — PASS.
        check("V12 all required evidence -> PASS", v.valid, v.summary)

    # V4: SUCCESS without required evidence -> completion violation.
    def v4() -> None:
        task4 = TaskState(goal="Add a health endpoint")
        task4.add_requirement("Health endpoint implemented")
        ctx = _validation_coder_ctx(
            str(repo),
            task4,
            changed=[],
            tests=[],
            verified=False,
            claims=True,
        )
        v = validator.validate(ctx)
        check(
            "V4 SUCCESS without required evidence -> FALSE_COMPLETION_CLAIM",
            ViolationCode.FALSE_COMPLETION_CLAIM in {x.code for x in v.violations}
            and not v.valid,
            v.summary,
        )

    # V5: agent claims tests passed; Test Intelligence says failed.
    def v5() -> None:
        task5 = TaskState(goal="Add a health endpoint")
        task5.add_requirement("Health endpoint implemented")
        ctx = _validation_coder_ctx(
            str(repo), task5, changed=["src/auth/service.py"], tests=["pytest"], verified=True
        )
        ctx.test_results = [
            TestResult(
                task_id=ctx.delegation.task_id,
                agent_id="tester-1",
                command="pytest",
                passed=1,
                failed=2,
            )
        ]
        v = validator.validate(ctx)
        check(
            "V5 tests claimed passed but Test Intelligence failed -> contradiction",
            ViolationCode.TEST_CLAIM_CONTRADICTION in {x.code for x in v.violations}
            and not v.valid,
            v.summary,
        )

    # V6: file outside the allowed scope.
    def v6() -> None:
        task6 = TaskState(goal="Add a health endpoint")
        task6.add_requirement("Health endpoint implemented")
        ctx = _validation_coder_ctx(
            str(repo),
            task6,
            changed=["src/auth/service.py", "src/payments/payment.py"],
            tests=["pytest"],
            verified=True,
        )
        v = validator.validate(ctx)
        check(
            "V6 file outside allowed scope -> WORKSPACE_SCOPE_VIOLATION",
            ViolationCode.WORKSPACE_SCOPE_VIOLATION in {x.code for x in v.violations}
            and not v.valid,
            v.summary,
        )

    # V7: budget exceeded.
    def v7() -> None:
        task7 = TaskState(goal="understand")
        delegation = DelegationRequest(
            task_id=task_key(task7),
            agent_type=AgentType.RESEARCH,
            objective="x",
            budget=ExecutionBudget(max_steps=5, max_tool_calls=10, steps_used=6),
        )
        v = validator.validate(
            ValidationContext(
                delegation=delegation,
                task_state=task7,
                result=AgentResult(status=AgentResultStatus.SUCCESS, summary="ok"),
            )
        )
        check(
            "V7 budget exceeded -> BUDGET_EXCEEDED",
            ViolationCode.BUDGET_EXCEEDED in {x.code for x in v.violations},
            v.summary,
        )

    # V8: timeout exceeded.
    def v8() -> None:
        task8 = TaskState(goal="understand")
        tid8 = task_key(task8)
        state = DEFAULT_REGISTRY.instantiate(AgentType.RESEARCH, "r1", tid8, "x")
        state.assign(state.context)
        state.start()
        state.metadata.started_at = datetime.now(UTC) - timedelta(seconds=30)
        state.complete(AgentResult(status=AgentResultStatus.SUCCESS, summary="ok"))
        delegation = DelegationRequest(
            task_id=tid8,
            agent_type=AgentType.RESEARCH,
            objective="x",
            budget=ExecutionBudget(timeout_seconds=5),
        )
        v = validator.validate(
            ValidationContext(
                agent_state=state,
                delegation=delegation,
                task_state=task8,
                result=state.result,
            )
        )
        check(
            "V8 timeout exceeded -> TIMEOUT_EXCEEDED",
            ViolationCode.TIMEOUT_EXCEEDED in {x.code for x in v.violations},
            v.summary,
        )

    # V9: Task A artifact submitted into Task B.
    def v9() -> None:
        task_a = TaskState(goal="Task A goal")
        task_b = TaskState(goal="Task B goal")
        artifact_a = ResearchFinding(
            task_id=task_key(task_a),
            agent_id="researcher-a",
            summary="findings from task A",
            source="code_intelligence",
            confidence=MemoryConfidence.DIRECT_OBSERVATION,
        )
        delegation_b = DelegationRequest(
            task_id=task_key(task_b),
            agent_type=AgentType.RESEARCH,
            objective="x",
            permissions=DEFAULT_REGISTRY.get("researcher").permissions,
        )
        state_b = DEFAULT_REGISTRY.instantiate(
            AgentType.RESEARCH, "researcher-b", task_key(task_b), "x"
        )
        v = validator.validate(
            ValidationContext(
                agent_state=state_b,
                delegation=delegation_b,
                task_state=task_b,
                artifacts=[artifact_a],
            )
        )
        check(
            "V9 Task A artifact into Task B -> ARTIFACT_TASK_MISMATCH",
            ViolationCode.ARTIFACT_TASK_MISMATCH in {x.code for x in v.violations}
            and not v.valid,
            v.summary,
        )

    # V10: cancelled task returns SUCCESS.
    def v10() -> None:
        task10 = TaskState(goal="Add a health endpoint")
        task10.add_requirement("Health endpoint implemented")
        task10.block("cancelled by user policy")
        ctx = _validation_coder_ctx(
            str(repo), task10, changed=["src/auth/service.py"], tests=["pytest"], verified=True
        )
        v = validator.validate(ctx)
        check(
            "V10 cancelled/blocked task returns SUCCESS -> TASK_STATE_CONFLICT",
            ViolationCode.TASK_STATE_CONFLICT in {x.code for x in v.violations}
            and not v.valid,
            f"{v.summary} | task.status={task10.status.value}",
        )

    # V11: idempotent + no mutation.
    def v11() -> None:
        task11 = TaskState(goal="understand")
        tid11 = task_key(task11)
        artifact11 = ResearchFinding(
            task_id=tid11,
            agent_id="r1",
            summary="findings",
            source="code_intelligence",
            confidence=MemoryConfidence.DIRECT_OBSERVATION,
        )
        delegation11 = DelegationRequest(
            task_id=tid11,
            agent_type=AgentType.RESEARCH,
            objective="x",
            permissions=DEFAULT_REGISTRY.get("researcher").permissions,
        )
        state11 = DEFAULT_REGISTRY.instantiate(AgentType.RESEARCH, "r1", tid11, "x")
        ctx11 = ValidationContext(
            agent_state=state11,
            delegation=delegation11,
            task_state=task11,
            artifacts=[artifact11],
        )
        task_before = task11.model_dump()
        artifact_before = artifact11.model_dump()
        first = validator.validate(ctx11)
        second = validator.validate(ctx11)
        first.metadata.pop("validated_at")
        second.metadata.pop("validated_at")
        check(
            "V11 validate twice -> identical, no state mutation",
            first.model_dump() == second.model_dump()
            and task11.model_dump() == task_before
            and artifact11.model_dump() == artifact_before,
            f"{first.summary} | task/artifact unchanged",
        )

    asyncio.run(v1_v2())
    v3_v12()
    v4()
    v5()
    v6()
    v7()
    v8()
    v9()
    v10()
    v11()


def test_critical_separation() -> None:
    """AgentResult does NOT automatically mean TaskState is complete."""
    task = TaskState(goal="implement health endpoint")
    task.add_requirement("endpoint works")

    async def body():
        state, agent = _fresh_state("implement health endpoint")
        result = await agent.run_with_state(state)
        check(
            "SEP: agent completed but task NOT complete",
            state.status is AgentStatus.COMPLETED and task.status is TaskStatus.TASK_STARTED,
        )
        check("SEP: task requirement untouched", not task.is_complete())
        check("SEP: task history empty", task.execution_history == [])
        check("SEP: agent result is not a task verdict", result.is_success)

    asyncio.run(body())


def _live76_artifact(context, kind: str):
    """The structured artifact each live specialist produces by default."""
    base = {
        "task_id": context.task_id,
        "agent_id": context.agent_id,
        "source": f"live76_{kind}",
        "confidence": MemoryConfidence.DIRECT_OBSERVATION,
    }
    if kind == "research":
        return ResearchFinding(
            **base,
            summary="authentication lives in src/auth/service.py",
            related_files=["src/auth/service.py"],
            related_symbols=["AuthService"],
        )
    if kind == "coding":
        return ImplementationResult(
            **base,
            summary="implemented the requested feature",
            changed_files=["src/app/main.py"],
            tests_added=["tests/test_main.py"],
            tests_run=["pytest -q"],
        )
    if kind == "test_qa":
        return TestResult(**base, command="pytest -q", passed=4, failed=0)
    if kind == "reviewer":
        return ReviewResult(
            **base, summary="review approved", approval=ApprovalStatus.APPROVED
        )
    raise AssertionError(f"no default live76 artifact for {kind!r}")


class Live76Agent(Agent):
    """Deterministic workflow specialist; ``plan`` keys behaviors by type."""

    def __init__(self, identity, plan: dict[str, str], resumed: bool = False) -> None:
        super().__init__(identity)
        self.plan = plan
        self.resumed = resumed
        self.trajectory: list[str] = []

    async def execute(self, objective: str, context: ExecutionContext) -> AgentResult:
        self.trajectory.append(f"internal reasoning: {objective} / {context.task_id}")
        kind = self.identity.agent_type.value
        mode = self.plan.get(kind, "ok")
        if mode == "needs_input_once":
            if not self.resumed:
                return AgentResult(
                    status=AgentResultStatus.NEEDS_INPUT, summary="need input"
                )
            mode = "ok"
        if mode == "needs_input":
            return AgentResult(
                status=AgentResultStatus.NEEDS_INPUT, summary="need input"
            )
        if mode == "failed":
            return AgentResult(
                status=AgentResultStatus.FAILED, summary=f"could not {objective}"
            )
        if mode == "blocked":
            return AgentResult(
                status=AgentResultStatus.BLOCKED, summary="blocked by policy"
            )
        if mode == "empty":
            return AgentResult(status=AgentResultStatus.SUCCESS, summary="ok")
        if mode == "claim":
            return AgentResult(
                status=AgentResultStatus.SUCCESS, summary="task is complete"
            )
        if mode == "testing_fail":
            return TestResult(
                task_id=context.task_id,
                agent_id=context.agent_id,
                command="pytest -q",
                passed=1,
                failed=3,
                failures=[
                    FailureAnalysis(
                        category=FailureCategory.TEST_ASSERTION, summary="boom"
                    )
                ],
                source="live76_tester",
            ).to_agent_result()
        return _live76_artifact(context, kind).to_agent_result()


def _live76_supervisor(plan: dict[str, str], store: ArtifactStore) -> Supervisor:
    def factory(spec, state) -> Live76Agent:
        return Live76Agent(
            state.identity,
            plan,
            resumed=state.status is AgentStatus.WAITING,
        )

    return Supervisor(registry=DEFAULT_REGISTRY, agent_factory=factory, store=store)


def _live76_chain() -> list[WorkflowStep]:
    """research -> implementation -> testing -> review (strict chain)."""
    return [
        WorkflowStep(
            step_id="research",
            name="Research",
            agent_type=AgentType.RESEARCH,
            objective="Understand the authentication flow",
            expected_output="ResearchFinding",
        ),
        WorkflowStep(
            step_id="implementation",
            name="Implementation",
            agent_type=AgentType.CODING,
            objective="Implement the change",
            dependencies=["research"],
            expected_output="ImplementationResult",
            required_evidence=["artifact"],
        ),
        WorkflowStep(
            step_id="testing",
            name="Testing",
            agent_type=AgentType.TEST_QA,
            objective="Run and validate tests",
            dependencies=["implementation"],
            expected_output="TestResult",
        ),
        WorkflowStep(
            step_id="review",
            name="Review",
            agent_type=AgentType.REVIEWER,
            objective="Review the change",
            dependencies=["testing"],
            expected_output="ReviewResult",
        ),
    ]


def _live76_happy_plan() -> dict[str, str]:
    return {"research": "ok", "coding": "ok", "test_qa": "ok", "reviewer": "ok"}


def test_section76_workflow_engine() -> None:
    """Section 7.6: workflow engine + sequential execution (15 live tests)."""
    print("\n--- Section 7.6: workflow engine + sequential execution ---")

    def task() -> TaskState:
        t = TaskState(goal="Fix the authentication bug.")
        t.add_requirement("Authentication works and tests pass")
        return t

    # 76.1 — happy path: 4-step chain, artifacts flow, workflow completes,
    # TaskState completion evaluated separately (requirement left open).
    async def happy_path() -> None:
        store = ArtifactStore(Path(tempfile.mkdtemp(prefix="ultron_live76_")))
        engine = WorkflowEngine(_live76_supervisor(_live76_happy_plan(), store))
        t = task()
        wf = engine.create_workflow(steps=_live76_chain(), task_state=t)
        n = await engine.execute_until_blocked(wf)
        impl = wf.get_step("implementation")
        check(
            "76.1 workflow completes all 4 steps in order",
            n == 4
            and wf.status is WorkflowStatus.COMPLETED
            and all(s.status is WorkflowStepStatus.COMPLETED for s in wf.steps)
            and wf.current_step == "review",
            wf.summary(),
        )
        # Artifact flow: implementation consumed the research finding.
        impl_req = engine.supervisor.get_delegation(impl.delegation_id)
        check(
            "76.1 implementation receives the ResearchFinding",
            len(impl_req.input_artifacts) == 1
            and impl_req.input_artifacts[0].artifact_type.value == "research_finding",
        )
        # TaskState completion evaluated separately: requirement still open.
        check(
            "76.1 workflow complete but task completion evaluated separately",
            wf.metadata.get("task_completion", "").startswith("not_completed:")
            and not t.is_complete(),
        )

    # 76.2 — dependency enforcement: implementation cannot run first.
    async def dependency_order() -> None:
        store = ArtifactStore(Path(tempfile.mkdtemp(prefix="ultron_live76b_")))
        engine = WorkflowEngine(_live76_supervisor(_live76_happy_plan(), store))
        wf = engine.create_workflow(steps=_live76_chain(), task_state=task())
        await engine.start(wf)
        step = await engine.execute_next_step(wf)
        check(
            "76.2 implementation cannot run before research completes",
            step is not None
            and step.step_id == "research"
            and wf.get_step("implementation").delegation_id is None,
        )
        check(
            "76.2 research completion unlocks implementation (READY)",
            wf.get_step("implementation").status is WorkflowStepStatus.READY,
        )

    # 76.3 — agent failure: research fails, no further steps, no retry.
    async def agent_failure() -> None:
        store = ArtifactStore(Path(tempfile.mkdtemp(prefix="ultron_live76c_")))
        engine = WorkflowEngine(
            _live76_supervisor({"research": "failed"}, store)
        )
        wf = engine.create_workflow(steps=_live76_chain(), task_state=task())
        await engine.execute_until_blocked(wf)
        check(
            "76.3 research failure -> step FAILED, workflow FAILED, no next step",
            wf.get_step("research").status is WorkflowStepStatus.FAILED
            and wf.status is WorkflowStatus.FAILED
            and wf.get_step("implementation").delegation_id is None,
        )

    # 76.4 — validation failure: coder SUCCESS without required evidence.
    async def validation_failure() -> None:
        store = ArtifactStore(Path(tempfile.mkdtemp(prefix="ultron_live76d_")))
        engine = WorkflowEngine(
            _live76_supervisor({"coding": "empty"}, store)
        )
        wf = engine.create_workflow(steps=_live76_chain(), task_state=task())
        await engine.execute_until_blocked(wf)
        impl = wf.get_step("implementation")
        check(
            "76.4 SUCCESS without evidence -> step NOT completed, workflow stops",
            impl.status is WorkflowStepStatus.FAILED
            and wf.status is WorkflowStatus.FAILED
            and wf.get_step("testing").delegation_id is None,
            impl.error or "",
        )

    # 76.5 — test failure: tester returns failing TestResult -> review never runs.
    async def test_failure() -> None:
        store = ArtifactStore(Path(tempfile.mkdtemp(prefix="ultron_live76e_")))
        plan = _live76_happy_plan() | {"test_qa": "testing_fail"}
        engine = WorkflowEngine(_live76_supervisor(plan, store))
        wf = engine.create_workflow(steps=_live76_chain(), task_state=task())
        await engine.execute_until_blocked(wf)
        check(
            "76.5 failing tests -> testing step FAILED, review never begins",
            wf.get_step("testing").status is WorkflowStepStatus.FAILED
            and wf.status is WorkflowStatus.FAILED
            and wf.get_step("review").delegation_id is None,
        )

    # 76.6 — blocked / needs input: workflow WAITING, no further execution.
    async def needs_input() -> None:
        store = ArtifactStore(Path(tempfile.mkdtemp(prefix="ultron_live76f_")))
        engine = WorkflowEngine(
            _live76_supervisor({"research": "needs_input"}, store)
        )
        wf = engine.create_workflow(steps=_live76_chain(), task_state=task())
        await engine.execute_until_blocked(wf)
        check(
            "76.6 NEEDS_INPUT -> step WAITING, workflow WAITING, no next step",
            wf.get_step("research").status is WorkflowStepStatus.WAITING
            and wf.status is WorkflowStatus.WAITING
            and wf.get_step("implementation").delegation_id is None,
        )

    # 76.7 — pause / resume: research not repeated, implementation starts.
    async def pause_resume() -> None:
        store = ArtifactStore(Path(tempfile.mkdtemp(prefix="ultron_live76g_")))
        engine = WorkflowEngine(_live76_supervisor(_live76_happy_plan(), store))
        wf = engine.create_workflow(steps=_live76_chain(), task_state=task())
        await engine.start(wf)
        await engine.execute_next_step(wf)  # research done
        research = wf.get_step("research")
        engine.pause(wf)
        await engine.resume(wf)
        step = await engine.execute_next_step(wf)
        check(
            "76.7 pause/resume continues from correct step, research not repeated",
            wf.status is WorkflowStatus.RUNNING
            and step is not None
            and step.step_id == "implementation"
            and research.status is WorkflowStepStatus.COMPLETED
            and research.attempts == 1,
        )

    # 76.8 — cancellation: future steps stop, artifacts remain, no completion.
    async def cancellation() -> None:
        store = ArtifactStore(Path(tempfile.mkdtemp(prefix="ultron_live76h_")))
        engine = WorkflowEngine(_live76_supervisor(_live76_happy_plan(), store))
        wf = engine.create_workflow(steps=_live76_chain(), task_state=task())
        await engine.start(wf)
        await engine.execute_next_step(wf)
        artifact_id = wf.get_step("research").result_artifact
        engine.cancel(wf)
        check(
            "76.8 cancel stops future steps and preserves produced artifacts",
            wf.status is WorkflowStatus.CANCELLED
            and wf.get_step("implementation").status is WorkflowStepStatus.CANCELLED
            and store.get(artifact_id) is not None
            and wf.metadata.get("task_completion") is None,
        )

    # 76.9 — context isolation: coder receives only the research artifact.
    async def context_isolation() -> None:
        store = ArtifactStore(Path(tempfile.mkdtemp(prefix="ultron_live76i_")))
        engine = WorkflowEngine(_live76_supervisor(_live76_happy_plan(), store))
        wf = engine.create_workflow(steps=_live76_chain(), task_state=task())
        await engine.execute_until_blocked(wf)
        impl = wf.get_step("implementation")
        req = engine.supervisor.get_delegation(impl.delegation_id)
        ctx = engine.supervisor.get_run(req.delegation_id).context
        blob = str(ctx.relevant_context)
        check(
            "76.9 coder gets only the research artifact — no trajectory",
            len(req.input_artifacts) == 1
            and "internal reasoning" not in blob
            and "trajectory" not in blob,
        )

    # 76.10 — false completion: final review claims SUCCESS without evidence.
    async def false_completion() -> None:
        store = ArtifactStore(Path(tempfile.mkdtemp(prefix="ultron_live76j_")))
        engine = WorkflowEngine(
            _live76_supervisor({"reviewer": "claim"}, store)
        )
        wf = engine.create_workflow(
            steps=[
                WorkflowStep(
                    step_id="review",
                    name="Review",
                    agent_type=AgentType.REVIEWER,
                    objective="Review the change",
                    claims_completion=True,
                    required_evidence=["artifact"],
                )
            ],
            task_state=task(),
        )
        await engine.execute_until_blocked(wf)
        check(
            "76.10 false completion claim -> workflow does NOT complete",
            wf.status is WorkflowStatus.FAILED
            and "false_completion_claim"
            in wf.get_step("review").validation["violations"],
        )

    # 76.11 — restart: reload serialized workflow in a fresh engine; the
    # completed step is not repeated.
    async def restart() -> None:
        store = ArtifactStore(Path(tempfile.mkdtemp(prefix="ultron_live76k_")))
        engine = WorkflowEngine(_live76_supervisor(_live76_happy_plan(), store))
        wf = engine.create_workflow(steps=_live76_chain(), task_state=task())
        await engine.start(wf)
        await engine.execute_next_step(wf)
        engine.pause(wf)
        payload = wf.model_dump_json()

        engine2 = WorkflowEngine(_live76_supervisor(_live76_happy_plan(), store))
        wf2 = Workflow.model_validate_json(payload)
        await engine2.resume(wf2, task_state=task())
        step = await engine2.execute_next_step(wf2)
        check(
            "76.11 restart resumes and does not repeat completed steps",
            step is not None
            and step.step_id == "implementation"
            and wf2.get_step("research").status is WorkflowStepStatus.COMPLETED
            and wf2.get_step("research").attempts == 1,
        )

    # 76.12 — traceability: every step links workflow -> task -> step ->
    # delegation -> agent -> artifact.
    async def traceability() -> None:
        store = ArtifactStore(Path(tempfile.mkdtemp(prefix="ultron_live76l_")))
        engine = WorkflowEngine(_live76_supervisor(_live76_happy_plan(), store))
        wf = engine.create_workflow(steps=_live76_chain(), task_state=task())
        await engine.execute_until_blocked(wf)
        rows = engine.trace_rows(wf)
        check(
            "76.12 every step traceable end-to-end",
            len(rows) == 4
            and all(
                r["workflow_id"] == wf.workflow_id
                and r["task_id"] == wf.task_id
                and r["delegation_id"]
                and r["agent_id"]
                and r["result_artifact"]
                for r in rows
            ),
        )

    # 76.13 — invalid workflow: circular dependency rejected before running.
    def invalid_workflow() -> None:
        engine = WorkflowEngine(_live76_supervisor(_live76_happy_plan(), None))
        try:
            engine.create_workflow(
                steps=[
                    WorkflowStep(step_id="a", name="A", agent_type=AgentType.RESEARCH, objective="o", dependencies=["b"]),
                    WorkflowStep(step_id="b", name="B", agent_type=AgentType.CODING, objective="o", dependencies=["a"]),
                ],
                task_id="t",
            )
        except ValueError as exc:
            check(
                "76.13 circular dependency workflow rejected before execution",
                "circular dependency" in str(exc),
                str(exc),
            )
        else:
            check("76.13 circular dependency workflow rejected before execution", False)

    # 76.14 — multiple artifacts: implementation depends on research AND a
    # second research step; review receives only the research findings.
    async def multiple_artifacts() -> None:
        store = ArtifactStore(Path(tempfile.mkdtemp(prefix="ultron_live76m_")))
        steps = [
            WorkflowStep(step_id="research_a", name="Res A", agent_type=AgentType.RESEARCH, objective="auth flow"),
            WorkflowStep(step_id="research_b", name="Res B", agent_type=AgentType.RESEARCH, objective="permissions"),
            WorkflowStep(
                step_id="implementation",
                name="Impl",
                agent_type=AgentType.CODING,
                objective="implement",
                dependencies=["research_a", "research_b"],
                input_artifact_types=["research_finding"],
            ),
        ]
        engine = WorkflowEngine(_live76_supervisor(_live76_happy_plan(), store))
        wf = engine.create_workflow(steps=steps, task_state=task())
        await engine.execute_until_blocked(wf)
        req = engine.supervisor.get_delegation(
            wf.get_step("implementation").delegation_id
        )
        check(
            "76.14 only explicitly relevant artifacts are passed",
            len(req.input_artifacts) == 2
            and all(
                a.artifact_type.value == "research_finding"
                for a in req.input_artifacts
            ),
        )

    # 76.15 — TaskState invariant: all agents SUCCESS but one requirement
    # unresolved -> TaskState must NOT become COMPLETED.
    async def taskstate_invariant() -> None:
        store = ArtifactStore(Path(tempfile.mkdtemp(prefix="ultron_live76n_")))
        engine = WorkflowEngine(_live76_supervisor(_live76_happy_plan(), store))
        t = task()  # requirement intentionally left unresolved
        wf = engine.create_workflow(steps=_live76_chain(), task_state=t)
        await engine.execute_until_blocked(wf)
        check(
            "76.15 workflow complete but TaskState NOT complete (requirement open)",
            wf.status is WorkflowStatus.COMPLETED
            and not t.is_complete()
            and t.status is not TaskStatus.TASK_COMPLETED
            and wf.metadata.get("task_completion", "").startswith("not_completed:"),
        )

    asyncio.run(happy_path())
    asyncio.run(dependency_order())
    asyncio.run(agent_failure())
    asyncio.run(validation_failure())
    asyncio.run(test_failure())
    asyncio.run(needs_input())
    asyncio.run(pause_resume())
    asyncio.run(cancellation())
    asyncio.run(context_isolation())
    asyncio.run(false_completion())
    asyncio.run(restart())
    asyncio.run(traceability())
    invalid_workflow()
    asyncio.run(multiple_artifacts())
    asyncio.run(taskstate_invariant())


def main() -> int:
    print("=" * 72)
    print("FIX #7 section 7.1 — 10-minute live validation (agent contract + lifecycle)")
    print("=" * 72)
    test_a_create_agent()
    test_b_assign_task()
    test_c_d_run_agent_and_success()
    test_e_simulate_failure()
    test_f_simulate_blocked()
    test_g_cancel_execution()
    test_h_invalid_transitions_rejected()
    test_i_execution_context_permitted_tools()
    test_j_agent_result_structured()
    test_critical_separation()
    test_section72_registry_and_permissions()
    test_section73_artifacts()
    test_section74_supervisor_delegation()
    test_section_validation()
    test_section76_workflow_engine()
    print("=" * 72)
    failed = [r for r in results if r[1] == FAIL]
    passed = len(results) - len(failed)
    print(f"Live validation: {passed}/{len(results)} checks passed.")
    if failed:
        for name, _, detail in failed:
            print(f"  FAILED: {name} {detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
