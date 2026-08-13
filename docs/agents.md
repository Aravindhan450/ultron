# Ultron – Agents System

Agents are the “workers” that turn a user request into actual work.  
They sit on top of the **Engine** and use **Tools + Memory**.

## Agent Types

| Agent            | Style              | Status          | When to use                                      | Strengths                          | Limitations                     |
|------------------|--------------------|-----------------|--------------------------------------------------|------------------------------------|---------------------------------|
| **Simple**       | Single-shot        | ✅ Implemented  | Direct questions, no tools needed                | Fast, cheap                        | Cannot use tools                |
| **ReAct**        | Reason + Act loop  | ✅ Implemented  | Most tool-using tasks                            | Good balance of reasoning & action | Can loop if poorly prompted     |
| **Orchestrator** | Multi-agent boss   | 🚧 Planned      | Complex tasks that need several specialists      | Coordinates others                 | Higher latency                  |
| **CodeAct**      | Code-centric       | 🚧 Planned      | Tasks best solved by writing & running code      | Extremely powerful for coding      | Needs sandbox                   |
| **Operative**    | Long-running       | 🚧 Planned      | Background jobs, monitoring, multi-step missions | Persistent, stateful               | More complex state management   |
| **Monitor**      | Observer           | 🚧 Planned      | Watch for events / conditions and react          | Reactive, low overhead             | Not for open-ended goals        |

## How Agents Work (High Level)

1. **Intelligence** layer chooses the best model + agent type for the request.
2. Agent receives:
   - User message (or voice transcript)
   - Relevant memory
   - Available tools (filtered by permission + relevance)
3. Agent reasons (and optionally calls tools).
4. Every tool call is designed to go through:
   - Risk Classifier → Permission system
   - GuardrailsEngine (secrets, PII, boundary checks)
   - Optional sandbox

   The security boundary (risk classifier + guardrails) is wired into the
   agent tool-call flow: every tool call in the simple and ReAct agents is
   routed through ``boundary.check()`` before execution. Verdicts: ``deny``
   → the action is hard-blocked (never offered for confirmation),
   ``confirm`` → interactive approval via the PendingAction flow, ``allow``
   → auto-execution (read-only / LOW-risk actions, or a permissive mode).
5. Final answer is returned (and spoken if in voice mode).

## ReAct Pattern (Most Common) — ✅ Implemented

ReAct (**Re**ason + **Act**) is the workhorse pattern for tool-using tasks.
The LLM drives the whole loop itself:

    Thought → Action (tool call) → Observation → Thought → ... → Final Answer

`ReActAgent` (`src/ultron/core/agents/react.py`) decides *when* a tool is
needed, emits a JSON tool call, reads the observation, and iterates until it
has enough information to answer. Unlike `SimpleAgent` (deterministic regex
detectors + a single-shot LLM fallback), the model plans the full multi-step
tool sequence — a much better fit for open-ended requests.

Key details:

- The system prompt is built **at call time from the live Tool Registry
  schema**, so newly registered tools are immediately visible to the model.
- Tool calls are fenced JSON blocks (`{"tool": "...", "arguments": {...}}`);
  `extract_tool_call()` parses them robustly (fenced or bare, including
  arguments with nested braces).
- **Safety model:** read-only / low-risk tools (`read_file`, web search, page
  fetch, memory lookups, read-only SQL, GET requests) execute directly inside
  the loop. State-modifying actions (`run_command`, `write_file`, non-read-only
  SQL, POST/PUT/DELETE requests) **never execute silently** — the agent returns
  a `PendingAction` so the CLI shows an interactive confirmation first.
- A `max_iterations` cap (default 10) prevents runaway loops if the model
  keeps calling tools without ever reaching a final answer.
- **Parallel batching:** when the model needs several independent read-only
  lookups at once, it can emit a single `run_tool_batch` call whose
  `calls_json` is a JSON array of `{"tool", "arguments"}` — the batch runs
  auto-allowed members concurrently, gates each member through the same
  security boundary (deny never runs, confirm never runs silently), and
  returns one synthesized report as the Observation. See
  [docs/parallel-tools.md](docs/parallel-tools.md).

### Selecting the ReAct agent

The CLI defaults to `simple`. Launch with ReAct instead:

    ultron chat --agent react

(short form: `ultron chat -a react`). You can also switch between the two
mid-session without restarting — type `/agent` for an interactive picker or
`/agent react` to switch directly. The prompt shows the active agent
(`[model | react] You:`), and the switch preserves your chosen model.

## CodeAct Pattern

The agent writes Python (or shell) code, runs it in a sandbox, sees the output, and iterates.

## Orchestration Layer (Fix #7)

`src/ultron/core/orchestration/` is the multi-agent orchestration layer.
Built incrementally; **sections 7.1 (agent contract + lifecycle), 7.2
(agent registry + permissions), 7.3 (structured agent results +
artifacts), 7.4 (supervisor + specialist delegation), the deterministic
validation layer, and 7.6 (workflow engine + sequential execution) are
implemented** — parallel execution, dynamic routing, workspace locks and
agent-to-agent messaging are later sections and are intentionally not
implemented yet.

Section 7.6 (workflow engine — `workflow.py`):

- **Models**: `Workflow` (workflow_id, task_id, ordered steps, status,
  current_step, events) and `WorkflowStep` (step_id, agent_type,
  objective, dependencies, input artifacts, required evidence, claims
  completion flag, allowed scope) with their own transition-validated
  state machines — `Workflow: PENDING → RUNNING → PAUSED/WAITING →
  RUNNING → COMPLETED` (FAILED / BLOCKED / CANCELLED terminal),
  `WorkflowStep: PENDING → READY → RUNNING → WAITING → RUNNING →
  COMPLETED`. `BLOCKED` is terminal by design (mirrors
  `TaskState.block()`); only PAUSED and WAITING workflows are resumable.
- **Dependency validation**: missing / unknown / self / circular /
  cross-workflow dependencies, duplicate step or workflow ids,
  unregistered agent types, and empty workflows are rejected before any
  step may execute.
- **Sequential execution**: `WorkflowEngine` routes every step through
  the existing `Supervisor` (never instantiates agents itself) —
  `create_workflow / validate_workflow / start / execute_next_step /
  execute_until_blocked / pause / resume / cancel / get_status`. A step
  runs only when all its dependencies are COMPLETED; completed steps are
  never repeated, including across serialization/reload (artifacts are
  re-resolved from the `ArtifactStore`).
- **Validation gate**: every `AgentResult` is checked by
  `OrchestrationValidator` (lifecycle, permissions, budget, timeout,
  result schema, artifacts, workspace scope, TaskState consistency,
  completion claims, ownership) plus deterministic engine gates
  (`required_evidence` and failing `TestResult` artifacts) — a SUCCESS
  claim without evidence never completes a step.
- **Failure mapping**: agent FAILED/CANCELLED → step + workflow FAILED /
  CANCELLED; agent NEEDS_INPUT → step + workflow WAITING (resumable);
  agent BLOCKED → step + workflow BLOCKED (terminal). No automatic
  retries.
- **TaskState integration**: the workflow owns orchestration only.
  Completion invokes the existing `TaskState.mark_complete()` API — the
  engine never forces `task.status`; when requirements remain the task
  stays incomplete while the workflow reports its own completion.
- **Artifact flow**: a step receives only the result artifacts of its
  completed dependencies (optionally filtered by
  `input_artifact_types`) — the only cross-agent channel, never internal
  trajectories.
- **Observability**: `WorkflowEvent`s (CREATED / STARTED / READY /
  STARTED / COMPLETED / FAILED / BLOCKED / WAITING / PAUSED / RESUMED /
  CANCELLED) logged via `get_logger`, plus `trace_rows()` linking
  workflow → task → step → delegation → agent → artifact.

Section 7.1 pieces:

- **Lifecycle** (`lifecycle.py`): explicit, transition-validated agent
  lifecycle `PENDING → ASSIGNED → RUNNING → WAITING → RUNNING →
  COMPLETED`, with failure states `FAILED / BLOCKED / CANCELLED` reachable
  from any active state. Terminal states accept no further transitions;
  `COMPLETED` is only reachable from `RUNNING`. Invalid transitions raise
  `ValueError`.
- **Identity** (`AgentType`, `AgentIdentity`): agent_id + agent_type
  (`supervisor`, `research`, `coding`, `test_qa`, `reviewer`, `security`).
- **Result** (`AgentResult`): structured outcome — `SUCCESS / FAILED /
  BLOCKED / CANCELLED / NEEDS_INPUT` with summary, artifacts, evidence,
  changed_files, tests, blockers, recommendations, metadata. Never bare
  strings as the communication protocol.
- **Budget** (`ExecutionBudget`): per-run limits on steps / tool calls /
  wall-clock; enforcement (incl. the advisory `timed_out()`) is the
  agent's responsibility at this layer.
- **Execution context** (`ExecutionContext`): the agent's *scoped view* —
  task_id, workspace, deny-by-default `allowed_tools`, permission profile,
  budget, current plan step, cancellation flag. It never duplicates
  TaskState.
- **Runtime record** (`AgentState`): identity + objective + context +
  lifecycle status + result + audit metadata, with lifecycle methods
  (`assign/start/wait/resume/complete/fail/block/cancel`).
- **Contract** (`Agent` ABC): every orchestrated agent implements
  `async execute(objective, context) -> AgentResult` and may override
  `cancel()`; `run_with_state()` drives a state to a terminal lifecycle
  state and can resume a WAITING state.

**Critical invariant:** completing an agent run never completes a TaskState
— agent completion and task completion are separate concepts. A supervisor
(later section) must verify a task against its TaskState and plan before
marking it complete.

Section 7.2 pieces:

- **Permissions** (`permissions.py`): a frozen `AgentPermissions` profile
  per agent type — `read` / `write` / `test` / `shell` / `network` levels
  (using the security boundary's own `Decision` enum) plus an `allowed_tools`
  whitelist that is **deny-by-default** (a tool not listed is denied, and
  `allowed_tools` is a tuple so even in-place mutation is impossible).
  `run_command` is classified deterministically: test commands (`pytest`,
  `npm test`, `cargo test`, ...) are TEST, read-only commands (`ls`, `git
  diff`, ...) are READ, everything else is SHELL. CONFIRM-level actions
  delegate to the real `SecurityBoundary` (risk tier + guardrails + mode) —
  the LLM can never bypass or change this.
- **Registry** (`registry.py`): `AgentSpec` (name, capabilities, allowed
  tools, permissions, max budget, risk level) + `AgentRegistry` (register /
  get / list / validate / capability-check / instantiate). Duplicate
  registrations are rejected; unknown agent types fail safely (`KeyError`).
  `DEFAULT_REGISTRY` pre-registers the six baselines — supervisor and
  researcher read/search only, coder writes/tests/shells subject to security
  (CONFIRM), tester read/search/test with no application-code writes,
  reviewer read/search/git-diff with no writes, security read/search/analysis
  with no writes.
- **Runtime-controlled permissions**: the registry is constructed at import
  time and never handed to an agent; agents receive only the scoped
  `ExecutionContext`, whose `check_action()` enforces the frozen profile and
  records verdicts to the same JSON-lines security audit trail.

**Critical invariant (§7.2):** the LLM can NEVER modify its own permissions
— profiles are frozen, the registry is runtime-owned, and no agent API
accepts permission changes.

Section 7.3 pieces (structured agent results + artifacts, `artifacts.py`):

- **Artifact base** (`AgentArtifact`): every artifact carries provenance —
  `task_id` + `agent_id` (ownership), `artifact_id` (deterministic
  `task:agent:type`), `timestamp`, `summary`, `evidence`, `source`,
  `confidence` (reuses Fix #6's `MemoryConfidence` — direct observation is
  never confused with an LLM guess), `related_files` / `related_symbols`,
  and free-form `metadata`.
- **Five artifact types**: `ResearchFinding` (architecture findings,
  uncertainties, recommendations), `ImplementationResult` (changed
  files/symbols, tests added/run, blockers), `TestResult` (reuses Fix #5's
  `CommandResult` + `FailureAnalysis` — **no second test-result system**),
  `ReviewResult` (`ReviewFinding` list with severity, required changes,
  `ApprovalStatus`), and `SecurityFinding` (severity, issue, blocking flag,
  links to the security layer's `GuardrailFinding`).
- **Serialization**: lossless pydantic JSON round-trips with strict
  artifact-type dispatch — unknown types and malformed payloads fail
  loudly; `AgentResult.artifact` is a discriminated union so an envelope
  restored from JSON yields the concrete artifact type, never the base
  class.
- **Envelope** (`to_agent_result()`): folds an artifact into the §7.1
  `AgentResult` (summary, evidence, changed_files, tests,
  recommendations/blockers) — the compact protocol between agents while
  the full structured artifact rides along on `result.artifact`.
- **Persistence** (`ArtifactStore`): JSONL store inside a directory;
  artifacts survive a process restart, corrupted records are skipped, and
  artifacts are queryable by task (`load_for_task`) and agent
  (`load_for_agent`). `task_key()` derives a deterministic task id from a
  TaskState's goal until the supervisor section introduces real ids.

**Critical invariant (§7.3):** agents communicate through structured
artifacts, never by dumping their internal conversation or trajectory — a
researcher's huge internal trajectory never travels onward; only its
`ResearchFinding` does.

Section 7.4 pieces (supervisor + specialist delegation, `delegation.py`):

- **DelegationRequest**: the unit of delegated work — task_id /
  parent_task_id, delegation_id, agent_type, objective, `input_artifacts`
  (structured artifacts are the ONLY cross-agent channel), constraints,
  the frozen permission profile (attached from the registry spec),
  expected_output, budget, timeout_seconds. The delegation lifecycle
  **reuses `AgentStatus`** (`PENDING → ASSIGNED → RUNNING → COMPLETED`, plus
  FAILED / BLOCKED / CANCELLED and WAITING for NEEDS_INPUT pauses) — no
  second state machine, no duplicated transition table.
- **Supervisor** (`Supervisor`): receives a TaskState, selects the
  specialist from the registry (unknown types → `KeyError`), creates the
  DelegationRequest (per-run budget copy + frozen permissions), dispatches
  inside a scoped ExecutionContext built by `AgentRegistry.instantiate`
  (permissions can never be widened), **enforces the timeout** via
  `asyncio.wait_for` + the budget, honors cancellation (mid-run cancels
  only set the context flag; the CANCELLED transition happens in finalize),
  records the structured result + artifact (persisted to the store),
  updates the TaskState (`last_observation` / `TaskError` — never a
  completion claim), and `decide()`s whether orchestration should continue
  (CONTINUE / NEEDS_INPUT / FAILED; COMPLETE reserved for the workflow
  section). Sequential only — no parallel execution.
- **Context isolation** (`task_brief`): a specialist receives only the
  string-only task brief (goal, task type, step tracking, requirements,
  recent errors) + artifact summaries + constraints — never the transcript,
  execution history, other agents' internal reasoning, or the entire
  TaskState.
- **Resume**: a NEEDS_INPUT delegation pauses (WAITING) and `dispatch()`
  resumes the SAME run when re-invoked (reuses the §7.1 WAITING → RUNNING
  machinery); the relevant context is refreshed with any new artifacts.

**Critical invariant (§7.4):** the supervisor is deterministic and has no
LLM — it routes, scopes, enforces and records. It NEVER marks a task
complete; completing a delegation only ever changes the delegation/agent
records, and `decide()` only reports whether more work may follow.

Validation layer (`validation.py`):

- **Deterministic, read-only, no LLM** — `OrchestrationValidator` runs
  modular checks over a `ValidationContext` (agent state, result,
  delegation, TaskState, artifacts, recorded tool uses, workspace, scope,
  required evidence, test results). It never executes, never mutates, and
  is idempotent: it reports, the supervisor decides.
- **Check families**: lifecycle (every recorded transition legal per the
  §7.1 state machine; result status matches the lifecycle status),
  delegation (well-formed + registered type), permissions/tools (every
  recorded tool use whitelisted and category-allowed — deterministic, the
  security boundary's verdicts are inspected, never re-issued),
  budget (used ≤ max; at-limit is a warning, over-limit a violation),
  timeout (elapsed ≤ configured; missing timing data is a warning),
  result schema per status (SUCCESS needs evidence, FAILED needs failure
  info, BLOCKED needs a blocker, NEEDS_INPUT needs the required input),
  artifacts (schema, provenance, agent ownership, task association),
  workspace scope (changed files inside the workspace AND the allowed
  scope — never reverted, only reported), TaskState consistency (a
  blocked/failed/cancelled task must never "succeed"; a completed task
  must have no unresolved steps), completion claims (a SUCCESS claim is
  only justified when required evidence exists and requirements/plan are
  satisfied → `FALSE_COMPLETION_CLAIM`), and ownership (delegation ↔
  task, run ↔ delegation, artifact ↔ task/agent).
- **Result model** (`ValidationResult`): PASS / WARNING / FAIL / BLOCKED,
  with per-check `ValidationCheck`s, `ValidationViolation`s carrying
  stable machine-readable `ViolationCode`s (`INVALID_LIFECYCLE_TRANSITION`,
  `UNAUTHORIZED_TOOL`, `UNAUTHORIZED_FILE_ACCESS`, `BUDGET_EXCEEDED`,
  `TIMEOUT_EXCEEDED`, `INVALID_AGENT_RESULT`, `INVALID_ARTIFACT`,
  `ARTIFACT_OWNERSHIP_VIOLATION`, `ARTIFACT_TASK_MISMATCH`,
  `TASK_STATE_CONFLICT`, `WORKSPACE_SCOPE_VIOLATION`,
  `FALSE_COMPLETION_CLAIM`, `TEST_CLAIM_CONTRADICTION`) plus evidence and
  the agent/task/delegation ids, and `ValidationWarning`s for non-fatal
  findings (missing provenance, at-budget-limit, missing timing data).
  All verdicts are logged through `get_logger` with no secrets.

**Critical invariant (validation):** agent claims are NOT proof — a
SUCCESS result never implies task completion. Validation checks whether
required evidence exists and the task's completion criteria are actually
satisfied, and reports `FALSE_COMPLETION_CLAIM` / `TASK_STATE_CONFLICT`
when they are not.

## Orchestrator Pattern

- Breaks a big goal into sub-tasks.
- Assigns them to specialized agents (or the same agent with different prompts).
- Collects results and synthesizes the final answer.

## Security Rules for Agents

- Agents **never** bypass the Permission & Approval system.
- High/Critical risk tools always require human confirmation.
- Sandboxed agents run in containers or WASM when possible.
- All tool calls and decisions are audited: every allow/confirm/deny verdict is
  appended to the JSON-lines trail at `~/.ultron/security_audit.jsonl` (UTC
  timestamp, tier, decision, mode, reason, guardrail findings).

## Natural-Language → Tool Routing (FIX #8 / #8.5)

The deterministic routing layer lives in `src/ultron/core/nlp/` and is wired
into `SimpleAgent.run()` (Step 4.98) **before** the generic shell-command
detector and the LLM fallback, so dedicated tools win and natural-language
wrappers never leak into the shell.

### Pipeline

    USER REQUEST → route_request() → UserIntent (category/tool/arguments)
    → security boundary → tool execution → interpret_command_result()

### Components

- `nlp/intent.py` — `IntentCategory` + `UserIntent` + deterministic
  `route_request()`: filesystem ops, code intelligence (definition /
  references / symbol / lexical / semantic search), project commands
  (tests/build/lint/typecheck/format/start/stop), info→command ("what is the
  current directory" → `pwd`), terminal normalization.  Returns `None` for
  genuinely ambiguous prose so it falls through to existing detectors.
- `nlp/normalize.py` — `normalize_terminal_command()` strips outer NL wrappers
  ("Execute:", "Run the command `…`", "Please run…", "Can you execute…?",
  "Use the terminal and execute…") but preserves inner content
  (`echo "Execute: pwd"` stays intact); prose is never treated as a command.
- `nlp/workspace.py` — `WorkspaceContext`: resolves location phrases
  ("current directory", "here", "this folder", ".", "project root") to the
  actual workspace root, joins relative paths against the project root,
  detects git/environment roots, and reads git-changed files (read-only) for
  affected-test selection.
- `nlp/project.py` — `discover_project_command()` (config-evidence-only
  commands) + `resolve_test_command()` / `resolve_explicit_test_command()`
  (project-aware test commands resolved into the virtualenv / poetry / uv —
  bare `pytest` on PATH is never assumed).
- `nlp/interpret.py` — deterministic result interpretation (never fabricates
  success from raw output).
- `nlp/observe.py` — bounded in-process ring buffer of routed actions
  (intent → tool → arguments → security → result) for diagnostics/tests.

### Routing priority

1. Dedicated domain tool (filesystem / code intelligence / test intelligence)
2. Existing intelligence subsystem
3. Structured executor
4. Terminal fallback

Test requests route to `handle_test()` (full suite, venv-resolved) or
`handle_relevant_tests()` (git changed files → `select_affected_tests`), and
"Run the full test suite" is caught deterministically — none of them depend
on the LLM.  Security still evaluates the **final** normalized arguments, and
`python -m pytest` / `.venv/bin/python -m pytest` are treated read-only like
bare `pytest`.

### Code-intelligence query resolution

Repository questions go through a deterministic resolution layer
(`coding/intelligence/resolve.py`) so that natural-language, mis-cased, and
multi-word symbol queries resolve to verified evidence instead of
speculation:

- **Symbol normalization** — `normalize_symbol_phrase()` generates candidate
  spellings from one input: `taskstate`, `task state`, `Task State`,
  `task-state`, `task_state` → candidates include `taskstate`, `Taskstate`,
  `task_state`, `task-state`, `TaskState`, … so users never need to know
  Python identifier casing.
- **Resolution cascade** — every lookup walks the hierarchy
  `exact indexed symbol → case-insensitive symbol → identifier normalization
  → lexical source search → references (INFERRED) → semantic (INFERRED)`.
  Exact symbol questions never jump straight to semantic search.
- **Case-insensitive index** — `index.find_definition()` /
  `find_symbol()` / `find_references()` accept a `case_insensitive=True`
  kwarg (SQL `LOWER()` match) wired through the `CodeIntelligence` facade.
- **Evidence-grounded answers** — results carry an explicit `status` of
  `VERIFIED` (indexed definition / verified reference) or `INFERRED` (lexical
  or semantic match).  Only verified locations are presented as definitive;
  lexical/semantic hits are labeled as such.  The agent never reports
  `"src/foo/supervisor.py is likely the definition"` from a filename
  convention.
- **Multi-word symbols** — `coding executor` / `CodingExecutor` /
  `codingexecutor` / `coding_executor` all resolve to `CodingExecutor`;
  intent classifiers are article-tolerant (`find where **the** supervisor is
  defined`) and distinguish definition vs. reference vs. semantic vs. lexical
  operations (`where is X defined` ≠ `where is X used` ≠ `who calls X`).
- **`SYMBOL_INSPECTION`** — "what does X do" routes to `report_symbol()`,
  which reads the verified definition source rather than answering from
  model memory.

Unknown symbols return an explicit no-verified-definition message — never a
speculative file path.

### Repository-question routing + code-search synthesis

- **Reference extraction is symbol-only** — "Where is TaskState used?",
  "find references to taskstate", "who uses X", "what references X",
  "where is X referenced/called" all extract ONLY the symbol (`TaskState`),
  never the grammatical wrapper (`is TaskState`).  The `is/are` after
  `where` is a required group, so the symbol phrase cannot swallow it.
- **Repository questions never web-search** — "how does X work/delegate/
  execute/interact", "how is X implemented", "explain how X works", and
  "why does X reject/fail" route to `code_investigation` (a new registered
  tool, read-only LOW) before the LLM fallback can misclassify them as
  `web_search`.  Explicit external queries ("search the web for X", "what
  is the latest Python release") are untouched.
- **Synthesis instead of dumps** — "where is X implemented/handled" and
  "how does X work" produce a ranked investigation: a verified definition
  is the primary implementation, with supporting components (imports /
  dependents of the defining file) and relevant tests; conceptual subjects
  ("command execution") fall back to lexical + semantic evidence ranked
  `src/` > `tests/` > `docs/` > scripts.  A nonexistent component returns
  "No repository evidence found" — never a filename-convention guess.
- **The same gate applies to the ReAct loop** — `route_llm_tool_call()`
  (`agents/react.py`) corrects misrouted LLM tool calls before execution:
  `search_web` on a repository question → `code_investigation`;
  question-shaped arguments on generic code tools → the specific
  capability (`find_definition`/`find_references`) with the extracted
  symbol; and — on the first tool call of a turn only — the *turn's*
  original request decides when the model stripped the argument to a bare
  symbol (`code_search(query='taskstate')` on "where is taskstate used?"
  → `find_references`).  Redirects are restricted to read-only code-intel
  tools and still flow through the security boundary.  `CodeIntelligenceBridge`
  delegates to the same `resolve.py` path as the registered tools, so both
  agents produce identical evidence-grounded answers.  `search_web` now
  gates identically to `web_search` (LOW/allow).

## Adding a New Agent

1. Create a class that inherits from `BaseAgent` (see `ReActAgent` for a
   reference implementation of a tool-loop agent).
2. Implement the `run()` (or `astream()`) method.
3. Add the agent's type name to `SUPPORTED_AGENTS` in
   `src/ultron/core/agents/__init__.py` and wire it into the `get_agent()`
   factory — the CLI (`--agent`) and the `/agent` slash command both validate
   against that single constant.
4. Optionally teach the Intelligence layer when to choose it.

## Recommended Default Mapping

| User Intent                     | Preferred Agent     |
|---------------------------------|---------------------|
| Simple Q&A                      | Simple              |
| “Do X for me” (tools needed)    | ReAct               |
| Coding / data analysis          | CodeAct             |
| Complex multi-step project      | Orchestrator        |
| “Keep watching for Y”           | Monitor / Operative |
| Background mission              | Operative           |

The entries above that reference agent types marked 🚧 Planned will apply as
those agents land.
