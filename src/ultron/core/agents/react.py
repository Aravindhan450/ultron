"""
ultron.core.agents.react
~~~~~~~~~~~~~~~~~~~~~~~~

ReAct (Reason + Act) agent — the workhorse pattern for tool-using tasks.

The agent alternates between:

    Thought -> Action (JSON tool call) -> Observation -> ... -> Final Answer

Unlike SimpleAgent (deterministic regex detectors + a single-shot LLM
fallback), ReActAgent lets the LLM drive the whole loop: it decides when a
tool is needed, calls it, reads the observation, and iterates until it has
enough information to answer. This makes it a better fit for open-ended,
multi-step requests where the exact tool sequence isn't predictable.

Safety model (consistent with the rest of Ultron):

- Every tool call is routed through the security boundary
  (``boundary.check()``) before execution. A guardrail hard-block (secret
  exfiltration, unsafe URL, path escape) yields ``deny`` and the action never
  runs.
- Read-only / low-risk tools (read_file, web search, page fetch, memory
  lookups, read-only SQL, GET requests) are ``allow`` and execute directly
  inside the loop.
- State-modifying actions (run_command, write_file/overwrite, non-read-only
  SQL, POST/PUT/DELETE requests) are ``confirm``: the agent returns a
  ChatMessage carrying a PendingAction payload so main.py can show an
  interactive confirmation first — unless a permissive security mode marks
  them ``allow``, in which case they run directly. Agents never bypass the
  Permission & Approval system.
"""

import json
import re
from typing import Any

from ultron.core.agents.base import BaseAgent
from ultron.core.agents.security import (
    blocked_message,
    check_action,
    is_allow,
    is_denied,
)
from ultron.core.agents.simple import (
    _generic_target_content,
    execute_tool,
    handle_file_write,
    handle_http,
    handle_parallel,
)
from ultron.core.coding.context import CodeContext
from ultron.core.coding.edits import EDIT_TOOL_ACTIONS, record_tool_result
from ultron.core.coding.workspace import discover_workspace
from ultron.core.intelligence.prompt_assembly import (
    build_response_guidance,
    polish_response,
)
from ultron.core.logging import get_logger
from ultron.core.memory.session_memory import SessionMemory
from ultron.core.tools.definitions import (
    ToolCapability as _ToolCapability,
)
from ultron.core.tools.definitions import (
    code_intel_tool_names,
    generic_code_tool_names,
    web_tool_names,
)
from ultron.core.tools.registry import get_tool, get_tools_schema
from ultron.core.types import (
    ChatMessage,
    FailureStrategy,
    PendingAction,
    PlanStep,
    Role,
    StepStatus,
    TaskState,
    history_to_openai_format,
)

logger = get_logger("ultron.agents.react")

# Cap on reasoning steps per turn — prevents runaway loops if the model keeps
# emitting tool calls without ever reaching a final answer.
DEFAULT_MAX_ITERATIONS = 10

# Code-intelligence tools the ReAct routing correction may redirect to.
# Redirects never target state-modifying tools: the runtime only moves an
# LLM tool call toward the dedicated read-only capability.  DERIVED from the
# canonical definitions table (STEP 2A) — never an independent list.
_CODE_INTEL_TOOLS = code_intel_tool_names()

# Tools the turn-level correction may replace when the turn's original request
# classifies to a specific symbol capability (see route_llm_tool_call): generic
# code tools AND web search — a repo-question turn must never produce a web
# call, even when the model stripped the argument down to a bare symbol that
# does not itself classify (search_web(query='taskstate') on a "where is
# taskstate used?" turn -> find_references). Genuine web turns never classify
# to a specific symbol tool, so they are never touched.  DERIVED from canonical
# capability metadata.
_TURN_CORRECTABLE_TOOLS = frozenset(generic_code_tool_names() | web_tool_names())

# Specific symbol capabilities: these answer "where is X used/defined" with
# VERIFIED index evidence — never a raw lexical dump.  A capability-level set
# (not tool names): the tool is discovered from the canonical registry via
# the selection result (STEP 2C).
_SPECIFIC_SYMBOL_CAPABILITIES = frozenset(
    {
        _ToolCapability.DEFINITION_LOOKUP,
        _ToolCapability.REFERENCE_LOOKUP,
    }
)


def route_llm_tool_call(
    tool_name: str, arguments: dict, user_input: str | None = None
) -> tuple[str, dict] | None:
    """
    Deterministic correction for the ReAct loop's tool calls.

    The LLM decides WHAT the user wants; the runtime decides WHICH tool and
    with WHAT arguments. This layer redirects a tool call the model misrouted:

    - ``search_web`` on a repository question ("How does the Supervisor
      delegate work?") -> ``code_investigation`` — repository questions never
      hit the web in the LLM-driven loop, exactly as in the deterministic
      SimpleAgent path;
    - a question-shaped argument on a code-intelligence tool
      (``code_search(query="Where is TaskState defined?")``) -> the specific
      tool ``route_request`` picks (``find_definition``, ...) with the
      correctly extracted symbol;
    - a generic code tool (``code_search``/``semantic_search``) whose bare
      argument is a symbol, while the TURN's original request classifies to a
      specific symbol capability (``user_input="Where is taskstate used?"``
      -> ``find_references(name="taskstate")``) — the model may emit a bare
      symbol that does not itself classify, but the user's question does, and
      the runtime decides the tool.

    ``user_input`` is the turn's original user message (optional): when it
    classifies to a specific symbol tool, a generic first tool call is
    corrected regardless of what the model extracted. Callers pass it only
    for the first tool call of a turn, so mid-loop generic searches (legit
    exploration) are never overridden.

    Returns ``(corrected_tool, corrected_arguments)`` to execute instead, or
    ``None`` to run the call as-is.  Redirects are restricted to read-only
    code-intelligence tools — never ``run_command`` or any state-modifying
    action, and never a security downgrade (the corrected call still flows
    through the boundary).
    """
    from ultron.core.capabilities import select_capability
    from ultron.core.nlp.intent import route_request

    # (1) Turn-level correction: the user's actual request classifies to a
    # specific symbol capability, but the model reached for a generic code
    # tool. The runtime wins — the capability selection (ONE path, STEP 2C)
    # decides the capability, and the canonical registry supplies the tool,
    # with the correctly extracted symbol.
    if user_input and user_input.strip():
        turn_intent = route_request(user_input.strip())
        if turn_intent is not None:
            selection = select_capability(turn_intent.intent_type)
            if (
                selection.is_resolved()
                and selection.primary in _SPECIFIC_SYMBOL_CAPABILITIES
                and tool_name in _TURN_CORRECTABLE_TOOLS
            ):
                corrected_tool = selection.preferred_tool
                if corrected_tool is not None:
                    return corrected_tool, dict(turn_intent.arguments or {})

    query = str(arguments.get("query") or arguments.get("name") or "").strip()
    if not query:
        return None

    intent = route_request(query)
    if intent is None or not intent.tool:
        return None
    if intent.tool not in _CODE_INTEL_TOOLS:
        return None

    if tool_name in web_tool_names():
        # A repository question must never be executed as a web search.
        if intent.tool != tool_name:
            return intent.tool, dict(intent.arguments or {})
        return None

    if tool_name in _CODE_INTEL_TOOLS and intent.tool != tool_name:
        # The argument is a natural-language question; route it to the
        # specific capability (and the correctly extracted symbol/query).
        return intent.tool, dict(intent.arguments or {})
    return None


def build_system_prompt() -> str:
    """
    Builds the ReAct system prompt with the live Tool Registry JSON schema.

    The schema is generated at call time so newly registered tools are
    immediately visible to the model on the next turn.
    """
    tools_schema = json.dumps(get_tools_schema(), indent=2)
    return (
        "You are ULTRON, an autonomous local AI assistant that solves tasks by "
        "alternating reasoning and tool use (the ReAct pattern).\n\n"
        f"Available Tools:\n{tools_schema}\n\n"
        "INSTRUCTIONS:\n"
        "1. Operate in a Thought -> Action -> Observation loop.\n"
        "2. When you need information or must perform an action, respond with:\n"
        "   Thought: <your reasoning>\n"
        "   ```json\n"
        '   {"tool": "<tool_name>", "arguments": { ... }}\n'
        "   ```\n"
        "   Use exactly the tool names and argument keys shown above. "
        "Do NOT add conversational text after the JSON block.\n"
        "3. After each Action you will receive an Observation. Continue the "
        "loop until you have enough information to answer.\n"
        "4. When you have enough information, answer directly in natural "
        "language. Do NOT include a JSON block in your final answer.\n"
        "5. Never invent tool results — only use facts from the Observations "
        "you actually received.\n"
        "6. State-modifying actions (commands, file writes, non-read-only "
        "database queries, POST/PUT/DELETE HTTP requests) are routed to the "
        "user for confirmation automatically — still emit the tool call "
        "normally when one is needed.\n"
        "7. Use the fewest tool calls needed to answer well.\n"
        "8. When several independent read-only lookups are needed at once "
        "(multiple files, sites, or searches), prefer a single "
        "`run_tool_batch` call whose `calls_json` argument is a JSON array "
        "of {\"tool\": ..., \"arguments\": {...}} — it executes them "
        "concurrently and synthesizes the results into one observation.\n"
        f"\n\n{build_response_guidance()}"
    )


def _parse_json_object(text: str) -> Any:
    """
    Parses a JSON object, falling back to the first '{' .. last '}' span.

    Naive non-greedy extraction can truncate tool calls whose arguments
    contain nested braces (e.g. a JSON string body). Retrying on the outer
    brace span makes extraction robust to that case.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        first = text.find("{")
        last = text.rfind("}")
        if first != -1 and last > first:
            try:
                return json.loads(text[first : last + 1])
            except json.JSONDecodeError:
                return None
        return None


def extract_tool_call(text: str) -> dict[str, Any] | None:
    """
    Extracts a JSON tool-call block from an LLM response.

    Accepts either a fenced block (```json { ... } ```) or a bare JSON object.
    Returns a dict with 'tool' and 'arguments' keys, or None when the response
    does not contain a parseable tool call (i.e. it is a final answer).
    """
    candidates: list[str] = []

    fence_match = re.search(r"```(?:json)?\s*(.+?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence_match:
        candidates.append(fence_match.group(1).strip())

    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        candidates.append(stripped)

    for candidate in candidates:
        data = _parse_json_object(candidate)
        if isinstance(data, dict) and "tool" in data:
            return data
    return None


# ---------------------------------------------------------------------------
# TaskState integration helpers
# ---------------------------------------------------------------------------

def _observation_succeeded(observation: str) -> bool:
    """
    Conservative success check for a tool observation.

    ``run_command`` reports failures as ``Exit code: 1`` (no ``Error`` prefix),
    so a bare startswith("Error") check would record a failed command as
    success and poison the verification evidence.
    """
    text = str(observation)
    if text.startswith(("Error", "Blocked by security")):
        return False
    if "cancelled by user" in text.lower():
        return False
    match = re.search(r"Exit code:\s*(\d+)", text)
    if match:
        return match.group(1) == "0"
    return True


def _needs_verification(task: TaskState | None) -> bool:
    """
    True when a final-looking answer must be verified against the TaskState
    before it can be accepted — the task is actively tracked and not yet
    explicitly complete.

    With a structured plan attached, verification is always required until
    the task is explicitly complete: an unverified final answer can never
    complete a planned task, and completion is only ever decided by
    TaskState + the plan — never by the model's word alone.
    """
    if task is None or task.is_complete():
        return False
    if task.plan is not None:
        return True
    return task.requires_verification or bool(task.remaining_requirements())


def _activate_plan_step(task: TaskState) -> None:
    """
    Marks the plan's next runnable step RUNNING (if a plan is attached).

    The current step is the RUNNING step when one exists, otherwise the next
    PENDING step whose dependencies are satisfied. Starting a step also
    records it as the task's ``current_step``.
    """
    if task.plan is None:
        return
    step = task.current_plan_step()
    if step is None or step.status is not StepStatus.PENDING:
        return
    step.status = StepStatus.RUNNING
    task.set_current_step(step.id)


def _resume_task(task: TaskState) -> TaskState:
    """
    Closes an interrupted confirmation turn: records the confirmed action's
    observation into the task transcript/history and resumes the task.

    Called by run() when a task is passed back in after main.py executed the
    pending action. The observation becomes a TOOL message the model sees, so
    it understands e.g. "TodoList exists, but the application is not complete".

    With a structured plan attached, the current plan step returns from
    WAITING_CONFIRMATION back to RUNNING so execution continues on the same
    step — never a restart from step 1.
    """
    observation = task.last_observation
    if observation is not None:
        action_type = task.pending_action.action_type if task.pending_action else "tool"
        target = task.pending_action.target if task.pending_action else ""
        task.context.append(
            ChatMessage(role=Role.TOOL, name=action_type, content=observation)
        )
        succeeded = _observation_succeeded(observation)
        task.record_tool_execution(
            tool_name=action_type,
            target=target,
            success=succeeded,
            detail=str(observation),
        )
        if task.code_context is not None:
            record_tool_result(
                task.code_context.tracker,
                action_type,
                target,
                str(observation),
                step=task.current_step,
                success=succeeded,
            )
            # Record confirmed actions into the executor too, so a failed
            # confirmed command counts against the repair budget exactly like
            # an inline one.
            _confirmed_arguments = (
                {"command": target} if action_type == "run_command" else {}
            )
            task.code_context.executor.record_observation(
                action_type, _confirmed_arguments, str(observation), succeeded
            )
            # Fix #4: a successful edit invalidates the code index — the next
            # intelligence query must refresh before the agent reasons from
            # stale source information.
            if succeeded and action_type in EDIT_TOOL_ACTIONS:
                task.code_context.intelligence.mark_dirty()
        if task.plan is not None:
            step = next(
                (
                    s
                    for s in task.plan.steps
                    if s.status is StepStatus.WAITING_CONFIRMATION
                ),
                None,
            )
            if step is not None:
                step.status = StepStatus.RUNNING
                task.set_current_step(step.id)
        else:
            task.set_current_step(len(task.execution_history))
        task.last_observation = None
        task.pending_action = None
        task.resume()
    return task


def _activate_task(
    task: TaskState | None,
    goal: str,
    messages: list[ChatMessage],
    task_start: int,
    response: str,
    action: PendingAction,
) -> TaskState:
    """
    Creates (or reuses) the task for a gated state-changing action.

    On first use the transcript is seeded from the messages after the user's
    goal message; the current assistant tool-call response is appended so the
    transcript survives the confirmation round-trip. The task is set to
    WAITING_CONFIRMATION and flagged for verification.
    """
    if task is None:
        task = TaskState(goal=goal)
        task.context = list(messages[task_start:])
        task.requires_verification = True
        _seed_task_history(task, messages, task_start)
    # Coding edits get workspace context so the task knows where it is and
    # what changed — attached regardless of when the task was created (a task
    # may already exist from an earlier non-coding confirmation), and it
    # survives confirmation because it lives on the TaskState itself.
    if task.code_context is None and action.action_type in EDIT_TOOL_ACTIONS:
        task.code_context = CodeContext(workspace=discover_workspace())
        task.code_context.attach_task(task)
    if task.plan is not None:
        step = task.current_plan_step()
        if step is not None and step.status is StepStatus.RUNNING:
            step.status = StepStatus.WAITING_CONFIRMATION
    task.context.append(ChatMessage(role=Role.ASSISTANT, content=response))
    task.pending_action = action
    task.wait_for_confirmation()
    return task


def _seed_task_history(
    task: TaskState, messages: list[ChatMessage], task_start: int
) -> None:
    """
    Copies tool observations that predate task creation into the history.

    When a task is created at the first gated action, any earlier read-only
    tool work in the transcript is part of the evidence — mirror it into
    ``execution_history`` so verification sees the complete picture.
    """
    for i in range(task_start, len(messages)):
        msg = messages[i]
        if msg.role != Role.TOOL or not msg.name or msg.name == "task_verification":
            continue
        target = ""
        if i - 1 >= task_start and messages[i - 1].role == Role.ASSISTANT:
            call = extract_tool_call(messages[i - 1].content)
            if isinstance(call, dict) and isinstance(call.get("arguments"), dict):
                target, _ = _generic_target_content(msg.name, call["arguments"])
        task.record_tool_execution(
            tool_name=msg.name,
            target=target,
            success=_observation_succeeded(msg.content),
            detail=str(msg.content),
        )
    task.set_current_step(len(task.execution_history))


def _task_messages(
    task: TaskState, history: list[ChatMessage] | None
) -> list[ChatMessage]:
    """
    Rebuilds the model context for a resumed task: the persisted transcript,
    prefixed with the caller's leading SYSTEM message (persona) when present.
    """
    base = list(task.context)
    if history and history[0].role == Role.SYSTEM:
        base.insert(0, history[0])
    return base


def _build_task_context_block(
    task: TaskState | None, session: SessionMemory | None = None
) -> str:
    """
    Explicit continuation context: goal, status, requirements, completed work.
    Injected into the system prompt so the model always knows the overall task
    is not finished until TaskState says so.
    """
    if task is None:
        return ""
    lines = [
        "CURRENT TASK (you are working toward this goal; do not stop until it is complete):",
        f"Goal: {task.goal}",
        f"Status: {task.status.value}",
        f"Step: {task.current_step}/{task.total_steps or '?'} completed",
        f"Requirements: {len(task.completed_requirements)}/{len(task.requirements)} complete",
    ]
    for requirement in task.requirements:
        mark = "[x]" if requirement.completed else "[ ]"
        lines.append(f"  {mark} {requirement.description}")
    if task.execution_history:
        lines.append("Work completed so far:")
        for entry in task.execution_history:
            status = "ok" if entry.success else "FAILED"
            lines.append(
                f"  - {entry.tool_name}({entry.target or ''}) [{status}] "
                f"{entry.detail[:200]}"
            )
    plan_block = _build_plan_context_block(task)
    if plan_block:
        lines.append("")
        lines.append(plan_block)
    # Fix #3: deterministic CodingExecutor guidance — how to accomplish the
    # current step (exploration needed, validation commands, repair budget).
    if task.code_context is not None:
        guidance = task.code_context.executor.step_guidance(task)
        if guidance:
            lines.append("")
            lines.append(guidance)
        # Fix #4: code-intelligence guidance — tool-selection strategy for
        # the step plus a bounded targeted-context block (definitions /
        # references for candidate symbols), never a repository dump.
        intelligence = task.code_context.executor.intelligence_guidance(task)
        if intelligence:
            lines.append("")
            lines.append(intelligence)
    # Fix #6: bounded, prioritized RELEVANT MEMORY (project facts about this
    # workspace, session continuity). The ContextManager applies the budget
    # and excludes stale/superseded records; current task/plan/observation
    # facts above always outrank memory.
    memory_block = _build_memory_context_block(task, session)
    if memory_block:
        lines.append("")
        lines.append(memory_block)
    return "\n".join(lines)


def _memory_task_terms(task: TaskState) -> list[str]:
    """Relevance keywords from the task goal + current step (deterministic)."""
    texts = [str(getattr(task, "goal", "") or "")]
    step = task.current_plan_step() if callable(getattr(task, "current_plan_step", None)) else None
    if step is not None:
        texts.append(str(getattr(step, "description", "") or ""))
    terms: list[str] = []
    for text in texts:
        terms.extend(re.split(r"[^A-Za-z0-9_]+", text))
    return [term for term in terms if len(term) >= 3][:16]


def _build_memory_context_block(
    task: TaskState | None, session: SessionMemory | None
) -> str:
    """
    Fix #6: the bounded RELEVANT MEMORY section for the current task.

    Project memory (workspace-scoped, valid records only) and session memory
    are assembled by the ContextManager under its budget + priority rules.
    Returns '' when neither store has anything relevant — memory never
    floods the prompt.
    """
    if task is None or task.code_context is None:
        return ""
    from ultron.core.memory.context_manager import ContextManager

    store = task.code_context.ensure_project_memory()
    if store is None and (session is None or session.is_empty):
        return ""
    records = store.recall(limit=40) if store is not None else []
    workspace = ""
    if task.code_context.workspace is not None:
        workspace = str(
            getattr(task.code_context.workspace, "project_root", "") or ""
        )
    return ContextManager().memory_block(
        project_records=records,
        session=session,
        workspace=workspace,
        task_terms=_memory_task_terms(task),
    )


def _sync_task_memory(task: TaskState | None) -> None:
    """
    Fix #6: deterministic memory formation + reconciliation (idempotent).

    Runs once per agent turn: recorded code-intelligence symbol lookups are
    promoted to project memory, and stored symbol facts are reconciled
    against the CURRENT code index (the repository wins over older memory).
    Formation never writes trivial tool outputs and never bypasses the
    store's secret guard.
    """
    if task is None or task.code_context is None:
        return
    if task.code_context.ensure_project_memory() is None:
        return
    from ultron.core.memory.formation import (
        reconcile_project_memory,
        remember_intelligence_facts,
    )

    store = task.code_context.ensure_project_memory()
    bridge = task.code_context.intelligence
    remember_intelligence_facts(store, bridge)
    reconcile_project_memory(store, bridge)


def _build_plan_context_block(task: TaskState) -> str:
    """
    Structured-plan context injected when the task carries a validated plan.

    The plan is the source of truth: the model sees the CURRENT step (with
    its dependencies and completion criteria), the completed steps, and the
    remaining steps. It is instructed to work only on the current step — a
    step completes only when its criteria are verified, and a single tool
    call never completes a step or the task.
    """
    plan = task.plan
    if plan is None:
        return ""
    step = task.current_plan_step()
    lines = [
        "STRUCTURED PLAN (source of truth):",
        f"Task type: {task.task_type.value if task.task_type else 'unknown'}",
        f"Workspace: {plan.workspace.value}",
    ]
    if step is not None:
        lines.append(f"Current step {step.id}/{len(plan.steps)}: {step.description}")
        if step.purpose:
            lines.append(f"  Purpose: {step.purpose}")
        if step.expected_outcome:
            lines.append(f"  Expected outcome: {step.expected_outcome}")
        if step.dependencies:
            lines.append(
                f"  Depends on: {', '.join(f'step {d}' for d in step.dependencies)}"
            )
        lines.append("  Completion criteria (all must be verified to finish this step):")
        for criterion in step.completion_criteria:
            lines.append(f"    [ ] {criterion}")
        retry = f" (retry x{step.retry_policy})" if step.retry_policy else ""
        lines.append(f"  Failure policy: {step.failure_strategy.value}{retry}")
    else:
        lines.append("Current step: none (all steps terminal)")
    completed = plan.completed_steps()
    if completed:
        lines.append("Completed steps:")
        for s in completed:
            lines.append(f"  [x] {s.id}. {s.description}")
    remaining = [s for s in plan.remaining_steps() if step is None or s.id != step.id]
    if remaining:
        lines.append("Remaining steps:")
        for s in remaining:
            deps = (
                f" (after {', '.join(str(d) for d in s.dependencies)})"
                if s.dependencies
                else ""
            )
            lines.append(f"  [ ] {s.id}. {s.description}{deps}")
    lines.append(
        "INSTRUCTIONS: work ONLY on the current plan step. Do not skip ahead; "
        "a step is complete only when all its completion criteria are verified "
        "— a single tool call never completes a step or the task."
    )
    return "\n".join(lines)


def _work_summary(task: TaskState) -> str:
    """Compact list of recorded tool executions for verification prompts."""
    if task.execution_history:
        lines = []
        for i, entry in enumerate(task.execution_history, start=1):
            status = "succeeded" if entry.success else "FAILED"
            lines.append(f"{i}. {entry.tool_name} -> {status}: {entry.detail[:300]}")
        return "\n".join(lines)
    return "(no tool executions recorded)"


def _verification_evidence_block(task: TaskState) -> str:
    """CodingExecutor evidence appended to verification prompts ('' when none)."""
    if task.code_context is None:
        return ""
    evidence = task.code_context.executor.verification_evidence(task)
    return f"\n\nCoding evidence:\n{evidence}" if evidence else ""


def _build_verification_prompt(task: TaskState, proposed_answer: str) -> str:
    """
    Asks the model to propose the goal's completion criteria and mark which
    are satisfied by the work actually performed. TaskState stays the
    authority: only criteria the model marks satisfied become completed
    requirements, and the task completes only when ALL are satisfied.
    """
    return (
        "You are verifying whether an autonomous assistant has completed the user's task.\n\n"
        f"Goal: {task.goal}\n\n"
        f"Work performed so far:\n{_work_summary(task)}\n\n"
        f"The assistant's proposed answer: {proposed_answer}\n\n"
        "Decide which completion criteria the goal implies, and mark each as satisfied "
        "ONLY if the recorded work actually fulfills it. Do not invent work that was "
        "not performed, and do not claim success while criteria remain unmet.\n"
        "Respond with ONLY a JSON array, e.g.:\n"
        '[{"description": "application files exist", "satisfied": true}, '
        '{"description": "application can run", "satisfied": false}]'
        f"{_verification_evidence_block(task)}"
    )


def _parse_requirements_json(text: str) -> list[dict] | None:
    """Parses a verification JSON array, tolerating surrounding prose/fences."""
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        first = text.find("[")
        last = text.rfind("]")
        if first == -1 or last <= first:
            return None
        try:
            data = json.loads(text[first : last + 1])
        except json.JSONDecodeError:
            return None
    if not isinstance(data, list):
        return None
    return [d for d in data if isinstance(d, dict)]


def _build_plan_verification_prompt(task: TaskState, proposed_answer: str) -> str:
    """
    Plan-aware verification prompt: marks the CURRENT step's completion
    criteria and the plan's overall criteria against the recorded work.

    The plan is the source of truth — the model proposes which criteria are
    satisfied, but the step only advances (SUCCEEDED) when ALL of its criteria
    are marked satisfied, and the task completes only when every step is
    terminal AND every overall requirement is complete.
    """
    step = task.current_plan_step()
    lines = [
        "You are verifying whether an autonomous assistant has completed the ",
        "CURRENT PLAN STEP of the user's task.\n",
        f"Goal: {task.goal}",
        f"Task type: {task.task_type.value if task.task_type else 'unknown'}",
        "",
    ]
    if step is not None:
        lines.append(
            f"Current plan step ({step.id}/{len(task.plan.steps)}): {step.description}"
        )
        if step.purpose:
            lines.append(f"  Purpose: {step.purpose}")
        if step.expected_outcome:
            lines.append(f"  Expected outcome: {step.expected_outcome}")
        lines.append("  Completion criteria:")
        for criterion in step.completion_criteria:
            lines.append(f"    - {criterion}")
    else:
        lines.append("Current plan step: none (all steps terminal).")
    lines.extend(["", "Overall task completion criteria (all must be satisfied):"])
    for requirement in task.requirements:
        mark = "satisfied" if requirement.completed else "unsatisfied"
        lines.append(f"  - [{mark}] {requirement.description}")
    lines.extend(
        [
            "",
            f"Work performed so far:\n{_work_summary(task)}",
            "",
            f"The assistant's proposed answer: {proposed_answer}",
            "",
            "Decide which completion criteria are satisfied ONLY if the recorded work ",
            "actually fulfills them. Do not invent work. Do not claim a step complete ",
            "while any of its criteria remain unmet.",
            "Respond with ONLY a JSON object:",
            '{"step_criteria": [{"description": "...", "satisfied": true}], ',
            '"plan_criteria": [{"description": "...", "satisfied": true}], ',
            '"step_failed": false, "plan_revision": null}',
            "- step_criteria: the CURRENT step's completion criteria, each marked.",
            "- plan_criteria: the overall task completion criteria above, each marked.",
            "- step_failed: true only if the current step cannot be completed as planned.",
            "- plan_revision: OPTIONAL replacement steps (same shape as plan steps: ",
            "  id, description, purpose, dependencies, expected_outcome, ",
            "  completion_criteria, failure_strategy, retry_policy) covering the ",
            "  REMAINING work, or null. Never include completed steps.",
        ]
    )
    evidence = _verification_evidence_block(task)
    if evidence:
        lines.append(evidence)
    return "\n".join(lines)


def _parse_plan_verification(text: str) -> dict | None:
    """Parses a plan verification JSON object, tolerating prose/fences."""
    data = _parse_json_object(str(text))
    return data if isinstance(data, dict) else None


def _parse_revision_steps(items) -> list[PlanStep] | None:
    """
    Parses a proposed adaptive plan revision into PlanStep objects.

    Returns None when the payload is not a valid list of step objects, so
    callers keep the current plan rather than executing an invalid revision.
    """
    if not isinstance(items, list):
        return None
    steps: list[PlanStep] = []
    for item in items:
        if not isinstance(item, dict):
            return None
        try:
            step_id = int(item.get("id", 0))
            description = str(item.get("description", "")).strip()
            if not step_id or not description:
                return None
            strategy = str(item.get("failure_strategy", "stop")).lower()
            try:
                failure_strategy = FailureStrategy(strategy)
            except ValueError:
                failure_strategy = FailureStrategy.STOP
            steps.append(
                PlanStep(
                    id=step_id,
                    description=description,
                    purpose=str(item.get("purpose", "")).strip(),
                    dependencies=[
                        int(d)
                        for d in item.get("dependencies", [])
                        if isinstance(d, int)
                    ],
                    expected_outcome=str(item.get("expected_outcome", "")).strip(),
                    completion_criteria=[
                        str(c).strip()
                        for c in item.get("completion_criteria", [])
                        if str(c).strip()
                    ],
                    failure_strategy=failure_strategy,
                    retry_policy=int(item.get("retry_policy", 0) or 0),
                )
            )
        except (ValueError, TypeError):
            return None
    return steps or None


def _retries_exhausted(step: PlanStep) -> bool:
    """True when a RETRY step has consumed its retry budget."""
    return (
        step.failure_strategy is FailureStrategy.RETRY
        and step.retry_policy > 0
        and step.attempts >= step.retry_policy
    )


def _cascade_skipped(task: TaskState) -> None:
    """
    Marks PENDING steps that can no longer run as SKIPPED, transitively.

    A step whose prerequisite was SKIPPED (or failed under a continue-style
    policy) can never execute. Leaving it PENDING would strand the plan —
    ``is_satisfied()`` could never become True and completion could never be
    reached — so it is marked SKIPPED like its prerequisite.
    """
    changed = True
    while changed:
        changed = False
        for candidate in task.plan.steps:
            if candidate.status is not StepStatus.PENDING:
                continue
            if any(
                (dep := task.plan.step(d)) is not None
                and dep.status in (StepStatus.SKIPPED, StepStatus.FAILED)
                for d in candidate.dependencies
            ):
                candidate.status = StepStatus.SKIPPED
                changed = True


class ReActAgent(BaseAgent):
    """
    Reason + Act agent that loops over Thought/Action/Observation until it
    produces a final answer (or exceeds max_iterations).

    Attributes:
        engine: The LLM backend (BaseEngine implementation).
        max_iterations: Maximum tool-call turns per user message.
    """

    def __init__(self, engine, max_iterations: int = DEFAULT_MAX_ITERATIONS) -> None:
        super().__init__(engine)
        self.max_iterations = max_iterations

    async def run(
        self,
        user_input: str,
        history: list[ChatMessage] | None = None,
        task: TaskState | None = None,
        session: SessionMemory | None = None,
    ) -> ChatMessage:
        """
        Executes the ReAct loop for a single user turn (or a task continuation).

        Returns a ChatMessage whose content is either the model's final answer
        or (for state-modifying actions) carries a PendingAction so the CLI
        can prompt the user for confirmation before anything executes.

        Task integration: when ``task`` is passed in (after main.py executed a
        confirmed action), the interrupted turn is closed with the action's
        observation and the loop resumes from the persisted task transcript —
        so a successful ``mkdir`` is an observation, never task completion.
        A final-looking answer is only accepted once the task is explicitly
        complete (see ``_verify_task``); otherwise the agent keeps working.

        Fix #6 session integration: ``session`` (optional) is a
        :class:`SessionMemory` the agent records continuity into (recent
        request, active workspace, task refs) and reads back from when
        assembling the bounded memory context block.
        """
        system_prompt = build_system_prompt()

        # Structured output enforcement: inject the exact schema when the
        # user asked for a machine-readable shape, and enforce it on the
        # final answer below.
        from ultron.core.intelligence.structured_output import (
            enforce_reply,
            schema_prompt_block,
        )
        schema_block = schema_prompt_block(user_input)
        if schema_block:
            system_prompt += "\n\n" + schema_block

        fresh_turn = task is None or not task.context
        if fresh_turn:
            # Fresh turn (or a freshly prepared planned task with an empty
            # transcript): seed the conversation from history, inject the
            # system prompt, then append the user input last (so the task
            # transcript slice below starts exactly at the user's goal).
            messages = list(history) if history else []
            if session is not None:
                session.note_request(user_input)
        else:
            # Continuation after a confirmed action: close the interrupted
            # turn with the observation, then resume from the task transcript.
            task = _resume_task(task)
            messages = _task_messages(task, history)

        # Fix #6: deterministic memory formation + reconciliation against the
        # current code index (idempotent, secret-guarded).
        _sync_task_memory(task)

        # Explicit continuation context: goal, status, requirements, completed
        # work — so the model always knows the overall task is not finished
        # until TaskState says it is.
        task_block = _build_task_context_block(task, session)
        if task_block:
            system_prompt += "\n\n" + task_block

        # Inject the ReAct system prompt at the front of the conversation,
        # preserving any existing system context (e.g. persona instructions).
        if messages and messages[0].role == Role.SYSTEM:
            messages[0] = ChatMessage(
                role=Role.SYSTEM,
                content=f"{messages[0].content}\n\n{system_prompt}",
            )
        else:
            messages.insert(0, ChatMessage(role=Role.SYSTEM, content=system_prompt))

        if fresh_turn:
            messages.append(ChatMessage(role=Role.USER, content=user_input))
            # The task transcript starts at the user's goal message.
            task_start = len(messages) - 1
            if task is not None:
                task.context = list(messages[task_start:])
        else:
            task_start = 0

        # With a structured plan attached, mark the first runnable step as
        # RUNNING so the model always has a concrete "current step".
        if task is not None:
            _activate_plan_step(task)
            # Fix #6 session continuity: bind the active workspace + task so a
            # follow-up request can reuse project context instead of
            # rediscovering the repository.
            if session is not None:
                workspace = task.code_context.workspace if task.code_context else None
                root = getattr(workspace, "project_root", None) if workspace else None
                if root:
                    session.set_workspace(str(root))
                if task.goal:
                    session.note_task((task.goal or "")[:60])

        first_tool_call = True
        for _ in range(self.max_iterations):
            response = (await self.engine.generate(history_to_openai_format(messages))) or ""
            tool_call = extract_tool_call(response)

            # No tool call → the model proposes a final answer. If the task is
            # still being verified, do NOT accept the answer on the model's
            # word alone — verify against TaskState first.
            if tool_call is None:
                if _needs_verification(task):
                    if task is not None and task.plan is not None:
                        accepted, answer = await self._verify_plan_task(
                            task, user_input, response, messages
                        )
                    else:
                        accepted, answer = await self._verify_task(
                            task, user_input, response, messages
                        )
                    if accepted:
                        return answer
                    continue
                # Fix #6: capture this final turn's intelligence facts.
                _sync_task_memory(task)
                return ChatMessage(
                    role=Role.ASSISTANT,
                    content=polish_response(enforce_reply(user_input, response)),
                    task_state=task,
                )

            tool_name = tool_call.get("tool")
            arguments = tool_call.get("arguments") or {}
            # Malformed model output must never crash the agent — coerce
            # non-dict arguments (e.g. a list) to an empty dict.
            if not isinstance(arguments, dict):
                arguments = {}

            # CodingExecutor deterministic gate: a state-changing action that
            # has already failed identically (or a NEW action once the repair
            # budget is spent) is blocked BEFORE execution — the block is fed
            # back as an observation so the model is forced to change approach.
            outcome = self._coding_gate(task, tool_name, arguments)
            if outcome is None:
                # Deterministic routing correction: the LLM decides what the
                # user wants, the runtime decides which tool and arguments.
                # Redirect a misrouted call (repo question -> web search, or
                # a question-shaped argument on a generic code tool) toward
                # the dedicated read-only capability. The turn's original
                # request is consulted only on the FIRST tool call, so
                # mid-loop generic searches (legit exploration) are never
                # overridden.
                corrected = route_llm_tool_call(
                    tool_name, arguments, user_input=user_input if first_tool_call else None
                )
                if corrected is not None:
                    tool_name, arguments = corrected
                first_tool_call = False
                outcome = self._route_tool(tool_name, arguments, user_input)

            # Confirmation-gated action — hand control back to the CLI with a
            # PendingAction payload AND the live task, so the original goal
            # survives the confirmation round-trip.
            if isinstance(outcome, ChatMessage) and outcome.pending_action is not None:
                task = _activate_task(
                    task, user_input, messages, task_start, response, outcome.pending_action
                )
                return ChatMessage(
                    role=Role.ASSISTANT,
                    content=outcome.content,
                    pending_action=outcome.pending_action,
                    task_state=task,
                )

            # Read-only routes (e.g. a GET via handle_http) return a plain
            # ChatMessage; unwrap it so the result feeds back as an Observation
            # and the loop can continue reasoning toward a final answer.
            if isinstance(outcome, ChatMessage):
                outcome = outcome.content

            # Record the assistant's action and the tool observation so the
            # model can reason over its own previous steps — and mirror them
            # into the task transcript + history when a task is active, so
            # verification sees every piece of work performed.
            messages.append(ChatMessage(role=Role.ASSISTANT, content=response))
            messages.append(ChatMessage(role=Role.TOOL, name=tool_name, content=str(outcome)))
            if task is not None:
                task.context.append(ChatMessage(role=Role.ASSISTANT, content=response))
                task.context.append(ChatMessage(role=Role.TOOL, name=tool_name, content=str(outcome)))
                target, _ = _generic_target_content(tool_name, arguments)
                succeeded = _observation_succeeded(str(outcome))
                task.record_tool_execution(
                    tool_name=tool_name,
                    target=target,
                    success=succeeded,
                    detail=str(outcome),
                )
                if task.code_context is not None:
                    record_tool_result(
                        task.code_context.tracker,
                        tool_name,
                        target,
                        str(outcome),
                        success=succeeded,
                    )
                    task.code_context.executor.record_observation(
                        tool_name, arguments, str(outcome), succeeded
                    )
                    # Fix #4: successful edits invalidate the code index.
                    if succeeded and tool_name in EDIT_TOOL_ACTIONS:
                        task.code_context.intelligence.mark_dirty()
                if task.plan is not None and not succeeded:
                    step = task.current_plan_step()
                    if step is not None and step.status is StepStatus.RUNNING:
                        step.attempts += 1
                        step.error = str(outcome)[:300]

        # Iteration budget exhausted with an active task — never claim success.
        if task is not None and (
            task.requires_verification
            or task.remaining_requirements()
            or (task.plan is not None and not task.plan.is_satisfied())
        ):
            remaining = task.remaining_requirements()
            if task.plan is not None and not task.plan.is_satisfied():
                # With a structured plan the plan is the source of truth —
                # report the remaining plan steps first, then any unmet
                # overall requirements as additional detail.
                steps = task.plan.remaining_steps()
                names = ", ".join(f"step {s.id} ({s.description})" for s in steps)
                detail = f"The task is incomplete — remaining plan steps: {names}."
                if remaining:
                    reqs = ", ".join(r.description for r in remaining)
                    detail += f" Unmet overall requirements: {reqs}."
            elif remaining:
                names = ", ".join(r.description for r in remaining)
                detail = f"The task is incomplete — remaining requirements: {names}."
            else:
                detail = "The task is incomplete — requirements were never fully verified."
            if task.plan is not None:
                step = task.current_plan_step()
                if step is not None and step.status is StepStatus.RUNNING:
                    step.status = StepStatus.FAILED
            task.record_failure(
                f"Exceeded the maximum of {self.max_iterations} reasoning steps "
                "with work remaining"
            )
            # Fix #6: capture the incomplete-turn intelligence facts.
            _sync_task_memory(task)
            return ChatMessage(
                role=Role.ASSISTANT,
                content=(
                    f"I could not complete the task within {self.max_iterations} steps. "
                    f"{detail}"
                ),
                task_state=task,
            )
        return ChatMessage(
            role=Role.ASSISTANT,
            content=(
                f"I exceeded my maximum of {self.max_iterations} reasoning steps "
                "without reaching a final answer. Please try rephrasing or "
                "simplifying the request."
            ),
            task_state=task,
        )

    async def _verify_task(
        self,
        task: TaskState,
        user_input: str,
        proposed_answer: str,
        messages: list[ChatMessage],
    ) -> tuple[bool, ChatMessage | None]:
        """
        One completion-verification pass against the TaskState.

        The model proposes completion criteria and marks which are satisfied by
        the recorded work; TaskState is the authority — the final answer is
        only accepted when every requirement is marked complete
        (``task.is_complete()``). Otherwise the model's proposal is recorded as
        an observation and the loop continues toward the goal.

        Returns ``(accepted, final_message)``.
        """
        from ultron.core.intelligence.structured_output import enforce_reply

        prompt = _build_verification_prompt(task, proposed_answer)
        raw = (await self.engine.generate([{"role": "user", "content": prompt}])) or ""
        requirements = _parse_requirements_json(raw)

        # Record the model's proposed answer either way, so a continuation has
        # full context.
        messages.append(ChatMessage(role=Role.ASSISTANT, content=proposed_answer))
        task.context.append(ChatMessage(role=Role.ASSISTANT, content=proposed_answer))

        def _note(content: str) -> None:
            messages.append(
                ChatMessage(role=Role.TOOL, name="task_verification", content=content)
            )
            task.context.append(
                ChatMessage(role=Role.TOOL, name="task_verification", content=content)
            )

        if requirements is None:
            _note(
                "Task verification could not parse a completion-criteria response; "
                "the task is still incomplete. Continue working toward the goal."
            )
            return False, None

        for item in requirements:
            description = str(item.get("description", "")).strip()
            if not description:
                continue
            satisfied = bool(item.get("satisfied"))
            try:
                if satisfied:
                    task.mark_requirement_complete(description)
                else:
                    task.mark_requirement_incomplete(description)
            except ValueError:
                # First time this criterion is proposed — add it, then mark it.
                task.add_requirement(description)
                if satisfied:
                    task.mark_requirement_complete(description)

        remaining = task.remaining_requirements()
        if remaining:
            names = ", ".join(f"'{r.description}'" for r in remaining)
            _note(
                f"Verification: task incomplete. Remaining requirements: {names}. "
                "Continue working toward the goal."
            )
            return False, None

        task.mark_complete()
        # Fix #6: capture this completing turn's intelligence facts.
        _sync_task_memory(task)
        return True, ChatMessage(
            role=Role.ASSISTANT,
            content=polish_response(enforce_reply(user_input, proposed_answer)),
            task_state=task,
        )

    async def _verify_plan_task(
        self,
        task: TaskState,
        user_input: str,
        proposed_answer: str,
        messages: list[ChatMessage],
    ) -> tuple[bool, ChatMessage | None]:
        """
        Plan-aware completion verification against a structured TaskPlan.

        The plan is the source of truth: the model marks the CURRENT step's
        completion criteria and the plan's overall criteria against the
        recorded work. A step advances to SUCCEEDED only when ALL of its
        criteria are satisfied; the task completes only when every step is
        terminal AND every overall requirement is complete. The model can
        never claim "done" while plan work remains (no skipping A → F).

        Failure policy: when a step is reported failed (or its retry budget is
        exhausted) the step's ``failure_strategy`` decides the outcome —
        STOP terminates the task, RETRY permits more attempts, SKIP skips the
        step, CONTINUE records the failure and keeps going.

        Adaptive planning: an optional ``plan_revision`` payload replaces the
        remaining steps (validated by TaskState) while completed steps stay
        recorded.

        Returns ``(accepted, final_message)``.
        """
        from ultron.core.intelligence.structured_output import enforce_reply

        prompt = _build_plan_verification_prompt(task, proposed_answer)
        raw = (await self.engine.generate([{"role": "user", "content": prompt}])) or ""
        data = _parse_plan_verification(raw)

        # Record the model's proposed answer either way, so a continuation has
        # full context.
        messages.append(ChatMessage(role=Role.ASSISTANT, content=proposed_answer))
        task.context.append(ChatMessage(role=Role.ASSISTANT, content=proposed_answer))

        def _note(content: str) -> None:
            messages.append(
                ChatMessage(role=Role.TOOL, name="task_verification", content=content)
            )
            task.context.append(
                ChatMessage(role=Role.TOOL, name="task_verification", content=content)
            )

        if data is None:
            _note(
                "Plan verification could not be parsed; the task is still "
                "incomplete. Continue working toward the current plan step."
            )
            return False, None

        # Mark the plan-level criteria against the task requirements.
        for item in data.get("plan_criteria") or []:
            if not isinstance(item, dict):
                continue
            description = str(item.get("description", "")).strip()
            if not description:
                continue
            satisfied = bool(item.get("satisfied"))
            try:
                if satisfied:
                    task.mark_requirement_complete(description)
                else:
                    task.mark_requirement_incomplete(description)
            except ValueError:
                task.add_requirement(description)
                if satisfied:
                    task.mark_requirement_complete(description)

        step = task.current_plan_step()
        if step is None:
            # No runnable step: complete only when the plan is fully satisfied
            # (every step terminal) AND every overall requirement is met.
            # "No runnable step" can also mean stranded PENDING steps whose
            # dependencies failed/skipped — those must not complete the task.
            if not task.plan.is_satisfied():
                pending = task.plan.remaining_steps()
                names = ", ".join(
                    f"step {s.id} ('{s.description}')" for s in pending
                )
                _note(
                    f"Verification: task incomplete — plan steps are not all "
                    f"terminal: {names}. Continue working toward the goal."
                )
                return False, None
            remaining = task.remaining_requirements()
            if remaining:
                names = ", ".join(f"'{r.description}'" for r in remaining)
                _note(
                    f"Verification: task incomplete. Remaining requirements: "
                    f"{names}. Continue working toward the goal."
                )
                return False, None
            task.mark_complete()
            # Fix #6: capture this completing turn's intelligence facts.
            _sync_task_memory(task)
            return True, ChatMessage(
                role=Role.ASSISTANT,
                content=polish_response(enforce_reply(user_input, proposed_answer)),
                task_state=task,
            )

        step_criteria = [
            item
            for item in (data.get("step_criteria") or [])
            if isinstance(item, dict) and str(item.get("description", "")).strip()
        ]
        step_failed = bool(data.get("step_failed"))
        # The CURRENT step advances only when ALL of ITS OWN completion
        # criteria are marked satisfied — criteria from other steps are
        # ignored, so the model can never skip ahead (A -> F).
        satisfied_descriptions = {
            str(item.get("description")).strip()
            for item in step_criteria
            if bool(item.get("satisfied"))
        }
        all_met = bool(step.completion_criteria) and all(
            criterion in satisfied_descriptions
            for criterion in step.completion_criteria
        )

        if step_failed or _retries_exhausted(step):
            # Apply the step's failure strategy.
            return self._apply_step_failure_strategy(
                task, step, data, proposed_answer, _note, user_input
            )

        if not all_met:
            unmet = [
                criterion
                for criterion in step.completion_criteria
                if criterion not in satisfied_descriptions
            ]
            names = ", ".join(f"'{u}'" for u in (unmet or ["all criteria"]))
            _note(
                f"Verification: step {step.id} ('{step.description}') is not "
                f"complete. Unmet step criteria: {names}. Continue working on "
                "this step; do not claim completion until every criterion is "
                "verified."
            )
            return False, None

        # Current step complete — mark it SUCCEEDED first, THEN apply any
        # adaptive revision so completed work is preserved, then advance.
        task.plan.set_step_status(step.id, StepStatus.SUCCEEDED, result=proposed_answer[:300])

        revision = data.get("plan_revision")
        if isinstance(revision, list) and revision:
            new_steps = _parse_revision_steps(revision)
            if new_steps is not None and task.adapt_plan(new_steps):
                _note(
                    "Plan revised: remaining steps updated to "
                    f"{len(new_steps)} replacement step(s). Continue with the "
                    "updated plan."
                )
            else:
                _note(
                    "A proposed plan revision was rejected (structurally "
                    "invalid); the current plan stays in force."
                )

        next_step = task.plan.next_step()
        if next_step is not None:
            next_step.status = StepStatus.RUNNING
            task.set_current_step(next_step.id)
            _note(
                f"Verification: step {step.id} complete. Next step: "
                f"{next_step.id} ('{next_step.description}'). Continue working "
                "toward it."
            )
            return False, None

        # All steps done: complete only when the plan is satisfied AND every
        # overall requirement is met — never while a step is still stranded.
        if not task.plan.is_satisfied():
            pending = task.plan.remaining_steps()
            names = ", ".join(
                f"step {s.id} ('{s.description}')" for s in pending
            )
            _note(
                f"Verification: task incomplete — plan steps are not all "
                f"terminal: {names}. Continue working toward the goal."
            )
            return False, None
        remaining = task.remaining_requirements()
        if remaining:
            names = ", ".join(f"'{r.description}'" for r in remaining)
            _note(
                f"Verification: all plan steps are complete but overall "
                f"requirements remain: {names}. Continue working toward the "
                "goal."
            )
            return False, None
        task.mark_complete()
        # Fix #6: capture this completing turn's intelligence facts.
        _sync_task_memory(task)
        return True, ChatMessage(
            role=Role.ASSISTANT,
            content=polish_response(enforce_reply(user_input, proposed_answer)),
            task_state=task,
        )

    def _apply_step_failure_strategy(
        self,
        task: TaskState,
        step: PlanStep,
        data: dict,
        proposed_answer: str,
        note,
        user_input: str,
    ) -> tuple[bool, ChatMessage | None]:
        """
        Applies a failed step's ``failure_strategy`` to the plan + task.

        STOP (default) and exhausted RETRY terminate the task with an explicit
        failure report — never a success claim. RETRY keeps the step RUNNING
        while attempts remain. SKIP marks the step SKIPPED and advances. 
        CONTINUE records the failure and advances.
        """
        strategy = step.failure_strategy
        reason = str(data.get("step_failed_reason") or "step failed")[:300]

        if strategy is FailureStrategy.RETRY and not _retries_exhausted(step):
            note(
                f"Step {step.id} failed but the plan allows retry "
                f"(attempt {step.attempts}/{step.retry_policy}). Do not repeat "
                "the identical failed action — inspect the failure and adjust "
                "your approach."
            )
            return False, None

        if strategy in (FailureStrategy.SKIP, FailureStrategy.CONTINUE):
            # SKIP marks the step SKIPPED. CONTINUE records the failure (as a
            # SKIPPED step with the error attached — the plan stays
            # satisfiable) and keeps going with independent work. Either way,
            # any PENDING step that depended on this one can never run, so it
            # is cascaded to SKIPPED to keep the plan consistent.
            status = StepStatus.SKIPPED
            task.plan.set_step_status(step.id, status, error=reason)
            _cascade_skipped(task)
            next_step = task.plan.next_step()
            if next_step is not None:
                next_step.status = StepStatus.RUNNING
                task.set_current_step(next_step.id)
            word = "skipped" if strategy is FailureStrategy.SKIP else "failed"
            note(
                f"Step {step.id} {word} per plan policy ({reason}). "
                + (
                    f"Next step: {next_step.id} ('{next_step.description}'). "
                    "Continue working toward it."
                    if next_step is not None
                    else "No remaining runnable steps."
                )
            )
            return False, None

        # STOP (default) or retries exhausted → the task cannot succeed.
        task.plan.set_step_status(step.id, StepStatus.FAILED, error=reason)
        remaining = task.remaining_steps()
        task.record_failure(f"Plan step {step.id} ('{step.description}') failed: {reason}")
        detail = ""
        if remaining:
            names = ", ".join(f"step {s.id} ({s.description})" for s in remaining)
            detail = f" Remaining plan steps: {names}."
        # Fix #6: capture the failure turn's intelligence facts.
        _sync_task_memory(task)
        return True, ChatMessage(
            role=Role.ASSISTANT,
            content=(
                f"I could not complete plan step {step.id} "
                f"('{step.description}'): {reason}. The task is incomplete."
                f"{detail}"
            ),
            task_state=task,
        )

    def _route_tool(self, tool_name: str, arguments: dict[str, Any], user_input: str) -> str | ChatMessage:
        """
        Executes a tool call, gated by the security boundary.

        Every tool call is routed through ``boundary.check()`` first:

        - ``deny`` (guardrail hard block: secret exfiltration, unsafe URL,
          path escape) → the action never executes; a blocked message is fed
          back as an Observation.
        - ``confirm`` (state-modifying, or HIGH/CRITICAL tier under the active
          mode) → a PendingAction is returned so the CLI asks the user first.
        - ``allow`` (read-only / LOW tier, or a permissive mode) → the tool
          executes directly inside the loop.

        Returns either a tool result string (fed back as an Observation) or a
        ChatMessage carrying a pending_action for the CLI to confirm.
        """
        # --- State-modifying actions: gated by the boundary verdict ---
        if tool_name == "run_command":
            # Never pass raw model text to the shell: normalize any
            # natural-language wrapper and refuse prose that is not
            # command-shaped ("Execute: pwd" -> ``pwd``).
            from ultron.core.nlp.normalize import normalize_terminal_command
            raw_cmd = str(arguments.get("command", "")).strip()
            normalized = normalize_terminal_command(raw_cmd)
            if not normalized:
                return (
                    "Error: run_command requires a command-shaped argument; "
                    "the supplied text is not a shell command."
                )
            cmd = normalized
            verdict = check_action("run_command", cmd)
            if is_denied(verdict):
                return blocked_message(verdict)
            if is_allow(verdict):
                return execute_tool("run_command", command=cmd)
            return ChatMessage(
                role=Role.ASSISTANT,
                content=f"Command execution requested: '{cmd}'",
                pending_action=PendingAction(action_type="run_command", target=cmd),
            )


        if tool_name == "run_parallel":
            # A parallel batch is gated command-by-command (any denial blocks
            # the whole batch; any confirmation routes it through a single
            # interactive approval listing every command). Tolerate a stray
            # string argument the same way the tool itself does.
            cmds_arg = arguments.get("commands", [])
            if isinstance(cmds_arg, str):
                cmds_arg = [cmds_arg]
            cmds = [str(c).strip() for c in cmds_arg]
            cmds = [c for c in cmds if c]
            if not cmds:
                return "Error: run_parallel requires a non-empty 'commands' list."
            return handle_parallel(cmds)

        if tool_name == "run_tool_batch":
            # Inter-tool parallel batch. Like run_parallel this routes
            # directly: every member is gated individually *inside* the tool
            # (deny never runs, confirm never runs silently), so the batch is
            # only as safe as its most dangerous member. Routing here instead
            # of the generic path avoids a redundant outer content scan of the
            # raw calls_json (which could false-flag an http:// member URL).
            from ultron.core.intelligence.parallel_tools import (
                run_tool_batch as _run_batch,
            )

            calls_json = arguments.get("calls_json") or arguments.get("calls")
            if isinstance(calls_json, (list, tuple)):
                calls_json = json.dumps(calls_json)
            if not calls_json:
                return (
                    "Error: run_tool_batch requires a 'calls_json' argument — "
                    "a JSON array of {\"tool\": ..., \"arguments\": {...}}."
                )
            return _run_batch(str(calls_json))

        if tool_name == "write_file":
            # handle_file_write runs the same boundary gate: deny → blocked,
            # allow (permissive mode) → direct write, confirm → PendingAction.
            return handle_file_write(
                str(arguments.get("filename", "")),
                str(arguments.get("content", "")),
                user_input=user_input,
            )

        # Fix #3 coding file operations: gated exactly like write_file. The
        # target is the file path and the content is what the boundary scans
        # (the new text for edits/appends, the destination for rename), so
        # path escapes and secret writes are denied for these too.
        if tool_name in {
            "create_file",
            "replace_file",
            "replace_in_file",
            "append_to_file",
            "delete_file",
            "rename_file",
        }:
            return self._route_coding_file_op(tool_name, arguments)

        if tool_name == "make_http_request":
            # handle_http runs the same boundary gate: deny → blocked; GET is
            # auto-allowed; POST/PUT/DELETE are confirmed unless permissive.
            return handle_http(
                str(arguments.get("method", "GET")),
                str(arguments.get("url", "")),
                arguments.get("body"),
            )

        if tool_name == "run_query":
            sql = str(arguments.get("sql", arguments.get("query", "")))
            verdict = check_action("run_query", sql)
            if is_denied(verdict):
                return blocked_message(verdict)
            if is_allow(verdict):
                func = get_tool("run_query")
                return func(sql) if func else "Error: Tool 'run_query' not found in registry."
            return ChatMessage(
                role=Role.ASSISTANT,
                content=f"Database query execution requested: '{sql}'",
                pending_action=PendingAction(action_type="db_query", target=sql),
            )

        # --- Everything else: read-only / low-risk execution, still gated ---
        func = get_tool(tool_name)
        if func is None:
            return f"Error: unknown tool '{tool_name}'. Choose one of the available tools."

        target, content = _generic_target_content(tool_name, arguments)
        verdict = check_action(tool_name, target, content)
        if is_denied(verdict):
            return blocked_message(verdict)

        try:
            return func(**arguments)
        except Exception as exc:  # noqa: BLE001 — arbitrary tool surface
            logger.debug(f"Tool '{tool_name}' raised an exception: {exc}")
            return f"Error executing tool '{tool_name}': {exc}"

    def _coding_gate(
        self,
        task: TaskState | None,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> str | None:
        """
        CodingExecutor deterministic safety gate, applied BEFORE routing.

        Returns a blocking observation string when the action must NOT
        execute, or None to proceed normally:

        - a state-changing action that has already failed identically more
          than the budget allows is blocked (no blind repetition);
        - any NEW state-changing action is blocked once the repair budget is
          exhausted (no endless repair attempts).

        The returned message is fed back as an observation, so the model sees
        it, learns the constraint, and must change its approach — the gate
        never executes the tool and never bypasses the security boundary.
        """
        if task is None or task.code_context is None:
            return None
        executor = task.code_context.executor
        message = executor.gate_action(tool_name, arguments)
        if message is not None:
            return message
        return executor.gate_new_action_with_exhausted_budget(tool_name)

    def _route_coding_file_op(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> str | ChatMessage:
        """
        Gates a Fix #3 coding file operation through the security boundary.

        Maps the tool's arguments to the (target, content) pair the boundary
        expects, then applies the same verdict policy as write_file:

        - ``deny`` (path escape, secret content) → hard block, never offered
        - ``allow`` (permissive mode) → execute directly
        - ``confirm`` → PendingAction for the CLI to ask the user first
        """
        file_path = str(arguments.get("file_path", arguments.get("path", ""))).strip()
        target = file_path
        content: str | None = None

        if tool_name == "replace_in_file":
            # Scan the replacement text the same way write_file content is.
            content = str(arguments.get("new", ""))
        elif tool_name in ("append_to_file", "create_file", "replace_file"):
            content = str(arguments.get("content", ""))
        elif tool_name == "rename_file":
            # Scan the destination path too (targets a file location).
            content = str(arguments.get("new_path", ""))

        if not target:
            return f"Error: {tool_name} requires a non-empty file path."

        verdict = check_action(tool_name, target, content)
        if is_denied(verdict):
            return blocked_message(verdict)
        if is_allow(verdict):
            return execute_tool(tool_name, **arguments)

        # Encode the tool arguments in the pending action so the CLI can
        # reconstruct the exact call after approval. replace_in_file needs
        # both old and new text, so they travel as a JSON payload.
        if tool_name == "replace_in_file":
            payload = json.dumps(
                {"old": str(arguments.get("old", "")), "new": str(arguments.get("new", ""))}
            )
        else:
            payload = content or ""
        return ChatMessage(
            role=Role.ASSISTANT,
            content=f"File operation requested: {tool_name} on '{target}'",
            pending_action=PendingAction(
                action_type=tool_name, target=target, content=payload
            ),
        )
