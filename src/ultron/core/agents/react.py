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
from ultron.core.intelligence.prompt_assembly import (
    build_response_guidance,
    polish_response,
)
from ultron.core.logging import get_logger
from ultron.core.tools.registry import get_tool, get_tools_schema
from ultron.core.types import ChatMessage, PendingAction, Role, history_to_openai_format

logger = get_logger("ultron.agents.react")

# Cap on reasoning steps per turn — prevents runaway loops if the model keeps
# emitting tool calls without ever reaching a final answer.
DEFAULT_MAX_ITERATIONS = 10


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

    async def run(self, user_input: str, history: list[ChatMessage] | None = None) -> ChatMessage:
        """
        Executes the ReAct loop for a single user turn.

        Returns a ChatMessage whose content is either the model's final answer
        or (for state-modifying actions) carries a PendingAction so the CLI
        can prompt the user for confirmation before anything executes.
        """
        messages = list(history) if history else []
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

        # Inject the ReAct system prompt at the front of the conversation,
        # preserving any existing system context (e.g. persona instructions).
        if messages and messages[0].role == Role.SYSTEM:
            messages[0] = ChatMessage(
                role=Role.SYSTEM,
                content=f"{messages[0].content}\n\n{system_prompt}",
            )
        else:
            messages.insert(0, ChatMessage(role=Role.SYSTEM, content=system_prompt))

        messages.append(ChatMessage(role=Role.USER, content=user_input))

        for _ in range(self.max_iterations):
            response = (await self.engine.generate(history_to_openai_format(messages))) or ""
            tool_call = extract_tool_call(response)

            # No tool call → the model is answering directly; that's the
            # answer (structured output is enforced when a schema was asked
            # for — repaired + [structured] notes, never silent deviation).
            if tool_call is None:
                return ChatMessage(
                    role=Role.ASSISTANT,
                    content=polish_response(enforce_reply(user_input, response)),
                )

            tool_name = tool_call.get("tool")
            arguments = tool_call.get("arguments") or {}
            # Malformed model output must never crash the agent — coerce
            # non-dict arguments (e.g. a list) to an empty dict.
            if not isinstance(arguments, dict):
                arguments = {}

            outcome = self._route_tool(tool_name, arguments, user_input)

            # Confirmation-gated action — hand control back to the CLI with a
            # PendingAction payload instead of executing anything silently.
            if isinstance(outcome, ChatMessage) and outcome.pending_action is not None:
                return outcome

            # Read-only routes (e.g. a GET via handle_http) return a plain
            # ChatMessage; unwrap it so the result feeds back as an Observation
            # and the loop can continue reasoning toward a final answer.
            if isinstance(outcome, ChatMessage):
                outcome = outcome.content

            # Record the assistant's action and the tool observation so the
            # model can reason over its own previous steps.
            messages.append(ChatMessage(role=Role.ASSISTANT, content=response))
            messages.append(ChatMessage(role=Role.TOOL, name=tool_name, content=str(outcome)))

        return ChatMessage(
            role=Role.ASSISTANT,
            content=(
                f"I exceeded my maximum of {self.max_iterations} reasoning steps "
                "without reaching a final answer. Please try rephrasing or "
                "simplifying the request."
            ),
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
            cmd = str(arguments.get("command", "")).strip()
            if not cmd:
                return "Error: run_command requires a non-empty 'command' argument."
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
