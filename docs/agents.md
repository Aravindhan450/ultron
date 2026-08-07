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

### Selecting the ReAct agent

The CLI defaults to `simple`. Launch with ReAct instead:

    ultron chat --agent react

(short form: `ultron chat -a react`). You can also switch between the two
mid-session without restarting — type `/agent` for an interactive picker or
`/agent react` to switch directly. The prompt shows the active agent
(`[model | react] You:`), and the switch preserves your chosen model.

## CodeAct Pattern

The agent writes Python (or shell) code, runs it in a sandbox, sees the output, and iterates.

## Orchestrator Pattern

- Breaks a big goal into sub-tasks.
- Assigns them to specialized agents (or the same agent with different prompts).
- Collects results and synthesizes the final answer.

## Security Rules for Agents

- Agents **never** bypass the Permission & Approval system.
- High/Critical risk tools always require human confirmation.
- Sandboxed agents run in containers or WASM when possible.
- All tool calls and decisions are audited.

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
