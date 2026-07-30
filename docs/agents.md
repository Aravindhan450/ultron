# Ultron – Agents System

Agents are the “workers” that turn a user request into actual work.  
They sit on top of the **Engine** and use **Tools + Memory**.

## Agent Types

| Agent            | Style              | When to use                                      | Strengths                          | Limitations                     |
|------------------|--------------------|--------------------------------------------------|------------------------------------|---------------------------------|
| **Simple**       | Single-shot        | Direct questions, no tools needed                | Fast, cheap                        | Cannot use tools                |
| **Orchestrator** | Multi-agent boss   | Complex tasks that need several specialists      | Coordinates others                 | Higher latency                  |
| **ReAct**        | Reason + Act loop  | Most tool-using tasks                            | Good balance of reasoning & action | Can loop if poorly prompted     |
| **CodeAct**      | Code-centric       | Tasks best solved by writing & running code      | Extremely powerful for coding      | Needs sandbox                   |
| **Operative**    | Long-running       | Background jobs, monitoring, multi-step missions | Persistent, stateful               | More complex state management   |
| **Monitor**      | Observer           | Watch for events / conditions and react          | Reactive, low overhead             | Not for open-ended goals        |

## How Agents Work (High Level)

1. **Intelligence** layer chooses the best model + agent type for the request.
2. Agent receives:
   - User message (or voice transcript)
   - Relevant memory
   - Available tools (filtered by permission + relevance)
3. Agent reasons (and optionally calls tools).
4. Every tool call goes through:
   - Risk Classifier → Permission system
   - GuardrailsEngine (secrets, PII, boundary checks)
   - Optional sandbox
5. Final answer is returned (and spoken if in voice mode).

## ReAct Pattern (Most Common)
Thought → Action (tool call) → Observation → Thought → ... → Final Answer


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

1. Create a class that inherits from `BaseAgent`.
2. Implement the `run()` (or `astream()`) method.
3. Register it in the agent registry.
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
