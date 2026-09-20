# Phase 3.5 Real-User Model-in-the-Loop Validation Report

## 1. Executive Summary
**Overall Verdict: PARTIAL PASS**

The Phase 3.5 Ultron architecture successfully executes real user CLI sessions, correctly manages model lifecycles (loading/unloading GGUFs), and strictly enforces the security boundary. However, the hardcoded task classification logic in `AgentRuntime` prevents the `PRIMARY` model (`Qwen3-8B`) from ever being selected for new tasks.

## 2. Validation Matrix

| Scenario | Objective | Verdict | Evidence / Notes |
|----------|-----------|---------|------------------|
| **A** | FAST everyday task (Gemma) | **PASS** | `gemma-3-4b-it` successfully loaded and answered "Explain the difference between TCP and UDP". |
| **B** | REASONING / PLANNING (Qwen3) | **FAIL** | `classify_task_deterministic` classified the design request as `MULTI_STEP` (`MODERATE` complexity). The `ModelRouter` scores `FAST` and `PRIMARY` equally for `MODERATE` tasks, and breaks ties alphabetically (`gemma` beats `qwen3`). `Qwen3-8B` is never selected. |
| **C** | REAL CODING TASK (Qwen2.5-Coder) | **PASS** | `qwen2.5-coder-7b-instruct` successfully loaded for a file creation task (matched `_FILE_RE` / Coding logic). |
| **D** | MULTI-TURN REASONING | **PASS** | Context was maintained across multiple prompts in the same session without unloading the active model. |
| **E** | REAL TOOL-USE / ReAct | **PASS** | The agent successfully generated a tool invocation (e.g., `run_command`). |
| **F** | SECURITY CONFIRMATION | **PASS** | The CLI intercept correctly paused execution, displayed `? Do you want to allow this action?`, and resumed upon `Yes`. |
| **G** | SECURITY REJECTION | **PASS** | The CLI intercept correctly blocked execution upon `No`. |
| **H** | REPAIR | **UNVERIFIED** | Unable to reliably trigger ReAct tool execution failure in the current prompt format due to overly cautious clarification logic. |
| **I** | ESCALATION | **UNVERIFIED** | Depends on H. |
| **J** | MODEL SWITCHING | **PASS** | `model_transitions.log` confirms `llama-server` successfully spun down and started up new models (e.g., from Gemma to Qwen2.5-Coder) when switching task types. |

## 3. Key Findings

1. **Routing Logic Flaw**: The deterministic classifier in `AgentRuntime._build_routing_request` sets `ComplexityLevel.SIMPLE` for almost everything, and `MODERATE` for `MULTI_STEP`. It never uses `COMPLEX`. Because the router breaks ties alphabetically, the `PRIMARY` model (`Qwen3`) is effectively starved and never selected for general reasoning tasks.
2. **Security Boundary is Robust**: The `prompt_toolkit` based confirmation intercept in `ultron chat` works flawlessly, successfully halting the ReAct loop to ask for user permission before executing high-tier actions.
3. **Model Lifecycle Works**: The `llama-server` process management correctly unloads the previous model and loads the new one when a context switch is required, respecting the configured ports.

## 4. Next Steps
Phase 3.5 is effectively complete in terms of architecture. The routing starvation issue (B) and failure state triggers (H, I) should be addressed in Phase 4 via prompt refinement or minor adjustments to `AgentRuntime` classification heuristics, rather than structural redesigns.
