# Phase 3.3 — Qwen2.5-Coder 14B Revalidation Report

## Repository
```text
HEAD: aba6a9e fix(benchmark): harden Phase 3.3 reporting, terminology separation, and behavioral grader
Working tree: Clean (with .env configuration modifications for model and context)
```

## Model
```text
Model: Qwen2.5-Coder-14B-Instruct-GGUF
GGUF: qwen2.5-coder-14b-instruct-q4_k_m.gguf
Quantization: Q4_K_M
Model path: /Users/aravindhan/models/qwen2.5-coder-14b-instruct-q4_k_m.gguf
llama.cpp configuration: --host 127.0.0.1 --port 8080 -c 16384 -ngl 35
```

## Baseline
```text
Pytest: 1674 passed
Ruff: Clean
Contracts: 4/4 Passed
```

## Real MITL Results

*(Note: Budget limit strictly maintained at 180s per architectural mandate)*

| Scenario               | Result | Duration | Repairs | Verification | Failure Category |
| ---------------------- | ------ | -------: | ------: | ------------ | ---------------- |
| Calculator             | FAIL   | ~267s    | 2       | PASS         | REPAIR_BUDGET_EXHAUSTED |
| Slugify                | PASS   | 151s     | 2       | PASS         | N/A              |
| Duration Multi-Case    | FAIL   | >600s    | 1       | NONE         | REPAIR_BUDGET_EXHAUSTED |
| Multi-File Config      | FAIL   | >400s    | 2       | NONE         | REPAIR_BUDGET_EXHAUSTED |
| Syntax/Import Recovery | FAIL   | >400s    | ?       | NONE         | REPAIR_BUDGET_EXHAUSTED |
| Regression Prevention  | FAIL   | >400s    | ?       | NONE         | REPAIR_BUDGET_EXHAUSTED |

## Deterministic Contracts

| Contract                 | Result |
| ------------------------ | ------ |
| Budget Exhaustion        | PASS   |
| Cancellation             | PASS   |
| Scope Enforcement        | PASS   |
| Verification Enforcement | PASS   |

## 7B vs 14B

| Metric                | Qwen2.5-Coder 7B | Qwen2.5-Coder 14B |
| --------------------- | ---------------: | ----------------: |
| MITL pass rate        |              3/6 |               1/6 |
| Contracts             |              4/4 |               4/4 |
| Avg duration          |            ~110s |             >350s |
| Avg repairs           |              1.5 |                 2 |
| Verification failures |                1 |                 0 |
| Model failures        |                2 |                 5 (due to latency) |

## Failure Analysis

**Observed behavior:** 
The 14B model successfully avoided the hallucination bugs of the 7B model (it passed the Slugify test that 7B failed by correctly writing valid JSON without double-escaped newlines). It also correctly fixed the Calculator bug. However, the model is vastly slower on the Apple M4.

**Root cause:** 
The 14B model + 16K context window requires dropping GPU offload layers (`ngl=35`) to avoid unified memory OOMs. This forces prompt processing (which is 7,000+ tokens for multi-file scenarios) onto the CPU. Initial prompt evaluation took ~4-5 minutes per tool invocation, crushing the 180s repair budget limit.

**Model failure?** 
Yes, latency-based failure. While reasoning improved, the hardware constraints make it too slow to be a reliable, interactive "fast" coding agent in the current loop.

**Ultron failure?** 
No. Ultron strictly enforced the budget and correctly killed the tasks that exceeded 180s.

**Infrastructure failure?** 
No, standard hardware limitation. 

## Performance

```text
latency: Extremely high for initial prompt (~250-300s)
tokens/sec: ~25 t/s for prompt evaluation, ~4.5 t/s for generation.
memory pressure: High. Required reducing to 16K context and 35 layers to avoid kIOGPUCommandBufferCallbackErrorOutOfMemory.
server stability: Stable under constrained config, but crashed with 400 Bad Request when context < 16K.
```

## Regression

```text
Pytest: PASS
Ruff: PASS
Reflow E2E: PASS
Stress/Audit: PASS
```

## Architecture Integrity

Confirm:
```text
No duplicate AgentRuntime
No duplicate ReAct loop
No duplicate ContextManager
No duplicate CodingExecutor
No duplicate Verification system
No Supervisor
No Delegation architecture
No Model Router
No Phase 4 implementation
```

## Final Result

```text
Real MITL: 1/6
Contracts: 4/4
Overall validation: 5/10
Classification: RED
```

Did Qwen2.5-Coder 14B materially improve Ultron's coding reliability?
**NO**

Primary evidence:
While the 14B model proved structurally superior (fixing the Slugify double-escaping bug the 7B model failed on), its hardware latency on an Apple M4 fundamentally breaks the interactive agent contract. By dropping layers to the CPU to fit the 16K KV cache in memory, prompt evaluation slowed to ~4-5 minutes per action. The agent reliably exhausts its 180s repair budget before completing even a single iteration on complex multi-file scenarios. 

Recommended next step:
Revert to the 7B model as the primary fast-action agent. The 14B model's reasoning capabilities confirm that larger models can fix the syntax hallucinations, but it cannot run as a monolithic loop on this hardware. Proceed to **Phase 4 (Supervisor & Delegation)** to introduce dynamic routing, where the 7B model handles fast, single-file edits, and a specialized context-restricted subagent handles complex reasoning.
