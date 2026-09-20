# Ultron REPAIR Model Selection and Re-Entry Path Validation Report

**Date:** 2026-09-21  
**Status:** PASS  
**Scope:** REPAIR model-selection, runtime re-entry, and end-to-end user-side CLI Model-in-the-Loop validation.

---

## 1. Executive Summary

The REPAIR model-selection and re-entry path in Ultron has been fully fixed and verified end-to-end through the real user CLI:

```bash
ultron chat
```

When a tool fails repeatedly and exceeds the retry limit during CLI execution, Ultron does **not** abort or terminate the task. Instead, it:
1. Catches the tool execution failure.
2. Formulates a `TaskState` with `transition_to_repair()`.
3. Re-enters `AgentRuntime.execute()` with the task state marked as `REPAIR`.
4. Invokes `ModelRouter.route()`, which scores and selects `qwen2.5-coder-7b-instruct` (coding specialist) based on existing routing policies without hardcoding.
5. Transitions to `ReActAgent` to autonomously reason about the failure, execute the corrective repair action (`write_file` / `replace_in_file`), prompt the user through the `SecurityBoundary`, verify the repaired command (`python error.py -> 1`), and output the final verified success response.

---

## 2. Root Cause Analysis & Architectural Fixes

### 2.1 Missing REPAIR Loop Interception in `main.py`
- **Issue:** In `main.py`, tool execution failures were evaluated only if `task` was already non-null. In the `SimpleAgent` path, `task` was `None`, so failures fell through to `else: break`, terminating immediately.
- **Fix:** Added deterministic failure detection via `is_step_failure` and `action_type == "execute_plan"` in `main.py`. Upon exhaustion of retries, `TaskState` is initialized with `CodeContext`, `task.record_failure()` and `task.transition_to_repair()` are invoked, and control re-enters `runtime.execute()` with `repair_agent = get_agent("react")`.

### 2.2 Preserving State Across Task Confirmation
- **Issue:** When transitioning a failed task into REPAIR, `task.status` was `TASK_FAILED`. Subsequent confirmation pauses and resumes triggered `ValueError: Cannot pause a task that is not TASK_RUNNING`.
- **Fix:** Added `TaskState.transition_to_repair()` in [`src/ultron/core/types.py`](file:///Users/aravindhan/ultron/src/ultron/core/types.py) to transition `TASK_FAILED -> TASK_RUNNING` while preserving `errors` and `execution_history`.

### 2.3 Role & Context Normalization for Local LLMs
- **Issue:** Multi-turn ReAct observations formatted as `role: "tool"` or consecutive same-role messages triggered Jinja template syntax errors (`HTTP 400 Bad Request`) on llama.cpp servers.
- **Fix:** In [`src/ultron/core/engine/llama_cpp.py`](file:///Users/aravindhan/ultron/src/ultron/core/engine/llama_cpp.py), normalized `role: "tool"` to `role: "user"` with `Observation ({tool_name}): ...` prefixes, and merged consecutive same-role messages.

### 2.4 Extra LLM Trailing Braces Decoding
- **Issue:** LLMs occasionally append extra closing braces (e.g. `{"tool": ... }}}`). `json.loads` threw `JSONDecodeError: Extra data`, discarding the tool call.
- **Fix:** In [`src/ultron/core/agents/react.py`](file:///Users/aravindhan/ultron/src/ultron/core/agents/react.py), updated `_parse_json_object` to slice to `exc.pos` on `JSONDecodeError`, cleanly recovering valid tool calls.

### 2.5 `ContextSourceType.CHANGES_AND_DIFF` Enum Member
- **Issue:** When `code_context.tracker` recorded file modifications after an edit, context assembly raised `AttributeError: type object 'ContextSourceType' has no attribute 'CHANGES_AND_DIFF'`.
- **Fix:** Added `CHANGES_AND_DIFF = "changes_and_diff"` to `ContextSourceType` in [`src/ultron/core/context/models.py`](file:///Users/aravindhan/ultron/src/ultron/core/context/models.py).

### 2.6 Robust Parameter Binding for `write_file` in ReAct
- **Issue:** ReAct's `_route_tool` looked exclusively for `arguments.get("filename")`. When the model provided `file_path`, the argument was read as empty string `""`.
- **Fix:** Updated `_route_tool` in [`src/ultron/core/agents/react.py`](file:///Users/aravindhan/ultron/src/ultron/core/agents/react.py) to accept `file_path`, `filename`, or `path`.

---

## 3. Real CLI Model-in-the-Loop Trace

**Command executed:**
```bash
ultron chat
```

**User prompt:**
```text
Create a script error.py that contains 'print(1/0)'. Then execute it using run_command. When it crashes, fix it by replacing it with 'print(1)' and run it again.
```

**Observed CLI Execution Transcript:**
```text
❯ Create a script error.py that contains 'print(1/0)'. Then execute it using run_command. When it crashes, fix it by replacing it with 'print(1)' and run it again.
INFO:ultron.runtime:Task classified: complexity=simple, coding=True, context=normal
INFO:ultron.runtime:ModelRouter decision: selected=qwen2.5-coder-7b-instruct (role=coding), reason=Coding task; coding specialist has strongest capability match.
INFO:ultron.engine.server:Spawning llama-server: /opt/homebrew/bin/llama-server -m /Users/aravindhan/models/qwen2.5-coder-7b-instruct-q8_0.gguf --host 127.0.0.1 --port 8080 -c 16384 -ngl 35
INFO:ultron.engine.server:llama-server is ready at http://127.0.0.1:8080
INFO:ultron.engine.lifecycle:Model qwen2.5-coder-7b-instruct successfully loaded.

Plan Approval Required
✻ Execute 5-step plan
  📋 Plan — 5 steps · 3 tools
1. write file 'error.py' — 🛡 needs approval
2. run command 'python error.py' — 🛡 needs approval
3. remember fact 'error.py caused a ZeroDivisionError' — ⚡ auto
4. write file 'error.py' — 🛡 needs approval
5. run command 'python error.py' — 🛡 needs approval
Dependencies:
  • step 1 creates 'error.py', used by step 2's command
  • step 1 creates 'error.py', used by step 5's command
  • step 4 creates 'error.py', used by step 5's command
Permissions: 1 auto · 4 confirm · 0

? Do you want to allow this action? Yes, allow

✻ execute_plan
Step 1 (write_file): Successfully wrote 10 characters to '/Users/aravindhan/ultron/error.py'.
Step 2 (run_command): FAILED after 3 attempts. Last error: Exit code: 1
Error Output:
Traceback (most recent call last):
  File "/Users/aravindhan/ultron/error.py", line 1, in <module>
    print(1/0)
          ~^~
ZeroDivisionError: division by zero
 finished in 0.04 s · CPU 0.04 s · peak ~1 MB

!  Tool execution failed after retries. Transitioning to autonomous REPAIR...
INFO:ultron.runtime:Task classified: complexity=simple, coding=True, context=normal
INFO:ultron.runtime:ModelRouter decision: selected=qwen2.5-coder-7b-instruct (role=coding), reason=Coding task; coding specialist has strongest capability match. | Repair state on coding task; specialist preferred.
INFO:ultron.engine.lifecycle:Model qwen2.5-coder-7b-instruct is already loaded.

INFO:ultron.agents.react:ReAct iteration 1 model response: 'Thought: The script caused a ZeroDivisionError because it attempted to divide by zero (`print(1/0)`). We need to replace the content of `error.py` with `print(1)` to fix it...'
INFO:ultron.agents.react:ReAct iteration 1 extracted tool_call: {'tool': 'write_file', 'arguments': {'file_path': '/Users/aravindhan/ultron/error.py', 'content': 'print(1)', 'overwrite': True}}
INFO:ultron.security:action=write_file tier=high decision=confirm

Confirmation Required
✻ Overwrite existing file  — /Users/aravindhan/ultron/error.py

? Do you want to allow this action? Yes, allow
✻ overwrite_file
Successfully wrote 8 characters to '/Users/aravindhan/ultron/error.py'.

INFO:ultron.runtime:Task classified: complexity=simple, coding=True, context=normal
INFO:ultron.runtime:ModelRouter decision: selected=qwen2.5-coder-7b-instruct (role=coding), reason=Coding task; coding specialist has strongest capability match. | Repair state on coding task; specialist preferred.
INFO:ultron.engine.lifecycle:Model qwen2.5-coder-7b-instruct is already loaded.

INFO:ultron.agents.react:ReAct iteration 1 model response: 'Thought: The script has been successfully fixed. Now we can run it again to ensure it works as expected...'
INFO:ultron.agents.react:ReAct iteration 1 extracted tool_call: {'tool': 'run_command', 'arguments': {'command': 'python /Users/aravindhan/ultron/error.py'}}
INFO:ultron.security:action=run_command tier=high decision=confirm

Confirmation Required
✻ Execute terminal command  — python /Users/aravindhan/ultron/error.py

? Do you want to allow this action? Yes, allow
✻ run_command
Exit code: 0
Output:
1
 finished in 0.04 s · CPU 0.02 s · peak ~1 MB

INFO:ultron.runtime:Task classified: complexity=simple, coding=True, context=normal
INFO:ultron.runtime:ModelRouter decision: selected=qwen2.5-coder-7b-instruct (role=coding), reason=Coding task; coding specialist has strongest capability match. | Repair state on coding task; specialist preferred.
INFO:ultron.engine.lifecycle:Model qwen2.5-coder-7b-instruct is already loaded.
INFO:ultron.agents.react:ReAct iteration 1 model response: 'The script has been successfully fixed and executed. The output is `1`, indicating that the script ran without errors.\n\nFinal answer: The script has been successfully executed, and the output is `1`.'

╭─────────────────────────────────────────────────────── ULTRON ───────────────────────────────────────────────────────╮
│ The script has been successfully fixed and executed. The output is 1, indicating that the script ran without errors. │
│                                                                                                                      │
│ Final answer: The script has been successfully executed, and the output is 1.                                        │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

---

## 4. Test Suite and Regression Verification

1. **Targeted Repair Tests:**
   [`tests/test_repair_model_selection.py`](file:///Users/aravindhan/ultron/tests/test_repair_model_selection.py)
   - `test_transition_to_repair_state_lifecycle` — PASSED
   - `test_routing_request_repair_vs_escalation` — PASSED
   - `test_model_router_repair_selects_coding_specialist` — PASSED
   - `test_model_router_escalation_selects_primary` — PASSED
   - `test_repair_runtime_reentry_workflow` — PASSED

2. **Full Repository Regression:**
   - `ruff check .` — **0 errors** (clean)
   - `pytest -q` — **1,759 passed, 6 deselected** in 35.26s

---

## 5. Conclusion

The REPAIR model selection and runtime re-entry path operates in full alignment with the project architecture:
- Tool retry limit exhaustion triggers autonomous REPAIR instead of terminating.
- ModelRouter dynamically selects the specialist model (`qwen2.5-coder-7b-instruct`) without hardcoding.
- Security boundary confirmation cards are faithfully presented to the user at each state-modifying action.
- The repair and re-run actions complete successfully and verify the outcome.
