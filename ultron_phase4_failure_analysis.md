# Ultron Phase 4 — Failure Analysis

## 1. Executive Summary

During Phase 4 real-world capability and reliability validation of Project Ultron, testing was performed using the live `ultron chat` CLI interface running against real local LLMs managed by `ModelLifecycleManager` on `llama-server`. A total of 10 real-world scenarios spanning conversational discovery, live web search, multi-model routing escalation/demotion, code generation, debugging loops, file inspection, and security boundaries were executed.

Two distinct failures were encountered and deeply analyzed:
1. **Task Classification Ambiguity on Compound Requests (RWT-05)**: Creation requests containing explanatory clauses were misclassified as `RESEARCH` or `INFORMATIONAL` instead of `SOFTWARE_ENGINEERING`.
2. **ReAct Deadlock on Read-Only Tool Failures (RWT-06)**: When an initial `read_file` failed with a missing file error, the ReAct agent repeated the failing tool call until hitting the `max_iterations (20)` boundary.

---

## 2. Failure Case 1: Deterministic Classifier Rule Precedence (RWT-05)

### Incident Description
- **Prompt**: `"Create a Python application that displays the current time in Chennai in real time... Create all required files... and explain how to run it."`
- **Expected Outcome**: Task classified as `SOFTWARE_ENGINEERING` or `MULTI_STEP`, activating tool execution to create files.
- **Actual Outcome**: Classified as `RESEARCH` / `INFORMATIONAL`. SimpleAgent outputted Python code and explanation directly into the chat response box without creating files.

### Root Cause Analysis
In `src/ultron/core/intelligence/task_classification.py`, the function `_classify_deterministic` evaluated rules in the following sequence:
```python
if _RESEARCH_RE.search(text) and _CODE_NOUNS_RE.search(text):
    return TaskType.RESEARCH
if _INFORMATIONAL_RE.search(text):
    return TaskType.INFORMATIONAL
...
if _SE_VERBS_RE.search(text) and _SE_NOUNS_RE.search(text):
    return TaskType.SOFTWARE_ENGINEERING
```
Because the user's prompt ended with `"and explain how to run it"`, `"explain"` matched `_RESEARCH_RE` and `"application"` matched `_CODE_NOUNS_RE`, triggering `TaskType.RESEARCH` before the software engineering check was reached.

### Applied Correction
A dedicated regular expression `_SE_CREATION_RE` was introduced:
```python
_SE_CREATION_RE = re.compile(
    r"\b(create|build|implement|develop|scaffold)\b.*?\b(app|application|project|service|cli|tool|script|program)\b"
)
```
The `_RESEARCH_RE` and `_INFORMATIONAL_RE` checks were updated with an exclusion guard `and not _SE_CREATION_RE.search(text)`. This guarantees that when a user explicitly requests to build/create an application or program, incidental explanatory clauses do not downgrade the task into passive research.

---

## 3. Failure Case 2: ReAct Deadlock on Missing File Read (RWT-06)

### Incident Description
- **Prompt**: `"Create a Python program that calculates the average of numbers from a file. Use a small test dataset, run it, identify any errors, fix them, and verify the final output."`
- **Agent**: `ReActAgent` using `qwen2.5-coder-7b-instruct-q8_0`.
- **Observed Behavior**:
  1. The coder model decided to inspect `test_numbers.txt` first: `Action: read_file(file_path="test_numbers.txt")`.
  2. The tool returned: `Observation: Error: file not found at test_numbers.txt`.
  3. Instead of transitioning to `Action: write_file(file_path="test_numbers.txt", content="10\n20\n30")`, the model outputted text reasoning about the numbers and then repeated `Action: read_file(file_path="test_numbers.txt")`.
  4. The loop repeated 19 times until reaching `Execution exceeded max_iterations (20)`.

### Root Cause Analysis
1. **Read-Only Action Exclusion from Coding Gate**: Ultron's `_coding_gate` interceptor only tracks state-modifying actions (`create_file`, `write_file`, `patch_file`, `run_command`). Read-only actions like `read_file` are not intercepted by the repair state machine.
2. **Local Model In-Context Behavior**: When `read_file` returns a file not found error, small local coder models (7B) often get caught in an in-context attention loop where the preceding observation anchors the next generated action back to the same tool call unless the observation explicitly guides action diversion.

### Recommendation
Under the strict Phase 4 Feature Freeze, no intrusive modifications to the ReAct agent loop or prompt templates were made without baseline validation. A low-risk enhancement for future iterations is adding an actionable hint to read-only tool error observations (e.g., `Error: file not found at test_numbers.txt. If this file does not exist yet, you must create it using write_file before reading it.`) or enforcing a repeated-failing-tool circuit breaker in `ReActAgent`.
