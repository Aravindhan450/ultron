# Ultron External Workspace & Sandbox Validation Report

**Date:** 2026-09-22  
**Target Environment Variable:** `ULTRON_WORKSPACE=~/UltronWorkspace`  
**Test Harness:** macOS Darwin / Python 3.12.13 / pytest 9.1.1 / ruff 0.16.x  
**Status:** COMPLETED & VERIFIED  

---

## 1. Executive Summary

Ultron previously operated primarily inside its own source/project repository directory, creating generated projects, virtual environments, and SQLite databases within the Ultron codebase.

In this phase, we implemented a dedicated **External Workspace Confinement** layer configured via `ULTRON_WORKSPACE=~/UltronWorkspace`. The external workspace ensures that all software engineering and artifact generation tasks create, execute, and verify projects independently on the user's main drive (`~/UltronWorkspace/<project>/`) without polluting or modifying Ultron's source tree.

Strict path safety and SecurityBoundary rules were maintained:
1. Dynamic expansion of `~` and environment variables without hardcoded usernames.
2. Robust path normalization and symlink resolution.
3. Path traversal attacks (`../`, `../../`, absolute escapes outside allowed bases) are strictly detected and hard-blocked by Guardrails.
4. Terminal commands executed via `run_command` execute with `cwd` set to the active artifact project directory.
5. All 1,785 automated tests pass (100%), and ruff linting remains clean (0 errors).

---

## 2. Architecture & Path Flow

```text
User Request ("Build a Python expense tracker application.")
     │
     ▼
AgentRuntime / Classifier (classify_task)
     │
     ▼
Artifact Planner (build_artifact_plan)
     │  - Slugify goal: "python-expense-tracker-application"
     │  - Resolve external root: ~/UltronWorkspace/python-expense-tracker-application/
     │  - Create directory: mkdir(parents=True, exist_ok=True)
     │  - Context activation: set_active_project_dir(resolved_project_dir)
     ▼
ReAct Execution Loop
     │
     ├── write_file / create_file / append_to_file / replace_in_file
     │     │  - resolve_project_path(rel_path) -> ~/UltronWorkspace/.../file.py
     │     │  - is_path_safe() verifies confinement inside get_allowed_base_dirs()
     │     ▼
     ├── run_command("python -m ...")
     │     │  - Default execution cwd -> get_active_project_dir()
     │     ▼
     └── Verification Step & UI Reporting
           │  - Empirical runtime observation
           │  - Final response reports: "Location: ~/UltronWorkspace/<project>"
```

---

## 3. Implementation Details

### A. Configuration & Path Resolution (`config.py`, `paths.py`)
- Added `workspace: str | None = None` to `Settings` in `src/ultron/core/config.py`.
- Added `get_configured_workspace() -> Path | None` in `src/ultron/core/tools/paths.py`:
  - Expands `~` using `expanduser().resolve()`.
  - Normalizes path and resolves symlinks.
  - Returns `None` if `ULTRON_WORKSPACE` is unset or empty.
- Added `get_allowed_base_dirs() -> list[Path]`:
  - Contains `ALLOWED_BASE_DIR` (Ultron process CWD) and the configured external workspace.
- Updated `is_path_safe(path)`:
  - Validates that the resolved absolute path is relative to at least one allowed base directory.
  - Detects path escapes and directory traversals (`../`).
- Added `set_active_project_dir(path)` and `get_active_project_dir() -> Path | None`:
  - Maintains the active artifact directory context.
- Added `resolve_project_path(path) -> Path`:
  - Resolves relative paths against the active project directory when set, otherwise process CWD.

### B. Security & File Access Policy (`file_policy.py`)
- Updated `FilePolicy._relative(path)` in `src/ultron/security/file_policy.py`:
  - Checks relative containment against all allowed base directories (`get_allowed_base_dirs()`).
  - Ensures Guardrails and SecurityBoundary evaluate actions in the external workspace identically to repository actions.

### C. Tool Execution & Working Directory (`command_runner.py`, File Tools)
- `_run_one(command, timeout, cwd=None)` and `run_command(command, cwd=None)`:
  - If `cwd` is not explicitly passed, defaults to `get_active_project_dir()`.
  - Subprocesses spawn with their working directory inside the external project folder.
- `write_file`, `read_file`, `create_file`, `replace_file`, `replace_in_file`, `list_directory`, `search_files`:
  - All relative file paths resolve against `resolve_project_path(...)`.

### D. Artifact Planning & CLI Response (`task_planning.py`, `main.py`)
- `build_artifact_plan(goal, workspace, project_dir=None)`:
  - Derives project folder slug from goal.
  - Resolves project root under `ULTRON_WORKSPACE` if configured, creates folder, and sets active project dir.
  - Stores `project_dir` in `TaskPlan`.
- `main.py`:
  - When rendering final assistant response for an artifact plan, appends `Location: <project_dir>` to the output.

---

## 4. Test & Verification Results

### A. Unit & Regression Tests (`pytest`)
- Created `tests/test_external_workspace.py` with 6 comprehensive test cases:
  1. `test_configured_workspace_resolution`: Tests `~` expansion to home directory and unset fallback.
  2. `test_allowed_base_dirs_includes_workspace`: Tests inclusion of external workspace in base directories.
  3. `test_is_path_safe_allows_both_bases`: Confirms operations inside CWD and external workspace are both permitted.
  4. `test_path_traversal_hard_blocked_by_security_boundary`: Verifies `../` escaping workspace is rejected by `check_file_access`.
  5. `test_artifact_plan_resolves_in_external_workspace`: Validates plan scaffolding under `~/UltronWorkspace/<slug>`.
  6. `test_file_operations_and_command_runner_in_active_project`: Verifies `write_file`, `read_file`, and `run_command` (testing `pwd`) execute within the active project directory.
- **Full Test Suite Results**:
  ```text
  ================ 1785 passed, 6 deselected in 34.06s ================
  ```
- **Linter Results (`ruff check .`)**:
  ```text
  All checks passed!
  ```

### B. Live Real-World CLI Validation
- Executed `ULTRON_WORKSPACE=~/UltronWorkspace ultron chat` in tmux.
- Sent user command: `"Build a Python expense tracker application."`
- **Observed Filesystem Changes on Host**:
  ```text
  $ ls -la ~/UltronWorkspace/python-expense-tracker-application/src/ultron/app/
  total 24
  -rw-r--r--  1 aravindhan  staff  138 22 Sep 12:01 expense_tracker.py
  -rw-r--r--  1 aravindhan  staff  158 22 Sep 12:01 models.py
  -rw-r--r--  1 aravindhan  staff  552 22 Sep 12:01 views.py
  ```
- Ultron source tree remained 100% clean of generated artifact files (`git status` showed no untracked project files in Ultron repository root).

---

## 5. Conclusion

The Ultron External Workspace implementation satisfies all requirements:
- Clean isolation between Ultron source code and generated projects.
- Dynamic path resolution without machine-specific hardcoding.
- Strict path traversal security preserved.
- Full automated test suite passes with zero regressions.
