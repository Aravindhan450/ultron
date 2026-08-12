"""ultron.core.coding
~~~~~~~~~~~~~~~~~~~~~

Coding workspace + execution context (Fix #3, stage 1).

Gives the coding agent reliable awareness of the repository/environment it
operates on, WITHOUT yet implementing the autonomous repair loop:

- :mod:`workspace` — CodingWorkspace discovery (project type, languages,
  package manager, build/test system, git state) + deterministic
  ``list_directory`` / ``search_files`` tools.
- :mod:`observations` — structured observations (file content, command,
  test, build, error, diff, repository state) instead of one giant string.
- :mod:`command` — structured ``CommandResult`` (exit code, stdout, stderr,
  duration, timeout) parsed from the existing run_command output.
- :mod:`edits` — safe edit operations (create / replace / targeted edit /
  append / delete / rename) + modification tracking (``FileModification`` /
  ``ModificationTracker``) with optional git status/diff integration.
- :mod:`context` — CodeContext: workspace + relevant files + observations +
  modifications, attached to a TaskState (duck-typed; never duplicates task
  state).

Security: the edit and read tools are registered with the shared registry
and classified by the existing SecurityBoundary — no privileged bypass.
"""

from ultron.core.coding.command import (
    CommandResult,
    capture_command,
    parse_command_output,
)
from ultron.core.coding.context import CodeContext
from ultron.core.coding.edits import (
    EditAction,
    FileModification,
    ModificationTracker,
    append_to_file,
    create_file,
    delete_file,
    record_tool_result,
    rename_file,
    replace_file,
    replace_in_file,
)
from ultron.core.coding.executor import (
    CodingExecutor,
    ExplorationState,
    FailureAnalysis,
    FailureCategory,
    RepairBudget,
    ValidationCommands,
    classify_failure,
    infer_validation_commands,
)
from ultron.core.coding.intelligence_bridge import (
    CodeIntelligenceBridge,
    IntelligenceQuery,
)
from ultron.core.coding.observations import Observation, ObservationKind
from ultron.core.coding.workspace import (
    CodingWorkspace,
    ProjectProfile,
    discover_workspace,
    discover_workspace_summary,
    git_diff,
    git_status,
    list_directory,
    search_files,
)

__all__ = [
    "CodeContext",
    "CodeIntelligenceBridge",
    "CodingExecutor",
    "CodingWorkspace",
    "CommandResult",
    "EditAction",
    "ExplorationState",
    "FailureAnalysis",
    "FailureCategory",
    "FileModification",
    "IntelligenceQuery",
    "ModificationTracker",
    "Observation",
    "ObservationKind",
    "ProjectProfile",
    "RepairBudget",
    "ValidationCommands",
    "append_to_file",
    "capture_command",
    "classify_failure",
    "create_file",
    "delete_file",
    "discover_workspace",
    "discover_workspace_summary",
    "git_diff",
    "git_status",
    "infer_validation_commands",
    "list_directory",
    "parse_command_output",
    "record_tool_result",
    "rename_file",
    "replace_file",
    "replace_in_file",
    "search_files",
]
