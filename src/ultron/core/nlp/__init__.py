"""ultron.core.nlp
~~~~~~~~~~~~~~~~~

Natural-language understanding → intent → tool selection → argument
extraction → normalization for the agent tool-routing layer.

The layer is deliberately deterministic where possible: it decides WHAT the
user wants and WHICH tool + arguments satisfy it, then hands the final
action to the security boundary.  The LLM is never asked to produce raw
shell text, and natural-language wrappers never leak into executable
arguments.

Modules:

- :mod:`normalize` — command normalization (wrapper stripping, quote-safe)
- :mod:`intent` — ``IntentCategory`` + ``UserIntent`` + deterministic
  ``route_request()`` classifier
- :mod:`project` — natural-language → project-specific command discovery
- :mod:`capabilities` — tool capability metadata + schema-aware selection
- :mod:`interpret` — deterministic result interpretation
- :mod:`observe` — structured action records (observability)
"""

from ultron.core.nlp.capabilities import (
    TOOL_CAPABILITIES,
    ToolCapability,
    select_tool,
)
from ultron.core.nlp.intent import IntentCategory, UserIntent, route_request
from ultron.core.nlp.interpret import interpret_command_result
from ultron.core.nlp.normalize import (
    detect_explicit_test_command,
    normalize_terminal_command,
)
from ultron.core.nlp.observe import ActionRecord, recent_actions, record_action
from ultron.core.nlp.project import (
    ResolvedTestCommand,
    discover_project_command,
    resolve_explicit_test_command,
    resolve_test_command,
)
from ultron.core.nlp.workspace import (
    WorkspaceContext,
    git_changed_files,
    resolve_location_path,
    resolve_workspace,
)

__all__ = [
    "TOOL_CAPABILITIES",
    "ActionRecord",
    "IntentCategory",
    "ResolvedTestCommand",
    "ToolCapability",
    "UserIntent",
    "WorkspaceContext",
    "detect_explicit_test_command",
    "discover_project_command",
    "git_changed_files",
    "interpret_command_result",
    "normalize_terminal_command",
    "recent_actions",
    "record_action",
    "resolve_explicit_test_command",
    "resolve_location_path",
    "resolve_test_command",
    "resolve_workspace",
    "route_request",
    "select_tool",
]
