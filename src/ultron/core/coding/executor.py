"""ultron.core.coding.executor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CODING EXECUTOR — the deterministic policy + safety layer for carrying out
coding plan steps (Fix #3, stage 2).

Separation of responsibilities (one coherent execution architecture):

    Planner         -> "what needs to happen?"          (Fix #2, LLM)
    CodingExecutor  -> "how do I accomplish this step?" (this module, deterministic)
    Tools           -> "perform the operation"          (registry)
    TaskState       -> "what has happened?"             (Fix #1)
    Verifier        -> "is the goal actually satisfied?"(Fix #1/#2, LLM + TaskState)

The CodingExecutor is NOT a new agent and NOT a second execution loop. The
ReActAgent loop remains the single reasoning loop; this module gives that
loop deterministic coding intelligence it cannot reason about reliably:

- :func:`classify_failure` — classify a failed command/build/test into a
  category (syntax, compilation, test assertion, dependency, configuration,
  environment, runtime, permission, timeout, unknown) so the model can react
  appropriately instead of blindly repeating the command.
- :class:`RepairBudget` — safe limits for the repair loop: maximum repair
  attempts and a hard ban on repeating the *identical* failed action. When a
  fingerprint repeats beyond ``max_identical_actions`` the executor blocks
  the action before it executes.
- :func:`infer_validation_commands` — the right test/build/lint command for
  the detected project (pytest, npm test, cargo test, go test, mvn/gradle,
  ...) determined from the workspace profile, never hardcoded per-task.
- :class:`ExplorationState` — what has been inspected (files read, searches,
  tree listings) so the executor can demand deliberate exploration before
  modification and can summarize context for the loop.
- :class:`CodingExecutor` — ties it together: per-step guidance injected
  into the model context, gating of repeated identical failures, recording
  of observations into the repair budget + exploration state, and the
  pre-completion diff/modification report used by final verification.

Security: this module never executes tools and never bypasses the boundary.
Every actual action still flows through ``boundary.check()`` in the agent
loop. All budgets are per-task and serialized on the CodeContext, so they
survive confirmation and agent continuation.
"""

from __future__ import annotations

import json
import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Failure classification
# ---------------------------------------------------------------------------


class FailureCategory(str, Enum):
    """Deterministic category of a failed command/build/test."""

    SYNTAX = "syntax"
    COMPILATION = "compilation"
    TEST_ASSERTION = "test_assertion"
    DEPENDENCY = "dependency"
    CONFIGURATION = "configuration"
    ENVIRONMENT = "environment"
    RUNTIME = "runtime"
    PERMISSION = "permission"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


class FailureAnalysis(BaseModel):
    """One classified command failure."""

    category: FailureCategory
    command: str = ""
    summary: str = ""
    evidence: str = ""  # the most relevant line(s) of output
    repair_hint: str = ""
    # Fix #5 failure localization — populated when the output identifies a
    # file/line (traceback, compiler) and/or the failing test node.
    file: str | None = None
    line: int | None = None
    test_name: str | None = None

    def location_display(self) -> str:
        """Compact location: 'file:line (test_name)' or '' when unknown."""
        parts = []
        if self.file:
            parts.append(f"{self.file}:{self.line}" if self.line else self.file)
        if self.test_name:
            parts.append(self.test_name)
        return " ".join(parts).strip()

    def to_prompt_line(self, max_len: int = 300) -> str:
        head = f"[{self.category.value}] {self.command or 'command'}"
        loc = self.location_display()
        if loc:
            head = f"{head} @ {loc}"
        if self.summary:
            return f"{head}: {self.summary[: max_len - len(head) - 2]}"
        return head


class FailureLocation(BaseModel):
    """Deterministic file/line/test extraction from failure output (Fix #5)."""

    file: str | None = None
    line: int | None = None
    test_name: str | None = None


# Deterministic localization patterns — matched in priority order.
_TRACEBACK_FRAME_RE = re.compile(r'File\s+"([^"]+)",\s*line\s+(\d+)')
# Modern pytest frame: ``math_util.py:2: in add`` (the implementation frame).
_PY_IN_FRAME_RE = re.compile(r"([\w./\\-]+\.py):(\d+):\s+in\s+[\w.]+")
_PYTEST_FAILED_NODE_RE = re.compile(
    r"FAILED\s+([\w./\\-]+\.py::[\w./\\\[\]]+)"
)
_PYTEST_SHORT_NODE_RE = re.compile(
    r"([\w./\\-]+\.py::[\w./\\\[\]]+)"
)
_GO_TEST_FAIL_RE = re.compile(r"---\s+FAIL:\s*([\w./-]+)\s+\(([^)]+)\)")
_GO_FILE_LINE_RE = re.compile(r"([\w./-]+\.go):(\d+):")
_PY_FILE_LINE_RE = re.compile(r"([\w./\\-]+\.py):(\d+):")
_JEST_FAIL_RE = re.compile(r"FAIL\s+([\w./-]+\.(?:test|spec)\.(?:ts|js|tsx|jsx))")


def localize_failure(stdout: str = "", stderr: str = "") -> FailureLocation:
    """
    Extracts (file, line, test_name) from failure output deterministically.

    Priority: traceback / implementation frame (``file:line``, the repair
    target) -> pytest FAILED / short node (test name + test file) -> go FAIL
    -> generic go/py ``file:line`` -> jest FAIL. Returns a
    :class:`FailureLocation` (fields optional); never raises on malformed
    input.
    """
    haystack = f"{stderr or ''}\n{stdout or ''}"
    loc = FailureLocation()

    frame = _TRACEBACK_FRAME_RE.search(haystack)
    if frame:
        loc.file = frame.group(1)
        try:
            loc.line = int(frame.group(2))
        except ValueError:
            loc.line = None
    elif (py_frame := _PY_IN_FRAME_RE.search(haystack)):
        loc.file = py_frame.group(1)
        try:
            loc.line = int(py_frame.group(2))
        except ValueError:
            loc.line = None

    node = _PYTEST_FAILED_NODE_RE.search(haystack) or _PYTEST_SHORT_NODE_RE.search(
        haystack
    )
    if node:
        loc.test_name = node.group(1)
        if loc.file is None:
            loc.file = node.group(1).split("::", 1)[0]

    go_fail = _GO_TEST_FAIL_RE.search(haystack)
    if go_fail:
        loc.test_name = loc.test_name or go_fail.group(1)

    if loc.file is None:
        go_line = _GO_FILE_LINE_RE.search(haystack)
        if go_line:
            loc.file = go_line.group(1)
            try:
                loc.line = int(go_line.group(2))
            except ValueError:
                loc.line = None

    if loc.file is None:
        py_line = _PY_FILE_LINE_RE.search(haystack)
        if py_line:
            loc.file = py_line.group(1)
            try:
                loc.line = int(py_line.group(2))
            except ValueError:
                loc.line = None

    if loc.file is None:
        jest = _JEST_FAIL_RE.search(haystack)
        if jest:
            loc.file = jest.group(1)

    return loc


_RE_CATEGORY: list[tuple[FailureCategory, list[str]]] = [
    (FailureCategory.TIMEOUT, [r"timed\s*out", r"timeout\s*expired", r"killed\s*after"]),
    (
        FailureCategory.PERMISSION,
        [
            r"permission\s+denied",
            r"operation\s+not\s+permitted",
            r"not\s+permitted",
            r"access\s+denied",
            r"eacces",
        ],
    ),
    (
        FailureCategory.DEPENDENCY,
        [
            r"no\s+module\s+named",
            r"module\s+not\s+found",
            r"cannot\s+find\s+(?:the\s+)?package",
            r"could\s+not\s+resolve",
            r"unmet\s+dependency",
            r"failed\s+to\s+resolve",
            r"npm\s+err!",
            r"importerror",
            r"error:\s+cannot\s+find\s+module",
        ],
    ),
    (
        FailureCategory.CONFIGURATION,
        [
            r"config(?:uration)?\s*error",
            r"invalid\s+config",
            r"missing\s+config",
            r"unknown\s+option",
            r"no\s+such\s+option",
            r"environment\s+variable",
            r"is\s+not\s+set",
            r"not\s+configured",
        ],
    ),
    (
        FailureCategory.SYNTAX,
        [
            r"syntaxerror",
            r"invalid\s+syntax",
            r"syntax\s+error",
            r"unexpected\s+token",
            r"parse\s*error",
            r"parsing\s+error",
            r"unexpected\s+eof",
        ],
    ),
    (
        FailureCategory.COMPILATION,
        [
            r"compilation\s+failed",
            r"build\s+failed",
            r"cannot\s+find\s+symbol",
            r"undefined\s+reference",
            r"undefined:\s+\w+",  # Go: "undefined: foo"
            r"could\s+not\s+compile",
            r"no\s+rule\s+to\s+make\s+target",
            r"fatal\s+error",
            r"\bts\d{4,5}\b",
            r"error\[e\d{4}\]",
        ],
    ),
    (
        FailureCategory.ENVIRONMENT,
        [
            r"command\s+not\s+found",
            r"not\s+recognized\s+as",
            r"connection\s+refused",
            r"cannot\s+connect",
            r"could\s+not\s+connect",
            r"certificate",
            r"\bssl\b",
            r"no\s+such\s+host",
        ],
    ),
    (
        FailureCategory.TEST_ASSERTION,
        [
            r"assertionerror",
            r"\d+\s+failed",
            r"tests?\s+failed",
            r"failed\s+tests?",
            r"failures?=",
            r"expected\s+.+?(?:but|got|to)",
            r"ran\s+\d+\s+tests?",
        ],
    ),
    (
        FailureCategory.RUNTIME,
        [
            r"traceback\s+\(most\s+recent\s+call\s+last\)",
            r"runtimeerror",
            r"valueerror",
            r"keyerror",
            r"indexerror",
            r"typeerror",
            r"attributeerror",
            r"nameerror",
            r"\bpanic:",
            r"segmentation\s+fault",
            r"core\s+dumped",
            r"process\s+exited\s+with\s+code",
            r"exception\s+in",
        ],
    ),
]

_REPAIR_HINTS: dict[FailureCategory, str] = {
    FailureCategory.SYNTAX: "Fix the syntax error at the flagged location, then rerun.",
    FailureCategory.COMPILATION: "Resolve the compilation/build error (missing symbol, bad reference), then rebuild.",
    FailureCategory.TEST_ASSERTION: "Read the failing test, trace the implementation, fix the behavior, then rerun the affected test.",
    FailureCategory.DEPENDENCY: "Inspect the dependency configuration, ensure the package is declared/installable, then rerun.",
    FailureCategory.CONFIGURATION: "Inspect the referenced configuration and correct the invalid setting.",
    FailureCategory.ENVIRONMENT: "Verify the required tool/service is available in this environment, or adapt the approach.",
    FailureCategory.RUNTIME: "Inspect the traceback, locate the faulty code path, and fix the runtime error.",
    FailureCategory.PERMISSION: "The action was blocked by permissions — choose an allowed alternative or ask for approval.",
    FailureCategory.TIMEOUT: "The command exceeded the time budget — narrow its scope or investigate why it hangs.",
    FailureCategory.UNKNOWN: "Inspect the full error output to identify the cause before acting.",
}


def classify_failure(
    command: str,
    exit_code: int | None = None,
    stdout: str = "",
    stderr: str = "",
    timed_out: bool = False,
) -> FailureAnalysis:
    """
    Classifies a failed command into a :class:`FailureCategory`.

    Deterministic regex scan over (stderr + stdout); ``timed_out`` short
    circuits to TIMEOUT. Unclassified failures fall back to UNKNOWN with a
    generic hint. Never raises.
    """
    if timed_out:
        return FailureAnalysis(
            category=FailureCategory.TIMEOUT,
            command=command,
            summary="command exceeded the time budget",
            repair_hint=_REPAIR_HINTS[FailureCategory.TIMEOUT],
        )
    haystack = f"{stderr or ''}\n{stdout or ''}".lower()
    location = localize_failure(stdout=stdout, stderr=stderr)
    for category, patterns in _RE_CATEGORY:
        for pattern in patterns:
            match = re.search(pattern, haystack)
            if match:
                line = match.group(0)
                evidence = _first_relevant_line(stdout, stderr, line)
                return FailureAnalysis(
                    category=category,
                    command=command,
                    summary=f"matched '{line.strip()[:120]}'",
                    evidence=evidence,
                    repair_hint=_REPAIR_HINTS[category],
                    file=location.file,
                    line=location.line,
                    test_name=location.test_name,
                )
    return FailureAnalysis(
        category=FailureCategory.UNKNOWN,
        command=command,
        summary=f"exit code {exit_code or '?'} with no classified error pattern",
        evidence=_first_relevant_line(stdout, stderr),
        repair_hint=_REPAIR_HINTS[FailureCategory.UNKNOWN],
        file=location.file,
        line=location.line,
        test_name=location.test_name,
    )


def _first_relevant_line(stdout: str, stderr: str, needle: str | None = None) -> str:
    """First non-empty line around *needle*, else the first non-empty line."""
    for block in (stderr, stdout):
        if not block:
            continue
        if needle and needle in block:
            idx = block.find(needle)
            start = block.rfind("\n", 0, idx)
            return block[start + 1 :].splitlines()[0][:300]
        for line in block.splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("["):  # skip resource footers
                return stripped[:300]
    return ""


# ---------------------------------------------------------------------------
# Validation command inference (deterministic from the workspace profile)
# ---------------------------------------------------------------------------


class ValidationCommands(BaseModel):
    """The project's expected test / build / lint commands."""

    test: str | None = None
    build: str | None = None
    lint: str | None = None

    def non_empty(self) -> list[str]:
        return [c for c in (self.test, self.build, self.lint) if c]


def _node_package_scripts(root: str) -> dict[str, str]:
    """Reads ``package.json`` scripts deterministically (bounded, never raises)."""
    try:
        from pathlib import Path

        text = Path(root, "package.json").read_text(encoding="utf-8", errors="replace")
        if len(text) > 200_000:
            return {}
        payload = json.loads(text)
        return {
            str(k): str(v)
            for k, v in (payload.get("scripts") or {}).items()
            if isinstance(v, str)
        }
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


def infer_validation_commands(workspace: Any) -> ValidationCommands:
    """
    Infers the test/build/lint commands for a workspace from its detected
    profile (project type, package manager, test framework, config files).

    Pure inspection of the already-discovered profile + ``package.json``
    scripts — never executes anything.
    """
    if workspace is None:
        return ValidationCommands()
    project_type = getattr(workspace, "project_type", "unknown") or "unknown"
    package_manager = getattr(workspace, "package_manager", None)
    test_framework = getattr(workspace, "test_framework", None)
    build_system = getattr(workspace, "build_system", None)
    root = getattr(workspace, "project_root", None) or ""

    if project_type == "python":
        # -p no:cacheprovider avoids stale .pytest_cache state between runs
        # after an edit (fresh verification is essential for a repair loop).
        test = "pytest -q -p no:cacheprovider" if test_framework == "pytest" else None
        lint = "ruff check ."
        return ValidationCommands(test=test, lint=lint)

    if project_type == "node":
        scripts = _node_package_scripts(root)
        pm = package_manager or "npm"
        test = scripts.get("test") or ("npm test" if pm == "npm" else f"{pm} test")
        build = scripts.get("build") or (
            f"{pm} run build" if build_system or "build" in scripts else None
        )
        lint = scripts.get("lint") or None
        return ValidationCommands(test=test, build=build, lint=lint)

    if project_type == "rust":
        return ValidationCommands(
            test="cargo test", build="cargo build", lint="cargo clippy"
        )

    if project_type == "go":
        return ValidationCommands(
            test="go test ./...", build="go build ./...", lint="gofmt -l ."
        )

    if project_type == "java":
        gradle = (package_manager or "").startswith("gradle")
        return ValidationCommands(
            test="gradle test" if gradle else "mvn test",
            build="gradle build" if gradle else "mvn package",
        )

    return ValidationCommands()


# ---------------------------------------------------------------------------
# Repair budget
# ---------------------------------------------------------------------------


def _action_fingerprint(tool_name: str, arguments: dict) -> str:
    """Canonical fingerprint of a tool call for identical-action detection."""
    try:
        payload = json.dumps(arguments, sort_keys=True, default=str)
    except (TypeError, ValueError):
        payload = str(arguments)
    return f"{tool_name}|{payload}"


class RepairBudget(BaseModel):
    """
    Safe limits for the repair loop.

    - ``max_repair_attempts`` — total failed actions allowed before the step
      is considered beyond repair.
    - ``max_identical_actions`` — a hard ban: the *exact same* failed action
      may be repeated at most this many times, then the executor blocks it.
    - ``command_timeout_s`` — advisory bound surfaced to the loop.

    State is per-task and serialized with the CodeContext, so it survives
    confirmation and agent continuation.
    """

    max_repair_attempts: int = 4
    max_identical_actions: int = 2
    command_timeout_s: int = 120
    failed_fingerprints: dict[str, int] = Field(default_factory=dict)
    failure_count: int = 0

    def record_failure(self, tool_name: str, arguments: dict) -> int:
        """Records one failed action; returns how many times it has failed."""
        fingerprint = _action_fingerprint(tool_name, arguments)
        self.failed_fingerprints[fingerprint] = self.failed_fingerprints.get(
            fingerprint, 0
        ) + 1
        self.failure_count += 1
        return self.failed_fingerprints[fingerprint]

    def identical_failures(self, tool_name: str, arguments: dict) -> int:
        """How many times this exact action has already failed."""
        return self.failed_fingerprints.get(
            _action_fingerprint(tool_name, arguments), 0
        )

    def repeat_blocked(self, tool_name: str, arguments: dict) -> bool:
        """True when this identical action has failed too many times to repeat."""
        return self.identical_failures(tool_name, arguments) >= self.max_identical_actions

    def exhausted(self) -> bool:
        """True when the repair budget for this task is spent."""
        return self.failure_count >= self.max_repair_attempts

    def summary(self) -> str:
        if self.failure_count == 0:
            return "no failures recorded"
        return (
            f"{self.failure_count}/{self.max_repair_attempts} failures; "
            f"{len(self.failed_fingerprints)} distinct failing action(s)"
        )


# ---------------------------------------------------------------------------
# Exploration state
# ---------------------------------------------------------------------------


# Fix #4 code-intelligence tools — recorded as searches for observability.
_INTELLIGENCE_TOOLS = frozenset(
    {
        "code_search",
        "find_symbol",
        "find_definition",
        "find_references",
        "get_imports",
        "get_dependents",
        "semantic_search",
        "code_index_status",
    }
)


class ExplorationState(BaseModel):
    """Tracks what the executor has inspected so far."""

    files_read: list[str] = Field(default_factory=list)
    searches: list[str] = Field(default_factory=list)
    tree_listings: int = 0

    def record(self, tool_name: str, arguments: dict) -> None:
        """Records one inspection tool call into the exploration state."""
        if tool_name == "read_file":
            path = str(arguments.get("file_path", ""))
            if path and path not in self.files_read:
                self.files_read.append(path)
        elif tool_name == "search_files":
            query = str(arguments.get("query", ""))
            if query and query not in self.searches:
                self.searches.append(query)
        elif tool_name == "list_directory":
            self.tree_listings += 1
        elif tool_name in _INTELLIGENCE_TOOLS:
            query = str(
                arguments.get("query")
                or arguments.get("name")
                or arguments.get("file_path")
                or ""
            )
            label = f"{tool_name}: {query}" if query else tool_name
            if label not in self.searches:
                self.searches.append(label)

    def has_inspected(self) -> bool:
        return bool(self.files_read or self.searches or self.tree_listings)

    def summary(self) -> str:
        bits = []
        if self.tree_listings:
            bits.append(f"{self.tree_listings} tree listing(s)")
        if self.files_read:
            bits.append(f"{len(self.files_read)} file(s) read")
        if self.searches:
            bits.append(f"{len(self.searches)} search(es)")
        return ", ".join(bits) if bits else "nothing inspected yet"


# Keywords that mark a plan step as requiring deliberate exploration before
# any modification (matching the outcome-oriented step wording Fix #2 uses).
_EXPLORATION_KEYWORDS = (
    "inspect",
    "explore",
    "understand",
    "locate",
    "identify",
    "analy",
    "review",
    "trace",
    "investigate",
    "determine",
    "read",
    "look",
)

# Tools that change state — the only ones subject to identical-action gating.
_STATE_CHANGING_TOOLS = frozenset(
    {
        "run_command",
        "write_file",
        "create_file",
        "replace_file",
        "replace_in_file",
        "append_to_file",
        "delete_file",
        "rename_file",
    }
)

_INSPECTION_TOOLS = frozenset(
    {"read_file", "search_files", "list_directory"}
) | _INTELLIGENCE_TOOLS


def _step_needs_exploration(step: Any) -> bool:
    """True when a step's description/purpose clearly calls for inspection."""
    text = " ".join(
        str(part or "")
        for part in (
            getattr(step, "description", ""),
            getattr(step, "purpose", ""),
            getattr(step, "expected_outcome", ""),
        )
    ).lower()
    return any(keyword in text for keyword in _EXPLORATION_KEYWORDS)


# Word -> preferred intelligence tool for the tool-selection strategy.
_INTELLIGENCE_HINTS: list[tuple[tuple[str, ...], str]] = [
    (
        ("definition", "where is", "defined", "declare", "implemented"),
        (
            "Prefer find_definition (exact symbol lookup) over grep for "
            "locating symbols."
        ),
    ),
    (
        ("reference", "caller", "usage", "used", "rename", "calls"),
        (
            "Prefer find_references to enumerate usages/callers instead of "
            "scanning files manually."
        ),
    ),
    (
        ("import", "depend", "module", "package"),
        "Prefer get_imports / get_dependents for dependency relationships.",
    ),
    (
        ("trace", "endpoint", "handler", "route", "flow"),
        (
            "Trace with find_definition then find_references (and get_imports) "
            "to follow the call path."
        ),
    ),
    (
        ("concept", "semantic", "related", "middleware", "architecture"),
        (
            "Prefer semantic_search for conceptual matches; exact symbol "
            "lookup is insufficient for natural-language questions."
        ),
    ),
]


def _intelligence_tool_hint(step: Any) -> str:
    """
    Deterministic tool-selection hint for the current step (Fix #4).

    Chooses the most precise available mechanism based on the step wording:
    exact symbol lookup before semantic search, semantic before raw lexical
    grep. Falls back to generic guidance when no hint matches.
    """
    text = " ".join(
        str(part or "")
        for part in (
            getattr(step, "description", ""),
            getattr(step, "purpose", ""),
            getattr(step, "expected_outcome", ""),
        )
    ).lower()
    for keywords, hint in _INTELLIGENCE_HINTS:
        if any(keyword in text for keyword in keywords):
            return hint
    return (
        "Use find_definition / find_references for exact symbols, "
        "code_search for text, semantic_search for concepts — never "
        "search the whole repository when an index query suffices."
    )


# ---------------------------------------------------------------------------
# The CodingExecutor
# ---------------------------------------------------------------------------


class CodingExecutor(BaseModel):
    """
    Deterministic coding policy layer consulted by the ReAct loop.

    Methods are duck-typed against TaskState / PlanStep / CodeContext so this
    module never imports agent or task modules (avoids import cycles) and
    stays usable wherever a task carries a CodeContext.
    """

    budget: RepairBudget = Field(default_factory=RepairBudget)
    exploration: ExplorationState = Field(default_factory=ExplorationState)
    failures: list[FailureAnalysis] = Field(default_factory=list)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_observation(
        self,
        tool_name: str,
        arguments: dict,
        observation: str,
        succeeded: bool,
    ) -> None:
        """Records a tool observation: exploration for inspection tools, and
        classified failures for state-changing tools."""
        if tool_name in _INSPECTION_TOOLS and succeeded:
            self.exploration.record(tool_name, arguments)
        if not succeeded and tool_name in _STATE_CHANGING_TOOLS:
            self.budget.record_failure(tool_name, arguments)
            analysis = self._classify_observation(tool_name, arguments, observation)
            if analysis is not None:
                self.failures.append(analysis)
                self.failures = self.failures[-12:]  # bounded history

    def _classify_observation(
        self, tool_name: str, arguments: dict, observation: str
    ) -> FailureAnalysis | None:
        """Classifies a command observation; commands are the main source."""
        if tool_name != "run_command":
            return None
        text = str(observation)
        match = re.search(r"Exit code:\s*(-?\d+)", text)
        exit_code = int(match.group(1)) if match else None
        timed_out = "timed out" in text.lower()
        command = str(arguments.get("command", ""))
        return classify_failure(
            command=command,
            exit_code=exit_code,
            stdout=text,
            stderr="",
            timed_out=timed_out,
        )

    # ------------------------------------------------------------------
    # Gating
    # ------------------------------------------------------------------

    def gate_action(self, tool_name: str, arguments: dict) -> str | None:
        """
        Returns a blocking observation string (or None) for an action that
        must NOT execute.

        A state-changing action that has already failed the same way more
        than ``max_identical_actions`` times is blocked before it runs — the
        loop feeds the returned message back as an observation instead of
        executing, so the model is forced to change its approach.
        """
        if tool_name not in _STATE_CHANGING_TOOLS:
            return None
        if not self.budget.repeat_blocked(tool_name, arguments):
            return None
        count = self.budget.identical_failures(tool_name, arguments)
        return (
            "Error: This exact action has already failed "
            f"{count} time(s). Do not repeat it identically — inspect the "
            "failure, adjust your approach, and only then retry."
        )

    def gate_new_action_with_exhausted_budget(self, tool_name: str) -> str | None:
        """
        Blocks NEW state-changing actions once the repair budget is spent.
        """
        if tool_name in _STATE_CHANGING_TOOLS and self.budget.exhausted():
            return (
                "Error: The repair budget for this task is exhausted "
                f"({self.budget.summary()}). Do not attempt further "
                "state-changing actions; report the task as incomplete."
            )
        return None

    # ------------------------------------------------------------------
    # Guidance + reporting
    # ------------------------------------------------------------------

    def step_guidance(self, task: Any) -> str:
        """
        Deterministic guidance for the CURRENT plan step, injected into the
        model context so the loop knows how to accomplish the step.

        Includes: whether the step requires exploration first, the project's
        validation commands, the repair budget state, and the exploration
        state so far.
        """
        lines = ["CODING EXECUTOR GUIDANCE:"]
        step = None
        plan = getattr(task, "plan", None)
        if plan is not None:
            step = task.current_plan_step() if hasattr(task, "current_plan_step") else None

        workspace = None
        code_context = getattr(task, "code_context", None)
        if code_context is not None:
            workspace = getattr(code_context, "workspace", None)

        if step is not None:
            needs_exploration = _step_needs_exploration(step)
            if needs_exploration and not self.exploration.has_inspected():
                lines.append(
                    "  This step requires understanding first: use "
                    "list_directory / search_files / read_file or the code "
                    "intelligence tools (find_definition / find_references / "
                    "code_search / semantic_search) to inspect the relevant "
                    "code BEFORE modifying anything."
                )
                lines.append("  " + _intelligence_tool_hint(step))
            elif needs_exploration and self.exploration.has_inspected():
                lines.append(
                    "  Exploration in progress — keep inspecting until you "
                    "understand the relevant code."
                )
            else:
                lines.append("  This step is an implementation/validation step.")
        else:
            lines.append("  No active plan step.")

        commands = infer_validation_commands(workspace)
        if commands.non_empty():
            parts = []
            if commands.test:
                parts.append(f"tests: {commands.test}")
            if commands.build:
                parts.append(f"build: {commands.build}")
            if commands.lint:
                parts.append(f"lint: {commands.lint}")
            lines.append("  Validation commands for this project: " + "; ".join(parts))
            lines.append(
                "  Run the relevant validation after modifying code; a coding "
                "task is not complete while its build/tests fail."
            )

        if self.exploration.has_inspected():
            lines.append(f"  Explored so far: {self.exploration.summary()}.")
        lines.append(f"  Repair budget: {self.budget.summary()}.")
        return "\n".join(lines)

    def intelligence_guidance(self, task: Any) -> str:
        """
        Code-intelligence guidance for the current step, injected into the
        model context (Fix #4 integration).

        When the code context has an enabled intelligence bridge, this
        returns:

        - the preferred tool-selection strategy for the step (exact symbol
          lookup over grep, semantic only for conceptual questions), and
        - a BOUNDED targeted context block (definitions/references for the
          step's candidate symbols) so the model never receives a repository
          dump.

        Returns '' when there is no code context or the bridge is disabled
        (e.g. workspace outside the allowed base dir) — guidance simply
        degrades to the plain exploration advice.
        """
        code_context = getattr(task, "code_context", None)
        if code_context is None:
            return ""
        bridge = getattr(code_context, "intelligence", None)
        if bridge is None or not getattr(bridge, "enabled", False):
            return ""

        lines = ["CODE INTELLIGENCE:"]
        step = (
            task.current_plan_step() if hasattr(task, "current_plan_step") else None
        )
        if step is not None:
            lines.append("  " + _intelligence_tool_hint(step))
        else:
            lines.append(
                "  Use find_definition / find_references for exact symbols, "
                "code_search for text, semantic_search for concepts."
            )
        # The targeted context block is only computed for steps that call for
        # exploration (mirrors step_guidance) — implementation steps just get
        # the tool-selection hint, so we never pay for index queries on pure
        # implementation/validation steps.
        if step is not None and _step_needs_exploration(step):
            try:
                block = bridge.context_block(task)
            except Exception:  # noqa: BLE001 — intelligence must never crash the loop
                block = ""
            if block:
                lines.append(block)
        usage = bridge.usage_summary()
        if usage and "no code-intelligence queries" not in usage:
            lines.append(f"  Intelligence usage: {usage}.")
        return "\n".join(lines)

    def pre_completion_report(self, task: Any) -> str:
        """
        Diff/modification report for final verification: what changed, via
        git status/diff when available, else the tracked modifications.
        Also flags modifications that appear UNRELATED to the task's
        relevant files so the verifier can catch accidental edits.
        """
        code_context = getattr(task, "code_context", None)
        if code_context is None:
            return ""

        tracker = getattr(code_context, "tracker", None)
        relevant_files = list(getattr(code_context, "relevant_files", []) or [])
        workspace = getattr(code_context, "workspace", None)

        parts: list[str] = ["MODIFICATION REPORT (what the assistant changed):"]
        modified = [m for m in (tracker.modifications if tracker else [])]
        if modified:
            for mod in modified:
                status = "ok" if mod.success else f"FAILED: {mod.error}"
                parts.append(f"  - {mod.describe()} [{status}]")
        else:
            parts.append("  - no file modifications recorded")

        # git integration where available.
        root = getattr(workspace, "project_root", None) if workspace else None
        if root:
            from ultron.core.coding.workspace import git_diff, git_status

            status = git_status(str(root))
            diff = git_diff(str(root), max_chars=3000)
            if status:
                parts.append("  git status (short):")
                for line in status.splitlines()[:12]:
                    parts.append(f"    {line}")
            if diff:
                parts.append("  git diff (truncated):")
                parts.append("    " + diff.replace("\n", "\n    "))

        # Unrelated-modification detection: changed paths outside the task's
        # relevant files (and outside the project's own source/test dirs).
        source_dirs = getattr(workspace, "source_dirs", []) if workspace else []
        test_dirs = getattr(workspace, "test_dirs", []) if workspace else []
        expected_prefixes = tuple(
            p for p in (*relevant_files, *source_dirs, *test_dirs) if p
        )
        unrelated = []
        for mod in modified:
            if mod.success and expected_prefixes and not mod.path.startswith(
                expected_prefixes
            ):
                unrelated.append(mod.path)
        if unrelated:
            parts.append(
                "  WARNING — modifications outside the task's relevant files: "
                + ", ".join(sorted(set(unrelated)))
                + ". Verify these are intentional before completing."
            )
        return "\n".join(parts)

    def verification_evidence(self, task: Any) -> str:
        """
        Compact evidence appended to the completion-verification prompt:
        modified files, classified failures, validation commands and diff
        status — so the verifier decides completion with the actual state,
        never the model's word alone.
        """
        code_context = getattr(task, "code_context", None)
        workspace = getattr(code_context, "workspace", None) if code_context else None
        tracker = getattr(code_context, "tracker", None) if code_context else None

        lines: list[str] = []
        modified = [m for m in (tracker.modifications if tracker else [])]
        if modified:
            lines.append(
                "Modified files: "
                + ", ".join(sorted({m.path for m in modified if m.success}))
            )
        if self.failures:
            lines.append(
                "Classified failures: "
                + "; ".join(a.to_prompt_line() for a in self.failures[-3:])
            )
        commands = infer_validation_commands(workspace)
        if commands.test:
            lines.append(f"Validation command: {commands.test}")
        if workspace is not None and getattr(workspace, "is_git_repo", False):
            lines.append("Workspace is a git repo — consult git status/diff.")
        # Fix #4: surface what the code-intelligence layer actually queried
        # so verification evidence reflects how the agent found its facts.
        code_context = getattr(task, "code_context", None)
        if code_context is not None:
            bridge = getattr(code_context, "intelligence", None)
            if bridge is not None:
                usage = bridge.usage_summary()
                if usage and "no code-intelligence queries" not in usage:
                    lines.append("Code intelligence: " + usage)
        return "\n".join(lines) if lines else ""
