"""Generic capability task generation (Phases 5-7 of STEP 3, refined in 3.1).

The generator consumes the canonical vocabulary — ``ToolCapability`` and its
``CapabilityContract`` — and a dynamically discovered subject pool, then
produces varied natural-language tasks for the *actual model* to execute.

STEP 3.1 decoupling: a task's VALIDITY comes from repository/capability
ground truth (contract, subject, operation wording, success criteria) —
never from the deterministic intent router.  The router is still executed
and recorded as a DIAGNOSTIC signal (``router_capability`` /
``router_agreement`` on the case); a router disagreement never rejects a
semantically valid task.

Development and holdout tasks use STRUCTURALLY DIFFERENT template families
(direct/explicit vs indirect/conversational/implicit) with independent
seeds, so the holdout measures capability understanding, not development
prompt grammar.  One semantic entity is tracked separately from its surface
forms (``entity_id`` vs ``surface_form``), and entity/capability selection
is coverage-aware (prefer under-tested subjects).

Capability coverage is classified up front so the framework can report —
never silently exclude — capabilities it will not auto-generate (unsafe,
external, or meta capabilities).
"""

from __future__ import annotations

import random
import re
from collections import Counter
from collections.abc import Callable
from enum import Enum
from pathlib import Path

from ultron.core.capabilities.contracts import contract_for
from ultron.core.capabilities.selector import SelectionState, select_for_request
from ultron.core.tools.definitions import ToolCapability
from ultron.validation.model import (
    CapabilityTestCase,
    Difficulty,
    GenerationStrategy,
    TestSource,
    TestSplit,
)
from ultron.validation.subjects import Subject

# ---------------------------------------------------------------------------
# Capability testability classification.
#
# This is *generation policy* — it decides whether the framework will build
# tasks for a capability, not what the capability means.  It contains no tool
# names, no risk levels, no read-only flags: those stay in TOOL_DEFINITIONS.
# ---------------------------------------------------------------------------


class Testability(str, Enum):
    """How (and whether) the framework auto-generates tasks for a capability."""

    READ_ONLY = "read_only"  # fully automatable, safe, deterministic
    EXECUTION = "requires_execution"  # real execution (tests/lint/typecheck/build)
    EXTERNAL = "external"  # needs live external information / network
    UNSAFE = "unsafe"  # modifies files/state or launches processes
    NOT_TESTABLE = "not_testable"  # no safe deterministic generation path


CAPABILITY_TESTABILITY: dict[ToolCapability, Testability] = {
    # --- read-only repository/code-intelligence capabilities ---------------
    ToolCapability.DEFINITION_LOOKUP: Testability.READ_ONLY,
    ToolCapability.REFERENCE_LOOKUP: Testability.READ_ONLY,
    ToolCapability.SYMBOL_SEARCH: Testability.READ_ONLY,
    ToolCapability.SYMBOL_INSPECTION: Testability.READ_ONLY,
    ToolCapability.CODE_SEARCH: Testability.READ_ONLY,
    ToolCapability.SEMANTIC_SEARCH: Testability.READ_ONLY,
    ToolCapability.REPOSITORY_INSPECTION: Testability.READ_ONLY,
    ToolCapability.REPOSITORY_INVESTIGATION: Testability.READ_ONLY,
    ToolCapability.DEPENDENCY_ANALYSIS: Testability.READ_ONLY,
    ToolCapability.FILE_SEARCH: Testability.READ_ONLY,
    ToolCapability.FILE_READ: Testability.READ_ONLY,
    ToolCapability.FILE_INSPECTION: Testability.READ_ONLY,
    ToolCapability.DIRECTORY_LIST: Testability.READ_ONLY,
    ToolCapability.GIT_OPERATION: Testability.READ_ONLY,
    ToolCapability.TERMINAL_EXECUTION: Testability.READ_ONLY,
    ToolCapability.MEMORY_QUERY: Testability.READ_ONLY,
    ToolCapability.GRAPH_REASONING: Testability.READ_ONLY,
    ToolCapability.RESOURCE_MONITORING: Testability.READ_ONLY,
    ToolCapability.DEBUG_ENVIRONMENT: Testability.READ_ONLY,
    ToolCapability.STRUCTURED_OUTPUT: Testability.READ_ONLY,
    ToolCapability.API_SCHEMA_LEARNING: Testability.READ_ONLY,
    ToolCapability.PARALLEL_BATCH: Testability.READ_ONLY,
    # --- real execution (safe, slow) ----------------------------------------
    ToolCapability.TEST_EXECUTION: Testability.EXECUTION,
    ToolCapability.LINT: Testability.EXECUTION,
    ToolCapability.TYPECHECK: Testability.EXECUTION,
    ToolCapability.BUILD: Testability.EXECUTION,
    # --- external information / network --------------------------------------
    ToolCapability.WEB_SEARCH: Testability.EXTERNAL,
    ToolCapability.HTTP_REQUEST: Testability.EXTERNAL,
    ToolCapability.PAGE_FETCH: Testability.EXTERNAL,
    # --- unsafe for automated execution --------------------------------------
    ToolCapability.FILE_WRITE: Testability.UNSAFE,
    ToolCapability.FILE_CREATE: Testability.UNSAFE,
    ToolCapability.FILE_DELETE: Testability.UNSAFE,
    ToolCapability.FILE_RENAME: Testability.UNSAFE,
    ToolCapability.DIRECTORY_CREATE: Testability.UNSAFE,
    ToolCapability.FORMAT: Testability.UNSAFE,
    ToolCapability.INSTALL: Testability.UNSAFE,
    ToolCapability.APPLICATION_START: Testability.UNSAFE,
    ToolCapability.APPLICATION_STOP: Testability.UNSAFE,
    ToolCapability.MEMORY_UPDATE: Testability.UNSAFE,
    ToolCapability.MEMORY_ASSOCIATION: Testability.UNSAFE,
    ToolCapability.PLAN_MANAGEMENT: Testability.UNSAFE,
    # --- no safe deterministic generation path -------------------------------
    ToolCapability.CODING_REQUEST: Testability.NOT_TESTABLE,  # meta: multi-step code modification
    ToolCapability.INFORMATION_REQUEST: Testability.NOT_TESTABLE,  # context-dependent (repo vs external)
    ToolCapability.DATABASE_QUERY: Testability.NOT_TESTABLE,  # depends on live database state
}


def testability(capability: ToolCapability) -> Testability:
    return CAPABILITY_TESTABILITY.get(capability, Testability.NOT_TESTABLE)


# ---------------------------------------------------------------------------
# Entity naming variants (surface forms of ONE semantic entity).
# ---------------------------------------------------------------------------


def symbol_variants(name: str) -> tuple[str, ...]:
    """Casing/format variants of one symbol (surface forms, same entity).

    ``TaskState`` -> ``TaskState``, ``taskstate``, ``TASKSTATE``,
    ``Task State``, ``task state``, ``task_state``.
    ``task_state`` (snake input) -> ``task_state``, ``TaskState``, ``task state``.
    Deterministic; only semantically valid forms are emitted.
    """
    variants = {name, name.lower(), name.upper()}
    words = re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+", name)
    if len(words) > 1:
        title = " ".join(w.capitalize() for w in words if w)
        variants.add(title)
        variants.add(title.lower())
        # Underscore form only: hyphenated forms of CamelCase identifiers are
        # not valid references to the entity and break the intent detectors.
        variants.add("_".join(w.lower() for w in words))
    if "_" in name:
        words2 = [w for w in name.split("_") if w]
        variants.add("".join(w.capitalize() for w in words2))
        variants.add(" ".join(words2))
    return tuple(sorted(v for v in variants if v))


# ---------------------------------------------------------------------------
# Operation words per capability.
#
# Task-construction metadata: a valid task for a capability must EXPRESS that
# capability's operation.  These words are the capability's own vocabulary
# (names + contract purpose), used ONLY for semantic validity of generated
# tasks — never for evaluating the model.
# ---------------------------------------------------------------------------

_OPERATION_WORDS: dict[ToolCapability, tuple[str, ...]] = {
    ToolCapability.DEFINITION_LOOKUP: ("defined", "definition", "declared", "declare"),
    ToolCapability.REFERENCE_LOOKUP: ("reference", "references", "referenced", "used", "usage", "usages", "call", "calls", "called by", "invoke", "invokes"),
    ToolCapability.SYMBOL_SEARCH: ("symbol", "class", "function", "named", "called"),
    ToolCapability.SYMBOL_INSPECTION: ("what does", "what is", "about", "do", "explain"),
    ToolCapability.CODE_SEARCH: ("search", "grep", "occurrence", "appear", "mentions", "mentions"),
    ToolCapability.SEMANTIC_SEARCH: ("code", "handles", "responsible", "semantic", "deals", "located", "happen"),
    ToolCapability.REPOSITORY_INSPECTION: ("inspect", "structure", "structured", "overview", "implement", "implemented", "describe"),
    ToolCapability.REPOSITORY_INVESTIGATION: ("how", "works", "work", "implemented", "explain", "mechanism", "under the hood", "delegate", "break down"),
    ToolCapability.FILE_SEARCH: ("files", "file", "named", "matching", "called"),
    ToolCapability.FILE_READ: ("read", "show", "contents", "open", "display", "inside", "pull up"),
    ToolCapability.FILE_INSPECTION: ("inspect", "details", "about", "tell"),
    ToolCapability.DIRECTORY_LIST: ("list", "files", "contents", "in", "what is in", "live"),
    ToolCapability.GIT_OPERATION: ("status", "diff", "commit", "log", "repo", "repository", "state", "changes"),
    ToolCapability.TERMINAL_EXECUTION: ("run", "execute", "directory", "working directory", "date", "time", "pwd"),
    ToolCapability.MEMORY_QUERY: ("remember", "know", "memory", "about"),
    ToolCapability.GRAPH_REASONING: ("know", "knowledge", "graph", "about"),
    ToolCapability.RESOURCE_MONITORING: ("resource", "resources", "usage", "cpu", "memory", "system"),
    ToolCapability.DEBUG_ENVIRONMENT: ("env", "environment", "configuration", "config", "variable", "variables"),
    ToolCapability.STRUCTURED_OUTPUT: ("structured", "json", "summary"),
    ToolCapability.API_SCHEMA_LEARNING: ("schema", "api", "endpoint", "surface"),
    ToolCapability.PARALLEL_BATCH: ("parallel", "batch", "at once", "together", "inspection", "inspections", "checks"),
    ToolCapability.TEST_EXECUTION: ("test", "tests", "suite"),
    ToolCapability.LINT: ("lint", "linter", "style", "code style"),
    ToolCapability.TYPECHECK: ("type", "types", "typecheck", "type-clean"),
    ToolCapability.BUILD: ("build", "compile", "compiles"),
}

# Words that would make a task mean the OPPOSITE of reading/investigating.
_CONFLICT_WORDS = (
    "delete",
    "remove",
    "create",
    "write",
    "rename",
    "move",
    "install",
    "launch",
    "start",
    "stop",
    "kill",
    "format",
    "overwrite",
    "modify",
    "edit",
    "change",
    "destroy",
    "erase",
)


def _expresses_operation(case: CapabilityTestCase) -> bool:
    """The task text must express the expected capability's operation."""
    words = _OPERATION_WORDS.get(case.expected_capability)
    if not words:
        return True  # no vocabulary declared -> trust the template family
    return any(re.search(rf"\b{re.escape(w)}\b", case.task, re.IGNORECASE) for w in words)


def _has_conflicting_operation(case: CapabilityTestCase) -> bool:
    """A task must not ask for a conflicting operation (e.g. \"delete\" for read)."""
    text = case.task.lower()
    if case.subject:
        for v in symbol_variants(case.subject):
            text = text.replace(v.lower(), "")
    return any(re.search(rf"\b{w}\b", text) for w in _CONFLICT_WORDS)


# ---------------------------------------------------------------------------
# Template families per capability per strategy.
#
# DEVELOPMENT templates are direct and explicit; HOLDOUT templates are
# indirect, conversational, and multi-clause with implicit capability.  The
# two families are structurally different — not the same table with a
# different seed.  No template references a specific symbol.
# ---------------------------------------------------------------------------

_SAFE_TERMINAL_COMMANDS: tuple[str, ...] = (
    "pwd",
    "git status",
    "git log --oneline -5",
    "ls",
)


def _make_template(fmt: str) -> Callable[[Subject, str, str], str]:
    def render(subject: Subject, variant: str, command: str = "") -> str:
        return fmt.format(s=variant, p=subject.rel_path, n=subject.name, c=command)

    return render


def _build_table(templates: dict[ToolCapability, dict[str, tuple[str, ...]]]) -> dict[ToolCapability, dict[str, Callable[[Subject, str, str], str]]]:
    built: dict[ToolCapability, dict[str, Callable[[Subject, str, str], str]]] = {}
    for cap, families in templates.items():
        table: dict[str, Callable[[Subject, str, str], str]] = {}
        for tid, fmts in families.items():
            for i, fmt in enumerate(fmts):
                table[f"{cap.value}.{tid}.{i}"] = _make_template(fmt)
        built[cap] = table
    return built


_DEV_TEMPLATES: dict[ToolCapability, dict[str, tuple[str, ...]]] = {
    ToolCapability.DEFINITION_LOOKUP: {
        "where_defined": ("Where is {s} defined?", "Where is {s} defined in the codebase?"),
        "find_where_defined": ("Find where {s} is defined", "Find where {s} is defined in the project"),
        "definition_of": ("Show me the definition of {s}", "What is the definition of {s}?"),
        "declared": ("Where is {s} declared?", "Find where {s} is declared"),
    },
    ToolCapability.REFERENCE_LOOKUP: {
        "where_used": ("Where is {s} used?", "Where is {s} used in the project?"),
        "references": ("Find references to {s}", "Find all references to {s}", "Find references to {s} in the codebase"),
        "usages": ("Find usages of {s}", "Where are the usages of {s}?"),
        "who_uses": ("Who uses {s}?", "Who calls {s}?"),
        "referenced": ("Where is {s} referenced?", "Where is {s} called?"),
    },
    ToolCapability.SYMBOL_SEARCH: {
        "find_symbol": ("Find the symbol {s}", "Look up the symbol {s}"),
        "find_class": ("Find the class {s}", "Show me the class {s}"),
    },
    ToolCapability.SYMBOL_INSPECTION: {
        "what_does": ("What does {s} do?", "What does {s} do in this repository?"),
        "what_is": ("What is {s}?", "Tell me about {s}"),
    },
    ToolCapability.CODE_SEARCH: {
        "search_code": ("Search the code for {s}", "Search the codebase for {s}"),
        "grep": ("Grep the repository for {s}", "Grep the code for {s}"),
        "occurrences": ("Find every occurrence of {s} in the source", "Where does {s} appear in the repository?"),
    },
    ToolCapability.SEMANTIC_SEARCH: {
        "responsible": ("Find the code responsible for {s}", "Find code that handles {s}"),
        "semantic": ("Search semantically for {s}", "Semantically search the repository for {s}"),
        "where_does": ("Where does {s} happen?", "Where does {s} live in the codebase?"),
    },
    ToolCapability.REPOSITORY_INSPECTION: {
        "inspect_module": ("Inspect {s}", "Show me how {s} is structured"),
        "describe": ("Describe {s} in this repository", "What files implement {s}?"),
    },
    ToolCapability.REPOSITORY_INVESTIGATION: {
        "how_works": ("How does {s} work?", "How does {s} work in this repository?"),
        "how_implemented": ("How is {s} implemented?", "Explain how {s} is implemented"),
        "where_implemented": ("Where is {s} implemented?", "Find where {s} is implemented"),
        "explain": ("Explain how {s} works", "Explain how {s} operates"),
        "delegate": ("How does {s} delegate?", "How does {s} interact with other components?"),
    },
    ToolCapability.FILE_SEARCH: {
        "files_named": ("Search for files named {s}", "Find files named {s}"),
        "files_matching": ("Search for files matching {s}", "Find files matching {s}"),
    },
    ToolCapability.FILE_READ: {
        "read": ("Read {p}", "Read the file {p}"),
        "contents": ("Show me the contents of {p}", "Display the contents of {p}"),
        "open": ("Open {p}", "Show {p}"),
    },
    ToolCapability.FILE_INSPECTION: {
        "inspect": ("Inspect the file {p}", "Show details about {p}"),
    },
    ToolCapability.DIRECTORY_LIST: {
        "list": ("List the files in {p}", "List the contents of {p}"),
        "what_in": ("What is in {p}?", "Show me what is in {p}"),
    },
    ToolCapability.GIT_OPERATION: {
        "status": ("Show git status", "Show me the git status"),
        "diff": ("Show the current diff", "Show me the current git diff"),
        "log": ("Show the recent commit history", "Show the last few commits"),
    },
    ToolCapability.TERMINAL_EXECUTION: {
        "run": ("Run {c}", "Execute: {c}"),
        "info": ("What is the current working directory?", "What time is it?"),
    },
    ToolCapability.MEMORY_QUERY: {
        "memory": ("What do you remember about {s}?", "Do you remember anything about {s}?"),
    },
    ToolCapability.GRAPH_REASONING: {
        "graph": ("What do you know about {s}?", "What is your knowledge about {s}?"),
    },
    ToolCapability.RESOURCE_MONITORING: {
        "resources": ("Show me the current resource usage", "What are the system resources right now?"),
    },
    ToolCapability.DEBUG_ENVIRONMENT: {
        "env": ("Show the environment configuration", "What environment variables are set?"),
    },
    ToolCapability.STRUCTURED_OUTPUT: {
        "structured": ("Summarize {s} as structured output", "Give me a structured summary of {s}"),
    },
    ToolCapability.API_SCHEMA_LEARNING: {
        "schema": ("What is the schema of the {s} API?", "Describe the API schema for {s}"),
    },
    ToolCapability.PARALLEL_BATCH: {
        "batch": ("Run the checks for {s} in parallel", "Batch the inspection of {s}"),
    },
    ToolCapability.TEST_EXECUTION: {
        "full": ("Run the full test suite", "Run all the tests"),
        "tests": ("Run the tests", "Run the project tests"),
        "relevant": ("Run the relevant tests", "Run the tests relevant to the current changes"),
    },
    ToolCapability.LINT: {
        "lint": ("Run the linter", "Lint the codebase"),
    },
    ToolCapability.TYPECHECK: {
        "typecheck": ("Run the type checker", "Typecheck the project"),
    },
    ToolCapability.BUILD: {
        "build": ("Build the project", "Compile the project"),
    },
}


_HOLDOUT_TEMPLATES: dict[ToolCapability, dict[str, tuple[str, ...]]] = {
    ToolCapability.DEFINITION_LOOKUP: {
        "need_defined": ("I need to understand where {s} is declared, could you show me?", "While reading the codebase I came across {s} — where does it get defined?"),
        "help_locate": ("Can you help me locate where the entity known as {s} has its definition?", "Could you help me find the definition of something called {s}?"),
        "curious_origin": ("I'm trying to figure out where this component comes from: {s}. Where is it defined?", "Where would I find the declaration of {s} in the source tree?"),
    },
    ToolCapability.REFERENCE_LOOKUP: {
        "trace_uses": ("Can you trace where {s} is referenced in the project?", "Could you trace where this symbol is referenced?"),
        "understand_invoke": ("I'm trying to understand what invokes {s} — where are its usages?", "While debugging I need to know everything that touches {s}. Where is it used?"),
        "dependents": ("Which parts of the codebase depend on {s}? Find its references.", "Could you tell me where {s} is used elsewhere in the code?"),
    },
    ToolCapability.SYMBOL_SEARCH: {
        "looking_for": ("I'm looking for something called {s} in this repository — can you find it?", "Could you locate the symbol named {s} for me?"),
        "heard_of": ("There's a component I've heard about called {s}; where does it live?", "Where in the codebase would I find a thing named {s}?"),
    },
    ToolCapability.SYMBOL_INSPECTION: {
        "keep_seeing": ("I keep seeing {s} in the code — what does it actually do?", "Can you explain what this thing, {s}, is responsible for?"),
        "someone_mentioned": ("Someone mentioned {s}; tell me about it.", "What can you tell me about this {s} I found while reading?"),
    },
    ToolCapability.CODE_SEARCH: {
        "every_place": ("I'd like to see every place that mentions {s} — can you search the source?", "Could you grep around for {s} and show me what comes up?"),
        "text_appears": ("Where all does the text {s} appear in this repository?", "I want to know everywhere {s} shows up in the code."),
    },
    ToolCapability.SEMANTIC_SEARCH: {
        "deals_with": ("I need the code that deals with {s} — can you find it for me?", "What implementation handles {s}? Look it up."),
        "logic_located": ("Where in the project is the logic for {s} located?", "Which part of the system is responsible for {s}?"),
    },
    ToolCapability.REPOSITORY_INSPECTION: {
        "put_together": ("I want to understand how {s} is put together in this repo.", "Could you give me an overview of {s} and where its code lives?"),
        "structure": ("What can you tell me about the structure of {s}?", "How is the code for {s} organized?"),
    },
    ToolCapability.REPOSITORY_INVESTIGATION: {
        "wrap_head": ("I'm trying to wrap my head around how {s} works — can you break it down?", "Could you walk me through what {s} does under the hood?"),
        "mechanism": ("I need to understand the mechanism behind {s}. Where is it implemented?", "How exactly does {s} get things done in this codebase?"),
        "explain_delegation": ("Could you explain how {s} hands off work to other parts of the system?", "What happens when {s} runs — where does the implementation live?"),
    },
    ToolCapability.FILE_SEARCH: {
        "any_files": ("Do we have any files named {s} around here?", "Can you check whether there's a file called {s} in the project?"),
        "looking_file": ("I'm looking for a file by the name {s} — can you find it?", "Is there a file matching {s} anywhere in the repository?"),
    },
    ToolCapability.FILE_READ: {
        "whats_inside": ("Could you show me what's inside {p}?", "I'd like to see the contents of {p}, please."),
        "pull_up": ("Can you pull up {p} for me?", "Mind showing me the file {p}?"),
    },
    ToolCapability.FILE_INSPECTION: {
        "tell_about": ("What can you tell me about {p}?", "Could you give me some details on the file {p}?"),
    },
    ToolCapability.DIRECTORY_LIST: {
        "what_lives": ("What files live in {p}?", "Can you show me what's sitting in {p}?"),
        "overview_dir": ("I need an overview of {p} — what's in there?", "What would I find if I looked inside {p}?"),
    },
    ToolCapability.GIT_OPERATION: {
        "repo_state": ("What's the current state of the repo?", "Has anything changed locally? Show me the status."),
        "recent_changes": ("What were the recent changes in this project?", "Could you tell me what's different in the working tree?"),
    },
    ToolCapability.TERMINAL_EXECUTION: {
        "where_am_i": ("Which directory am I in right now?", "Can you tell me the current working directory?"),
        "todays_date": ("What's today's date?", "Could you run {c} for me?"),
    },
    ToolCapability.MEMORY_QUERY: {
        "happen_know": ("Do you happen to know anything about {s}?", "Is there anything stored about {s}?"),
    },
    ToolCapability.GRAPH_REASONING: {
        "graph_knowledge": ("What do you know about {s} already?", "Is {s} part of what you've learned?"),
    },
    ToolCapability.RESOURCE_MONITORING: {
        "looking_now": ("How are system resources looking right now?", "Could you check the current CPU and memory usage?"),
    },
    ToolCapability.DEBUG_ENVIRONMENT: {
        "env_config": ("What's the current environment configuration?", "Which environment variables are defined?"),
    },
    ToolCapability.STRUCTURED_OUTPUT: {
        "structured_form": ("Can you give me {s} in a structured form?", "Could you summarize {s} as structured data?"),
    },
    ToolCapability.API_SCHEMA_LEARNING: {
        "api_surface": ("What does the {s} API surface look like?", "Could you describe the schema exposed by {s}?"),
    },
    ToolCapability.PARALLEL_BATCH: {
        "at_once": ("Could you check {s} and run the related inspections at once?", "Can you handle the {s} checks together in one batch?"),
    },
    ToolCapability.TEST_EXECUTION: {
        "run_and_tell": ("Could you run the project's tests and tell me the result?", "I'd like to see whether the test suite passes."),
    },
    ToolCapability.LINT: {
        "style_check": ("Could you check the code style across the project?", "Are there any lint issues right now?"),
    },
    ToolCapability.TYPECHECK: {
        "verify_types": ("Could you verify the types in this project?", "Is the code type-clean at the moment?"),
    },
    ToolCapability.BUILD: {
        "confirm_build": ("Could you build the project and confirm it compiles?", "Does the project build successfully right now?"),
    },
}

_STRATEGY_TABLES: dict[GenerationStrategy, dict[ToolCapability, dict[str, Callable[[Subject, str, str], str]]]] = {
    GenerationStrategy.DEVELOPMENT_DIRECT: _build_table(_DEV_TEMPLATES),
    GenerationStrategy.HOLDOUT_INDIRECT: _build_table(_HOLDOUT_TEMPLATES),
}

_STRATEGY_SPLIT: dict[GenerationStrategy, TestSplit] = {
    GenerationStrategy.DEVELOPMENT_DIRECT: TestSplit.DEVELOPMENT,
    GenerationStrategy.HOLDOUT_INDIRECT: TestSplit.HOLDOUT,
}

# Capabilities whose templates need a symbol subject.
_SYMBOL_CAPABILITIES = {
    ToolCapability.DEFINITION_LOOKUP,
    ToolCapability.REFERENCE_LOOKUP,
    ToolCapability.SYMBOL_SEARCH,
    ToolCapability.SYMBOL_INSPECTION,
    ToolCapability.CODE_SEARCH,
    ToolCapability.SEMANTIC_SEARCH,
    ToolCapability.REPOSITORY_INSPECTION,
    ToolCapability.REPOSITORY_INVESTIGATION,
    ToolCapability.MEMORY_QUERY,
    ToolCapability.GRAPH_REASONING,
    ToolCapability.STRUCTURED_OUTPUT,
    ToolCapability.API_SCHEMA_LEARNING,
    ToolCapability.PARALLEL_BATCH,
}

# Capabilities whose templates use the file's path, not a symbol variant.
_FILE_CAPABILITIES = {
    ToolCapability.FILE_READ,
    ToolCapability.FILE_INSPECTION,
}

# Capabilities whose subject is a file *stem* ("search for files named X").
_FILE_STEM_CAPABILITIES = {ToolCapability.FILE_SEARCH}

_DIRECTORY_CAPABILITIES = {ToolCapability.DIRECTORY_LIST}

# Capabilities whose templates need no subject at all.
_SUBJECT_FREE_CAPABILITIES = {
    ToolCapability.GIT_OPERATION,
    ToolCapability.TERMINAL_EXECUTION,
    ToolCapability.RESOURCE_MONITORING,
    ToolCapability.DEBUG_ENVIRONMENT,
    ToolCapability.TEST_EXECUTION,
    ToolCapability.LINT,
    ToolCapability.TYPECHECK,
    ToolCapability.BUILD,
}

_SUBJECT_KIND_FOR_CAPABILITY: dict[ToolCapability, str] = {
    ToolCapability.FILE_READ: "file",
    ToolCapability.FILE_INSPECTION: "file",
    ToolCapability.DIRECTORY_LIST: "directory",
    ToolCapability.FILE_SEARCH: "file",
}

_DIFFICULTY: dict[ToolCapability, Difficulty] = {
    ToolCapability.DEFINITION_LOOKUP: Difficulty.BASIC,
    ToolCapability.REFERENCE_LOOKUP: Difficulty.BASIC,
    ToolCapability.SYMBOL_SEARCH: Difficulty.BASIC,
    ToolCapability.FILE_READ: Difficulty.BASIC,
    ToolCapability.DIRECTORY_LIST: Difficulty.BASIC,
    ToolCapability.GIT_OPERATION: Difficulty.BASIC,
    ToolCapability.TERMINAL_EXECUTION: Difficulty.BASIC,
    ToolCapability.SYMBOL_INSPECTION: Difficulty.INTERMEDIATE,
    ToolCapability.CODE_SEARCH: Difficulty.INTERMEDIATE,
    ToolCapability.FILE_SEARCH: Difficulty.INTERMEDIATE,
    ToolCapability.SEMANTIC_SEARCH: Difficulty.INTERMEDIATE,
    ToolCapability.REPOSITORY_INSPECTION: Difficulty.INTERMEDIATE,
    ToolCapability.REPOSITORY_INVESTIGATION: Difficulty.ADVANCED,
}


# ---------------------------------------------------------------------------
# Independent task validity (Part 2 — no intent router involved).
# ---------------------------------------------------------------------------


def validate_task(case: CapabilityTestCase, root: str | Path | None = None) -> tuple[bool, str]:
    """Validates a generated task against repository/capability ground truth.

    NEVER calls the deterministic intent router.  Checks, in order:

      1. the capability has a canonical contract;
      2. success criteria are evaluable (contract defines them);
      3. the subject is present when the capability needs one, and its kind
         matches the capability's subject requirement;
      4. the subject is actually mentioned in the task text;
      5. the task expresses the expected capability's operation;
      6. the task does not ask for a conflicting operation;
      7. the subject's repository path exists (when known and root is given).

    Returns ``(True, "ok")`` or ``(False, reason)``.
    """
    contract = case.contract
    if contract is None:
        return False, f"no canonical contract for capability {case.expected_capability.value}"
    if not getattr(contract, "success_criteria", ()):
        return False, f"success criteria not evaluable for {case.expected_capability.value}"

    needs_subject = case.expected_capability in (
        _SYMBOL_CAPABILITIES | _FILE_CAPABILITIES | _DIRECTORY_CAPABILITIES | _FILE_STEM_CAPABILITIES
    )
    if needs_subject:
        if not case.subject:
            return False, f"capability {case.expected_capability.value} requires a subject but task has none"
        expected_kind = _SUBJECT_KIND_FOR_CAPABILITY.get(case.expected_capability)
        if (
            expected_kind
            and case.subject_kind
            and case.subject_kind != expected_kind
        ):
            return False, (
                f"subject kind {case.subject_kind!r} is not appropriate for "
                f"{case.expected_capability.value} (needs {expected_kind!r})"
            )
        if case.subject:
            variants = symbol_variants(case.subject)
            if not any(v.lower() in case.task.lower() for v in variants):
                return False, f"subject '{case.subject}' not mentioned in task '{case.task}'"

    if not _expresses_operation(case):
        return False, (
            f"task '{case.task}' does not express the operation of "
            f"{case.expected_capability.value}"
        )
    if _has_conflicting_operation(case):
        return False, f"task '{case.task}' asks for a conflicting operation"

    if root is not None and case.subject_path and not (Path(root) / case.subject_path).exists():
        return False, f"subject path '{case.subject_path}' does not exist in repository"
    return True, "ok"


def record_router_diagnostic(case: CapabilityTestCase) -> None:
    """Records the deterministic router's view as a DIAGNOSTIC (never a gate).

    Populates ``router_capability`` (resolved capability value, or the
    selection state when unresolved) and ``router_agreement`` (whether the
    router resolved to the expected capability).  The task is kept even when
    the router disagrees — that disagreement is evaluation data.
    """
    selection = select_for_request(case.task)
    if selection.state is SelectionState.RESOLVED:
        case.router_capability = selection.primary.value if selection.primary else selection.state.value
        case.router_agreement = selection.primary == case.expected_capability
    elif selection.state is SelectionState.AMBIGUOUS:
        case.router_capability = "ambiguous:" + ",".join(c.value for c in selection.ambiguity)
        case.router_agreement = case.expected_capability in selection.ambiguity
    else:
        case.router_capability = selection.state.value
        case.router_agreement = False


# ---------------------------------------------------------------------------
# The generator.
# ---------------------------------------------------------------------------


class TaskGenerator:
    """Builds varied, validated capability tasks from a subject pool.

    Deterministic for a given seed and strategy: same repository + seed +
    strategy -> same plan.  ``strategy`` selects the template family
    (development-direct vs holdout-indirect); entity selection is
    coverage-aware (prefer under-tested entities).
    """

    def __init__(
        self,
        subjects: list[Subject],
        *,
        seed: int = 7,
        strategy: GenerationStrategy = GenerationStrategy.DEVELOPMENT_DIRECT,
    ) -> None:
        self.subjects = subjects
        self.strategy = strategy
        self.rng = random.Random(seed)
        self._tables = _STRATEGY_TABLES[strategy]
        self._split = _STRATEGY_SPLIT[strategy]
        self._last_case_id = 0
        # Coverage bookkeeping (Part 6): prefer under-tested entities.
        self._entity_usage: Counter = Counter()
        self._capability_usage: Counter = Counter()

    def _next_id(self, capability: ToolCapability) -> str:
        self._last_case_id += 1
        return f"{self.strategy.value}.{capability.value}-{self._last_case_id}"

    def _pick_subject(self, capability: ToolCapability) -> Subject | None:
        """Coverage-aware subject selection for one capability."""
        kind = _SUBJECT_KIND_FOR_CAPABILITY.get(capability)
        if kind is not None:
            pool = [s for s in self.subjects if s.kind == kind]
        elif capability in _SYMBOL_CAPABILITIES:
            pool = [s for s in self.subjects if s.kind in ("class", "function", "enum")]
        else:
            return None
        if not pool:
            return None
        # Prefer entities used least so far (coverage, not PASS rate).
        pool.sort(key=lambda s: (self._entity_usage[s.name], s.name))
        return pool[0]

    def _render(self, capability: ToolCapability, subject: Subject | None) -> CapabilityTestCase | None:
        table = self._tables.get(capability)
        if not table:
            return None
        template_id = self.rng.choice(list(table.keys()))
        render = table[template_id]
        if capability in _SUBJECT_FREE_CAPABILITIES:
            if capability is ToolCapability.TERMINAL_EXECUTION and "run" in template_id:
                command = self.rng.choice(_SAFE_TERMINAL_COMMANDS)
                text = self._render_subject_free(render, command)
            else:
                text = self._render_subject_free(render, "")
            subject_ref = None
            variant = ""
            entity_id = None
            subject_kind = None
            subject_path = None
        else:
            if subject is None:
                return None
            if capability in _FILE_STEM_CAPABILITIES:
                stem = Path(subject.rel_path).stem
                variant = stem
                text = render(subject, stem)
                subject_ref = stem
                entity_id = stem
                subject_kind = "file"
                subject_path = subject.rel_path
            elif capability in _FILE_CAPABILITIES or capability in _DIRECTORY_CAPABILITIES:
                text = render(subject, subject.rel_path)
                variant = subject.rel_path
                subject_ref = subject.name
                entity_id = subject.rel_path
                subject_kind = subject.kind
                subject_path = subject.rel_path
            else:
                variants = symbol_variants(subject.name)
                if not variants:
                    return None
                variant = self.rng.choice(variants)
                text = render(subject, variant)
                subject_ref = subject.name
                entity_id = subject.name
                subject_kind = subject.kind
                subject_path = subject.rel_path

        contract = contract_for(capability)
        evidence: tuple[str, ...] = ()
        if contract is not None and contract.evidence_required:
            evidence = tuple(k.value for k in contract.evidence_required)
        case = CapabilityTestCase(
            case_id=self._next_id(capability),
            capability=capability,
            task=text,
            expected_capability=capability,
            subject=subject_ref,
            entity_id=entity_id,
            surface_form=variant,
            subject_kind=subject_kind,
            subject_path=subject_path,
            strategy=self.strategy,
            expected_behavior=(contract.purpose if contract is not None else None),
            evidence_requirements=evidence,
            difficulty=_DIFFICULTY.get(capability, Difficulty.BASIC),
            test_source=TestSource.GENERATED,
            split=self._split,
            template_id=template_id,
        )
        return case

    def _render_subject_free(self, render: Callable[[Subject, str, str], str], command: str) -> str:
        return render(Subject(name="", kind="", rel_path=""), "", command)

    def generate_for(
        self,
        capability: ToolCapability,
        *,
        per_capability: int = 3,
        allow_execution: bool = False,
    ) -> list[CapabilityTestCase]:
        """Generates and validates up to ``per_capability`` tasks for one capability.

        Tasks are accepted by the independent validity check only; the
        deterministic router is executed afterward as a diagnostic.
        """
        testability_of = testability(capability)
        if testability_of is Testability.UNSAFE or testability_of is Testability.NOT_TESTABLE:
            return []
        if testability_of is Testability.EXTERNAL:
            return []
        if testability_of is Testability.EXECUTION and not allow_execution:
            return []
        cases: list[CapabilityTestCase] = []
        attempts = 0
        while len(cases) < per_capability and attempts < per_capability * 10:
            attempts += 1
            subject = self._pick_subject(capability)
            case = self._render(capability, subject)
            if case is None:
                continue
            ok, _ = validate_task(case)
            if not ok:
                continue
            record_router_diagnostic(case)  # diagnostic only — never a gate
            cases.append(case)
            if case.entity_id:
                self._entity_usage[case.entity_id] += 1
            self._capability_usage[capability] += 1
        return cases

    def generate_plan(
        self,
        capabilities: list[ToolCapability] | None = None,
        *,
        per_capability: int = 3,
        allow_execution: bool = False,
        max_tasks: int | None = None,
    ) -> list[CapabilityTestCase]:
        """Generates a diverse validated plan across capabilities.

        Diversity control: tasks are assigned round-robin across the requested
        capabilities (one task per capability per round) so no capability
        monopolizes a budget-constrained plan; within a capability the
        coverage-aware subject picker spreads entities.  The seeded RNG varies
        wording, surface forms, and template families per task.
        """
        caps = capabilities or [
            c for c in ToolCapability if testability(c) is Testability.READ_ONLY
        ]
        pools: dict[ToolCapability, list[CapabilityTestCase]] = {}
        for cap in caps:
            pools[cap] = self.generate_for(cap, per_capability=per_capability, allow_execution=allow_execution)
        plan: list[CapabilityTestCase] = []
        cap_order = list(caps)
        round_index = 0
        while True:
            if max_tasks is not None and len(plan) >= max_tasks:
                break
            progressed = False
            for cap in cap_order:
                pool = pools[cap]
                if round_index < len(pool):
                    plan.append(pool[round_index])
                    progressed = True
                    if max_tasks is not None and len(plan) >= max_tasks:
                        break
            round_index += 1
            if not progressed:
                break
        return plan
