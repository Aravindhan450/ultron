"""ultron.core.nlp.intent
~~~~~~~~~~~~~~~~~~~~~~~~~

Structured intent model + deterministic routing for the natural-language →
tool layer.

:class:`UserIntent` is a machine-readable statement of *what the user wants*:
an intent category, extracted entities, a confidence level, the chosen tool,
and the exact arguments.  The LLM is not involved in producing these — the
detectors here are deterministic, so the same wording always routes the same
way and wrappers never leak into arguments.

:func:`route_request` is the entry point consulted by the agent loop for
requests the specialised detectors did not claim.  It covers:

- filesystem operations (list / delete / make-directory / rename / search)
- code intelligence (definition / references / symbol / code search /
  semantic search)
- project commands (tests / build / lint / typecheck / format / start /
  stop / backend / frontend) via :func:`discover_project_command`
- info → command mapping (\"what is the current directory\" -> pwd)

:func:`route_request` returns None for requests it does not confidently
own, so the existing detectors / LLM fallback keep their current priority.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ultron.core.nlp.normalize import normalize_terminal_command


class IntentCategory(str, Enum):
    """Broad, tool-ecosystem-aware intent categories."""

    INFORMATION_REQUEST = "information_request"
    REPOSITORY_INSPECTION = "repository_inspection"
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    FILE_CREATE = "file_create"
    FILE_DELETE = "file_delete"
    FILE_RENAME = "file_rename"
    DIRECTORY_LIST = "directory_list"
    DIRECTORY_CREATE = "directory_create"
    FILE_SEARCH = "file_search"
    CODE_SEARCH = "code_search"
    SYMBOL_SEARCH = "symbol_search"
    DEFINITION_LOOKUP = "definition_lookup"
    REFERENCE_LOOKUP = "reference_lookup"
    SYMBOL_INSPECTION = "symbol_inspection"
    SEMANTIC_SEARCH = "semantic_search"
    REPOSITORY_INVESTIGATION = "repository_investigation"
    TERMINAL_EXECUTION = "terminal_execution"
    TEST_EXECUTION = "test_execution"
    APPLICATION_START = "application_start"
    APPLICATION_STOP = "application_stop"
    BUILD = "build"
    INSTALL = "install"
    GIT_OPERATION = "git_operation"
    LINT = "lint"
    TYPECHECK = "typecheck"
    FORMAT = "format"
    CODING_REQUEST = "coding_request"
    MEMORY_QUERY = "memory_query"
    MEMORY_UPDATE = "memory_update"
    UNKNOWN = "unknown"


# Confidence levels.  HIGH = deterministic exact match; MEDIUM = pattern
# match with a valid fallback; LOW = ambiguous, ask the user.
HIGH = "high"
MEDIUM = "medium"
LOW = "low"


@dataclass
class UserIntent:
    """Structured representation of one natural-language request."""

    intent_type: IntentCategory
    objective: str
    entities: dict[str, str] = field(default_factory=dict)
    confidence: str = HIGH
    requires_tool: bool = True
    requires_confirmation: bool | None = None
    ambiguity: str | None = None
    tool: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PATH_TOKEN = r"[\w./~-]+"

_FILENAME_RE = re.compile(r"(?<!\w)([\w./-]+\.[a-zA-Z0-9]+)(?!\w)")

def _first_filename(text: str) -> str | None:
    """Returns the first filename-looking token (word chars, dots, slashes)."""
    m = _FILENAME_RE.search(text)
    return m.group(1) if m else None


def _after_keyword(text: str, keywords: tuple[str, ...]) -> str | None:
    """Returns the token following one of *keywords*, or None."""
    pattern = rf"\b(?:{'|'.join(keywords)})\s+({_PATH_TOKEN})"
    m = re.search(pattern, text, re.IGNORECASE)
    return m.group(1) if m else None


def _named_target(text: str, verbs: tuple[str, ...]) -> str | None:
    """Extracts the target after 'called X' / 'named X' following a verb."""
    pattern = (
        rf"\b(?:{'|'.join(verbs)})\s+(?:a\s+)?(?:new\s+)?"
        rf"(?:directory|folder|file)?\s*(?:called|named)\s+({_PATH_TOKEN})"
    )
    m = re.search(pattern, text, re.IGNORECASE)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Detectors (each returns UserIntent | None)
# ---------------------------------------------------------------------------

# Natural-language locations that mean "the workspace root".  These are
# resolved to the actual project root at dispatch time by the workspace
# context — never passed to the tool as the literal phrase.
_LOCATION_PHRASE_RE = re.compile(
    r"\b(?:current\s+)?(?:working\s+)?(?:directory|folder|dir)\b"
    r"|\bhere\b"
    r"|\bthis\s+(?:directory|folder|dir)\b"
    r"|\bproject\s+(?:directory|root)\b"
    r"|\bworkspace(?:\s+root)?\b",
    re.IGNORECASE,
)

# Words that can follow "in/of/at" but are NOT paths ("in the ...",
# "of my ...") — the directory-listing extractor must not return these.
_PATH_STOPWORDS = frozenset(
    {
        "the", "a", "an", "my", "your", "our", "their", "its", "his", "her",
        "this", "that", "these", "those", "some", "any", "all", "every",
        "current", "here", "same", "other", "each", "both", "either",
        "neither", "one", "two", "few", "many", "much", "more", "most",
    }
)


def detect_directory_list(text: str) -> UserIntent | None:
    """\"list the files here\" / \"show this directory\" / \"what's in src\".

    Location phrases (\"current directory\" / \"here\" / \"this folder\" /
    \"project root\") resolve to ``.`` (the workspace root marker), which the
    dispatch layer turns into the actual project root — never into the literal
    phrase.  An explicit path (\"what's in src\") is kept as a relative path.
    """
    if not re.search(
        r"\b(?:list|show|display)\s+(?:me\s+)?(?:the\s+)?(?:files?|contents|"
        r"directory|folders?|dirs?)\b"
        r"|\bwhat(?:'s|\s+is)\s+in\b"
        r"|\bshow\s+(?:me\s+)?this\s+directory\b",
        text,
        re.IGNORECASE,
    ):
        return None

    # Location phrase -> workspace root marker.  Checked first so "list the
    # files in the current directory" never yields path="the".
    if _LOCATION_PHRASE_RE.search(text):
        path = "."
    else:
        token = _after_keyword(text, ("in", "of", "at"))
        if token and token.lower().strip("./") not in _PATH_STOPWORDS:
            path = token
        else:
            path = "."
    return UserIntent(
        intent_type=IntentCategory.DIRECTORY_LIST,
        objective=f"List directory '{path}'",
        entities={"path": path},
        tool="list_directory",
        arguments={"path": path},
    )


def detect_file_delete(text: str) -> UserIntent | None:
    """\"delete test.txt\" / \"remove notes.md\"."""
    m = re.search(
        rf"\b(?:delete|remove|rm)\s+(?:the\s+)?(?:file\s+)?({_PATH_TOKEN})",
        text,
        re.IGNORECASE,
    )
    if not m:
        return None
    target = m.group(1)
    if not re.search(r"\.", target):
        return None  # "delete the file" without a name is ambiguous
    return UserIntent(
        intent_type=IntentCategory.FILE_DELETE,
        objective=f"Delete '{target}'",
        entities={"file_path": target},
        tool="delete_file",
        arguments={"file_path": target},
    )


def detect_make_directory(text: str) -> UserIntent | None:
    """\"make a directory called TodoList\" / \"create folder src/lib\"."""
    m = re.search(
        rf"\b(?:make|create|mkdir)\s+(?:a\s+|an\s+)?(?:new\s+)?"
        rf"(?:directory|folder|dir)\s*(?:called|named)?\s*(?:{_PATH_TOKEN})",
        text,
        re.IGNORECASE,
    )
    if not m:
        return None
    # Require an explicit name, not "make a directory" with no target.
    rest = text[m.end():].strip()
    if rest and not re.match(r"^[.!?]?\s*$", rest):
        return None
    target = m.group(0).split()[-1]
    if target.lower() in {"directory", "folder", "dir", "a", "an", "new", "the"}:
        return None
    return UserIntent(
        intent_type=IntentCategory.DIRECTORY_CREATE,
        objective=f"Create directory '{target}'",
        entities={"path": target},
        tool="run_command",
        arguments={"command": f"mkdir -p {target}"},
    )


def detect_file_rename(text: str) -> UserIntent | None:
    """\"rename foo.py to bar.py\" / \"move a.txt to b.txt\"."""
    m = re.search(
        rf"\b(?:rename|move)\s+({_PATH_TOKEN})\s+to\s+({_PATH_TOKEN})",
        text,
        re.IGNORECASE,
    )
    if not m:
        return None
    return UserIntent(
        intent_type=IntentCategory.FILE_RENAME,
        objective=f"Rename '{m.group(1)}' to '{m.group(2)}'",
        entities={"file_path": m.group(1), "new_path": m.group(2)},
        tool="rename_file",
        arguments={"file_path": m.group(1), "new_path": m.group(2)},
    )


def detect_file_search(text: str) -> UserIntent | None:
    """\"search for files named config\" / \"find files matching auth\"."""
    m = re.search(
        rf"\b(?:search|find)\s+(?:for\s+)?(?:the\s+)?files?\s+"
        rf"(?:named|called|matching)\s+({_PATH_TOKEN})",
        text,
        re.IGNORECASE,
    )
    if not m:
        return None
    return UserIntent(
        intent_type=IntentCategory.FILE_SEARCH,
        objective=f"Search files for '{m.group(1)}'",
        entities={"query": m.group(1)},
        tool="search_files",
        arguments={"query": m.group(1), "path": "."},
    )


# A symbol phrase: one or more identifier-ish words, tolerant of the
# optional article ("the supervisor") and multi-word names ("coding
# executor", "task state", "orchestration validator").  Parentheses are
# allowed for call-like references ("authenticate()"); the first class
# includes them so a trailing "()" stays part of the symbol.
_SYMBOL_PHRASE = r"[\w.()]+(?:\s+[\w.()]+)*?"
# Optional leading article consumed BEFORE the symbol capture.
_ARTICLE = r"(?:the\s+|a\s+|an\s+)?"


def detect_definition_lookup(text: str) -> UserIntent | None:
    """\"where is TaskState defined\" / \"find where the supervisor is
    defined\" / \"where is coding executor declared\" / \"definition of
    authenticate\".

    Article-tolerant and multi-word so these never fall through to the LLM
    (which previously guessed file paths from naming conventions).
    """
    m = re.search(
        r"\bwhere\s+(?:is|are)\s+(?:the\s+|a\s+|an\s+)?(" + _SYMBOL_PHRASE + r")\s+"
        r"(?:is\s+|are\s+)?(?:defined|declared)\b"
        r"|\b(?:find|show)\s+where\s+(?:the\s+|a\s+|an\s+)?(" + _SYMBOL_PHRASE + r")\s+"
        r"(?:is\s+|are\s+)?(?:defined|declared)\b"
        r"|\bwhere\s+(?:the\s+|a\s+|an\s+)?(" + _SYMBOL_PHRASE + r")\s+"
        r"(?:is|are)\s+(?:defined|declared)\b"
        r"|\b(?:the\s+)?definition\s+of\s+(" + _SYMBOL_PHRASE + r")\b",
        text,
        re.IGNORECASE,
    )
    if not m:
        return None
    name = next((g for g in m.groups() if g), None)
    if not name:
        return None
    return UserIntent(
        intent_type=IntentCategory.DEFINITION_LOOKUP,
        objective=f"Find definition of '{name}'",
        entities={"name": name},
        tool="find_definition",
        arguments={"name": name},
    )


def detect_reference_lookup(text: str) -> UserIntent | None:
    """\"references to TaskState\" / \"usages of X\" / \"what calls X\" /
    \"who uses X\" / \"where is X used\" / \"find where X is used\" /
    \"where is X referenced/called\".

    The extracted entity is ONLY the symbol — the grammatical wrapper
    (\"where **is** X used\") never leaks into the capture.  The \"is/are\"
    after \"where\" is a required/greedy group so the symbol phrase cannot
    swallow it (the historical \"is TaskState\" bug).
    """
    m = re.search(
        r"\b(?:find\s+)?(?:all\s+)?references?\s+to\s+" + _ARTICLE + "(" + _SYMBOL_PHRASE + r")\b"
        r"|\b(?:find\s+)?(?:all\s+)?usages?\s+of\s+" + _ARTICLE + "(" + _SYMBOL_PHRASE + r")\b"
        r"|\bwhere\s+(?:is|are)\s+" + _ARTICLE + "(" + _SYMBOL_PHRASE + r")\s+(?:is|are)?\s*(?:used|referenced|called)\b"
        r"|\bfind\s+where\s+" + _ARTICLE + "(" + _SYMBOL_PHRASE + r")\s+(?:is|are)?\s*(?:used|referenced|called)\b"
        r"|\b(?:what|who)\s+(?:calls|uses|references?)\s+" + _ARTICLE + "(" + _SYMBOL_PHRASE + r")\b"
        r"|\bcallers?\s+of\s+" + _ARTICLE + "(" + _SYMBOL_PHRASE + r")\b",
        text,
        re.IGNORECASE,
    )
    if not m:
        return None
    name = next((g for g in m.groups() if g), None)
    if not name:
        return None
    return UserIntent(
        intent_type=IntentCategory.REFERENCE_LOOKUP,
        objective=f"Find references to '{name}'",
        entities={"name": name},
        tool="find_references",
        arguments={"name": name},
    )


def detect_symbol_inspection(text: str) -> UserIntent | None:
    """\"what does CodingExecutor do?\" — locate the symbol (definition +
    references) rather than answering from model memory."""
    m = re.search(
        r"\bwhat\s+does\s+" + _ARTICLE + "(" + _SYMBOL_PHRASE + r")\s+do\??\s*$",
        text,
        re.IGNORECASE,
    )
    if not m:
        return None
    name = m.group(1).strip()
    return UserIntent(
        intent_type=IntentCategory.SYMBOL_INSPECTION,
        objective=f"Inspect symbol '{name}'",
        entities={"name": name},
        tool="report_symbol",
        arguments={"name": name},
    )


def detect_symbol_lookup(text: str) -> UserIntent | None:
    """\"find the symbol UserService\" / \"show symbols in auth\"."""
    m = re.search(
        r"\b(?:find|show|look\s+up)\s+(?:the\s+)?(?:symbol|class|function)\s+"
        r"([\w.]+)\b",
        text,
        re.IGNORECASE,
    )
    if not m:
        return None
    return UserIntent(
        intent_type=IntentCategory.SYMBOL_SEARCH,
        objective=f"Find symbol '{m.group(1)}'",
        entities={"name": m.group(1)},
        tool="find_symbol",
        arguments={"name": m.group(1)},
    )


def detect_code_search(text: str) -> UserIntent | None:
    """\"search the code for X\" / \"search the codebase for X\" — raw
    lexical search.

    \"where X is implemented/handled\" is NOT raw search: it is a
    repository investigation (definition + synthesis) and is routed by
    :func:`detect_repository_investigation`'s sibling branch below.
    """
    m = re.search(
        r"\b(?:search|grep)\s+(?:the\s+)?(?:code|codebase|repository)\s+"
        r"(?:for\s+)?(.+?)\s*[.!?]?\s*$",
        text,
        re.IGNORECASE,
    )
    if m:
        query = m.group(1).strip()
        if query:
            return UserIntent(
                intent_type=IntentCategory.CODE_SEARCH,
                objective=f"Code search for '{query}'",
                entities={"query": query},
                tool="code_search",
                arguments={"query": query},
            )

    # "where (is) X implemented" / "find where X is implemented" /
    # "where is X handled" — semantic-looking lookup with a multi-word
    # subject ("command execution"), routed to the investigation tool so
    # the answer synthesizes a primary implementation instead of dumping
    # raw lexical matches.  Leading articles are stripped ("where is the
    # OrchestrationValidator implemented" -> "OrchestrationValidator").
    where = re.search(
        r"\b(?:find\s+|show\s+me\s+)?where\s+(?:is\s+|are\s+)?"
        r"(?:the\s+|a\s+|an\s+)?(.+?)\s+(?:is\s+|are\s+)?(?:implemented|handled)\b",
        text,
        re.IGNORECASE,
    )
    if where:
        subject = where.group(1).strip()
        subject = re.sub(r"^(?:the|a|an)\s+", "", subject, flags=re.IGNORECASE)
        if subject:
            return UserIntent(
                intent_type=IntentCategory.REPOSITORY_INVESTIGATION,
                objective=f"Find where '{subject}' is implemented",
                entities={"query": subject},
                tool="code_investigation",
                arguments={"query": subject},
            )
    return None


def detect_semantic_search(text: str) -> UserIntent | None:
    """\"search semantically for X\" / \"find code responsible for X\" /
    \"find code that does X\" / \"where does the supervisor delegate\"."""
    m = re.search(
        r"\bsemantic(?:ally)?\s+(?:search|find)\s+(?:for\s+)?(.+?)\s*[.!?]?\s*$"
        r"|\b(?:search|find)\s+semantic(?:ally)?\s+(?:for\s+)?(.+?)\s*[.!?]?\s*$"
        r"|\bfind\s+(?:the\s+)?code\s+(?:that\s+)?(?:is\s+)?(?:responsible\s+for\s+)?"
        r"(.+?)\s*[.!?]?\s*$"
        r"|\bwhere\s+does\s+(?:the\s+)?(.+?)\s+"
        r"(?:work|delegate|happen|live|execute)\b\s*[.!?]?\s*$",
        text,
        re.IGNORECASE,
    )
    if not m:
        return None
    query = next((g for g in m.groups() if g), None)
    if not query:
        return None
    query = query.strip()
    query = re.sub(r"^(?:the|a|an)\s+", "", query, flags=re.IGNORECASE)
    return UserIntent(
        intent_type=IntentCategory.SEMANTIC_SEARCH,
        objective=f"Semantic search for '{query}'",
        entities={"query": query},
        tool="semantic_search",
        arguments={"query": query},
    )


def detect_repository_investigation(text: str) -> UserIntent | None:
    """\"how does X work\" / \"how does the Supervisor delegate\" /
    \"how is command execution implemented\" / \"explain how X works\" /
    \"how does X interact with Y\" / \"why does the workflow validator reject\".

    These are repository/code-understanding questions: they must route to
    code intelligence (definition + relationships + synthesis) — never to
    web search.  The subject is passed to ``code_investigation`` which
    resolves a verified definition when one exists and otherwise falls back
    to ranked semantic evidence.
    """
    m = re.search(
        r"\bhow\s+does\s+(?:the\s+)?(.+?)\s+(?:work|delegate|execute|operate|behave)\b"
        r"|\bhow\s+does\s+(?:the\s+)?(.+?)\s+interact\s+with\s+[^?]+\??\s*$"
        r"|\bhow\s+(?:is|are)\s+(?:the\s+)?(.+?)\s+(?:implemented|structured)\b"
        r"|\bexplain\s+how\s+(?:the\s+)?(.+?)\s+(?:works|is\s+implemented|is\s+structured)\b"
        r"|\bwhy\s+does\s+(?:the\s+)?(.+?)\s+(?:reject|fail|error)\b",
        text,
        re.IGNORECASE,
    )
    if not m:
        return None
    subject = next((g for g in m.groups() if g), None)
    if not subject:
        return None
    subject = subject.strip()
    subject = re.sub(r"^(?:the|a|an)\s+", "", subject, flags=re.IGNORECASE)
    if not subject:
        return None
    return UserIntent(
        intent_type=IntentCategory.REPOSITORY_INVESTIGATION,
        objective=f"Investigate how '{subject}' works in this repository",
        entities={"query": subject},
        tool="code_investigation",
        arguments={"query": subject},
    )


# ---------------------------------------------------------------------------
# Project-command requests ("run the linter", "build the project")
# ---------------------------------------------------------------------------

# what -> discover_project_command() key; the actual command is resolved at
# dispatch time against the repository so nothing is invented.
_PROJECT_COMMAND_REQUESTS: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(r"\b(?:run\s+)?(?:the\s+)?(?:test\s+suite|all\s+tests?)\b", re.IGNORECASE), "test"),
    (re.compile(r"\b(?:run\s+|please\s+run\s+)?(?:the\s+)?linter\b", re.IGNORECASE), "lint"),
    (re.compile(r"\b(?:run\s+)?(?:the\s+)?type[- ]?check(?:er)?\b", re.IGNORECASE), "typecheck"),
    (re.compile(r"\bformat\s+(?:the\s+)?code\b", re.IGNORECASE), "format"),
    (re.compile(r"\b(?:build|compile)\s+(?:the\s+)?project\b", re.IGNORECASE), "build"),
    (re.compile(r"\b(?:start|launch|open|run)\s+(?:the\s+)?"
                r"(?:development\s+server|dev\s+server|backend|frontend|app(?:lication)?)\b", re.IGNORECASE), "start"),
    (re.compile(r"\b(?:stop|kill|shut\s+down)\s+(?:the\s+)?"
                r"(?:development\s+server|dev\s+server|backend|frontend|server|app(?:lication)?)\b", re.IGNORECASE), "stop"),
)


# Map discover key -> intent category.
_PROJECT_CATEGORY: dict[str, IntentCategory] = {
    "test": IntentCategory.TEST_EXECUTION,
    "lint": IntentCategory.LINT,
    "typecheck": IntentCategory.TYPECHECK,
    "format": IntentCategory.FORMAT,
    "build": IntentCategory.BUILD,
    "start": IntentCategory.APPLICATION_START,
    "stop": IntentCategory.APPLICATION_STOP,
}


def detect_project_command_request(text: str) -> UserIntent | None:
    """\"run the linter\" / \"build the project\" / \"start the backend\"."""
    for pattern, what in _PROJECT_COMMAND_REQUESTS:
        if pattern.search(text):
            return UserIntent(
                intent_type=_PROJECT_CATEGORY[what],
                objective=f"{what} the project",
                entities={"what": what},
                tool="run_command",
                arguments={"what": what},
            )
    return None


# ---------------------------------------------------------------------------
# Info → command mapping
# ---------------------------------------------------------------------------

_INFO_TO_COMMAND: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(r"\bwhat\s+is\s+the\s+current\s+(?:working\s+)?directory\b",
                re.IGNORECASE), "pwd"),
    (re.compile(r"\bwhere\s+am\s+i\b", re.IGNORECASE), "pwd"),
    (re.compile(r"\bshow\s+me\s+the\s+current\s+(?:working\s+)?directory\b",
                re.IGNORECASE), "pwd"),
    (re.compile(r"\bwhat\s+time\s+is\s+it\b", re.IGNORECASE), "date"),
    (re.compile(r"\bwho\s+am\s+i\b", re.IGNORECASE), "whoami"),
)


def detect_info_to_command(text: str) -> UserIntent | None:
    """Maps simple informational questions to a deterministic command."""
    for pattern, command in _INFO_TO_COMMAND:
        if pattern.search(text):
            return UserIntent(
                intent_type=IntentCategory.TERMINAL_EXECUTION,
                objective=f"Run '{command}' to answer the question",
                entities={"command": command},
                tool="run_command",
                arguments={"command": command},
            )
    return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def route_request(text: str, cwd: str | None = None) -> UserIntent | None:
    """
    Deterministically routes *text* to a tool-backed intent, or returns None
    when no deterministic detector owns it.

    Runs the specific detectors in priority order.  Each detector returns a
    fully-resolved :class:`UserIntent` (category, tool, arguments) or None.
    """
    if not text or not text.strip():
        return None
    raw = text.strip()

    # Specific detectors first: they are more precise than the terminal
    # catch-all, and their keywords must not be swallowed by it ("find" and
    # "make" are shell commands, but "find references to X" and "make a
    # directory called X" are code-intelligence / filesystem requests).
    detectors = (
        detect_info_to_command,
        detect_project_command_request,
        detect_directory_list,
        detect_file_delete,
        detect_file_rename,
        detect_make_directory,
        detect_file_search,
        detect_definition_lookup,
        detect_reference_lookup,
        detect_symbol_lookup,
        detect_symbol_inspection,
        detect_code_search,
        detect_semantic_search,
        detect_repository_investigation,
    )
    for detector in detectors:
        intent = detector(raw)
        if intent is not None:
            return intent

    # Terminal execution last: wrapper stripping, quote-safe.  Requests that
    # look like an explicit command ("pwd", "Execute: pwd", "Run git status")
    # land here; prose like "run the tests" returns None and is left to the
    # specialised test/lint detectors in the agent.
    command = normalize_terminal_command(raw)
    if command:
        return UserIntent(
            intent_type=IntentCategory.TERMINAL_EXECUTION,
            objective=f"Execute '{command}'",
            entities={"command": command},
            tool="run_command",
            arguments={"command": command},
        )
    return None
