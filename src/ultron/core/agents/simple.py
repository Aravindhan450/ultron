import asyncio
import json
import re

import httpx

from ultron.core.agents.base import BaseAgent
from ultron.core.agents.security import (
    blocked_message,
    check_action,
    is_allow,
    is_confirm,
    is_denied,
    security_mode,
)
from ultron.core.intelligence.prompt_assembly import (
    build_response_guidance,
    polish_response,
)
from ultron.core.logging import get_logger
from ultron.core.types import ChatMessage, PendingAction, Role, history_to_openai_format
from ultron.ui.theme import ACCENT

logger = get_logger("ultron.agents.simple")


# ---------------------------------------------------------------------------
# Shared execution + security-mapping helpers
# ---------------------------------------------------------------------------

def execute_tool(tool_name: str, **kwargs) -> str:
    """
    Executes a registered tool directly and returns its output as a string.

    Used by paths the security boundary classifies as auto-allowed (LOW risk
    or a permissive mode). Returns an error string if the tool is missing or
    raises, so the failure surfaces as a message instead of a crash.
    """
    from ultron.core.tools.registry import get_tool
    func = get_tool(tool_name)
    if not func:
        return f"Error: Tool '{tool_name}' not found in registry."
    try:
        return str(func(**kwargs))
    except Exception as exc:  # noqa: BLE001 — tools are arbitrary; a tool bug
        # must surface as a message, never crash the agent loop.
        return f"Error executing tool '{tool_name}': {exc}"


def _generic_target_content(tool_name: str, arguments: dict) -> tuple[str, str | None]:
    """
    Maps a tool call's arguments to the (target, content) pair the security
    boundary expects for classification and guardrail scanning.
    """
    if tool_name == "read_file":
        return str(arguments.get("file_path", "")), None
    if tool_name == "web_search":
        return str(arguments.get("query", "")), None
    if tool_name == "fetch_page_text":
        return str(arguments.get("url", "")), None
    if tool_name == "search_memories":
        return str(arguments.get("keyword", "")), None
    if tool_name == "add_memory":
        return "", str(arguments.get("fact", ""))
    if tool_name == "add_triple":
        return "", " ".join(
            str(arguments.get(key, "")) for key in ("subject", "predicate", "object")
        )
    if tool_name == "query_triples":
        target = next(
            (str(arguments.get(key, "")) for key in ("subject", "predicate", "object") if arguments.get(key)),
            "",
        )
        return target, None
    if tool_name == "search_triples":
        return str(arguments.get("keyword", "")), None
    if tool_name == "query_chain":
        return str(arguments.get("anchor", "")), None
    if tool_name == "run_parallel":
        # The batch is encoded newline-joined so the boundary can classify
        # and scan each command in the batch individually.
        return "\n".join(str(c) for c in arguments.get("commands", [])), None
    if tool_name == "retrieve":
        return str(arguments.get("url", "") or arguments.get("request", "")), None
    if tool_name == "check_connectivity":
        return str(arguments.get("url", "")), None
    if tool_name == "learn_api_schema":
        return str(arguments.get("base_url", "")), None
    if tool_name == "get_api_knowledge":
        return str(arguments.get("base_url", "")), None
    if tool_name == "forget_api":
        return str(arguments.get("base_url", "")), None
    if tool_name == "api_usage_hint":
        return str(arguments.get("url", "")), str(arguments.get("body"))
    if tool_name == "resource_forecast":
        return str(arguments.get("command", "")), None
    if tool_name == "check_resources":
        return "", None
    if tool_name == "memory_connections":
        return str(arguments.get("topic", "")), None
    if tool_name == "related_facts":
        return str(arguments.get("fact", "")), None
    if tool_name == "discover_connections":
        return "", None
    if tool_name == "explain_relation":
        return "", f"{arguments.get('a', '')} {arguments.get('b', '')}"
    if tool_name in {"enforce_schema", "schema_validate"}:
        return str(arguments.get("format", "")), str(arguments.get("text", ""))
    if tool_name == "list_schemas":
        return "", None
    if tool_name in {"preflight_plan", "analyze_dependencies"}:
        return "", str(arguments.get("steps_json", ""))
    if tool_name == "list_plan_actions":
        return "", None
    if tool_name == "get_debug_context":
        return "", None
    if tool_name == "diagnose_failure":
        return str(arguments.get("command", "")), str(arguments.get("text", ""))
    if tool_name == "check_dependency":
        return str(arguments.get("name", "")), None
    if tool_name == "run_tool_batch":
        # The batch payload is the content: the guardrails scan it for
        # secrets/URLs while the per-call gating inside run_tool_batch
        # classifies each member individually.
        return "", str(arguments.get("calls_json", ""))
    if tool_name == "synthesize_analysis":
        return "", str(arguments.get("results_json", ""))
    return "", None


def _step_target_content(step: dict) -> tuple[str, str | None]:
    """
    Maps a multi-step plan step to the (target, content) pair the boundary
    uses to gate the step before execution.
    """
    action = step.get("action", "")
    if action == "read_file":
        return str(step.get("filename", "")), None
    if action == "write_file":
        return str(step.get("filename", "")), str(step.get("content", ""))
    if action == "run_command":
        return str(step.get("command", "")), None
    if action == "make_http_request":
        return str(step.get("url", "")), str(step.get("body") or step.get("method", ""))
    if action == "run_query":
        return str(step.get("sql", "")), None
    if action == "add_memory":
        return "", str(step.get("fact", ""))
    return "", None


def _gate_command(command: str) -> ChatMessage | None:
    """
    Gates a shell command through the security boundary.

    Returns a blocked or auto-executed ChatMessage when the boundary denies or
    allows the command directly; returns None when the command needs
    interactive confirmation (the caller should emit a PendingAction).
    """
    verdict = check_action("run_command", command)
    if is_denied(verdict):
        return ChatMessage(role=Role.ASSISTANT, content=blocked_message(verdict))
    if is_allow(verdict):
        from ultron.core.tools.resource_monitor import (
            forecast_severity,
            forecast_warning,
        )

        severity = forecast_severity(command)
        warning = forecast_warning(command)
        if severity in {"heavy", "critical"}:
            # Resource escalation: a forecast-heavy command that would
            # otherwise auto-run is offered for confirmation with the
            # resource warning shown up front — except in permissive mode,
            # which promises no prompts: there it runs with the warning
            # attached to the reply instead.
            if security_mode() == "permissive":
                result = execute_tool("run_command", command=command)
                if warning:
                    result = f"{result}\n\n[resources] ⚠ {warning}"
                return ChatMessage(role=Role.ASSISTANT, content=result)
            return ChatMessage(
                role=Role.ASSISTANT,
                content=(
                    f"Command execution requested: '{command}'\n\n"
                    f"[resources] ⚠ {warning}"
                ),
                pending_action=PendingAction(
                    action_type="run_command", target=command
                ),
            )
        result = execute_tool("run_command", command=command)
        if severity == "moderate" and warning:
            result = f"{result}\n\n[resources] ⚠ note: {warning}"
        return ChatMessage(role=Role.ASSISTANT, content=result)
    return None

# ---------------------------------------------------------------------------
# Detectors
# Each function takes raw user_input and returns a matched value (filename,
# command string, fact, topic, bool) or None/False if no match.
# ---------------------------------------------------------------------------

def extract_any_filename(text: str) -> str | None:
    """
    Extracts ANY token matching a filename pattern (word chars, hyphens, slashes, ending with an extension)
    from text without requiring specific verb patterns.

    Safe to use AFTER intent classification has already confirmed a file action category.
    """
    filename_pattern = r'[\w./-]+\.[a-zA-Z0-9]+'
    match = re.search(rf'\b({filename_pattern})\b', text, re.IGNORECASE)
    if match:
        return match.group(1)
    return None

_IMAGE_EXT_RE = re.compile(r"\.(?:png|jpe?g|gif|webp|bmp|tiff?|heic)$", re.IGNORECASE)


def detect_image_intent(user_input: str) -> str | None:
    """
    Detects requests to analyze an uploaded image (e.g. "analyze this chart.png",
    "look at the graph in plot.png", "view the file data.bmp",
    "what's in screenshot.png").

    Returns the first image-extension filename when a vision verb is present,
    otherwise None. Ordinary file reads ("read config.json") keep their
    existing path because their extension is not an image one. Runs before the
    file-read detector so image files route to the vision model rather than
    the raw-file reader.
    """
    text = user_input.strip()
    has_vision_verb = re.search(
        r"\b(?:analyze|analyse|examine|inspect|view|read)\b"
        r"|\blook\s+at\b"
        r"|\bwhat(?:'s|\s+is)\s+in\b",
        text,
        re.IGNORECASE,
    )
    if not has_vision_verb:
        return None
    for match in re.finditer(r"[\w./-]+\.[a-zA-Z0-9]+", text):
        token = match.group(0)
        # A URL-ish token ("//example.com/chart.png") is not a local image path.
        if "://" in token or token.startswith("//"):
            continue
        if _IMAGE_EXT_RE.search(token):
            return token
    return None


def detect_greeting_intent(user_input: str) -> bool:
    """
    Detects if the user input is a simple conversational greeting (e.g. "hi", "hello", "hey", "hii", "greetings").
    Returns True if matched, otherwise False.
    """
    pattern = r'^\s*(?:hi+|hello+|hey+|greetings|good\s+(?:morning|afternoon|evening)|sup|yo)\s*[!.]*\s*$'
    return bool(re.search(pattern, user_input, re.IGNORECASE))

def detect_file_read_intent(user_input: str) -> str | None:
    """
    Detects if the user input contains a common file-reading pattern using regex.
    Returns the extracted filename if a match is found, otherwise None.

    This helper exists because relying on small local AI models to consistently emit
    tool calls for common phrasings can be unreliable. Deterministically capturing
    file-reading intents in code ensures fast and consistent performance.
    """
    # Regex to capture a filename (word characters, hyphens, slashes, ending with an extension like .txt, .py, .md)
    filename_pattern = r'[\w./-]+\.[a-zA-Z0-9]+'

    # Patterns for common file reading phrasings
    patterns = [
        rf'\bread\s+({filename_pattern})\b',
        rf'\bopen\s+({filename_pattern})\b',
        rf'\bshow\s+(?:me\s+)?({filename_pattern})\b',
        rf"\bwhat(?:'s|\s+is)\s+in\s+({filename_pattern})\b",
        rf'\bcontents\s+of\s+({filename_pattern})\b',
        rf'\bcat\s+({filename_pattern})\b',
    ]

    for pattern in patterns:
        match = re.search(pattern, user_input, re.IGNORECASE)
        if match:
            return match.group(1)

    return None

def detect_file_write_intent(user_input: str) -> tuple[str, str] | None:
    """
    Detects if the user input contains a common file-writing pattern using regex.
    Returns a tuple of (filename, content) if matched, otherwise None.

    Supported phrasings:
      - "write <content> to <filename>"
      - "save <content> to <filename>"
      - "create [a] file <filename> with <content>"
      - "create [a] [new] file named/called <filename> [and put/with <content> [inside/in it]]"
      - "make [a] file named/called <filename> [and put/with <content>]"
    """
    filename_pattern = r'[\w./-]+\.[a-zA-Z0-9]+'

    # Helper: strip wrapping quotes from a string
    def _strip_quotes(s: str) -> str:
        s = s.strip()
        if len(s) >= 2 and s[0] in ('"', "'") and s[-1] == s[0]:
            return s[1:-1]
        return s

    # --- Pattern group 1: verb + content + "to" + filename ---
    to_patterns = [
        rf'^\s*write\s+(?P<content>.+?)\s+to\s+(?P<filename>{filename_pattern})\s*$',
        rf'^\s*save\s+(?P<content>.+?)\s+to\s+(?P<filename>{filename_pattern})\s*$',
    ]
    for pattern in to_patterns:
        match = re.search(pattern, user_input, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group("filename").strip(), _strip_quotes(match.group("content"))

    # --- Pattern group 2: "create [a] file <filename> with <content>" ---
    m = re.search(
        rf'^\s*create\s+(?:a\s+)?file\s+(?P<filename>{filename_pattern})\s+with\s+(?P<content>.+?)\s*$',
        user_input, re.IGNORECASE | re.DOTALL
    )
    if m:
        return m.group("filename").strip(), _strip_quotes(m.group("content"))

    # --- Pattern group 3: "create/make [a/an] [new] file named/called <filename>" ---
    # Filename must end with a recognised extension.
    m = re.search(
        rf'(?:create|make)\s+(?:a\s+|an\s+)?(?:new\s+)?file\s+(?:named|called)\s+(?P<filename>{filename_pattern})',
        user_input, re.IGNORECASE
    )
    if m:
        filename = m.group("filename").strip()
        # Try to extract quoted content anywhere after the filename
        content_match = re.search(r'["\'](.+?)["\']', user_input[m.end():])
        content = content_match.group(1).strip() if content_match else ""
        return filename, content

    # --- Pattern group 4: "make/create <filename>" with optional content ---
    # Fallback: bare filename anywhere preceded by "create" or "make"
    m = re.search(
        rf'(?:create|make)\s+(?P<filename>{filename_pattern})',
        user_input, re.IGNORECASE
    )
    if m:
        filename = m.group("filename").strip()
        content_match = re.search(r'["\'](.+?)["\']', user_input[m.end():])
        content = content_match.group(1).strip() if content_match else ""
        return filename, content

    return None

def detect_command_intent(user_input: str) -> str | None:
    """
    Detects if the user input requests running a command (e.g., "run <command>" or "execute <command>").
    Returns the extracted command string if matched, otherwise None.
    """
    pattern = r'^\s*(?:run|execute)\s+(?P<command>.+)\s*$'
    match = re.search(pattern, user_input, re.IGNORECASE)
    if match:
        return match.group("command").strip()
    return None

def _split_commands(text: str) -> list[str]:
    """
    Splits a parallel-request command portion into individual commands.

    Separators are commas, the word "and", and newlines — but only outside
    single/double quotes, so `echo "fish and chips"` stays one command.
    Surrounding quotes are stripped per segment and blank segments dropped.
    """
    commands: list[str] = []
    current: list[str] = []
    quote: str | None = None
    i = 0
    n = len(text)

    def unwrap(segment: str) -> str:
        """Strips quotes only when they wrap the whole segment — quotes that
        are part of the command (e.g. `echo "fish and chips"`) are kept."""
        s = segment.strip()
        if len(s) >= 2 and s[0] in "\"'" and s[-1] == s[0]:
            return s[1:-1]
        return s

    while i < n:
        ch = text[i]
        if quote:
            current.append(ch)
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in "\"'":
            quote = ch
            current.append(ch)
            i += 1
            continue
        if ch in ",\n":
            segment = unwrap("".join(current))
            if segment:
                commands.append(segment)
            current = []
            i += 1
            continue
        if ch == " " and text[i : i + 5].lower() == " and ":
            segment = unwrap("".join(current))
            if segment:
                commands.append(segment)
            current = []
            i += 5
            continue
        current.append(ch)
        i += 1
    segment = unwrap("".join(current))
    if segment:
        commands.append(segment)
    return commands


def detect_parallel_intent(user_input: str) -> list[str] | None:
    """
    Detects requests to run multiple commands at once (e.g. "run X and Y in
    parallel", "execute a, b, c simultaneously").

    Returns the list of extracted commands if an explicit parallelism marker
    is present, otherwise None. The marker is required — a bare "run ls"
    stays sequential so nothing is silently parallelized.

    Supported markers: in parallel, simultaneously, concurrently,
    at the same time. Both trailing ("run X and Y in parallel") and leading
    ("in parallel, run X and Y") forms are matched.
    """
    text = user_input.strip()
    marker = (
        r"\b(?:in\s+parallel|simultaneously|concurrently|at\s+the\s+same\s+time)\b"
    )

    # Trailing form: "run <cmds> in parallel" / "execute <cmds> simultaneously"
    match = re.search(
        rf"^\s*(?:please\s+)?(?:run|execute)\s+(?P<cmds>.+?)\s+{marker}\s*(?:please)?\s*\.?\s*$",
        text,
        re.IGNORECASE,
    )
    if match:
        return _split_commands(match.group("cmds"))

    # Leading form: "in parallel, run <cmds>" / "simultaneously execute <cmds>"
    match = re.search(
        rf"^\s*(?:please\s+)?{marker}\s*,\s*(?:run|execute)\s+(?P<cmds>.+)\s*\.?\s*$",
        text,
        re.IGNORECASE,
    )
    if match:
        return _split_commands(match.group("cmds"))

    return None


# TLDs treated as bare-domain candidates ("check example.com and example.org").
# Deliberately excludes file-like extensions (json, txt, …) so "config.json"
# is never mistaken for a domain.
_BARE_DOMAIN_TLDS = frozenset(
    {
        "com", "org", "net", "io", "ai", "dev", "co", "me", "app",
        "gov", "edu", "info", "biz", "us", "uk", "de", "fr", "ca",
        "jp", "ru", "in", "xyz",
    }
)


def _extract_bare_domains(text: str) -> list[str]:
    """
    Extracts bare hostnames (example.com, docs.pandas.org) without a scheme.

    Conservative: the final label must be a known TLD so filenames like
    ``config.json`` are never captured. Returns lowercased domains, deduped.
    """
    domains: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(r"\b(?:[\w-]+\.)+[a-z]{2,}\b", text, re.IGNORECASE):
        token = match.group(0)
        tld = token.rsplit(".", 1)[-1].lower()
        if tld not in _BARE_DOMAIN_TLDS:
            continue
        if "://" in token or token.startswith("//"):
            continue
        lowered = token.lower()
        if lowered not in seen:
            seen.add(lowered)
            domains.append(lowered)
    return domains


def detect_tool_batch_intent(user_input: str) -> dict | None:
    """
    Detects requests that want results from several *different* tools at
    once — e.g. "read config.json and notes.txt", "check example.com and
    example.org", "search for pandas and numpy at the same time".

    Returns one of:
      {"calls": [{tool, arguments}, ...]}  — deterministically extracted batch
      {"planner": True}                     — ambiguous parallel request;
                                               the LLM planner builds the batch
      None                                   — not a parallel-tool request

    This is the *inter-tool* counterpart of ``detect_parallel_intent`` (which
    only parallelizes shell commands). Runs before the single-target
    detectors so a multi-target request is captured as a unit instead of
    being claimed by the first single-target match.
    """
    text = user_input.strip()

    # Shell-command parallel batches stay with detect_parallel_intent
    # ("run X and Y in parallel" must not become a tool batch).
    if detect_parallel_intent(text):
        return None

    calls: list[dict] = []
    seen: set[tuple] = set()

    def add(tool: str, arguments: dict) -> None:
        key = (tool, tuple(sorted(arguments.items())))
        if key not in seen:
            seen.add(key)
            calls.append({"tool": tool, "arguments": arguments})

    # 1) Multiple filenames -> read_file calls ("read X and Y").
    #    Image files are skipped — they route to the vision model instead.
    if re.search(r"\b(?:read|open|show|cat)\b", text, re.IGNORECASE):
        for fname in re.findall(r"[\w./-]+\.[a-zA-Z0-9]+", text):
            if "://" in fname or fname.startswith("//"):
                continue  # URL, not a local file
            if _IMAGE_EXT_RE.search(fname):
                continue
            add("read_file", {"file_path": fname})

    # 2) URLs (explicit scheme or bare domain) -> connectivity check / fetch.
    if re.search(r"\b(?:check|fetch|read|is\s+.+\bup|online)\b", text, re.IGNORECASE):
        urls: list[str] = []
        explicit = [
            re.sub(r"[\.,;\)]+$", "", u).strip()
            for u in re.findall(r"https?://[^\s]+", text, re.IGNORECASE)
        ]
        urls.extend(u for u in explicit if u)
        # Mask explicit URLs so bare-domain extraction never re-matches the
        # host inside them ("https://example.com" must yield one URL, not two).
        masked = re.sub(r"https?://[^\s]+", " ", text, flags=re.IGNORECASE)
        urls.extend(_extract_bare_domains(masked))
        fetch_mode = bool(re.search(r"\b(?:fetch|read)\b", text, re.IGNORECASE))
        for url in urls:
            if fetch_mode:
                add("fetch_page_text", {"url": url})
            else:
                add("check_connectivity", {"url": url})

    # 3) Repeated or comma-separated search queries ("search for A and B"
    #    splits only on explicit structure, so "fish and chips" stays whole).
    search_matches = list(
        re.finditer(r"\bsearch\s+(?:the\s+web\s+)?for\s+(.+?)(?=\s+(?:and|,)\s+search|$)", text, re.IGNORECASE)
    )
    if search_matches:
        queries = [m.group(1).strip().rstrip(".") for m in search_matches if m.group(1).strip()]
        for q in queries:
            add("web_search", {"query": q})

    if len(calls) >= 2:
        return {"calls": calls}

    # Explicit parallelism phrasing with no concrete targets -> LLM planner.
    if re.search(
        r"\b(?:at\s+the\s+same\s+time|simultaneously|concurrently|in\s+parallel|together)\b",
        text,
        re.IGNORECASE,
    ):
        return {"planner": True}

    return None


def detect_remember_intent(user_input: str) -> str | None:
    """
    Detects if the user input requests remembering a fact (e.g. "remember that ...", "remember ...").
    Returns the extracted fact string if matched, otherwise None.
    """
    pattern = r'^\s*(?:please\s+)?remember\s+(?:that\s+)?(?P<fact>.+)\s*$'
    match = re.search(pattern, user_input, re.IGNORECASE)
    if match:
        return match.group("fact").strip()
    return None

def detect_deduction_question(user_input: str) -> str | None:
    """
    Detects knowledge-graph reasoning questions (e.g. "what is the capital of
    a country that borders Germany"). Returns the question when it matches a
    deterministic reasoning template, otherwise None.

    These are answered from stored triples by graph traversal — the AI is
    never involved, so the answer is always grounded in remembered facts.
    """
    from ultron.core.tools.memory import graph

    text = user_input.strip()
    if graph.is_deduction_question(text):
        return text
    return None


def detect_memory_question(user_input: str) -> str | None:
    """
    Detects questions asking Ultron to recall stored facts about a topic.

    Strategy: two-gate approach instead of enumerating every exact phrase.
      Gate 1 — does this look like a recall question?
               Must contain "what" AND at least one recall verb:
               "remember", "tell", "know", "say".
      Gate 2 — does it have "about <topic>"?
               Extract everything after the LAST occurrence of the word
               "about" as the topic.  Using the last "about" handles
               constructions like "what did i tell you to remember about X"
               where an intermediate "about" could appear in the verb phrase.

    Matches (all case-insensitive):
      - "what did i tell you about FastAPI"
      - "what did i tell you to remember about testing"
      - "what do you remember about databases"
      - "what do you know about X"
      - "what did i say about Y"
      - informal variants like "what did i tell u about testing"

    Returns the extracted topic string, or None if either gate fails.

    Why handled in code rather than by the AI?  The system prompt injects ALL
    stored memories, so the AI can hallucinate connections between unrelated
    facts.  This path calls search_memories(topic) and builds the reply
    directly from DB rows — zero AI involvement, zero hallucination risk.
    """
    text = user_input.strip()

    # Gate 1: must contain "what" and at least one recall verb
    has_what = bool(re.search(r'\bwhat\b', text, re.IGNORECASE))
    has_recall_verb = bool(re.search(
        r'\b(?:remember|tell|know|say|told)\b', text, re.IGNORECASE
    ))
    if not (has_what and has_recall_verb):
        return None

    # Gate 2: must contain the word "about" followed by a non-empty topic.
    # re.finditer gives us all matches; we want the LAST one.
    about_matches = list(re.finditer(r'\babout\s+', text, re.IGNORECASE))
    if not about_matches:
        return None

    last_match = about_matches[-1]
    topic = text[last_match.end():].strip().rstrip('?').strip()

    return topic if topic else None

def detect_test_intent(user_input: str) -> bool:
    """
    Detects if the user input requests running tests (e.g. "run tests", "test my code").
    Returns True if matched, otherwise False.
    """
    pattern = r'\b(run\s+tests?|run\s+the\s+tests?|test\s+my\s+code|run\s+pytest)\b'
    return bool(re.search(pattern, user_input, re.IGNORECASE))

def detect_git_intent(user_input: str) -> str | None:
    """
    Maps common natural-language Git phrases to the actual git command to run.

    Returns the ready-to-execute command string if matched, otherwise None.

    Design note: Git commands are plain shell commands, so we reuse the
    existing run_command tool and PendingAction confirmation flow without
    any new infrastructure.  This detector is placed BEFORE the generic
    detect_command_intent so that phrases like "show diff" are caught here
    rather than being passed literally to the shell as `diff`.

    Commit phrasing optionally accepts a quoted message, e.g.:
      "commit this \"fixed login bug\""  ->  git add -A && git commit -m "fixed login bug"
      "commit changes"                   ->  git add -A && git commit -m "Update via Ultron"
    """
    text = user_input.strip()

    # --- git status ---
    if re.search(r'\b(what\s+changed|show\s+changes|git\s+status)\b', text, re.IGNORECASE):
        return "git status"

    # --- git diff ---
    if re.search(
        r'\b(show\s+(?:me\s+the\s+)?diff|what(?:\'s|\s+is)\s+different|git\s+diff)\b',
        text, re.IGNORECASE
    ):
        return "git diff"

    # --- git log ---
    if re.search(r'\b(git\s+log|show\s+commit\s+history|recent\s+commits)\b', text, re.IGNORECASE):
        return "git log --oneline -10"

    # --- git commit ---
    # Match "commit changes" or "commit this", optionally with a quoted message.
    commit_match = re.search(
        r'\bcommit\s+(?:changes|this)\b(?:.*?"(?P<msg>[^"]+)")?',
        text, re.IGNORECASE
    )
    if commit_match:
        msg = commit_match.group("msg")
        # Use the user's quoted message if provided; otherwise fall back to a
        # sensible default so the automated commit is clearly identifiable in history.
        commit_msg = msg.strip() if msg else "Update via Ultron"
        return f'git add -A && git commit -m "{commit_msg}"'

    return None

def detect_lint_intent(user_input: str) -> bool:
    """
    Detects if the user wants to lint / check their code for issues.

    Returns True if matched, otherwise False.

    Design note: Like git and test commands, linting is just a shell command
    under the hood.  Detecting it explicitly here (rather than letting it fall
    through to detect_command_intent) means we can map casual phrases like
    "check my code" to the correct ruff invocation without the user needing
    to remember the exact CLI syntax.
    """
    pattern = (
        r'\b('
        r'check\s+(?:my\s+)?code'
        r'|lint(?:\s+my\s+code)?'
        r'|run\s+linter'
        r'|find\s+issues'
        r'|check\s+for\s+errors'
        r'|run\s+ruff'
        r')\b'
    )
    return bool(re.search(pattern, user_input, re.IGNORECASE))

def detect_web_search_intent(user_input: str) -> str | None:
    """
    Detects if the user wants to search the web for information using regex.
    Matches phrases like "search for X", "look up X", "google X", "find info on X".

    Returns the extracted search query string if matched, otherwise None.
    """
    pattern = r'^\s*(?:please\s+)?(?:search\s+(?:the\s+web\s+)?for|look\s+up|google|find\s+info\s+on|search\s+for)\s+(?P<query>.+)\s*$'
    match = re.search(pattern, user_input, re.IGNORECASE)
    if match:
        return match.group("query").strip()
    return None

def detect_fetch_page_intent(user_input: str) -> str | None:
    """
    Detects if the user wants to fetch and read a web page URL.
    Matches phrases like "fetch this page X", "get the content of X", "read this url X".

    Returns the extracted URL if matched, otherwise None.
    """
    # Look for http:// or https:// URL in input preceded by fetch/read/get phrasing
    url_pattern = r'https?://[^\s]+'
    phrase_pattern = r'\b(?:fetch(?:\s+this\s+page)?|get(?:\s+the\s+content\s+of)?|read(?:\s+this\s+url)?|scrape)\b'

    if re.search(phrase_pattern, user_input, re.IGNORECASE):
        url_match = re.search(url_pattern, user_input, re.IGNORECASE)
        if url_match:
            raw_url = url_match.group(0)
            return re.sub(r'[\.,;\)]+$', '', raw_url).strip()

    return None

def detect_db_query_intent(user_input: str) -> str | None:
    """
    Detects if the user input requests running a database SQL query.
    Matches phrases like "run this query: X", "query the database: X", "execute sql: X", "run sql X".

    Returns the extracted raw SQL string if matched, otherwise None.
    """
    pattern = (
        r'^\s*(?:please\s+)?'
        r'(?:run\s+this\s+query[:\s]+'
        r'|query\s+the\s+database[:\s]+'
        r'|execute\s+sql[:\s]+'
        r'|run\s+sql\s+'
        r'|query[:\s]+'
        r'|run\s+query[:\s]+)'
        r'(?P<sql>.+)\s*$'
    )
    match = re.search(pattern, user_input, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group("sql").strip()
    return None

def extract_api_url(text: str) -> str | None:
    """
    Extracts a base URL for API schema-learning requests.

    Accepts http(s):// URLs, localhost/127.0.0.1 bases (kept as http://), and
    bare domains (normalized to https://) — the shapes users most often use
    to point Ultron at an API.
    """
    url_match = re.search(r"https?://[^\s]+", text, re.IGNORECASE)
    if url_match:
        return re.sub(r"[.,;\\)]+$", "", url_match.group(0)).strip()
    host_match = re.search(r"\b(?:localhost|127\.0\.0\.1)(?::\d+)?\b", text, re.IGNORECASE)
    if host_match:
        return "http://" + host_match.group(0).lower()
    domain_match = re.search(r"\b(?:[\w-]+\.)+[a-z]{2,}\b", text, re.IGNORECASE)
    if domain_match:
        return "https://" + domain_match.group(0).lower()
    return None


def detect_api_schema_intent(user_input: str) -> dict | None:
    """
    Detects API schema learning / inspection / reset requests:

      - "learn the api schema for http://localhost:8000"  -> learn
      - "fetch the schema for example.com"                  -> learn
      - "what do you know about the api at example.com"     -> knowledge
      - "what apis do you know"                             -> knowledge
      - "api usage hints for http://localhost:8000"         -> hints
      - "forget the api schema for http://localhost:8000"   -> forget

    Returns {"action": ..., "url": str | None}. The URL is None for
    knowledge requests that name no API (those list everything learned).
    Runs before the HTTP detector so schema-learning phrasing is never
    mistaken for a plain GET of the URL.
    """
    text = user_input.strip()

    if re.search(
        r"\blearn\s+(?:the\s+)?(?:api\s+)?schema\b"
        r"|\bfetch\s+(?:the\s+)?(?:api\s+)?schema\b"
        r"|\bdiscover\s+(?:the\s+)?(?:api\s+)?schema\b",
        text,
        re.IGNORECASE,
    ):
        action = "learn"
    elif re.search(r"\bapi\s+usage\s+hints?\b", text, re.IGNORECASE):
        action = "hints"
    elif re.search(
        r"\bforget\s+(?:the\s+)?(?:api\s+)?(?:schema|knowledge)\b"
        r"|\bclear\s+(?:the\s+)?(?:api\s+)?(?:schema|knowledge)\b",
        text,
        re.IGNORECASE,
    ):
        action = "forget"
    elif re.search(
        r"\bwhat\s+do\s+you\s+know\s+about\s+(?:the\s+)?apis?\b"
        r"|\bwhat\s+apis?\s+do\s+you\s+know\b"
        r"|\bapi\s+knowledge\b"
        r"|\bknown\s+apis?\b"
        r"|\bwhat\s+(?:have|'ve)\s+you\s+learned\s+about\s+(?:the\s+)?apis?\b",
        text,
        re.IGNORECASE,
    ):
        action = "knowledge"
    else:
        return None

    return {"action": action, "url": extract_api_url(text)}


def _extract_forecast_command(text: str) -> str | None:
    """Pulls a command out of a forecast request (quotes, 'for/of:', 'heavy is')."""
    quoted = re.search(r"[\"'](.+?)[\"']", text)
    if quoted:
        return quoted.group(1).strip()
    match = re.search(r"\b(?:for|of|:)\s+(.+)$", text, re.IGNORECASE)
    if match:
        return match.group(1).strip().rstrip("?").strip()
    match = re.search(r"\bheavy\s+is\s+(.+)$", text, re.IGNORECASE)
    if match:
        return match.group(1).strip().rstrip("?").strip()
    return None


def detect_resource_intent(user_input: str) -> dict | None:
    """
    Detects resource-monitoring requests:

      - "check system resources" / "how much memory is free"  -> check
      - "resource forecast for pip install" / "how heavy is find /" -> forecast

    Returns {"action": "check"|"forecast", "command": str | None}.
    Runs before the HTTP/retrieval detectors; resource phrasing carries no
    URLs so there is no overlap with the networking tools.
    """
    text = user_input.strip()

    if re.search(
        r"\b(?:check\s+(?:the\s+)?(?:system\s+)?resources?)\b"
        r"|\bsystem\s+resources?\b"
        r"|\b(?:cpu|memory|ram)\s+(?:usage|load|status)\b"
        r"|\bhow\s+(?:much|is)\s+(?:memory|ram|cpu)\b"
        r"|\bhow\s+is\s+my\s+system\b"
        r"|\bsystem\s+(?:status|load)\b",
        text,
        re.IGNORECASE,
    ):
        return {"action": "check", "command": None}

    if re.search(
        r"\b(?:resource\s+)?forecast\b"
        r"|\bhow\s+heavy\s+is\b"
        r"|\bhow\s+long\s+will\s+(?:this|it|that)\b"
        r"|\bwill\s+this\s+be\s+heavy\b"
        r"|\bhow\s+much\s+(?:cpu|memory)\s+will\s+this\s+use\b",
        text,
        re.IGNORECASE,
    ):
        return {"action": "forecast", "command": _extract_forecast_command(text)}

    return None


def detect_association_intent(user_input: str) -> dict | None:
    """
    Detects personalized-learning requests — asking how stored facts connect
    across domains:

      - "what connections do you see"                   -> connections
      - "connections for renaissance"                    -> connections (topic)
      - "how is renaissance art related to the medici"   -> relate (a, b)
      - "discover new connections" / "link my memories" -> discover

    Returns {"action", "a", "b", "topic"}. Runs after the memory-question
    detector (3.5) so recall ("what do you remember about X") keeps its
    existing path.
    """
    text = user_input.strip()

    if re.search(
        r"\bdiscover\b.*\bconnections?\b|\bnovel\s+connections?\b",
        text,
        re.IGNORECASE,
    ):
        return {"action": "discover", "a": "", "b": "", "topic": ""}

    for pattern in (
        r"\bhow\s+is\s+(?P<a>.+?)\s+related\s+to\s+(?P<b>.+?)\??\s*$",
        r"\bhow\s+does\s+(?P<a>.+?)\s+(?:relate|connect)\s+to\s+(?P<b>.+?)\??\s*$",
        r"\bis\s+(?P<a>.+?)\s+related\s+to\s+(?P<b>.+?)\??\s*$",
    ):
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return {
                "action": "relate",
                "a": match.group("a").strip(),
                "b": match.group("b").strip(),
                "topic": "",
            }

    if re.search(
        r"\bwhat\s+connections?\b"
        r"|\bshow\s+(?:me\s+)?connections?\b"
        r"|\bmemory\s+connections?\b"
        r"|\bconnections?\s+(?:between|among|for|about)\b"
        r"|\brelated\s+facts?\b"
        r"|\bhow\s+(?:are|do)\s+my\s+(?:memories|facts)\s+(?:connected|linked|relate)\b"
        r"|\blink(?:ed|s)?\s+(?:memories|facts|knowledge)\b",
        text,
        re.IGNORECASE,
    ):
        topic = ""
        topic_match = re.search(
            r"\bconnections?\s+(?:for|about)\s+(.+?)\??\s*$"
            r"|\brelated\s+facts?\s+(?:about|for)\s+(.+?)\??\s*$",
            text,
            re.IGNORECASE,
        )
        if topic_match:
            topic = (topic_match.group(1) or topic_match.group(2) or "").strip()
        return {"action": "connections", "a": "", "b": "", "topic": topic}

    return None


def detect_http_intent(user_input: str) -> tuple[str, str, str | None] | None:
    """
    Detects HTTP request intents using regular expressions and body/method rules.

    Rules & Behavior:
    1. URL Extraction: Locates 'http://' or 'https://' URLs and strips trailing punctuation.
    2. Body Detection: Looks for 'with body <payload>' or inline JSON '{...}'.
    3. Method Selection:
       - Priority 1: Explicit method keywords (POST, PUT, DELETE, PATCH, GET) in input.
       - Priority 2 (Body Override Rule): If a body/payload is present, defaults to POST even
         if conversational words like "get" or "fetch" appear (e.g. "get from X with body Y").
       - Default: GET (when no state-modifying verb or body is present).

    Returns a tuple of (method, url, optional_body) if matched, otherwise None.
    """
    text = user_input.strip()

    # 1. Search for URL (http:// or https://)
    url_match = re.search(r'https?://[^\s]+', text, re.IGNORECASE)
    if not url_match:
        return None

    raw_url = url_match.group(0)
    # Clean trailing punctuation from URL (e.g., '.', ',', ';', ')')
    url = re.sub(r'[\.,;\)]+$', '', raw_url).strip()

    # 2. Extract Body / Payload
    body: str | None = None

    # Check for explicit "with body <payload>" construct
    with_body_match = re.search(r'\bwith\s+body\s+(?P<body>.+)$', text, re.IGNORECASE | re.DOTALL)
    if with_body_match:
        body = with_body_match.group("body").strip()
    else:
        # Check for inline JSON block enclosed in {...}
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            body_candidate = json_match.group(0).strip()
            # Validate if it's parseable JSON
            try:
                json.loads(body_candidate)
                body = body_candidate
            except (json.JSONDecodeError, TypeError):
                body = body_candidate

    # Clean wrapping quotes if body was specified as '...' or "..."
    if body:
        wrapped_in_quotes = (body.startswith('"') and body.endswith('"')) or (
            body.startswith("'") and body.endswith("'")
        )
        if wrapped_in_quotes:
            body = body[1:-1].strip()

    # 3. Method Selection Logic
    # Priority 1: Explicit state-changing method keywords in user_input
    explicit_method_match = re.search(r'\b(POST|PUT|DELETE|PATCH)\b', text, re.IGNORECASE)
    if explicit_method_match:
        method = explicit_method_match.group(1).upper()
    elif body:
        # Priority 2: Body Override Rule — if body is present and no explicit POST/PUT/DELETE/PATCH was matched,
        # default to POST even if conversational "get" or "fetch" is present.
        method = "POST"
    elif re.search(r'\b(GET|fetch|show|read|retrieve)\b', text, re.IGNORECASE):
        method = "GET"
    else:
        method = "GET"

    return method, url, body

def detect_debug_intent(user_input: str) -> dict | None:
    """
    Detects debugging requests and extracts the failure target.

    Matches debugging phrasing: "debug this", "why is my script failing",
    "diagnose this error: …", "help me fix", "what went wrong", …

    Returns {"command": str, "error": str, "expected": str} where:
      - ``command``  — a quoted command, or the text after "debug "/"run "
      - ``error``    — pasted error/traceback text (quoted or after
        "error:"/"this error:"/"diagnose this:")
      - ``expected`` — the user's stated expectation after "expected"/
        "should …" (used for expected-vs-actual reconciliation)

    Runs after the HTTP/retrieval detectors (so URL-bearing inputs keep their
    networking path) and before the generic command detector.
    """
    text = user_input.strip()
    triggers = (
        r"\bdebug(?:ging)?\b"
        r"|\bdiagnos[ei]\b"
        r"|\bwhy\s+(?:is|did|does|do|can'?t)\b.*\b(?:fail|crash|break|not\s+work)\b"
        r"|\bwhy\s+.*\b(?:failing|crashed|broken|not\s+working)\b"
        r"|\bmy\s+(?:code|script|program|app|tests?)\s+(?:crashed|failed|is\s+broken|doesn'?t\s+work|isn'?t\s+working)\b"
        r"|\bhelp\s+(?:me\s+)?(?:fix|debug)\b"
        r"|\bfix\s+this\s+error\b"
        r"|\bwhat\s+went\s+wrong\b"
        r"|\binvestigate\s+(?:the\s+)?error\b"
        r"|\bwhat'?s\s+wrong\s+(?:with|in)\b"
    )
    if not re.search(triggers, text, re.IGNORECASE):
        return None

    command = ""
    error = ""
    expected = ""

    # Quoted segments: an error-looking quote is the pasted failure, otherwise
    # the first quote is the command to debug.
    for quoted in re.findall(r"[\"'](.+?)[\"']", text):
        if re.search(r"Traceback|ModuleNotFoundError|Error|error\b", quoted):
            if not error:
                error = quoted.strip()
        elif not command:
            command = quoted.strip()

    # Pasted error after "error:" / "this error:" / "diagnose this:"
    if not error:
        err_match = re.search(
            r"\b(?:diagnose\s+this\s+)?error\s*[:]\s*(.+)$",
            text,
            re.IGNORECASE,
        )
        if err_match:
            error = err_match.group(1).strip()

    # Bare target: "debug main.py" / "debug the script" / "run pytest"
    # (dots are allowed so "debug main.py" keeps its extension).
    if not command and not error:
        target_match = re.search(
            r"\b(?:debug|run|diagnose)\s+(?P<target>[^!?;]+)$",
            text,
            re.IGNORECASE,
        )
        if target_match:
            target = target_match.group("target").strip()
            # Vague targets ("debug this script") would run nonsense in the
            # shell — strip filler words until something concrete remains.
            # The stopword match consumes trailing whitespace OR the end of
            # the string so stripping always makes progress (no infinite loop).
            while True:
                filler = re.match(
                    r"^(?:this|my|the|it|that|script|code|program|app)(?:\s+|$)",
                    target,
                    re.IGNORECASE,
                )
                if not filler:
                    break
                target = target[filler.end():].strip()
            command = target

    # Stated expectation: "expected …" / "should print/output/be …"
    expected_match = re.search(
        r"\bexpected\s*[:]?\s*(.+?)(?:\s*\.\s*)?$"
        r"|\bshould\s+(?:print|output|be|produce|return)\s+(.+?)(?:\s*\.\s*)?$",
        text,
        re.IGNORECASE,
    )
    if expected_match:
        expected = (expected_match.group(1) or expected_match.group(2) or "").strip()

    return {"command": command, "error": error, "expected": expected}


def detect_multistep_intent(user_input: str) -> bool:
    """
    Heuristic detector for compound / multi-step requests.

    Returns True if the user input looks like it contains more than one
    distinct action (e.g. "read X, then write Y, then run Z").  We look for
    coordinating conjunctions and sequencing keywords that typically glue
    multiple imperatives together.

    This is intentionally broad — false positives are harmless because
    plan_task() will still produce a sensible (possibly single-step) plan,
    and the user must confirm before anything runs.
    """
    # Words / phrases that commonly connect sequential steps
    sequence_markers = [
        r'\bthen\b',
        r'\bafter\s+that\b',
        r'\bafterwards\b',
        r'\bnext\b',
        r'\bfollowed\s+by\b',
        r'\band\s+then\b',
        r'\balso\b',
        r'\bfinally\b',
        r'\blast(?:ly)?\b',
    ]
    # Must mention at least two action verbs to qualify as multi-step
    action_verbs = [
        r'\bread\b', r'\bopen\b', r'\bshow\b',
        r'\bwrite\b', r'\bcreate\b', r'\bsave\b',
        r'\brun\b', r'\bexecute\b', r'\btest\b',
        r'\bremember\b',
    ]

    has_sequence = any(
        re.search(m, user_input, re.IGNORECASE) for m in sequence_markers
    )
    if not has_sequence:
        return False

    verb_hits = sum(
        1 for v in action_verbs
        if re.search(v, user_input, re.IGNORECASE)
    )
    return verb_hits >= 2


# ---------------------------------------------------------------------------
# Planning helpers  (used by handle_multistep)
# ---------------------------------------------------------------------------

async def plan_task(user_input: str, engine) -> list[dict] | None:
    """
    Asks the LLM to decompose a compound user request into an ordered list
    of structured action steps.

    Supported action types: read_file, write_file, run_command, add_memory.

    Returns a list of dicts on success, or None if the LLM response cannot
    be parsed as valid JSON.
    """
    planning_prompt = (
        "You are a task-planning assistant.\n"
        "Break the following user request into a list of simple steps.\n"
        "Only use these action types: read_file, write_file, run_command, "
        "make_http_request, run_query, add_memory.\n"
        "Respond with ONLY a JSON array — no other text, no markdown fences.\n"
        "Use exactly these formats for each action type:\n"
        '  {"action": "read_file",   "filename": "..."}\n'
        '  {"action": "write_file",  "filename": "...", "content": "..."}\n'
        '  {"action": "run_command", "command": "..."}\n'
        '  {"action": "make_http_request", "method": "GET|POST", "url": "...", "body": "..."}\n'
        '  {"action": "run_query",   "sql": "..."}\n'
        '  {"action": "add_memory",  "fact": "..."}\n'
        "\n"
        f"User request: {user_input}"
    )

    try:
        raw = await engine.generate([{"role": "user", "content": planning_prompt}])
    except (httpx.HTTPError, OSError, ValueError):
        return None  # Engine error — fall back silently

    # Strip markdown code fences if the model wrapped its answer
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r'^```[\w]*\n?', '', raw)
        raw = re.sub(r'\n?```$', '', raw)
        raw = raw.strip()

    try:
        steps = json.loads(raw)
    except json.JSONDecodeError:
        return None  # Unparseable — fall back to normal flow

    if not isinstance(steps, list):
        return None

    # Validate that every step has a recognised action key
    valid_actions = {
        "read_file",
        "write_file",
        "run_command",
        "make_http_request",
        "run_query",
        "add_memory",
    }
    cleaned: list[dict] = []
    for step in steps:
        if isinstance(step, dict) and step.get("action") in valid_actions:
            cleaned.append(step)

    return cleaned if cleaned else None


def is_step_failure(action: str, result: str) -> bool:
    """
    Determines whether an action step result represents a failure.

    Design rationale:
    `run_command` needs special handling because a "successful" shell execution
    (no Python exception) can still represent a failed command via a non-zero exit
    code (e.g. "Exit code: 1\nOutput:..."), which is different from an "Error: ..."
    string returned when exceptions or missing tools occur.
    """
    res_str = str(result)
    if action == "run_command":
        if res_str.startswith("Error"):
            return True
        exit_code_match = re.search(r'Exit code:\s*(\d+)', res_str, re.IGNORECASE)
        return bool(exit_code_match and exit_code_match.group(1) != "0")

    return res_str.startswith("Error")


async def execute_plan(steps: list[dict]) -> list[str]:
    """
    Executes a list of structured action steps produced by plan_task().

    Design decisions:
    - Retries: Only `run_command` and `make_http_request` steps get automatic retries
      (up to 3 attempts with a 1-second pause between attempts) because network or
      timing glitches can cause temporary failures. File writes, reads, and memory
      mutations are NOT retried silently to prevent duplicate writes or corrupted state.
    - We stop on the FIRST unresolvable error rather than continuing with subsequent steps.
      This is an intentional predictability guarantee: if step 2 failed, the user can be
      certain step 3 never touched anything.
    """
    from ultron.core.tools.registry import get_tool

    results: list[str] = []
    retryable_actions = {"run_command", "make_http_request"}

    for i, step in enumerate(steps, start=1):
        action = step.get("action", "")
        label = f"Step {i} ({action})"

        # Security gate per step. A DENY verdict (secret exfiltration, unsafe
        # URL, path escape) hard-blocks the step and stops the plan — a step
        # that is blocked never runs, and nothing after it runs either.
        # CONFIRM steps run because the plan itself is the user's explicit
        # directive — except in strict mode, where anything above LOW risk is
        # skipped until the user approves it separately.
        target, content = _step_target_content(step)
        verdict = check_action(action, target, content)
        if is_denied(verdict):
            results.append(f"{label}: BLOCKED by security — {blocked_message(verdict)}")
            break
        if is_confirm(verdict) and security_mode() == "strict":
            results.append(f"{label}: skipped — requires your approval ({verdict.reason})")
            break

        max_attempts = 3 if action in retryable_actions else 1
        last_result = ""
        attempts_used = 0

        # Resource awareness in multi-step plans: annotate run_command steps
        # whose forecast is heavy or critical before they execute.
        step_note = ""
        if action == "run_command":
            from ultron.core.tools.resource_monitor import (
                forecast_severity,
                forecast_warning,
            )
            cmd_text = str(step.get("command", ""))
            if forecast_severity(cmd_text) in {"heavy", "critical"}:
                step_note = " \u26a0 " + (forecast_warning(cmd_text) or "")

        for attempt in range(1, max_attempts + 1):
            attempts_used = attempt
            try:
                if action == "read_file":
                    func = get_tool("read_file")
                    if not func:
                        result = "Error: Tool 'read_file' not found in registry."
                    else:
                        result = func(step["filename"])

                elif action == "write_file":
                    func = get_tool("write_file")
                    if not func:
                        result = "Error: Tool 'write_file' not found in registry."
                    else:
                        # overwrite=True: the user approved this plan, no second prompt needed
                        result = func(step["filename"], step["content"], overwrite=True)

                elif action == "run_command":
                    func = get_tool("run_command")
                    if not func:
                        result = "Error: Tool 'run_command' not found in registry."
                    else:
                        result = func(step["command"])

                elif action == "make_http_request":
                    func = get_tool("make_http_request")
                    if not func:
                        result = "Error: Tool 'make_http_request' not found in registry."
                    else:
                        result = func(
                            step.get("method", "GET"),
                            step["url"],
                            step.get("body")
                        )

                elif action == "run_query":
                    func = get_tool("run_query")
                    if not func:
                        result = "Error: Tool 'run_query' not found in registry."
                    else:
                        result = func(step["sql"])

                elif action == "add_memory":
                    func = get_tool("add_memory")
                    if not func:
                        result = "Error: Tool 'add_memory' not found in registry."
                    else:
                        result = func(step["fact"])

                else:
                    result = f"Error: unknown action type '{action}'"

            except Exception as exc:  # noqa: BLE001 — plan steps run arbitrary tools
                result = f"Error: {exc}"

            last_result = str(result)

            # If step succeeded (not a failure), break retry loop
            if not is_step_failure(action, last_result):
                break

            # If retrying, pause 1 second before the next attempt (non-blocking)
            if attempt < max_attempts:
                await asyncio.sleep(1)

        # Format output entry based on success/failure and attempt count
        if is_step_failure(action, last_result):
            if max_attempts > 1:
                entry = f"{label}{step_note}: FAILED after {attempts_used} attempts. Last error: {last_result}"
            else:
                entry = f"{label}{step_note}: {last_result}"
            results.append(entry)
            # Stop on first error — don't execute further steps.
            break
        else:
            if attempts_used > 1:
                entry = f"{label}{step_note}: [succeeded after {attempts_used} attempts] {last_result}"
            else:
                entry = f"{label}{step_note}: {last_result}"
            results.append(entry)

    return results


# ---------------------------------------------------------------------------
# Handlers
# Each function contains exactly the logic that was previously inline inside
# run() for the matching detector.  Keeping them here makes run() a clean
# index of "what Ultron can do" rather than a wall of nested if-blocks.
# ---------------------------------------------------------------------------

def handle_greeting() -> ChatMessage:
    """
    Returns a friendly conversational greeting response without executing
    any tools or querying the database.
    """
    return ChatMessage(
        role=Role.ASSISTANT,
        content="Hello! How can I help you today?"
    )

async def handle_image(path: str, user_input: str, engine) -> ChatMessage:
    """
    Analyzes an image with a vision-capable model.

    Reads the image file, base64-encodes it, and sends it to the engine as an
    Ollama image part alongside a prompt derived from the user's request.
    Gated by the security boundary like any file read (path escapes and
    secret-bearing content are denied before the file is touched).

    When the active model cannot see images, explains how to switch to a
    vision model instead of failing obscurely.
    """
    import base64
    from pathlib import Path

    # Security gate: same classification as a file read (path-escape deny).
    verdict = check_action("read_file", path)
    if is_denied(verdict):
        return ChatMessage(role=Role.ASSISTANT, content=blocked_message(verdict))

    image_file = Path(path)
    if not image_file.is_file():
        return ChatMessage(
            role=Role.ASSISTANT,
            content=f"Sorry, I couldn't find the image '{path}'.",
        )

    # Vision capability pre-check (only when the engine can report it).
    supports = getattr(engine, "supports_images", None)
    if supports is not None:
        try:
            supported = await supports()
        except (httpx.HTTPError, OSError, ValueError):
            supported = None
        if supported is False:
            model = getattr(engine, "model", "?")
            return ChatMessage(
                role=Role.ASSISTANT,
                content=(
                    f"The active model '{model}' can't see images yet. Pull a "
                    "vision-capable model and select it, e.g.:\n"
                    "  ollama pull llava\n"
                    "  /model  →  llava"
                ),
            )

    try:
        encoded = base64.b64encode(image_file.read_bytes()).decode("ascii")
    except OSError as exc:
        return ChatMessage(
            role=Role.ASSISTANT,
            content=f"Sorry, I couldn't read the image '{path}' ({exc}).",
        )

    prompt = (
        f"The user attached the image '{path}' and asked:\n{user_input}\n\n"
        "Analyze the image carefully. If it contains a diagram, chart, graph, "
        "table, or handwritten notes, interpret the visual data precisely and "
        "act on the user's request (e.g. write code implementing what is "
        "shown, or explain what the data means). Be specific and concrete."
    )
    messages = [{"role": "user", "content": prompt, "images": [encoded]}]
    try:
        content = await engine.generate(messages)
    except (httpx.HTTPError, OSError, ValueError) as exc:
        return ChatMessage(
            role=Role.ASSISTANT,
            content=f"Sorry, the image analysis failed ({exc}).",
        )
    return ChatMessage(
        role=Role.ASSISTANT,
        content=f"Here's my analysis of '{path}':\n\n{polish_response(content)}",
    )


def handle_file_read(filename: str) -> ChatMessage:
    """
    Reads a file, gated by the security boundary.

    File reads are LOW risk and auto-allowed in every security mode, so they
    execute directly. The guardrails deny reads that escape the allowed
    working directory (path escape) — those are blocked here before any
    filesystem access.
    """
    verdict = check_action("read_file", filename)
    if is_denied(verdict):
        return ChatMessage(role=Role.ASSISTANT, content=blocked_message(verdict))

    result = execute_tool("read_file", file_path=filename)
    if str(result).startswith("Error"):
        return ChatMessage(
            role=Role.ASSISTANT,
            content=f"Sorry, I could not find or read the file '{filename}'.",
        )
    return ChatMessage(
        role=Role.ASSISTANT,
        content=f"Here are the contents of '{filename}':\n\n{result}",
    )

def handle_file_write(filename: str, content: str, user_input: str = "") -> ChatMessage:
    """
    Signals main.py to show an interactive confirmation before writing a file.

    Flow & Rules:
    0. Guard: Reject blank filenames immediately.
    1. Smart Hint: If user used "overwrite", "replace", or "force", emit 'overwrite_file'.
    2. Existence Check: Use os.path.exists() — NOT write_file() — to detect if the file
       already exists. The old probe called write_file() which had the side-effect of
       creating the file before the user confirmed, causing "already exists" on approval.
    """
    import os

    # Guard 0: blank filename — fail fast before any filesystem access
    if not filename or not filename.strip():
        return ChatMessage(
            role=Role.ASSISTANT,
            content="Sorry, I couldn't determine the file name from your request. "
                    "Please specify a filename with an extension, e.g. 'notes.txt'.",
        )

    # Security gate: classify the write and scan the content (secrets, PII)
    # before anything touches the filesystem. A denial hard-blocks here — the
    # action is never offered for confirmation.
    verdict = check_action("write_file", filename, content)
    if is_denied(verdict):
        return ChatMessage(role=Role.ASSISTANT, content=blocked_message(verdict))

    # Permissive mode auto-allows file writes — execute directly.
    if is_allow(verdict):
        return ChatMessage(
            role=Role.ASSISTANT,
            content=execute_tool("write_file", filename=filename, content=content, overwrite=True),
        )

    # Resolve to absolute path the same way file_writer.py does
    abs_path = filename if os.path.isabs(filename) else os.path.join(os.getcwd(), filename)

    # Smart hint: user explicitly requested overwrite
    has_force_hint = bool(re.search(r'\b(overwrite|replace|force)\b', user_input, re.IGNORECASE))
    if has_force_hint:
        return ChatMessage(
            role=Role.ASSISTANT,
            content=f"File '{filename}' will be overwritten as requested.",
            pending_action=PendingAction(
                action_type="overwrite_file",
                target=filename,
                content=content
            )
        )

    # Non-destructive existence check — no write happens here
    if os.path.exists(abs_path) and not os.path.isdir(abs_path):
        return ChatMessage(
            role=Role.ASSISTANT,
            content=f"File '{filename}' already exists. Do you want to overwrite it?",
            pending_action=PendingAction(
                action_type="overwrite_file",
                target=filename,
                content=content
            )
        )

    # New file — signal main.py to confirm creation
    return ChatMessage(
        role=Role.ASSISTANT,
        content=f"File create requested: '{filename}'",
        pending_action=PendingAction(
            action_type="write_file",
            target=filename,
            content=content
        )
    )

def handle_remember(fact: str) -> ChatMessage:
    """
    Stores a new memory via the unified write path: sentences that parse as
    subject/predicate/object are stored as knowledge-graph triples, everything
    else falls back to the flat fact store. Gated by the security boundary
    (which blocks outgoing content carrying credential-like data).
    """
    verdict = check_action("add_memory", "", fact)
    if is_denied(verdict):
        return ChatMessage(role=Role.ASSISTANT, content=blocked_message(verdict))
    from ultron.core.tools.registry import get_tool
    func = get_tool("add_memory")
    tool_result = func(fact) if func else "Error: Tool 'add_memory' not found in registry."
    content = str(tool_result)
    try:
        # Personalized learning: correlate the new fact against everything
        # stored and announce any cross-domain connections. Best-effort —
        # the learning layer must never break remembering.
        from ultron.core.learning.associations import connect_new_fact
        announcement = connect_new_fact(fact)
        if announcement:
            content += f"\n\n{announcement}"
    except Exception:  # noqa: BLE001, S110 — learning layer is optional, best-effort only
        pass
    return ChatMessage(role=Role.ASSISTANT, content=content)

def handle_deduction_question(question: str) -> ChatMessage:
    """
    Answers a reasoning question from the knowledge graph by walking stored
    triples (answer_question / query_chain). Gated by the security boundary
    like every other tool action. When no stored facts support the deduction,
    says so honestly instead of guessing.
    """
    verdict = check_action("query_triples", question)
    if is_denied(verdict):
        return ChatMessage(role=Role.ASSISTANT, content=blocked_message(verdict))

    from ultron.core.tools.memory import graph

    answer = graph.answer_question(question)
    if answer:
        return ChatMessage(role=Role.ASSISTANT, content=answer)
    return ChatMessage(
        role=Role.ASSISTANT,
        content=(
            "I can't deduce that from what I have stored yet — I only answer "
            "reasoning questions from knowledge-graph facts I've actually remembered."
        ),
    )


def handle_memory_question(topic: str) -> ChatMessage:
    """
    Answers a memory-recall question by fetching only facts that match the
    topic keyword — never all memories.  Unions the flat fact store with the
    knowledge graph (edges where the topic is the subject or object).  Built
    directly from DB rows so the AI is never involved and cannot hallucinate.
    """
    verdict = check_action("search_memories", topic)
    if is_denied(verdict):
        return ChatMessage(role=Role.ASSISTANT, content=blocked_message(verdict))
    from ultron.core.tools.registry import get_tool
    func = get_tool("search_memories")
    matches = func(topic) if func else []

    from ultron.core.tools.memory import graph
    graph_matches = graph.recall_about(topic)

    # Deduplicate while preserving order (flat facts first, then graph edges).
    combined = list(dict.fromkeys(matches + graph_matches))

    if combined:
        bullet_list = "\n".join(f"- {item}" for item in combined)
        return ChatMessage(
            role=Role.ASSISTANT,
            content=f"Here's what I have stored about '{topic}':\n{bullet_list}"
        )
    return ChatMessage(
        role=Role.ASSISTANT,
        content=f"I don't have anything stored about '{topic}'."
    )

def handle_test() -> ChatMessage:
    """
    Runs pytest -v, gated by the security boundary.

    pytest is a read-only command (LOW risk), so it auto-executes in every
    mode; a confirmation prompt only appears if the security mode classifies
    it otherwise.
    """
    blocked = _gate_command("pytest -v")
    if blocked is not None:
        return blocked
    return ChatMessage(
        role=Role.ASSISTANT,
        content="Test execution requested: 'pytest -v'",
        pending_action=PendingAction(
            action_type="run_command",
            target="pytest -v"
        )
    )

def handle_git(command: str) -> ChatMessage:
    """
    Runs a git command, gated by the security boundary.

    Read-only git commands (status/diff/log) are LOW risk and auto-execute;
    state-changing ones (e.g. ``git add -A && git commit``) need interactive
    confirmation via the PendingAction flow.
    """
    blocked = _gate_command(command)
    if blocked is not None:
        return blocked
    return ChatMessage(
        role=Role.ASSISTANT,
        content=f"Git command requested: '{command}'",
        pending_action=PendingAction(
            action_type="run_command",
            target=command
        )
    )

def handle_lint() -> ChatMessage:
    """
    Runs ruff in concise mode, gated by the security boundary.

    ruff is a read-only command (LOW risk), so it auto-executes in every mode.
    """
    ruff_cmd = "ruff check . --output-format=concise"
    blocked = _gate_command(ruff_cmd)
    if blocked is not None:
        return blocked
    return ChatMessage(
        role=Role.ASSISTANT,
        content=f"Lint check requested: '{ruff_cmd}'",
        pending_action=PendingAction(
            action_type="run_command",
            target=ruff_cmd
        )
    )

def handle_resource(action: str, command: str | None = None) -> ChatMessage:
    """
    Handles resource-monitoring requests, gated by the security boundary.

    - check    -> current system snapshot (CPU / load / memory)
    - forecast -> predicted resource profile of a command
    Both are read-only (LOW risk) and auto-execute; a missing forecast
    target asks for the command instead of guessing.
    """
    if action == "forecast" and not command:
        return ChatMessage(
            role=Role.ASSISTANT,
            content=(
                "Which command should I forecast? "
                "e.g. 'resource forecast for pip install'."
            ),
        )
    tool_name = "check_resources" if action == "check" else "resource_forecast"
    verdict = check_action(tool_name, command or "")
    if is_denied(verdict):
        return ChatMessage(role=Role.ASSISTANT, content=blocked_message(verdict))
    if action == "check":
        return ChatMessage(role=Role.ASSISTANT, content=execute_tool("check_resources"))
    return ChatMessage(
        role=Role.ASSISTANT,
        content=execute_tool("resource_forecast", command=command),
    )


def handle_debug(
    command: str = "",
    error: str = "",
    expected: str = "",
) -> ChatMessage:
    """
    Produces an environmental-state debug report for failing code.

    Three modes:
      1. Pasted error text -> the failure is diagnosed directly (no command
         executes; the security boundary never gets involved because nothing
         runs).
      2. A command -> gated through the security boundary exactly like a
         normal run_command (path-escape, secret and dangerous-pattern
         guardrails all apply), executed, and its result diagnosed.
      3. Neither -> the environment snapshot is shown and the user is asked
         what to debug.

    Every report couples the diagnosis with the exact environmental state
    (OS, Python, tool versions, declared-vs-installed dependencies) and any
    stated expectation, so fixes start from real data instead of guesses.
    """
    from ultron.core.intelligence.debug_context import (
        capture_environment,
        diagnose_failure,
        format_debug_report,
        format_environment,
    )

    command = (command or "").strip()
    error = (error or "").strip()
    expected = (expected or "").strip()

    # Mode 1: pasted error/traceback — diagnose directly, nothing runs.
    if error:
        diagnosis = diagnose_failure(error, command or None)
        return ChatMessage(
            role=Role.ASSISTANT,
            content=format_debug_report(
                command or None, diagnosis, expected or None
            ),
        )

    # Mode 2: run the failing command (gated) and diagnose its result.
    if command:
        verdict = check_action("run_command", command)
        if is_denied(verdict):
            return ChatMessage(role=Role.ASSISTANT, content=blocked_message(verdict))
        if is_confirm(verdict):
            return ChatMessage(
                role=Role.ASSISTANT,
                content=f"Debug command requested: '{command}'",
                pending_action=PendingAction(
                    action_type="run_command", target=command
                ),
            )
        result = execute_tool("run_command", command=command)
        diagnosis = diagnose_failure(result, command)
        return ChatMessage(
            role=Role.ASSISTANT,
            content=format_debug_report(command, diagnosis, expected or None),
        )

    # Mode 3: no target — show the environment and ask what to debug.
    return ChatMessage(
        role=Role.ASSISTANT,
        content=(
            format_environment(capture_environment())
            + "\n\nWhat would you like me to debug? Paste the error, or tell me "
            "the command to run (e.g. 'debug python main.py')."
        ),
    )


def handle_association(
    action: str,
    a: str = "",
    b: str = "",
    topic: str = "",
) -> ChatMessage:
    """
    Handles personalized-learning requests (cross-domain memory connections),
    gated by the security boundary like every other tool action. All four
    actions are read-only local operations (LOW risk) and auto-execute.
    """
    if action == "discover":
        verdict = check_action("discover_connections", "")
        if is_denied(verdict):
            return ChatMessage(role=Role.ASSISTANT, content=blocked_message(verdict))
        return ChatMessage(
            role=Role.ASSISTANT,
            content=execute_tool("discover_connections"),
        )

    if action == "relate":
        if not a or not b:
            return ChatMessage(
                role=Role.ASSISTANT,
                content=(
                    "Which two things should I relate? "
                    "e.g. 'how is renaissance art related to the medici'."
                ),
            )
        verdict = check_action("explain_relation", "", f"{a} {b}")
        if is_denied(verdict):
            return ChatMessage(role=Role.ASSISTANT, content=blocked_message(verdict))
        return ChatMessage(
            role=Role.ASSISTANT,
            content=execute_tool("explain_relation", a=a, b=b),
        )

    # connections
    verdict = check_action("memory_connections", topic or "")
    if is_denied(verdict):
        return ChatMessage(role=Role.ASSISTANT, content=blocked_message(verdict))
    return ChatMessage(
        role=Role.ASSISTANT,
        content=execute_tool("memory_connections", topic=topic),
    )


def handle_command(command: str) -> ChatMessage:
    """
    Runs a shell command, gated by the security boundary.

    Read-only commands (ls, cat, git status, pytest, …) are LOW risk and
    auto-execute; state-changing commands need interactive confirmation;
    dangerous patterns are escalated to CRITICAL and still require
    confirmation (the user keeps the final say).
    """
    blocked = _gate_command(command)
    if blocked is not None:
        return blocked
    content = f"Command execution requested: '{command}'"
    from ultron.core.tools.resource_monitor import forecast_warning
    warning = forecast_warning(command)
    if warning:
        content += f"\n\n[resources] ⚠ {warning}"
    return ChatMessage(
        role=Role.ASSISTANT,
        content=content,
        pending_action=PendingAction(
            action_type="run_command",
            target=command
        )
    )

def handle_parallel(commands: list[str]) -> ChatMessage:
    """
    Runs a batch of shell commands concurrently, gated command-by-command.

    The batch is only as safe as its most dangerous command, so every
    command is classified individually before anything runs:

      - any denial (credential in a command, unsafe pattern) hard-blocks the
        whole batch — nothing runs;
      - any command needing confirmation routes the whole batch through a
        single interactive PendingAction — the user approves the batch once,
        seeing every command;
      - when every command is auto-allowed, the batch executes immediately.
    """
    from ultron.core.tools.registry import get_tool

    # Normalize: a single "command" may carry embedded newlines (e.g. from an
    # LLM tool call). Each line is its own shell command, so gate and run them
    # per line — classification must match execution exactly.
    normalized: list[str] = []
    for cmd in commands:
        for line in str(cmd).splitlines():
            line = line.strip()
            if line:
                normalized.append(line)
    commands = normalized
    if not commands:
        return ChatMessage(
            role=Role.ASSISTANT,
            content="Error: no commands provided to run in parallel.",
        )

    blocked_reasons: list[str] = []
    needs_confirm = False
    for i, cmd in enumerate(commands):
        verdict = check_action("run_command", cmd)
        if is_denied(verdict):
            # Report the position and the reason, never the raw command text:
            # the guardrails just flagged it (e.g. it carries a credential),
            # so echoing it back would defeat the redaction stance used
            # everywhere else.
            blocked_reasons.append(
                f"  - command {i + 1}: {blocked_message(verdict)}"
            )
        elif is_confirm(verdict):
            needs_confirm = True

    if blocked_reasons:
        return ChatMessage(
            role=Role.ASSISTANT,
            content=(
                "Parallel execution blocked — one or more commands were denied "
                "by the security boundary:\n" + "\n".join(blocked_reasons)
            ),
        )

    # Resource awareness: a batch that would otherwise auto-run is escalated
    # to confirmation when any command is forecast-heavy/critical or the
    # batch is large (parallel execution multiplies CPU and memory).
    from ultron.core.tools.resource_monitor import forecast_severity

    heavy_cmds = [
        cmd for cmd in commands if forecast_severity(cmd) in {"heavy", "critical"}
    ]
    resource_note: str | None = None
    if heavy_cmds:
        resource_note = (
            "[resources] \u26a0 heavy commands in this batch: "
            + ", ".join(heavy_cmds)
            + " — running them together may spike CPU and memory."
        )
    elif len(commands) > 8:
        resource_note = (
            f"[resources] \u26a0 batch of {len(commands)} commands in parallel "
            "may spike CPU and memory."
        )

    if needs_confirm:
        listing = "\n".join(
            f"  {i + 1}. {cmd}" for i, cmd in enumerate(commands)
        )
        content = (
            f"Parallel execution requested ({len(commands)} commands "
            f"run simultaneously):\n{listing}"
        )
        if resource_note:
            content += f"\n\n{resource_note}"
        return ChatMessage(
            role=Role.ASSISTANT,
            content=content,
            pending_action=PendingAction(
                action_type="run_parallel",
                target="\n".join(commands),
            ),
        )

    if resource_note:
        # Permissive mode promises no prompts: run the batch and attach the
        # resource warning to the reply instead of escalating.
        if security_mode() == "permissive":
            func = get_tool("run_parallel")
            result = (
                func(commands)
                if func
                else "Error: Tool 'run_parallel' not found in registry."
            )
            return ChatMessage(
                role=Role.ASSISTANT,
                content=f"{result}\n\n{resource_note}",
            )
        listing = "\n".join(
            f"  {i + 1}. {cmd}" for i, cmd in enumerate(commands)
        )
        return ChatMessage(
            role=Role.ASSISTANT,
            content=(
                f"Parallel execution requested ({len(commands)} commands "
                f"run simultaneously):\n{listing}\n\n{resource_note}"
            ),
            pending_action=PendingAction(
                action_type="run_parallel",
                target="\n".join(commands),
            ),
        )

    func = get_tool("run_parallel")
    result = (
        func(commands)
        if func
        else "Error: Tool 'run_parallel' not found in registry."
    )
    return ChatMessage(role=Role.ASSISTANT, content=str(result))


async def handle_parallel_tools(
    user_input: str,
    engine,
    calls: list[dict] | None = None,
) -> ChatMessage:
    """
    Runs several *different* tools concurrently and returns one synthesized
    report (the inter-tool counterpart of ``handle_parallel``).

    ``calls`` is the deterministically-extracted batch when the detector
    produced one; otherwise the LLM planner (``plan_tool_batch``) turns the
    request into independent calls. Every call is gated through the security
    boundary inside ``run_tool_batch`` — deny verdicts never execute, confirm
    verdicts are surfaced as needing approval instead of running silently.
    """
    if calls is None:
        from ultron.core.intelligence.parallel_tools import plan_tool_batch

        calls = await plan_tool_batch(user_input, engine)
    if not calls:
        return ChatMessage(
            role=Role.ASSISTANT,
            content=(
                "I couldn't break that down into parallel tool calls. Try "
                "naming the sources explicitly, e.g. 'read config.json and "
                "notes.txt' or 'check example.com and the docs site'."
            ),
        )

    from ultron.core.intelligence.parallel_tools import run_tool_batch

    result = run_tool_batch(json.dumps(calls))
    return ChatMessage(role=Role.ASSISTANT, content=str(result))


def detect_retrieval_intent(user_input: str) -> dict | None:
    """
    Detects unified-retrieval requests: checking whether a site is online
    and/or reading the content of a URL.

    Returns {"request": ..., "url": ... | None} when the request is
    retrieval-shaped, otherwise None. Placed before the HTTP / web-search /
    fetch detectors so "is example.com online and read its headlines" is
    handled by the orchestrator as one unit instead of being partially
    matched per tool. Plain searches (no URL, no availability marker) and
    state-changing API calls keep their existing paths.
    """
    from ultron.core.tools.builtin.retrieval import (
        _CONNECTIVITY_MARKER,
        _CONTENT_MARKER,
        _DOMAIN_RE,
        extract_retrieval_url,
    )

    text = user_input.strip()
    url = extract_retrieval_url(text)
    if url is None and _CONNECTIVITY_MARKER.search(text):
        # Availability requests may name hosts with non-web TLDs (.local, .lan)
        # that the web-domain guard in extract_retrieval_url rejects — accept
        # any bare domain here since the request is explicitly about reachability.
        domain = _DOMAIN_RE.search(text)
        if domain:
            url = "https://" + domain.group(0).lower()
    if url is None:
        # No URL: only claim availability requests (which need the URL from a
        # follow-up clarification turn).
        if _CONNECTIVITY_MARKER.search(text):
            return {"request": text, "url": None}
        return None

    has_connectivity = bool(_CONNECTIVITY_MARKER.search(text))
    has_content = bool(_CONTENT_MARKER.search(text))
    if has_connectivity or has_content:
        return {"request": text, "url": url}
    return None


def handle_retrieve(request: str, url: str | None = None) -> ChatMessage:
    """
    Runs the unified retrieval orchestrator, gated by the security boundary.

    Retrieval is read-only (LOW risk) so it auto-executes in every mode; the
    guardrails deny unsafe URLs before any network request fires. When no URL
    can be found the orchestrator says so explicitly instead of guessing.
    """
    verdict = check_action("retrieve", url or "")
    if is_denied(verdict):
        return ChatMessage(role=Role.ASSISTANT, content=blocked_message(verdict))
    result = execute_tool("retrieve", request=request, url=url)
    return ChatMessage(role=Role.ASSISTANT, content=str(result))


def handle_api_schema(action: str, url: str | None = None) -> ChatMessage:
    """
    Handles API schema learning / inspection / reset requests, gated by the
    security boundary like every other tool action.

    - learn      -> fetch + mine the OpenAPI spec (read-only, LOW risk)
    - knowledge  -> summarize learned endpoints + detected drift
    - hints      -> usage prediction for a pending call
    - forget     -> clear everything learned about one API
    """
    if action == "knowledge" and not url:
        verdict = check_action("get_api_knowledge", "")
        if is_denied(verdict):
            return ChatMessage(role=Role.ASSISTANT, content=blocked_message(verdict))
        return ChatMessage(
            role=Role.ASSISTANT,
            content=execute_tool("get_api_knowledge", base_url=""),
        )

    if not url:
        return ChatMessage(
            role=Role.ASSISTANT,
            content=(
                "Which API? Send me its base URL "
                "(e.g. http://localhost:8000 or example.com)."
            ),
        )

    if action == "hints":
        verdict = check_action("api_usage_hint", url)
        if is_denied(verdict):
            return ChatMessage(role=Role.ASSISTANT, content=blocked_message(verdict))
        return ChatMessage(
            role=Role.ASSISTANT,
            content=execute_tool("api_usage_hint", method="GET", url=url),
        )

    tool_name = {
        "learn": "learn_api_schema",
        "knowledge": "get_api_knowledge",
        "forget": "forget_api",
    }.get(action)
    if tool_name is None:
        return ChatMessage(
            role=Role.ASSISTANT,
            content="Sorry, I didn't understand that schema request.",
        )
    verdict = check_action(tool_name, url)
    if is_denied(verdict):
        return ChatMessage(role=Role.ASSISTANT, content=blocked_message(verdict))
    return ChatMessage(
        role=Role.ASSISTANT,
        content=execute_tool(tool_name, base_url=url),
    )


def handle_http(method: str, url: str, body: str | None = None) -> ChatMessage:
    """
    Handles HTTP request intents, gated by the security boundary.

    Safety & Security Policy:
    1. The guardrails deny non-https / non-localhost URLs before anything runs.
    2. GET requests (auto-allowed, LOW risk) execute immediately.
    3. State-modifying requests (POST, PUT, DELETE, PATCH) are HIGH risk and
       emit a PendingAction confirmation — except in a permissive mode, where
       they auto-execute.
    """
    from ultron.core.tools.registry import get_tool

    method_upper = method.strip().upper()
    is_state_modifying = method_upper in {"POST", "PUT", "DELETE", "PATCH"}

    # Security gate: URL safety + method classification happen here.
    verdict = check_action("make_http_request", url, content=body or method)
    if is_denied(verdict):
        return ChatMessage(role=Role.ASSISTANT, content=blocked_message(verdict))

    # Permissive mode auto-allows state-changing requests.
    if is_state_modifying and is_allow(verdict):
        result = execute_tool("make_http_request", method=method_upper, url=url, body=body)
        return ChatMessage(role=Role.ASSISTANT, content=result)

    # Emit PendingAction confirmation prompt for state-modifying HTTP methods
    if is_state_modifying:
        details = f"HTTP {method_upper} {url}"
        if body:
            details += f"\nBody: {body}"

        return ChatMessage(
            role=Role.ASSISTANT,
            content=f"HTTP request requested:\n{details}",
            pending_action=PendingAction(
                action_type="run_command",
                target=f"http_request:{method_upper}:{url}" + (f":{body}" if body else "")
            )
        )

    # Read-only GET request — execute immediately via tool registry
    http_tool = get_tool("make_http_request")
    if not http_tool:
        result = "Error: Tool 'make_http_request' not found in registry."
    else:
        try:
            result = http_tool(method_upper, url, body)
        except (httpx.HTTPError, OSError, ValueError) as exc:
            result = f"Error: unexpected error during HTTP request ({exc})."

    return ChatMessage(role=Role.ASSISTANT, content=result)

async def handle_multistep(user_input: str, engine) -> ChatMessage:
    """
    Breaks a compound request into steps via the LLM planner, then runs the
    plan preflight: every step's permission, missing info, and dependencies
    are listed UP FRONT before anything executes.

    - Any blocked step → the plan is never offered (the preview explains why).
    - Any step needing approval or missing information → one approval prompt
      for the whole plan (steps JSON in the pending action).
    - All auto + complete → runs immediately with the preview as an intro.
    """
    steps = await plan_task(user_input, engine)

    if steps is None:
        # The LLM responded but its output wasn't valid JSON —
        # ask the user to simplify rather than silently falling through,
        # which could produce a confusing partial result.
        return ChatMessage(
            role=Role.ASSISTANT,
            content=(
                "I couldn't break that request into clear steps. "
                "Could you try rephrasing it more simply, or ask me "
                "to do one thing at a time?"
            )
        )

    from ultron.core.intelligence.planning import format_plan_preview, preflight_plan

    # Classify every step exactly once; the preview reuses the result.
    preflight = preflight_plan(steps)
    preview = format_plan_preview(steps, preflight)
    summary = preflight["summary"]

    # A security-blocked step (secret exfiltration, unsafe URL, path
    # escape) means the plan must not run at all — say why.
    if summary["blocked"] > 0:
        lines = [preview, "", "⛔ This plan contains blocked steps and will not run:"]
        for index, action, reason in preflight["blocked"]:
            lines.append(f"  • step {index} ({action}): {reason}")
        return ChatMessage(role=Role.ASSISTANT, content="\n".join(lines))

    # Any step needing approval (or missing information) → ask once, up
    # front, for the whole chain instead of prompting mid-execution. The
    # preview rides on the pending action so the CLI approval card shows it.
    if summary["confirm"] > 0 or summary["missing"] > 0:
        return ChatMessage(
            role=Role.ASSISTANT,
            content=preview,
            pending_action=PendingAction(
                action_type="execute_plan",
                target=json.dumps(steps),
                content=preview,
            ),
        )

    # Every step is auto-allowed and complete — run with the preview shown.
    step_results = await execute_plan(steps)
    return ChatMessage(
        role=Role.ASSISTANT,
        content=preview + "\n\n" + "\n".join(step_results),
    )

async def handle_llm_fallback(
    user_input: str,
    history: list[ChatMessage],
    engine,
) -> ChatMessage:
    """
    Fallback pipeline when deterministic regex detectors do not match plain-English queries.

    Injects the dynamic Tool Registry JSON Schema into the System Prompt.
    If the LLM emits a JSON tool call block:
      - Intercepts tool_name and arguments.
      - Checks state safety (e.g., write_file, run_command) and routes through PendingAction if required.
      - Executes the tool and returns the formatted response.
    If no tool call is emitted, returns standard conversational text.
    """
    from ultron.core.tools.registry import get_tool, get_tools_schema

    tools_schema = get_tools_schema()
    tools_json_str = json.dumps(tools_schema, indent=2)

    tool_instruction = (
        "You are ULTRON, an autonomous local AI assistant with access to tools.\n"
        "Available Tools:\n"
        f"{tools_json_str}\n\n"
        "INSTRUCTIONS:\n"
        "1. ONLY use a tool if the user explicitly requests a specific action that requires it "
        "(such as reading/writing a file, running a shell command, searching stored memories, or making an HTTP request).\n"
        "2. Do NOT call memory tools (like get_all_memories or search_memories) for general chit-chat, greetings, or conversational queries. Memory tools must ONLY be called when the user explicitly asks what is remembered or saved.\n"
        "3. When invoking a tool, respond ONLY with a JSON tool call block:\n"
        "```json\n"
        "{\n"
        '  "tool": "<tool_name>",\n'
        '  "arguments": { ... }\n'
        "}\n"
        "```\n"
        "Do NOT include any extra conversational text before or after the JSON block when calling a tool.\n"
        "4. If no tool is explicitly needed for the user's message, answer directly in natural conversational text.\n"
        f"{build_response_guidance()}"
    )

    # Structured output enforcement: when the user asked for a machine-
    # readable shape ("as JSON with fields …"), the exact schema is injected
    # here so the model knows the required shape before generating.
    from ultron.core.intelligence.structured_output import (
        enforce_reply,
        schema_prompt_block,
    )
    schema_block = schema_prompt_block(user_input)
    if schema_block:
        tool_instruction += "\n\n" + schema_block

    # Create a copy of the history list to avoid mutating original
    messages = list(history) if history else []
    messages.append(ChatMessage(role=Role.USER, content=user_input))

    if messages and messages[0].role == Role.SYSTEM:
        messages[0] = ChatMessage(
            role=Role.SYSTEM,
            content=f"{messages[0].content}\n\n{tool_instruction}"
        )
    else:
        messages.insert(0, ChatMessage(role=Role.SYSTEM, content=tool_instruction))

    openai_messages = history_to_openai_format(messages)
    response_content = await engine.generate(openai_messages)

    # Attempt to extract JSON tool call block
    tool_call_match = re.search(r'```(?:json)?\s*(\{\s*"tool":.*?\})\s*```', response_content, re.DOTALL | re.IGNORECASE)
    raw_json = tool_call_match.group(1) if tool_call_match else None

    if not raw_json and response_content.strip().startswith("{") and response_content.strip().endswith("}"):
        raw_json = response_content.strip()

    if raw_json:
        try:
            call_data = json.loads(raw_json)
            tool_name = call_data.get("tool")
            arguments = call_data.get("arguments", {})

            func = get_tool(tool_name)
            if func:
                # Security Check: every tool call is gated by the boundary
                # before execution — deny blocks, allow executes directly,
                # confirm routes through the PendingAction flow.
                if tool_name == "run_command":
                    cmd = arguments.get("command", "")
                    verdict = check_action("run_command", cmd)
                    if is_denied(verdict):
                        return ChatMessage(role=Role.ASSISTANT, content=blocked_message(verdict))
                    if is_allow(verdict):
                        return ChatMessage(
                            role=Role.ASSISTANT,
                            content=execute_tool("run_command", command=cmd),
                        )
                    return ChatMessage(
                        role=Role.ASSISTANT,
                        content=f"Command execution requested: '{cmd}'",
                        pending_action=PendingAction(
                            action_type="run_command",
                            target=cmd
                        )
                    )
                elif tool_name == "run_parallel":
                    cmds = [str(c) for c in arguments.get("commands", [])]
                    return handle_parallel(cmds)

                elif tool_name == "write_file":
                    fname = arguments.get("filename", "")
                    content = arguments.get("content", "")
                    return handle_file_write(fname, content, user_input=user_input)

                # Direct execution for read-only tools or non-destructive
                # operations — still gated so a denied action never runs.
                target, content = _generic_target_content(tool_name, arguments)
                verdict = check_action(tool_name, target, content)
                if is_denied(verdict):
                    return ChatMessage(role=Role.ASSISTANT, content=blocked_message(verdict))
                try:
                    tool_result = func(**arguments)
                except Exception as exc:  # noqa: BLE001 — arbitrary tool surface
                    tool_result = f"Error executing tool '{tool_name}': {exc}"

                return ChatMessage(
                    role=Role.ASSISTANT,
                    content=f"Executed tool '[bold {ACCENT}]{tool_name}[/bold {ACCENT}]':\n\n{tool_result}"
                )
        except (json.JSONDecodeError, IndexError, TypeError, ValueError) as exc:
            logger.debug(f"Failed to parse tool call JSON from LLM: {exc}")

    # Legacy TOOL_CALL: read_file: fallback compatibility — still gated by
    # the security boundary so a path-escape read can never execute here.
    if "TOOL_CALL: read_file:" in response_content:
        for line in response_content.splitlines():
            if "TOOL_CALL: read_file:" in line:
                file_path = line.split("TOOL_CALL: read_file:")[1].strip()
                verdict = check_action("read_file", file_path)
                if is_denied(verdict):
                    return ChatMessage(role=Role.ASSISTANT, content=blocked_message(verdict))
                read_file_func = get_tool("read_file")
                tool_result = read_file_func(file_path) if read_file_func else "Error: Tool 'read_file' not found."
                return ChatMessage(
                    role=Role.ASSISTANT,
                    content=f"Here are the contents of '{file_path}':\n\n{tool_result}"
                )

    # Standard natural text response — with structured-output enforcement:
    # when a schema was requested, the reply is validated + deterministically
    # repaired before it is shown ([structured] notes list every change).
    return ChatMessage(
        role=Role.ASSISTANT,
        content=polish_response(enforce_reply(user_input, response_content)),
    )


# ---------------------------------------------------------------------------
# AI Intent Classifier
# ---------------------------------------------------------------------------

async def classify_intent(user_input: str, engine) -> str:
    """
    Classifies user intent into a specific category using the AI engine.

    Design rationale:
    This two-stage approach (AI classifies broadly, code extracts precisely) avoids
    trusting the AI to accurately pull out exact values like filenames or URLs, which
    we know from earlier testing is unreliable — the AI is only asked the much easier
    question of "which general category," not "give me the precise details."
    """
    valid_categories = {
        "read_file",
        "write_file",
        "run_command",
        "http_request",
        "web_search",
        "fetch_page",
        "git",
        "run_tests",
        "lint",
        "remember",
        "memory_question",
        "none",
    }

    prompt = (
        "Classify the following user message into EXACTLY ONE of these categories:\n\n"
        "- read_file: user wants to view or read the contents of a local file, e.g. 'read main.py', 'show me config.json'\n"
        "- write_file: user wants to create, save, or write content to a file, e.g. 'write hello to test.txt', 'save this note in info.md'\n"
        "- run_command: user wants to execute a shell command, e.g. 'run ls', 'execute pwd'\n"
        "- http_request: user wants to send an HTTP API request (GET/POST/PUT/DELETE), e.g. 'post to http://localhost:8000'\n"
        "- web_search: user wants to search the web or google for online information, e.g. 'search for Python 3.12 release date', 'look up FastAPI docs'\n"
        "- fetch_page: user wants to fetch or read text from a specific web page URL, e.g. 'fetch this page https://example.com', 'read URL https://docs.python.org'\n"
        "- git: user wants to perform git operations like status, diff, log, or commit, e.g. 'check git status', 'show diff'\n"
        "- run_tests: user wants to run unit tests or pytest, e.g. 'test my code', 'run pytest'\n"
        "- lint: user wants to lint or check code quality, e.g. 'check for errors', 'run linter'\n"
        "- remember: user wants Ultron to store or remember a fact, e.g. 'remember that key is 123', 'please remember I like Python'\n"
        "- memory_question: user asks what facts or information Ultron remembers, or a reasoning question about stored knowledge, e.g. 'what do you know about databases', 'what is the capital of France', 'what did I tell you about FastAPI'\n"
        "- none: none of the above categories apply\n\n"
        f"User message: {user_input}\n\n"
        "Respond with ONLY the single category word, nothing else. If none of these clearly apply, respond with 'none'."
    )

    try:
        raw_response = await engine.generate([{"role": "user", "content": prompt}])
        category = raw_response.strip().lower()
        if category in valid_categories:
            return category
        return "none"
    except (httpx.HTTPError, OSError, ValueError):
        return "none"


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class SimpleAgent(BaseAgent):
    """
    A simple agent that passes user input and history directly to the engine.
    """

    def __init__(self, engine):
        super().__init__(engine)
        # Short-term "memory" for clarifying questions.
        # Stores {"category": <str>} when Ultron asks a clarifying question (e.g. "which file?").
        self._pending_clarification: dict | None = None

    async def run(self, user_input: str, history: list[ChatMessage] | None = None) -> ChatMessage:
        """
        Routes user input to the correct handler in priority order.

        Each detector is checked in order; the first one that matches handles
        the request and returns immediately.  This keeps the flow easy to scan
        and modify — add a new tool by adding one detect_*/handle_* pair and
        one check here.

        Design choice: rather than holding multi-turn state (like _pending_write
        or _pending_command) and matching text confirmation responses, we attach
        a `pending_action` payload to the ChatMessage.  This signals main.py to
        prompt the user immediately with an interactive questionary menu.
        """
        # Step -1: Check for pending clarification response from previous turn.
        # This creates a simple one-turn "memory" for clarifying questions, so replies
        # like "test.txt" are understood as directly answering "which file?" instead
        # of needing to independently match a full detection pattern on their own.
        if self._pending_clarification:
            pending = self._pending_clarification
            self._pending_clarification = None  # Clear immediately to avoid stale state

            cat = pending.get("category")
            if cat == "read_file":
                filename = detect_file_read_intent(user_input) or user_input.strip()
                if filename:
                    return handle_file_read(filename)

            elif cat == "write_file":
                write_match = detect_file_write_intent(user_input)
                if write_match:
                    return handle_file_write(write_match[0], write_match[1], user_input=user_input)
                # Fallback: treat user_input as filename with empty content
                target_file = user_input.strip()
                if target_file:
                    return handle_file_write(target_file, "", user_input=user_input)

            elif cat == "run_command":
                cmd = detect_command_intent(user_input) or user_input.strip()
                if cmd:
                    return handle_command(cmd)

            elif cat == "retrieve":
                from ultron.core.tools.builtin.retrieval import extract_retrieval_url
                url = extract_retrieval_url(user_input)
                if url:
                    return handle_retrieve(user_input, url)

            elif cat == "api_schema":
                action = pending.get("action", "learn")
                url = extract_api_url(user_input)
                if url:
                    return handle_api_schema(action, url)

        # Clear any leftover pending state if execution reached here
        self._pending_clarification = None

        # Step 0: compound / multi-step request — must run FIRST so that
        #         "write X then read Y" is handled as a unit and not partially
        #         matched by a single-action detector below.
        if detect_multistep_intent(user_input):
            return await handle_multistep(user_input, self.engine)

        # Step 0.5: greeting intent — fast-path conversational response bypassing tools/LLM
        if detect_greeting_intent(user_input):
            return handle_greeting()

        # Step 0.9: image analysis — before file reads so "analyze chart.png"
        #           routes to the vision model, not the raw-file reader.
        image_path = detect_image_intent(user_input)
        if image_path:
            return await handle_image(image_path, user_input, self.engine)

        # Step 0.95: inter-tool parallel batch — several independent sources
        #           at once ("read X and Y", "check A and B and C",
        #           "search for P and Q"). Runs BEFORE the single-target
        #           detectors so the request is captured as one parallel unit
        #           instead of being claimed by the first single match. The
        #           command-parallel path (detect_parallel_intent) is exempt
        #           inside the detector itself.
        batch_req = detect_tool_batch_intent(user_input)
        if batch_req:
            if batch_req.get("calls"):
                return await handle_parallel_tools(user_input, self.engine, calls=batch_req["calls"])
            return await handle_parallel_tools(user_input, self.engine)

        # Step 1: file-read intent
        filename = detect_file_read_intent(user_input)
        if filename:
            return handle_file_read(filename)

        # Step 2: file-write intent
        write_match = detect_file_write_intent(user_input)
        if write_match:
            return handle_file_write(write_match[0], write_match[1], user_input=user_input)

        # Step 3: store a memory fact
        fact = detect_remember_intent(user_input)
        if fact:
            return handle_remember(fact)

        # Step 3.4: knowledge-graph reasoning — answerable by deterministic
        #           graph traversal (query_chain), so it runs before recall.
        deduction_question = detect_deduction_question(user_input)
        if deduction_question:
            return handle_deduction_question(deduction_question)

        # Step 3.5: recall stored facts about a topic — handled in code (not AI)
        #           so only matching facts are shown and hallucination is impossible.
        topic = detect_memory_question(user_input)
        if topic:
            return handle_memory_question(topic)

        # Step 3.55: personalized learning — cross-domain memory connections
        #           ("what connections do you see", "how is X related to Y").
        association = detect_association_intent(user_input)
        if association:
            return handle_association(
                association["action"],
                a=association.get("a", ""),
                b=association.get("b", ""),
                topic=association.get("topic", ""),
            )

        # Step 3.7: parallel command execution — before the single-command
        #           detectors so an explicit "in parallel" request wins over
        #           partial matches of its individual commands.
        parallel_commands = detect_parallel_intent(user_input)
        if parallel_commands:
            return handle_parallel(parallel_commands)

        # Step 4: run tests
        if detect_test_intent(user_input):
            return handle_test()

        # Step 4.5: git operations — before generic command so "show diff"
        #           maps to the correct git command rather than the shell.
        git_cmd = detect_git_intent(user_input)
        if git_cmd:
            return handle_git(git_cmd)

        # Step 4.75: lint / code-check — before generic command so "check my
        #            code" maps to ruff rather than failing in the shell.
        if detect_lint_intent(user_input):
            return handle_lint()

        # Step 3.45: API schema inference — learn / inspect / forget API
        #           schemas. Before the memory-question detector so "what do
        #           you know about the api at X" routes to schema knowledge,
        #           and before the HTTP detector so "learn the schema for
        #           http://..." is never mistaken for a plain GET.
        schema_req = detect_api_schema_intent(user_input)
        if schema_req:
            action, url = schema_req["action"], schema_req["url"]
            if url or action == "knowledge":
                return handle_api_schema(action, url)
            self._pending_clarification = {"category": "api_schema", "action": action}
            return ChatMessage(
                role=Role.ASSISTANT,
                content=(
                    "Which API? Send me its base URL "
                    "(e.g. http://localhost:8000 or example.com)."
                ),
            )

        # Step 4.8: resource monitoring — system snapshots and command
        #           forecasts ("how is my system", "will this be heavy").
        resource_req = detect_resource_intent(user_input)
        if resource_req:
            return handle_resource(resource_req["action"], resource_req.get("command"))

        # Step 4.84: unified retrieval — one entry point for "is X online",
        #           "read X", or a combination; it plans the best networking
        #           strategy (connectivity / fetch / search / both) instead of
        #           guessing between separate web tools.
        retrieval = detect_retrieval_intent(user_input)
        if retrieval:
            if retrieval["url"]:
                return handle_retrieve(retrieval["request"], retrieval["url"])
            self._pending_clarification = {"category": "retrieve"}
            return ChatMessage(
                role=Role.ASSISTANT,
                content="Which website should I check? Send me the URL (or a domain like example.com).",
            )

        # Step 4.85: HTTP intent — before generic command detector
        http_match = detect_http_intent(user_input)
        if http_match:
            return handle_http(*http_match)

        # Step 4.88: Web search intent — before generic command detector.
        # Web searches are LOW risk and auto-allowed, so they execute directly
        # (the guardrails deny credential-like queries).
        search_query = detect_web_search_intent(user_input)
        if search_query:
            verdict = check_action("web_search", search_query)
            if is_denied(verdict):
                return ChatMessage(role=Role.ASSISTANT, content=blocked_message(verdict))
            return ChatMessage(
                role=Role.ASSISTANT,
                content=execute_tool("search_web", query=search_query),
            )

        # Step 4.9: Fetch web page intent — before generic command detector.
        # Page fetches are LOW risk and auto-allowed; unsafe URLs (non-https /
        # non-localhost) are denied by the guardrails before any request fires.
        fetch_url = detect_fetch_page_intent(user_input)
        if fetch_url:
            verdict = check_action("fetch_page_text", fetch_url)
            if is_denied(verdict):
                return ChatMessage(role=Role.ASSISTANT, content=blocked_message(verdict))
            return ChatMessage(
                role=Role.ASSISTANT,
                content=execute_tool("fetch_page_text", url=fetch_url),
            )

        # Step 4.95: Database query intent — before generic command detector.
        # The boundary verdict decides: read-only SQL auto-executes, destructive
        # or state-changing SQL is confirmed, and anything denied never runs.
        db_sql = detect_db_query_intent(user_input)
        if db_sql:
            from ultron.core.tools.builtin.database import run_query
            verdict = check_action("run_query", db_sql)
            if is_denied(verdict):
                return ChatMessage(role=Role.ASSISTANT, content=blocked_message(verdict))
            if is_confirm(verdict):
                return ChatMessage(
                    role=Role.ASSISTANT,
                    content=f"Database query execution requested: '{db_sql}'",
                    pending_action=PendingAction(
                        action_type="db_query",
                        target=db_sql
                    )
                )
            return ChatMessage(role=Role.ASSISTANT, content=run_query(db_sql))

        # Step 4.97: environmental-state debugging — "debug this",
        #            "why is my script failing", "diagnose this error: …".
        #            Runs before the generic command detector so debug
        #            phrasing produces a diagnosis + environment report
        #            instead of being passed to the shell verbatim.
        debug_req = detect_debug_intent(user_input)
        if debug_req:
            return handle_debug(
                debug_req.get("command", ""),
                error=debug_req.get("error", ""),
                expected=debug_req.get("expected", ""),
            )

        # Step 5: generic shell command ("run X" / "execute X")
        command = detect_command_intent(user_input)
        if command:
            return handle_command(command)

        # Step 5.5: AI Intent Classification Fallback
        # This two-stage approach (AI classifies broadly, code extracts precisely) avoids
        # trusting the AI to accurately pull out exact values like filenames or URLs, which
        # we know from earlier testing is unreliable — the AI is only asked the much easier
        # question of "which general category," not "give me the precise details."
        category = await classify_intent(user_input, self.engine)

        if category != "none":
            if category == "read_file":
                re_filename = detect_file_read_intent(user_input) or extract_any_filename(user_input)
                if re_filename:
                    return handle_file_read(re_filename)
                self._pending_clarification = {"category": category}
                return ChatMessage(
                    role=Role.ASSISTANT,
                    content="It sounds like you want to read a file — which file?"
                )

            elif category == "write_file":
                re_write_match = detect_file_write_intent(user_input)
                if re_write_match:
                    return handle_file_write(re_write_match[0], re_write_match[1], user_input=user_input)
                
                # Try extracting any filename from text
                extracted_fname = extract_any_filename(user_input)
                if extracted_fname:
                    self._pending_clarification = {"category": category}
                    return ChatMessage(
                        role=Role.ASSISTANT,
                        content=f"I found the filename '{extracted_fname}' — what content should I write to it?"
                    )

                self._pending_clarification = {"category": category}
                return ChatMessage(
                    role=Role.ASSISTANT,
                    content="It sounds like you want to write a file — which file and what content?"
                )

            elif category == "run_command":
                re_command = detect_command_intent(user_input)
                if re_command:
                    return handle_command(re_command)
                self._pending_clarification = {"category": category}
                return ChatMessage(
                    role=Role.ASSISTANT,
                    content="It sounds like you want to run a command — which command?"
                )

            elif category == "http_request":
                re_http_match = detect_http_intent(user_input)
                if re_http_match:
                    return handle_http(*re_http_match)

            elif category == "web_search":
                query = detect_web_search_intent(user_input) or user_input.strip()
                if query:
                    return ChatMessage(
                        role=Role.ASSISTANT,
                        content=f"Web search requested: '{query}'",
                        pending_action=PendingAction(
                            action_type="web_search",
                            target=query
                        )
                    )

            elif category == "fetch_page":
                url = detect_fetch_page_intent(user_input)
                if not url:
                    # Extract any http:// or https:// URL if present
                    url_match = re.search(r'https?://[^\s]+', user_input, re.IGNORECASE)
                    if url_match:
                        raw_url = url_match.group(0)
                        url = re.sub(r'[\.,;\)]+$', '', raw_url).strip()
                if url:
                    return ChatMessage(
                        role=Role.ASSISTANT,
                        content=f"Web page fetch requested: '{url}'",
                        pending_action=PendingAction(
                            action_type="fetch_page",
                            target=url
                        )
                    )
                return ChatMessage(
                    role=Role.ASSISTANT,
                    content="It sounds like you want to fetch a web page — which URL?"
                )

            elif category == "git":
                re_git_cmd = detect_git_intent(user_input)
                if re_git_cmd:
                    return handle_git(re_git_cmd)

            elif category == "run_tests":
                return handle_test()

            elif category == "lint":
                return handle_lint()

            elif category == "remember":
                re_fact = detect_remember_intent(user_input)
                if re_fact:
                    return handle_remember(re_fact)

            elif category == "memory_question":
                # Reasoning questions get the deterministic graph answer first;
                # otherwise fall back to topic recall across both stores.
                re_deduction = detect_deduction_question(user_input)
                if re_deduction:
                    return handle_deduction_question(re_deduction)
                re_topic = detect_memory_question(user_input)
                if re_topic:
                    return handle_memory_question(re_topic)

        # Step 6: nothing matched — fall through to the LLM
        return await handle_llm_fallback(user_input, history or [], self.engine)


