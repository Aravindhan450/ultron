import re
import json
import asyncio
from ultron.core.agents.base import BaseAgent
from ultron.core.agents.security import (
    blocked_message,
    check_action,
    is_allow,
    is_confirm,
    is_denied,
    security_mode,
)
from ultron.core.logging import get_logger
from ultron.core.types import ChatMessage, Role, PendingAction, history_to_openai_format

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
    except Exception as exc:
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
        return ChatMessage(role=Role.ASSISTANT, content=execute_tool("run_command", command=command))
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
            except Exception:
                body = body_candidate

    # Clean wrapping quotes if body was specified as '...' or "..."
    if body:
        if (body.startswith('"') and body.endswith('"')) or (body.startswith("'") and body.endswith("'")):
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
        "Only use these action types: read_file, write_file, run_command, add_memory.\n"
        "Respond with ONLY a JSON array — no other text, no markdown fences.\n"
        "Use exactly these formats for each action type:\n"
        '  {"action": "read_file",   "filename": "..."}\n'
        '  {"action": "write_file",  "filename": "...", "content": "..."}\n'
        '  {"action": "run_command", "command": "..."}\n'
        '  {"action": "add_memory",  "fact": "..."}\n'
        "\n"
        f"User request: {user_input}"
    )

    try:
        raw = await engine.generate([{"role": "user", "content": planning_prompt}])
    except Exception:
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
    valid_actions = {"read_file", "write_file", "run_command", "add_memory"}
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
        if exit_code_match and exit_code_match.group(1) != "0":
            return True
        return False

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
    import time
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

                elif action == "add_memory":
                    func = get_tool("add_memory")
                    if not func:
                        result = "Error: Tool 'add_memory' not found in registry."
                    else:
                        result = func(step["fact"])

                else:
                    result = f"Error: unknown action type '{action}'"

            except Exception as exc:
                result = f"Error: {exc}"

            last_result = str(result)

            # If step succeeded (not a failure), break retry loop
            if not is_step_failure(action, last_result):
                break

            # If retrying, pause 1 second before the next attempt
            if attempt < max_attempts:
                time.sleep(1)

        # Format output entry based on success/failure and attempt count
        if is_step_failure(action, last_result):
            if max_attempts > 1:
                entry = f"{label}: FAILED after {attempts_used} attempts. Last error: {last_result}"
            else:
                entry = f"{label}: {last_result}"
            results.append(entry)
            # Stop on first error — don't execute further steps.
            break
        else:
            if attempts_used > 1:
                entry = f"{label}: [succeeded after {attempts_used} attempts] {last_result}"
            else:
                entry = f"{label}: {last_result}"
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
    Stores a new fact in the SQLite memory store, gated by the security
    boundary (which blocks outgoing content carrying credential-like data).
    """
    verdict = check_action("add_memory", "", fact)
    if is_denied(verdict):
        return ChatMessage(role=Role.ASSISTANT, content=blocked_message(verdict))
    from ultron.core.tools.registry import get_tool
    func = get_tool("add_memory")
    tool_result = func(fact) if func else "Error: Tool 'add_memory' not found in registry."
    return ChatMessage(role=Role.ASSISTANT, content=str(tool_result))

def handle_memory_question(topic: str) -> ChatMessage:
    """
    Answers a memory-recall question by fetching only facts that match the
    topic keyword — never all memories.  Built directly from DB rows so the
    AI is never involved and cannot hallucinate stored facts.
    """
    verdict = check_action("search_memories", topic)
    if is_denied(verdict):
        return ChatMessage(role=Role.ASSISTANT, content=blocked_message(verdict))
    from ultron.core.tools.registry import get_tool
    func = get_tool("search_memories")
    matches = func(topic) if func else []

    if matches:
        bullet_list = "\n".join(f"- {fact}" for fact in matches)
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
    return ChatMessage(
        role=Role.ASSISTANT,
        content=f"Command execution requested: '{command}'",
        pending_action=PendingAction(
            action_type="run_command",
            target=command
        )
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
        except Exception as exc:
            result = f"Error: unexpected error during HTTP request ({exc})."

    return ChatMessage(role=Role.ASSISTANT, content=result)

async def handle_multistep(user_input: str, engine) -> ChatMessage:
    """
    Breaks a compound request into steps via the LLM planner, then executes
    them in order.  Returns a clear step-by-step result summary.
    If planning fails (unparseable JSON), returns a friendly fallback message.
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

    # Run each step in order, stopping immediately on any error.
    step_results = await execute_plan(steps)
    return ChatMessage(role=Role.ASSISTANT, content="\n".join(step_results))

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
        "4. If no tool is explicitly needed for the user's message, answer directly in natural conversational text."
    )

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
                except Exception as exc:
                    tool_result = f"Error executing tool '{tool_name}': {exc}"

                return ChatMessage(
                    role=Role.ASSISTANT,
                    content=f"Executed tool '[bold cyan]{tool_name}[/bold cyan]':\n\n{tool_result}"
                )
        except Exception as exc:
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

    # Standard natural text response
    return ChatMessage(role=Role.ASSISTANT, content=response_content)


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
        "- memory_question: user asks what facts or information Ultron remembers, e.g. 'what do you know about databases', 'what did I tell you about FastAPI'\n"
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
    except Exception:
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

        # Step 3.5: recall stored facts about a topic — handled in code (not AI)
        #           so only matching facts are shown and hallucination is impossible.
        topic = detect_memory_question(user_input)
        if topic:
            return handle_memory_question(topic)

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
                re_topic = detect_memory_question(user_input)
                if re_topic:
                    return handle_memory_question(re_topic)

        # Step 6: nothing matched — fall through to the LLM
        return await handle_llm_fallback(user_input, history or [], self.engine)


