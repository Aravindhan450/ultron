import re
import json
import asyncio
from ultron.core.agents.base import BaseAgent
from ultron.core.logging import get_logger
from ultron.core.types import ChatMessage, Role, PendingAction, history_to_openai_format

logger = get_logger("ultron.agents.simple")

# ---------------------------------------------------------------------------
# Detectors
# Each function takes raw user_input and returns a matched value (filename,
# command string, fact, topic, bool) or None/False if no match.
# ---------------------------------------------------------------------------

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

    Additional phrasings can be added to the pattern list below in the future.
    """
    filename_pattern = r'[\w./-]+\.[a-zA-Z0-9]+'

    patterns = [
        # Match: "write <content> to <filename>"
        rf'^\s*write\s+(?P<content>.+?)\s+to\s+(?P<filename>{filename_pattern})\s*$',
        # Match: "create a file <filename> with <content>" or "create file <filename> with <content>"
        rf'^\s*create\s+(?:a\s+)?file\s+(?P<filename>{filename_pattern})\s+with\s+(?P<content>.+?)\s*$',
        # Match: "save <content> to <filename>"
        rf'^\s*save\s+(?P<content>.+?)\s+to\s+(?P<filename>{filename_pattern})\s*$',
    ]

    for pattern in patterns:
        match = re.search(pattern, user_input, re.IGNORECASE | re.DOTALL)
        if match:
            filename = match.group("filename").strip()
            content = match.group("content").strip()
            # Remove wrapping quotes around content if present (e.g., 'hello' or "hello")
            if (content.startswith('"') and content.endswith('"')) or (content.startswith("'") and content.endswith("'")):
                content = content[1:-1]
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


async def execute_plan(steps: list[dict]) -> list[str]:
    """
    Executes a list of structured action steps produced by plan_task().

    Design decisions:
    - We stop on the FIRST error rather than continuing with subsequent steps.
      This is an intentional predictability guarantee: if step 2 failed, the
      user can be certain step 3 never touched anything.  Continuing past
      errors would leave the workspace in an unpredictable intermediate state.
    - Each individual tool call is wrapped in try/except so a Python exception
      (e.g. from a broken tool) is caught and recorded as an "Error: ..." entry
      rather than crashing the whole plan.
    - write_file is called with overwrite=True because the user already committed
      to this plan by making the request — we don't need a second confirmation.
    """
    from ultron.core.tools.registry import get_tool

    results: list[str] = []

    for i, step in enumerate(steps, start=1):
        action = step.get("action", "")
        label = f"Step {i} ({action})"

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

            elif action == "add_memory":
                func = get_tool("add_memory")
                if not func:
                    result = "Error: Tool 'add_memory' not found in registry."
                else:
                    result = func(step["fact"])

            else:
                result = f"Error: unknown action type '{action}'"

        except Exception as exc:
            # Catch any unexpected exception from the tool itself and record it
            # so the rest of our reporting logic can treat it uniformly.
            result = f"Error: {exc}"

        entry = f"{label}: {result}"
        results.append(entry)

        # Stop on first error — don't execute further steps.
        # The user should be able to trust that if this step failed,
        # nothing after it was touched.
        if str(result).startswith("Error"):
            break

    return results


# ---------------------------------------------------------------------------
# Handlers
# Each function contains exactly the logic that was previously inline inside
# run() for the matching detector.  Keeping them here makes run() a clean
# index of "what Ultron can do" rather than a wall of nested if-blocks.
# ---------------------------------------------------------------------------

def handle_file_read(filename: str) -> ChatMessage:
    """
    Signals main.py to show an interactive confirmation before reading a file.
    Reading a file exposes its contents, so it follows the same confirmation
    pattern as running commands — the user sees what will happen and can
    allow or cancel before anything is executed.
    """
    return ChatMessage(
        role=Role.ASSISTANT,
        content=f"File read requested: '{filename}'",
        pending_action=PendingAction(
            action_type="read_file",
            target=filename
        )
    )

def handle_file_write(filename: str, content: str) -> ChatMessage:
    """
    Signals main.py to show an interactive confirmation before writing a file.

    Two sub-cases, both go through confirmation:
      - New file  → action_type="write_file"   (handled in main.py, overwrite=False)
      - Existing  → action_type="overwrite_file" (existing flow, unchanged)

    We detect which case applies by attempting the write with overwrite=False
    first.  If it reports "already exists", we switch to the overwrite flow.
    This keeps the single-responsibility boundary intact: simple.py decides
    WHAT to do, main.py handles HOW to confirm and execute it.
    """
    from ultron.core.tools.registry import get_tool
    func = get_tool("write_file")

    # Probe whether the file already exists without writing anything.
    # write_file(..., overwrite=False) returns an error string if the file
    # exists, so we can branch on that without a separate filesystem check.
    probe = func(filename, content, overwrite=False) if func else ""

    if "already exists" in str(probe):
        # File exists — send to the existing overwrite-confirmation flow
        return ChatMessage(
            role=Role.ASSISTANT,
            content=f"File '{filename}' already exists and requires confirmation to overwrite.",
            pending_action=PendingAction(
                action_type="overwrite_file",
                target=filename,
                content=content
            )
        )

    # New file — signal main.py to confirm before creating it.
    # The file has NOT been written yet; main.py will call write_file on "Yes".
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
    """Stores a new fact in the SQLite memory store and confirms to the user."""
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
    """Signals main.py to confirm and run pytest -v via the run_command tool."""
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
    Signals main.py to confirm and run a git command via the run_command tool.
    Zero new infrastructure — the same PendingAction / confirmation flow used
    for all other shell commands.
    """
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
    Signals main.py to confirm and run ruff in concise mode.
    --output-format=concise produces one issue-per-line output that
    summarize_lint_output() in main.py can parse reliably.
    """
    ruff_cmd = "ruff check . --output-format=concise"
    return ChatMessage(
        role=Role.ASSISTANT,
        content=f"Lint check requested: '{ruff_cmd}'",
        pending_action=PendingAction(
            action_type="run_command",
            target=ruff_cmd
        )
    )

def handle_command(command: str) -> ChatMessage:
    """Signals main.py to confirm and run an arbitrary shell command."""
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
    Handles HTTP request intents.

    Safety & Security Policy:
    1. Restrict unencrypted 'http://' to localhost / 127.0.0.1 only. Enforce 'https://' for external hosts.
    2. Read-only GET requests (to localhost or external HTTPS endpoints) bypass interactive confirmation
       and execute immediately.
    3. State-modifying requests (POST, PUT, DELETE, PATCH) emit a PendingAction interactive confirmation request.
    """
    from ultron.core.tools.registry import get_tool

    # Security check: allow http:// ONLY for localhost / 127.0.0.1; require https:// for all external URLs.
    is_localhost = url.startswith("http://localhost") or url.startswith("http://127.0.0.1")
    is_https = url.startswith("https://")

    if not (is_localhost or is_https):
        return ChatMessage(
            role=Role.ASSISTANT,
            content="Error: only localhost or https URLs are allowed."
        )

    method_upper = method.strip().upper()
    is_state_modifying = method_upper in {"POST", "PUT", "DELETE", "PATCH"}

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
        "When the user asks you to perform an action that matches one of your available tools, "
        "respond ONLY with a JSON tool call block matching this structure:\n"
        "```json\n"
        "{\n"
        '  "tool": "<tool_name>",\n'
        '  "arguments": { ... }\n'
        "}\n"
        "```\n"
        "Do NOT include any extra conversational text before or after the JSON block when calling a tool.\n"
        "If no tool is needed for the user's message, answer directly in natural conversational text."
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
                # Security Check: Route state-modifying actions through PendingAction flow
                if tool_name == "run_command":
                    cmd = arguments.get("command", "")
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
                    return ChatMessage(
                        role=Role.ASSISTANT,
                        content=f"File create requested: '{fname}'",
                        pending_action=PendingAction(
                            action_type="write_file",
                            target=fname,
                            content=content
                        )
                    )

                # Direct execution for read-only tools or non-destructive operations
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

    # Legacy TOOL_CALL: read_file: fallback compatibility
    if "TOOL_CALL: read_file:" in response_content:
        for line in response_content.splitlines():
            if "TOOL_CALL: read_file:" in line:
                file_path = line.split("TOOL_CALL: read_file:")[1].strip()
                read_file_func = get_tool("read_file")
                tool_result = read_file_func(file_path) if read_file_func else "Error: Tool 'read_file' not found."
                return ChatMessage(
                    role=Role.ASSISTANT,
                    content=f"Here are the contents of '{file_path}':\n\n{tool_result}"
                )

    # Standard natural text response
    return ChatMessage(role=Role.ASSISTANT, content=response_content)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class SimpleAgent(BaseAgent):
    """
    A simple agent that passes user input and history directly to the engine.
    """

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
        # Step 0: compound / multi-step request — must run FIRST so that
        #         "write X then read Y" is handled as a unit and not partially
        #         matched by a single-action detector below.
        if detect_multistep_intent(user_input):
            return await handle_multistep(user_input, self.engine)

        # Step 1: file-read intent
        filename = detect_file_read_intent(user_input)
        if filename:
            return handle_file_read(filename)

        # Step 2: file-write intent
        write_match = detect_file_write_intent(user_input)
        if write_match:
            return handle_file_write(*write_match)

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

        # Step 5: generic shell command ("run X" / "execute X")
        command = detect_command_intent(user_input)
        if command:
            return handle_command(command)

        # Step 6: nothing matched — fall through to the LLM
        return await handle_llm_fallback(user_input, history or [], self.engine)
