"""ultron.core.intelligence.task_classification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GOAL UNDERSTANDING + TASK CLASSIFICATION — the first layer of the general
task pipeline:

    USER REQUEST -> GOAL UNDERSTANDING -> TASK CLASSIFICATION -> PLANNING
    -> PLAN VALIDATION -> EXECUTION -> TOOLS

It answers *what the user wants to accomplish* (task type + desired
outcome), never *which tool to call*.  The deterministic fast path covers
common phrasings cheaply and without an LLM; an optional LLM pass handles
ambiguous requests and extracts a proper outcome-oriented goal.

The classifier is pure — it performs no tool calls, executes nothing, and
mutates no state.  Tools never decide the user's goal here.
"""

from __future__ import annotations

import json
import re

from ultron.core.types import TaskClassification, TaskType

# ---------------------------------------------------------------------------
# Deterministic classification rules
# ---------------------------------------------------------------------------
# Ordered from most specific to most generic.  The first rule that fires
# wins.  Markers are deliberately conservative: a clear miss falls through
# to the LLM classifier instead of guessing.

_DEBUG_VERBS_RE = re.compile(
    r"\b(debug\w*|diagnos\w*|investigat\w*|troubleshoot\w*|reproduc\w*)\b"
)
_FIX_VERBS_RE = re.compile(r"\b(fix\w*|repair\w*|patch\w*|resolve\w*)\b")
_FAILURE_NOUNS_RE = re.compile(
    r"\b(bug\w*|broken|failing|failure|crash\w*|error\w*|flaky|exception\w*|"
    r"traceback|stack\s*trace|failed|not\s+working|doesn't\s+work|"
    r"does\s+not\s+work)\b"
)
_WHY_RE = re.compile(r"\bwhy\b")

_CODE_REVIEW_RE = re.compile(
    r"\b(review|audit)\b[^.!?]*(code|repo|repositor|codebase|project|security|"
    r"vulnerab|dependenc)|"
    r"\bcheck\s+(the\s+)?(code|repo|repositor|codebase|project)\s+for"
)

_RESEARCH_RE = re.compile(
    r"\b(analy\w*|understand\w*|research\w*|summariz\w*|explain\w*|"
    r"investigat\w*)\b|"
    r"\bhow\s+(do|does|is|are)\b|"
    r"\breferences\s+to\b"
)
_CODE_NOUNS_RE = re.compile(
    r"\b(repo\w*|repositor\w*|codebase|project|code|source|architecture|"
    r"service|module|function|component|application|app\b|api\b|"
    r"implementation|auth\w*|login\w*)\b"
)

_SYSTEM_RE = re.compile(
    r"\b(deploy\w*|launch\w*|publish\w*|rollout\w*)\b|"
    r"\b(start|stop|restart|reboot)\s+(the\s+)?"
    r"(server|service|daemon|app|application|redis|postgres|docker|container|nginx)|"
    r"\b(redis|postgres|docker|container|nginx|kubernetes|k8s|systemctl|"
    r"daemon|cron|vm\b|cluster|cloud|aws\b)\b"
)

_DATA_RE = re.compile(
    r"\b(database\w*|sql\b|query\w*|schema\w*|migration\w*|table\w*|record\w*|"
    r"row\w*|postgres|mysql|sqlite|mongodb|dataset|data\b)\b"
)

_SE_VERBS_RE = re.compile(
    r"\b(create|build|implement|develop|write|make|add|refactor\w*|upgrade\w*|"
    r"update|migrat\w*|convert|port\w*|rewrite|generate|scaffold|"
    r"set\s+up|fix\w*|rename)\b"
)
_SE_NOUNS_RE = re.compile(
    r"\b(app\b|application\w*|backend\w*|frontend\w*|api\b|service\w*|project\w*|"
    r"feature\w*|module\w*|component\w*|plugin\w*|library\w*|package\w*|cli\b|"
    r"tool\w*|website\w*|dashboard\w*|endpoint\w*|function\w*|class\w*|"
    r"microservice\w*|server\w*|documentation\b|docs\b|dependenc\w*|"
    r"auth\w*|login\w*|oauth|react|fastapi|django|flask|node\b|express|spring\b|"
    r"rails|angular|vue\b|graphql|rest\b|dockerfile)\b"
)

_ACTION_VERBS = {
    "create", "make", "write", "read", "find", "count", "save", "list", "add",
    "remove", "update", "delete", "copy", "move", "run", "start", "stop",
    "install", "configure", "set", "fix", "test", "build", "refactor", "search",
    "print", "download", "upload", "rename", "show", "get", "open", "close",
    "generate", "deploy", "convert", "extract", "merge", "split", "sort",
}
_SEQUENCE_MARKERS_RE = re.compile(
    r"\b(then|next|after\s+that|afterwards|subsequently|finally|first|"
    r"and\s+then|followed\s+by|after\s+which|in\s+parallel)\b"
)

_CONFIG_RE = re.compile(
    r"\b(configur\w*|set\s+up|settings|environment\s+variable\w*|\.env\b|"
    r"\benv\b|config\w*)\b"
)

_FILE_RE = re.compile(
    r"\b(file\w*|director\w*|folder\w*|mkdir|filename\w*)\b|"
    r"\b\w+\.(py|js|ts|jsx|tsx|txt|md|json|yaml|yml|toml|ini|cfg|html|css|"
    r"java|go|rs|cpp|rb|php|sql|sh|env|lock)\b"
)

_INFORMATIONAL_RE = re.compile(
    r"\b(what\s+is|what\s+are|what's|whats|who\s+is|who\s+are|when\s+did|"
    r"when\s+does|when\s+is|why\s+is|why\s+are|explain\b|define\b|tell\s+me|"
    r"describe\b|meaning\s+of|difference\s+between|list\s+of)\b"
)

_POLITE_PREFIX_RE = re.compile(
    r"^(please|can you|could you|would you|hey|hi|hello)" r"[,!\s]*"
    r"|^i (want|need|would like|'d like)( you)?( to)?" r"[,!\s]*",
    flags=re.IGNORECASE,
)

_SIMPLE_ACTION_RE = re.compile(
    r"\b(create|make|list|show|print|get|read|write|save|run|start|stop|"
    r"delete|remove|copy|move|rename|open|close|install|search|download|"
    r"upload|generate|find|count)\b"
)

_VAGUE_DEPLOY_RE = re.compile(
    r"\b(deploy|launch|publish|rollout)\s+(this|it|the\s+(app|application|"
    r"service|project))\.?$"
)


def _normalize(text: str) -> str:
    """Lowercases and collapses whitespace for marker matching."""
    return re.sub(r"\s+", " ", text.strip().lower())


def _count_action_verbs(text: str) -> int:
    """Counts distinct action verbs in a request (used for multi-step)."""
    return len(set(re.findall(r"[a-z']+", text)) & _ACTION_VERBS)


def _classify_deterministic(text: str) -> TaskType | None:
    """Returns the best task type from rules, or None when ambiguous."""
    if _DEBUG_VERBS_RE.search(text) or (
        _FIX_VERBS_RE.search(text) and _FAILURE_NOUNS_RE.search(text)
    ) or (_FAILURE_NOUNS_RE.search(text) and _WHY_RE.search(text)):
        return TaskType.DEBUGGING
    if _CODE_REVIEW_RE.search(text):
        return TaskType.CODE_REVIEW
    if _RESEARCH_RE.search(text) and _CODE_NOUNS_RE.search(text):
        return TaskType.RESEARCH
    if _INFORMATIONAL_RE.search(text):
        return TaskType.INFORMATIONAL
    if _SYSTEM_RE.search(text):
        return TaskType.SYSTEM_OPERATION
    if _DATA_RE.search(text):
        return TaskType.DATA_OPERATION
    # Multi-step outranks software engineering so filename-style nouns
    # ("app.txt", "TestDir") never masquerade as project terms.
    if _SEQUENCE_MARKERS_RE.search(text) and _count_action_verbs(text) >= 2:
        return TaskType.MULTI_STEP
    if _count_action_verbs(text) >= 3:
        return TaskType.MULTI_STEP
    if _SE_VERBS_RE.search(text) and _SE_NOUNS_RE.search(text):
        return TaskType.SOFTWARE_ENGINEERING
    if _CONFIG_RE.search(text):
        return TaskType.CONFIGURATION
    if _FILE_RE.search(text):
        return TaskType.FILE_OPERATION
    if _SIMPLE_ACTION_RE.search(text):
        return TaskType.SIMPLE_ACTION
    return None


def extract_goal(user_input: str, task_type: TaskType | None = None) -> str:
    """
    Derives the desired *outcome* from the user's request.

    The goal is independent of any tool call: \"Fix the failing
    authentication tests.\" becomes \"Make the authentication tests pass.\"
    Polite prefixes are stripped and the result is normalized; a few
    high-value outcome rewrites keep the goal readable.
    """
    text = user_input.strip()
    # Strip polite prefixes iteratively ("Can you please ..." -> "...").
    while True:
        stripped = _POLITE_PREFIX_RE.sub("", text, count=1).strip()
        if stripped == text:
            break
        text = stripped
    text = text or user_input.strip()

    match = re.match(r"^fix (?:the )?failing (.+?)\.?$", text, flags=re.IGNORECASE)
    if match:
        return f"Make the {match.group(1).strip()} pass."

    if text:
        text = text[0].upper() + text[1:]
    if task_type is TaskType.INFORMATIONAL:
        return text.rstrip(".!?") + "?"
    return text.rstrip(".!?") + "."


def _clarification_signal(text: str, task_type: TaskType) -> str | None:
    """Detects requests that cannot be planned safely without more info."""
    if task_type is TaskType.SYSTEM_OPERATION and _VAGUE_DEPLOY_RE.search(text):
        return "No deployment target or environment was specified."
    return None


def classify_task_deterministic(user_input: str) -> TaskClassification:
    """
    Classifies a request using rules only — no LLM, fully deterministic.

    Ambiguous requests default to INFORMATIONAL so callers can decide
    whether to consult the LLM classifier (see :func:`classify_task`).
    """
    text = _normalize(user_input)
    task_type = _classify_deterministic(text) or TaskType.INFORMATIONAL
    signal = _clarification_signal(text, task_type)
    return TaskClassification(
        task_type=task_type,
        goal=extract_goal(user_input, task_type),
        clarification_required=signal is not None,
        clarification_questions=[signal] if signal else [],
    )


# ---------------------------------------------------------------------------
# LLM classification (ambiguous requests only)
# ---------------------------------------------------------------------------

_CLASSIFY_PROMPT = """You are the goal-understanding layer of a local AI coding assistant.

Classify what the user wants to ACCOMPLISH. The task type describes the
nature of the goal, never the tool to call.

TASK TYPES (use exactly one):
- informational: answer or explain; no actions required
- simple_action: one straightforward action
- multi_step: general sequenced multi-action work
- software_engineering: build / implement / refactor / upgrade code
- debugging: diagnose and fix failures
- code_review: inspect code and report findings
- research: investigate / understand / analyze (e.g. explain an architecture)
- system_operation: deploy / run / manage services or infrastructure
- file_operation: file-system operations
- configuration: settings / env / configuration changes
- data_operation: database / query / schema / data work

Rules:
- "goal" is the desired OUTCOME as a concise statement, never a command
  like "run pytest". For "Fix the failing authentication tests." the goal
  is "Make the authentication tests pass."
- If the request lacks information needed to plan safely (for example
  "Deploy this." with no target or environment), set
  clarification_required to true and list the missing information.
- Respond with ONLY a JSON object, no markdown fences:
  {"task_type": "...", "goal": "...", "clarification_required": false,
   "clarification_questions": []}

User request: {user_input}
"""


def _parse_classification(raw: str) -> dict | None:
    """Parses and validates the LLM classification JSON."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[\w]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
        raw = raw.strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    task_type = str(payload.get("task_type", "")).lower()
    if task_type not in {t.value for t in TaskType}:
        return None
    if not str(payload.get("goal", "")).strip():
        return None
    return payload


async def classify_task(
    user_input: str, engine=None
) -> TaskClassification:
    """
    Full classification: deterministic fast path, with an LLM pass only for
    requests the rules could not place.

    ``engine`` is optional — pass None (or omit) to get a pure, fully
    deterministic classification with no LLM call.  When the rules already
    matched, the deterministic result wins (fast, cheap, testable).
    """
    text = _normalize(user_input)
    deterministic = classify_task_deterministic(user_input)

    if engine is None or _classify_deterministic(text) is not None:
        return deterministic

    try:
        prompt = _CLASSIFY_PROMPT.replace("{user_input}", user_input)
        raw = await engine.generate([{"role": "user", "content": prompt}])
    except Exception:  # noqa: BLE001 — engine failures fall back to deterministic
        return deterministic

    payload = _parse_classification(raw)
    if payload is None:
        return deterministic

    task_type = TaskType(str(payload["task_type"]))
    clarification_questions = [
        str(q) for q in payload.get("clarification_questions", []) if str(q).strip()
    ]
    return TaskClassification(
        task_type=task_type,
        goal=str(payload["goal"]).strip(),
        clarification_required=bool(payload.get("clarification_required", False)),
        clarification_questions=clarification_questions,
    )
