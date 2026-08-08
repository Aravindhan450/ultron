"""
ultron.core.tools.builtin.retrieval
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Unified retrieval interface.

Instead of guessing between ``search_web``, ``fetch_page_text`` and raw HTTP,
one entry point (``retrieve``) reads the intent out of the request text and
plans the best strategy — connectivity check, page fetch, web search, or a
combination (e.g. "is X online AND what are its headlines?").

The plan is deterministic (marker-based), matching the project's "code
decides, AI only falls back" philosophy. All networking reuses the existing
tools; every strategy is read-only and URL-safety-gated.
"""

import re
import time

import httpx

from ultron.core.tools.builtin.http_client import check_url_safety
from ultron.core.tools.builtin.web_search import fetch_page_text, search_web

# Connectivity intent markers: the request is asking about *availability*.
# Deliberately conservative — bare "up"/"down"/"live" are too ambiguous
# ("sign up", "download", "live demo"), so they are only matched inside
# explicit availability phrases.
_CONNECTIVITY_MARKER = re.compile(
    r"\b(?:online|offline|reachable|unreachable|connectivity)\b"
    r"|\bis\s+it\s+(?:up|down|working)\b"
    r"|\b(?:site|website|server|page|url|domain|host)\s+is\s+"
    r"(?:up|down|working|running|responding)(?!\s+to\b)\b"
    r"|\bis\s+(?:the\s+)?(?:site|website|server|page|url|domain|host)\s+"
    r"(?:up|down|working|running|responding)(?!\s+to\b)\b"
    r"|\S+(?:\.\w+|://)\S*\s+is\s+(?:up|down)(?!\s+to\b)\b"
    r"|\b(?:is|are)\s+\S+(?:\.\w+|://)\S*\s+(?:up|down)\b"
    r"|\b(?:up|down)\s+right\s+now\b"
    # "status of" is only a connectivity signal when it names a domain —
    # "status of the build" / "status of my PR" are non-network questions.
    r"|\bstatus\s+of\s+(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z]{2,}\b",
    re.IGNORECASE,
)

# Content intent markers: the request is asking about the *content* of a page.
_CONTENT_MARKER = re.compile(
    r"\b(?:read|fetch|scrape|parse)\b"
    r"|\bheadlines?\b"
    r"|\bmain\s+content\b"
    r"|\bcontent\s+of\b"
    r"|\bwhat(?:'s|\s+is)\s+on\b",
    re.IGNORECASE,
)

_URL_RE = re.compile(r"https?://[^\s]+", re.IGNORECASE)
_DOMAIN_RE = re.compile(
    r"(?<!\w)(?<![@.])\b(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z]{2,}\b",
    re.IGNORECASE,
)

# A bare domain only becomes a URL when it looks like a website: a web-ish
# TLD and not a file path (config.yaml, notes.txt, ...).
_FILE_EXT_RE = re.compile(
    r"\.(?:ya?ml|json|toml|ini|cfg|conf|py|js|ts|txt|md|rst|log|csv|xml|html?|sh|env|db|sqlite)$",
    re.IGNORECASE,
)
_WEB_TLD_RE = re.compile(
    r"\.(?:com|org|net|io|ai|dev|co|app|gov|edu|me|tv|info|biz|uk|de|fr|jp|in|au|ca|nz|eu|us|ru|br)$",
    re.IGNORECASE,
)
# Any non-http(s) scheme (ftp://, file://, ...) — its host must not be
# re-interpreted as a bare https domain.
_ANY_SCHEME_RE = re.compile(r"\b[a-z][a-z0-9+.-]*://", re.IGNORECASE)


def _is_web_domain(domain: str) -> bool:
    """True when a bare domain looks like a website rather than a file path."""
    if _FILE_EXT_RE.search(domain):
        return False
    return bool(_WEB_TLD_RE.search(domain))

# Verbs stripped from a query before it is sent to the search tool.
_SEARCH_VERB_RE = re.compile(
    r"^\s*(?:please\s+)?(?:search\s+(?:the\s+web\s+)?for|look\s+up|google|"
    r"find\s+info\s+on|search\s+for)\s+",
    re.IGNORECASE,
)


def check_connectivity(url: str) -> str:
    """
    Checks whether a website is reachable and returns a one-line status.

    Sends an httpx HEAD request (10s timeout, following redirects). If the
    server rejects HEAD at the transport level, falls back to a GET and only
    reads the status line. Reports the status code, reason phrase, and
    measured latency.

    Safety: reuses the shared URL-safety gate (localhost or https only).
    """
    clean_url = (url or "").strip()
    if not clean_url:
        return "Error: no URL provided to check connectivity."
    safety_error = check_url_safety(clean_url)
    if safety_error:
        return safety_error

    try:
        with httpx.Client(timeout=10.0, follow_redirects=True) as client:
            start = time.monotonic()
            try:
                response = client.head(clean_url)
            except httpx.ConnectError:
                # Some servers refuse HEAD at the connection level — fall back
                # to a GET and keep only the status line. (Timeouts must NOT
                # fall back: httpx's TimeoutException is a TransportError, so
                # catching RequestError here would turn a 10s hang into a 20s
                # double-wait before reporting the timeout.)
                response = client.get(clean_url)
            else:
                # Others reject HEAD with an HTTP status (405/501); the server
                # is clearly reachable, so retry with GET before judging it.
                if response.status_code in (405, 501):
                    response = client.get(clean_url)
            latency_ms = (time.monotonic() - start) * 1000
    except httpx.TimeoutException:
        return f"Error: {clean_url} did not respond within 10 seconds (likely offline or unreachable)."
    except httpx.RequestError as exc:
        return f"Error: {clean_url} is unreachable ({exc})."
    except (httpx.HTTPError, UnicodeDecodeError) as exc:
        return f"Error: failed to check {clean_url} ({exc})."

    # A response at all means the server is reachable — an error status is
    # "up but erroring", not "unreachable" (that is reserved for the Error
    # returns above).
    state = "online" if response.status_code < 400 else "up but erroring"
    return f"{clean_url} is {state} ({response.status_code} {response.reason_phrase}, {latency_ms:.0f}ms)"


def extract_retrieval_url(text: str) -> str | None:
    """
    Extracts an http(s) URL, or a website-looking bare domain normalized to
    https://, from text.

    Bare domains are only accepted when they look like a website (web-ish TLD,
    not a file path like config.yaml); hosts under ftp://, file://, etc. and
    email addresses are not treated as domains. Returns None when nothing
    looks like a web URL.
    """
    match = _URL_RE.search(text or "")
    if match:
        return re.sub(r"[.,;\)]+$", "", match.group(0)).strip()
    if _ANY_SCHEME_RE.search(text or ""):
        return None
    domain = _DOMAIN_RE.search(text or "")
    if domain:
        candidate = "https://" + domain.group(0).lower()
        if _is_web_domain(candidate):
            return candidate
    return None


def _plan(request: str, url: str | None) -> list[str]:
    """
    Decides the retrieval strategy from the request text and URL.

    Returns an ordered list of tool names to run.
    """
    text = (request or "").lower()
    wants_connectivity = bool(_CONNECTIVITY_MARKER.search(text))
    wants_content = bool(_CONTENT_MARKER.search(text))

    if wants_connectivity and wants_content:
        return ["check_connectivity", "fetch_page_text"]
    if wants_connectivity:
        return ["check_connectivity"]
    if wants_content:
        return ["fetch_page_text"]
    if url:
        return ["fetch_page_text"]
    return ["search_web"]


def _clean_search_query(request: str) -> str:
    """Strips the leading search verb phrase from a query before searching."""
    query = _SEARCH_VERB_RE.sub("", request.strip()).strip()
    return query or request.strip()


def retrieve(request: str, url: str | None = None) -> str:
    """
    Unified retrieval entry point.

    Reads the intent out of *request* and executes the best strategy — a
    connectivity check, a page fetch, a web search, or a combination — and
    returns a labelled combined report. See docs/retrieval.md.

    Parameters:
      - request: The user's retrieval request (search query, "is X online",
        "read the headlines of X", ...).
      - url: Optional explicit URL; extracted from the request when omitted.

    Returns:
      A formatted report of every strategy that ran.
    """
    if not (request or "").strip() and not (url or "").strip():
        return "Error: retrieve needs a request or a URL."

    request = (request or "").strip()
    url = ((url or "").strip() or extract_retrieval_url(request))

    plan = _plan(request, url)
    steps: list[str] = []

    for step in plan:
        if step == "search_web":
            result = search_web(_clean_search_query(request))
            steps.append(f"[search] {result}")
        elif step == "check_connectivity":
            if not url:
                return "Error: no URL given to check connectivity — please include a URL."
            steps.append(f"[connectivity] {check_connectivity(url)}")
        elif step == "fetch_page_text":
            if not url:
                return "Error: no URL given to fetch — please include a URL."
            steps.append(f"[page] {fetch_page_text(url)}")

    return "\n\n".join(steps)
