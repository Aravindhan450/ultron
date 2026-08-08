"""
Tests for the unified retrieval orchestrator.

retrieve decides the best networking strategy (connectivity check / page
fetch / search / combination) from the request text, and check_connectivity
answers "is this site up?". Both are read-only and URL-safety-gated through
the security boundary. Network tests run against a local http.server so no
external requests are made.
"""

import asyncio
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from ultron.core.agents import security as agent_security
from ultron.core.agents.simple import (
    SimpleAgent,
    _generic_target_content,
    detect_retrieval_intent,
)
from ultron.core.tools.builtin.retrieval import (
    _plan,
    check_connectivity,
    extract_retrieval_url,
    retrieve,
)
from ultron.security import SecurityBoundary
from ultron.security.models import RiskTier


class _Handler(BaseHTTPRequestHandler):
    """Serves a tiny news page over localhost."""

    def do_GET(self):
        body = (
            b"<html><head><title>Test News</title></head>"
            b"<body><h1>Breaking: Local Server Online</h1>"
            b"<p>Some article text here.</p></body></html>"
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *args):
        pass


@pytest.fixture
def local_server():
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


@pytest.fixture
def interactive_mode(monkeypatch):
    monkeypatch.setattr(
        agent_security, "_boundary", SecurityBoundary(mode="interactive")
    )


class FakeEngine:
    def __init__(self, responses=()):
        self._responses = list(responses)
        self.calls = []

    async def generate(self, messages, **kwargs):
        self.calls.append(messages)
        return self._responses.pop(0) if self._responses else ""


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Plan decisions
# ---------------------------------------------------------------------------


def test_plan_connectivity_only():
    assert _plan("is example.com online", "https://example.com") == [
        "check_connectivity"
    ]


def test_plan_content_only():
    assert _plan("read the headlines", "https://example.com") == ["fetch_page_text"]


def test_plan_combined_connectivity_and_content():
    assert _plan(
        "check if it is online and read its main headlines", "https://example.com"
    ) == ["check_connectivity", "fetch_page_text"]


def test_plan_url_defaults_to_fetch():
    assert _plan("", "https://example.com") == ["fetch_page_text"]


def test_plan_search_fallback():
    assert _plan("latest python news", None) == ["search_web"]


# ---------------------------------------------------------------------------
# check_connectivity
# ---------------------------------------------------------------------------


def test_check_connectivity_online(local_server):
    result = check_connectivity(local_server)
    assert "is online" in result
    assert "200" in result


def test_check_connectivity_unreachable():
    result = check_connectivity("https://127.0.0.1:1")
    assert "unreachable" in result or result.startswith("Error")


def test_check_connectivity_unsafe_url():
    result = check_connectivity("http://example.com")
    assert "only localhost or https" in result


# ---------------------------------------------------------------------------
# retrieve orchestrator
# ---------------------------------------------------------------------------


def test_retrieve_combined_report(local_server, monkeypatch):
    monkeypatch.setattr(
        "ultron.core.tools.builtin.retrieval.search_web", lambda q: "search stub"
    )
    result = retrieve(
        f"check if {local_server} is online and read its main headlines"
    )
    assert "[connectivity]" in result
    assert "is online" in result
    assert "[page]" in result
    assert "Breaking: Local Server Online" in result


def test_retrieve_search_fallback(monkeypatch):
    monkeypatch.setattr(
        "ultron.core.tools.builtin.retrieval.search_web",
        lambda q: f"results for {q}",
    )
    result = retrieve("latest python news")
    assert "[search]" in result
    assert "results for latest python news" in result


def test_retrieve_connectivity_without_url():
    result = retrieve("is it online")
    assert "no URL given" in result


def test_retrieve_empty_input():
    assert "needs a request or a URL" in retrieve("")


# ---------------------------------------------------------------------------
# URL extraction
# ---------------------------------------------------------------------------


def test_extract_retrieval_url_protocol():
    assert extract_retrieval_url("read https://example.com/news now") == (
        "https://example.com/news"
    )


def test_extract_retrieval_url_bare_domain():
    assert extract_retrieval_url("check if example.com is online") == (
        "https://example.com"
    )


def test_extract_retrieval_url_none():
    assert extract_retrieval_url("tell me a story") is None


def test_extract_retrieval_url_rejects_file_paths_emails_and_ftp():
    assert extract_retrieval_url("read the config.yaml file") is None
    assert extract_retrieval_url("my email is a@b.co please check") is None
    assert extract_retrieval_url("fetch ftp://example.com/file") is None


def test_extract_retrieval_url_keeps_web_domains():
    assert extract_retrieval_url("check if news.example.com is online") == (
        "https://news.example.com"
    )


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


def test_detect_retrieval_connectivity_bare_domain():
    m = detect_retrieval_intent("check if example.com is online")
    assert m is not None
    assert m["url"] == "https://example.com"


def test_detect_retrieval_url_content():
    m = detect_retrieval_intent("read the article at https://example.com/news")
    assert m is not None
    assert m["url"] == "https://example.com/news"


def test_detect_retrieval_combined():
    m = detect_retrieval_intent(
        "check if https://example.com is online and read its headlines"
    )
    assert m is not None
    assert m["url"] == "https://example.com"


def test_detect_retrieval_no_url_returns_request_only():
    m = detect_retrieval_intent("check if this website is online")
    assert m == {"request": "check if this website is online", "url": None}


def test_detect_retrieval_status_of_domain_only():
    # "status of" is only a connectivity signal when it names a domain.
    m = detect_retrieval_intent("check the status of example.com")
    assert m is not None
    assert m["url"] == "https://example.com"


def test_detect_retrieval_domain_is_up_phrasing():
    m = detect_retrieval_intent("check if example.com is up")
    assert m is not None
    assert m["url"] == "https://example.com"


def test_detect_retrieval_question_order_connectivity():
    m = detect_retrieval_intent("is the server down?")
    assert m is not None
    assert m["url"] is None  # needs the URL from a clarification turn


def test_detect_retrieval_question_order_running():
    m = detect_retrieval_intent("is the server running?")
    assert m is not None


def test_detect_retrieval_up_to_date_not_connectivity():
    # "up to date" must not be read as an availability question.
    assert detect_retrieval_intent("is the server up to date") is None
    assert detect_retrieval_intent("the page is up to date") is None


def test_detect_retrieval_non_web_tld_connectivity():
    # Connectivity requests accept hosts with non-web TLDs (.local, .lan).
    m = detect_retrieval_intent("check if my-site.local is online")
    assert m is not None
    assert m["url"] == "https://my-site.local"


def test_detect_retrieval_rejects_file_paths():
    # "read the config.yaml file" is a local-file request, not a web fetch.
    assert detect_retrieval_intent("read the config.yaml file") is None


@pytest.mark.parametrize(
    "text",
    [
        "search for python news",  # plain search keeps its own path
        "post to http://localhost:8000 with body {}",  # state-changing API call
        "get from https://api.example.com/v1",  # raw GET API call
        "what do you know about France",
        "run ls and pwd in parallel",
        "remember that Paris is in France",
        "what is the status of the build",  # non-network "status of" question
        "status of my PR",
        "read the config file",  # no URL, no availability marker
    ],
)
def test_detect_retrieval_negative(text):
    assert detect_retrieval_intent(text) is None


# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------


def test_classify_retrieval_low():
    boundary = SecurityBoundary(mode="interactive")
    assert boundary.classify_action("retrieve", "https://example.com") == RiskTier.LOW
    assert boundary.classify_action("retrieve", "latest python news") == RiskTier.LOW
    assert boundary.classify_action("check_connectivity", "https://example.com") == (
        RiskTier.LOW
    )


def test_guardrails_deny_unsafe_retrieve_url():
    from ultron.security.guardrails import GuardrailsEngine

    engine = GuardrailsEngine()
    assert engine.evaluate(action_type="retrieve", target="http://example.com").blocked
    assert engine.evaluate(
        action_type="check_connectivity", target="http://example.com"
    ).blocked


def test_guardrails_allow_search_query_without_url():
    from ultron.security.guardrails import GuardrailsEngine

    engine = GuardrailsEngine()
    assert not engine.evaluate(
        action_type="retrieve", target="latest python news"
    ).blocked
    assert not engine.evaluate(
        action_type="retrieve", target="https://example.com"
    ).blocked


# ---------------------------------------------------------------------------
# Argument mapping + registry
# ---------------------------------------------------------------------------


def test_generic_target_content_retrieve():
    target, content = _generic_target_content(
        "retrieve", {"url": "https://example.com"}
    )
    assert target == "https://example.com"
    assert content is None
    target, _ = _generic_target_content("retrieve", {"request": "latest news"})
    assert target == "latest news"
    target, _ = _generic_target_content(
        "check_connectivity", {"url": "https://example.com"}
    )
    assert target == "https://example.com"


def test_registry_registers_retrieval_tools():
    from ultron.core.tools.registry import get_tool

    assert get_tool("retrieve") is not None
    assert get_tool("check_connectivity") is not None


# ---------------------------------------------------------------------------
# Agent wiring
# ---------------------------------------------------------------------------


def test_agent_retrieval_autoexecutes(monkeypatch, interactive_mode):
    from ultron.core.tools import registry as reg

    monkeypatch.setitem(reg.TOOLS, "retrieve", lambda request, url=None: "fake result")
    agent = SimpleAgent(None)
    msg = _run(agent.run("check if example.com is online"))
    assert msg.pending_action is None
    assert "fake result" in msg.content


def test_agent_retrieval_clarification_then_executes(monkeypatch, interactive_mode):
    from ultron.core.tools import registry as reg

    monkeypatch.setitem(reg.TOOLS, "retrieve", lambda request, url=None: "fake result")
    agent = SimpleAgent(None)
    msg1 = _run(agent.run("check if this website is online"))
    assert "Which website" in msg1.content
    msg2 = _run(agent.run("news.example.com"))
    assert "fake result" in msg2.content


def test_agent_retrieval_denied_unsafe_url(interactive_mode):
    agent = SimpleAgent(None)
    msg = _run(agent.run("fetch http://example.com"))
    assert "Blocked by security" in msg.content
    assert msg.pending_action is None
