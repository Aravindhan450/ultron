"""
Tests for real-time API schema inference.

The learning loop: every make_http_request call records endpoint shapes and
parses validation errors for drift ("unknown field X, did you mean Y",
"X is required", type mismatches, 404-on-known-endpoint). High-confidence
renames are applied automatically to future calls. Tests run against a local
http.server — no external requests, and the knowledge DB is repointed at a
temp file for every test.
"""

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from ultron.core.agents import security as agent_security
from ultron.core.agents.simple import (
    SimpleAgent,
    _generic_target_content,
    detect_api_schema_intent,
    extract_api_url,
)
from ultron.core.learning import api_schema
from ultron.core.learning.api_schema import (
    ENDPOINT_REMOVED,
    RENAMED_FIELD,
    REQUIRED_FIELD,
    TYPE_CHANGE,
    _extract_json_keys,
    _infer_drift,
    _split_url,
    api_usage_hint,
    apply_hints,
    forget_api,
    get_api_knowledge,
    learn_api_schema,
    record_interaction,
)
from ultron.core.tools.builtin.http_client import make_http_request
from ultron.security import SecurityBoundary
from ultron.security.models import RiskTier

OPENAPI_SPEC = {
    "openapi": "3.0.0",
    "info": {"title": "Test API", "version": "1.0.0"},
    "paths": {
        "/users": {
            "post": {
                "parameters": [
                    {"name": "X-Trace", "in": "header"},
                    {"name": "verbose", "in": "query"},
                ],
                "requestBody": {
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "username": {"type": "string"},
                                    "email": {"type": "string"},
                                },
                            }
                        }
                    }
                },
            }
        }
    },
}


class _ApiHandler(BaseHTTPRequestHandler):
    """A tiny API that has 'renamed' user -> username."""

    def _reply(self, status: int, payload: dict):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/openapi.json":
            self._reply(200, OPENAPI_SPEC)
        else:
            self._reply(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) or b"{}"
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            self._reply(400, {"error": "invalid JSON body"})
            return
        if self.path.startswith("/users"):
            if "user" in payload:
                self._reply(
                    400,
                    {"error": "unknown field 'user', did you mean 'username'?"},
                )
            elif "username" in payload:
                self._reply(201, {"id": 1, "username": payload["username"]})
            else:
                self._reply(400, {"error": "field 'username' is required"})
        else:
            self._reply(404, {"error": "not found"})

    def log_message(self, *args):
        pass


@pytest.fixture
def local_api():
    server = HTTPServer(("127.0.0.1", 0), _ApiHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


@pytest.fixture(autouse=True)
def tmp_db(monkeypatch, tmp_path):
    """Every test uses its own schema-knowledge database."""
    monkeypatch.setattr(api_schema, "SCHEMA_DB_PATH", tmp_path / "schema.db")


@pytest.fixture
def interactive_mode(monkeypatch):
    monkeypatch.setattr(
        agent_security, "_boundary", SecurityBoundary(mode="interactive")
    )


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# URL + key helpers
# ---------------------------------------------------------------------------


def test_split_url_origin_and_path():
    assert _split_url("https://api.example.com/v1/users?page=2") == (
        "https://api.example.com",
        "/v1/users",
    )


def test_split_url_bare_host_defaults_to_https():
    assert _split_url("example.com") == ("https://example.com", "/")


def test_split_url_with_port():
    assert _split_url("http://localhost:8000/users") == (
        "http://localhost:8000",
        "/users",
    )


def test_extract_json_keys_dict_and_list():
    assert _extract_json_keys({"b": 1, "a": 2}) == ["a", "b"]
    assert _extract_json_keys([{"id": 1, "name": "x"}]) == ["id", "name"]
    assert _extract_json_keys([1, 2]) == []
    assert _extract_json_keys("nope") == []


# ---------------------------------------------------------------------------
# Drift inference
# ---------------------------------------------------------------------------


def test_infer_drift_renamed_with_suggestion():
    d = _infer_drift(
        "POST", ["user"], 400, "unknown field 'user', did you mean 'username'?"
    )
    assert d["drift_type"] == RENAMED_FIELD
    assert d["field"] == "user"
    assert d["correction"] == "username"
    assert d["confidence"] == "high"


def test_infer_drift_renamed_without_suggestion():
    d = _infer_drift("POST", ["login"], 400, "unexpected field 'login'")
    assert d["drift_type"] == RENAMED_FIELD
    assert d["field"] == "login"
    assert d["correction"] == ""
    assert d["confidence"] == "medium"


def test_infer_drift_required():
    d = _infer_drift("POST", [], 422, "field 'email' is required")
    assert d["drift_type"] == REQUIRED_FIELD
    assert d["field"] == "email"
    assert d["confidence"] == "high"


def test_infer_drift_rails_required():
    d = _infer_drift("POST", [], 400, "param is missing or the value is empty: user")
    assert d["drift_type"] == REQUIRED_FIELD
    assert d["field"] == "user"


def test_infer_drift_type_change():
    d = _infer_drift("POST", ["age"], 422, "field 'age' should be an integer")
    assert d["drift_type"] == TYPE_CHANGE
    assert d["field"] == "age"
    assert d["correction"] == "integer"
    assert d["confidence"] == "medium"


def test_infer_drift_type_change_got_for():
    d = _infer_drift(
        "POST", ["age"], 422, "expected an integer, but got string for field 'age'"
    )
    assert d["drift_type"] == TYPE_CHANGE
    assert d["field"] == "age"


def test_infer_drift_body_key_fallback():
    d = _infer_drift("POST", ["login"], 400, "Property 'login' is not permitted")
    assert d["drift_type"] == RENAMED_FIELD
    assert d["field"] == "login"
    assert d["confidence"] == "medium"


def test_infer_drift_ignores_server_errors():
    assert _infer_drift("POST", [], 500, "internal server error") is None
    assert _infer_drift("GET", [], 200, "ok") is None
    assert _infer_drift("POST", [], 401, "unauthorized") is None


# ---------------------------------------------------------------------------
# record_interaction + persistence
# ---------------------------------------------------------------------------


def test_record_success_learns_endpoint_shape():
    drift = record_interaction(
        "GET",
        "http://localhost:8000/users/1",
        None,
        200,
        '{"id": 1, "name": "alice"}',
    )
    assert drift is None
    knowledge = get_api_knowledge("http://localhost:8000")
    assert "GET /users/1" in knowledge
    assert "id, name" in knowledge


def test_success_call_merges_request_fields(local_api):
    # A later successful call must UNION, not clobber, previously-learned
    # fields (including spec-mined ones).
    learn_api_schema(local_api)
    record_interaction(
        "POST", f"{local_api}/users", {"username": "alice"}, 201, '{"id": 1}'
    )
    knowledge = get_api_knowledge(local_api)
    assert "email, username" in knowledge  # spec's email survived the call


def test_conflicting_correction_updates(local_api):
    # A fresh, different did-you-mean suggestion replaces the stale one.
    record_interaction(
        "POST",
        f"{local_api}/users",
        {"user": "alice"},
        400,
        "unknown field 'user', did you mean 'username'?",
    )
    record_interaction(
        "POST",
        f"{local_api}/users",
        {"user": "alice"},
        400,
        "unknown field 'user', did you mean 'login'?",
    )
    hint = api_usage_hint("POST", f"{local_api}/users")
    assert "'login'" in hint
    assert "'username'" not in hint


def test_apply_hints_rename_collision_no_clobber(local_api):
    record_interaction(
        "POST",
        f"{local_api}/users",
        {"user": "alice"},
        400,
        "unknown field 'user', did you mean 'username'?",
    )
    body = {"user": "alice", "username": "bob"}
    corrected, notes = apply_hints("POST", f"{local_api}/users", body)
    assert corrected is None  # the existing 'username' value is never clobbered
    assert any("already sends" in n for n in notes)


def test_learn_api_schema_malformed_url():
    result = learn_api_schema("https://")
    assert result.startswith("Error")


def test_record_drift_then_dedupe():
    body = {"user": "alice"}
    first = record_interaction(
        "POST",
        "http://localhost:8000/users",
        body,
        400,
        "unknown field 'user', did you mean 'username'?",
    )
    assert first is not None
    assert first["correction"] == "username"
    # The identical failure must not duplicate the drift row.
    second = record_interaction(
        "POST",
        "http://localhost:8000/users",
        body,
        400,
        "unknown field 'user', did you mean 'username'?",
    )
    assert second is None


def test_record_404_on_known_endpoint_is_removal(local_api):
    record_interaction(
        "POST", f"{local_api}/users", {"username": "alice"}, 201, '{"id": 1}'
    )
    drift = record_interaction(
        "POST", f"{local_api}/users", {"username": "alice"}, 404, "not found"
    )
    assert drift is not None
    assert drift["drift_type"] == ENDPOINT_REMOVED


def test_record_404_on_unknown_endpoint_is_ignored(local_api):
    drift = record_interaction(
        "POST", f"{local_api}/nope", {"x": 1}, 404, "not found"
    )
    assert drift is None


# ---------------------------------------------------------------------------
# apply_hints (usage prediction)
# ---------------------------------------------------------------------------


def test_apply_hints_applies_high_confidence_rename(local_api):
    record_interaction(
        "POST",
        f"{local_api}/users",
        {"user": "alice"},
        400,
        "unknown field 'user', did you mean 'username'?",
    )
    corrected, notes = apply_hints("POST", f"{local_api}/users", {"user": "alice"})
    assert corrected == {"username": "alice"}
    assert any("renamed" in n for n in notes)


def test_apply_hints_medium_confidence_rename_not_applied(local_api):
    record_interaction(
        "POST", f"{local_api}/users", {"login": "alice"}, 400, "unexpected field 'login'"
    )
    corrected, notes = apply_hints("POST", f"{local_api}/users", {"login": "alice"})
    assert corrected is None  # no suggestion -> never guess a replacement
    assert notes == []  # nothing actionable


def test_apply_hints_reports_missing_required(local_api):
    record_interaction(
        "POST", f"{local_api}/users", {}, 400, "field 'email' is required"
    )
    corrected, notes = apply_hints("POST", f"{local_api}/users", {"username": "x"})
    assert corrected is None
    assert any("email" in n and "required" in n for n in notes)


def test_apply_hints_no_drift_no_change():
    corrected, notes = apply_hints("POST", "http://localhost:9999/x", {"a": 1})
    assert corrected is None
    assert notes == []


def test_apply_hints_drift_scoped_to_method_and_path(local_api):
    record_interaction(
        "POST",
        f"{local_api}/users",
        {"user": "alice"},
        400,
        "unknown field 'user', did you mean 'username'?",
    )
    # GET on a different path is untouched.
    corrected, _ = apply_hints("GET", f"{local_api}/users", {"user": "alice"})
    assert corrected is None
    corrected, _ = apply_hints("POST", f"{local_api}/other", {"user": "alice"})
    assert corrected is None


# ---------------------------------------------------------------------------
# learn_api_schema (OpenAPI discovery + mining)
# ---------------------------------------------------------------------------


def test_learn_api_schema_mines_spec(local_api):
    result = learn_api_schema(local_api)
    assert "Learned the OpenAPI schema" in result
    knowledge = get_api_knowledge(local_api)
    assert "POST /users" in knowledge
    # Body properties + query param mined; header param excluded.
    assert "username" in knowledge
    assert "email" in knowledge
    assert "verbose" in knowledge
    assert "X-Trace" not in knowledge


def test_learn_api_schema_unsafe_url():
    result = learn_api_schema("http://example.com")
    assert result.startswith("Error")


def test_learn_api_schema_no_spec():
    class _NoSpecHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(404)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), _NoSpecHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_port}"
        result = learn_api_schema(url)
        assert "No OpenAPI document found" in result
    finally:
        server.shutdown()


def test_forget_api_clears_knowledge(local_api):
    learn_api_schema(local_api)
    assert "API knowledge" in get_api_knowledge(local_api)
    result = forget_api(local_api)
    assert "Forgotten" in result
    assert "No knowledge stored" in get_api_knowledge(local_api)


def test_api_usage_hint_tool(local_api):
    make_http_request("POST", f"{local_api}/users", '{"user": "alice"}')
    hint = api_usage_hint("POST", f"{local_api}/users", '{"user": "alice"}')
    assert "send 'username' instead" in hint


# ---------------------------------------------------------------------------
# make_http_request end-to-end (the learning loop)
# ---------------------------------------------------------------------------


def test_make_http_request_learns_then_auto_corrects(local_api):
    url = f"{local_api}/users"

    # First call: the API rejects 'user' — the tool detects + remembers.
    r1 = make_http_request("POST", url, '{"user": "alice"}')
    assert "400" in r1
    assert "[api schema] Detected a renamed_field" in r1
    assert "username" in r1

    # Second call with the same payload: auto-corrected to 'username' -> 201.
    r2 = make_http_request("POST", url, '{"user": "alice"}')
    assert "201" in r2
    assert "renamed" in r2.lower()

    # A missing required field is detected on the next attempt.
    r3 = make_http_request("POST", url, "{}")
    assert "400" in r3
    assert "Detected a required_field" in r3

    # The store now carries both drifts.
    knowledge = get_api_knowledge(local_api)
    assert "renamed_field" in knowledge
    assert "required_field" in knowledge


def test_make_http_request_get_still_works(local_api):
    r = make_http_request("GET", f"{local_api}/users/1")
    assert "404" in r  # server has no GET /users/1
    assert "[api schema]" not in r  # a 404 on an unseen endpoint is not drift


# ---------------------------------------------------------------------------
# Registry + security
# ---------------------------------------------------------------------------


def test_registry_registers_schema_tools():
    from ultron.core.tools.registry import get_tool

    for name in ("learn_api_schema", "api_usage_hint", "get_api_knowledge", "forget_api"):
        assert get_tool(name) is not None


def test_classify_schema_tools_low():
    boundary = SecurityBoundary(mode="interactive")
    for tool in ("learn_api_schema", "api_usage_hint", "get_api_knowledge", "forget_api"):
        assert boundary.classify_action(tool, "http://localhost:8000") == RiskTier.LOW


def test_guardrails_scan_new_tool_urls():
    from ultron.security.guardrails import GuardrailsEngine

    engine = GuardrailsEngine()
    # learn_api_schema makes an outbound fetch — unsafe URLs are denied.
    assert engine.evaluate(action_type="learn_api_schema", target="http://example.com").blocked
    assert not engine.evaluate(
        action_type="learn_api_schema", target="http://localhost:8000"
    ).blocked
    # The store-only tools never touch the network, so a host key is fine.
    for tool in ("api_usage_hint", "get_api_knowledge", "forget_api"):
        assert not engine.evaluate(action_type=tool, target="http://example.com").blocked


def test_generic_target_content_schema_tools():
    target, content = _generic_target_content(
        "learn_api_schema", {"base_url": "http://localhost:8000"}
    )
    assert target == "http://localhost:8000"
    assert content is None
    target, content = _generic_target_content(
        "api_usage_hint",
        {"url": "http://localhost:8000", "body": '{"user": 1}'},
    )
    assert target == "http://localhost:8000"
    assert content == '{"user": 1}'


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


def test_detect_api_schema_learn_with_url():
    m = detect_api_schema_intent("learn the api schema for http://localhost:8000")
    assert m == {"action": "learn", "url": "http://localhost:8000"}


def test_detect_api_schema_learn_bare_domain():
    m = detect_api_schema_intent("fetch the schema for example.com")
    assert m["action"] == "learn"
    assert m["url"] == "https://example.com"


def test_detect_api_schema_knowledge_no_url():
    m = detect_api_schema_intent("what apis do you know")
    assert m == {"action": "knowledge", "url": None}


def test_detect_api_schema_knowledge_with_url():
    m = detect_api_schema_intent("what do you know about the api at example.com")
    assert m["action"] == "knowledge"
    assert m["url"] == "https://example.com"


def test_detect_api_schema_hints():
    m = detect_api_schema_intent("api usage hints for http://localhost:8000/users")
    assert m["action"] == "hints"
    assert m["url"] == "http://localhost:8000/users"


def test_detect_api_schema_forget():
    m = detect_api_schema_intent("forget the api schema for http://localhost:8000")
    assert m["action"] == "forget"
    assert m["url"] == "http://localhost:8000"


def test_detect_api_schema_forget_requires_proximity():
    # "forget ... api ..." anywhere in a sentence is NOT a schema request.
    assert detect_api_schema_intent("forget what I told you about the api rate limit issue") is None


@pytest.mark.parametrize(
    "text",
    [
        "post to http://localhost:8000 with body {}",  # plain API call
        "get from https://api.example.com/v1",  # plain GET
        "what do you know about France",  # memory question, not an API
        "read the config file",
        "search for python news",
        "is example.com online",  # retrieval, not schema learning
    ],
)
def test_detect_api_schema_negative(text):
    assert detect_api_schema_intent(text) is None


def test_extract_api_url_localhost_and_domain():
    assert extract_api_url("http://localhost:8000") == "http://localhost:8000"
    assert extract_api_url("127.0.0.1:9000") == "http://127.0.0.1:9000"
    assert extract_api_url("api.example.com") == "https://api.example.com"
    assert extract_api_url("no url here") is None


# ---------------------------------------------------------------------------
# Agent wiring
# ---------------------------------------------------------------------------


def test_agent_learn_api_schema(monkeypatch, interactive_mode):
    from ultron.core.tools import registry as reg

    monkeypatch.setitem(
        reg.TOOLS, "learn_api_schema", lambda base_url: f"learned {base_url}"
    )
    agent = SimpleAgent(None)
    msg = _run(agent.run("learn the api schema for http://localhost:8000"))
    assert msg.pending_action is None
    assert "learned http://localhost:8000" in msg.content


def test_agent_api_knowledge_all(monkeypatch, interactive_mode):
    from ultron.core.tools import registry as reg

    monkeypatch.setitem(
        reg.TOOLS, "get_api_knowledge", lambda base_url="": "known apis: none"
    )
    agent = SimpleAgent(None)
    msg = _run(agent.run("what apis do you know"))
    assert "known apis: none" in msg.content


def test_agent_api_schema_clarification_then_executes(monkeypatch, interactive_mode):
    from ultron.core.tools import registry as reg

    monkeypatch.setitem(
        reg.TOOLS, "learn_api_schema", lambda base_url: f"learned {base_url}"
    )
    agent = SimpleAgent(None)
    msg1 = _run(agent.run("learn the api schema for this api"))
    assert "Which API" in msg1.content
    msg2 = _run(agent.run("http://localhost:8000"))
    assert "learned http://localhost:8000" in msg2.content


def test_agent_api_schema_unsafe_url_blocked(interactive_mode):
    agent = SimpleAgent(None)
    msg = _run(agent.run("learn the api schema for http://example.com"))
    assert "Blocked by security" in msg.content
    assert msg.pending_action is None
