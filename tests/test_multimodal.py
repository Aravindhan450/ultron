"""
Tests for multimodal image analysis.

detect_image_intent routes vision phrasing ("analyze chart.png") to the
vision model; handle_image base64-encodes the image, sends it to the engine
as a multimodal image part, and degrades gracefully when the active model can't
see images. A real 1x1 PNG is used; no external network is needed.
"""


import asyncio
import base64
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from ultron.core.agents.simple import SimpleAgent, detect_image_intent, handle_image

# A valid 1x1 transparent PNG.
PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


class FakeEngine:
    """Scripted engine: records the vision call, reports a canned answer."""

    def __init__(self, response="analysis result", vision=True):
        self._response = response
        self.calls = []
        self._vision = vision

    async def supports_images(self, model=None):
        return self._vision

    async def generate(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return self._response


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture
def image_file(tmp_path, monkeypatch):
    from ultron.core.tools import paths

    monkeypatch.setattr(paths, "ALLOWED_BASE_DIR", tmp_path)
    p = tmp_path / "chart.png"
    p.write_bytes(base64.b64decode(PNG_B64))
    return p


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("analyze this chart.png", "chart.png"),
        ("analyse the graph in plot.png", "plot.png"),
        ("look at the screenshot.jpg", "screenshot.jpg"),
        ("examine this diagram.webp", "diagram.webp"),
        ("what's in photo.PNG", "photo.PNG"),
        ("view the file data.bmp", "data.bmp"),
        ("analyze chart.png", "chart.png"),
    ],
)
def test_detect_image_intent_positive(text, expected):
    assert detect_image_intent(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "read config.json",  # not an image extension
        "read notes.txt",
        "show me main.py",
        "run ls and pwd in parallel",
        "analyze the data",  # no filename
        "analyze https://example.com/chart.png",  # a remote URL, not a local file
        "look at //cdn.example.com/diagram.png",
    ],
)
def test_detect_image_intent_negative(text):
    assert detect_image_intent(text) is None


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------


def test_handle_image_sends_base64_image_part(image_file):
    engine = FakeEngine(response="It is a bar chart trending upward.")
    msg = _run(handle_image(str(image_file), "analyze this chart", engine))
    assert msg.pending_action is None
    assert "bar chart trending upward" in msg.content

    assert len(engine.calls) == 1
    messages = engine.calls[0][0]
    assert messages[0]["role"] == "user"
    assert "analyze this chart" in messages[0]["content"]
    assert messages[0]["images"] == [base64.b64encode(image_file.read_bytes()).decode("ascii")]


def test_handle_image_vision_unavailable(image_file):
    engine = FakeEngine(vision=False)
    msg = _run(handle_image(str(image_file), "analyze this chart", engine))
    assert "does not support vision" in msg.content
    assert "llama-server" in msg.content
    assert engine.calls == []  # no request was sent


def test_handle_image_unknown_capability_proceeds(image_file):
    # supports_images returning None (couldn't check) must NOT show the
    # "no vision" hint — the request proceeds and real errors surface.
    engine = FakeEngine(vision=None)
    msg = _run(handle_image(str(image_file), "analyze this chart", engine))
    assert "does not support vision" not in msg.content
    assert "analysis result" in msg.content
    assert len(engine.calls) == 1


def test_handle_image_missing_file(tmp_path, monkeypatch):
    from ultron.core.tools import paths

    monkeypatch.setattr(paths, "ALLOWED_BASE_DIR", tmp_path)
    engine = FakeEngine()
    msg = _run(handle_image(str(tmp_path / "nope.png"), "analyze", engine))
    assert "couldn't find the image" in msg.content
    assert engine.calls == []


def test_handle_image_path_escape_denied(tmp_path, monkeypatch):
    from ultron.core.tools import paths

    monkeypatch.setattr(paths, "ALLOWED_BASE_DIR", tmp_path)
    outside = tmp_path.parent / "secret.png"
    engine = FakeEngine()
    msg = _run(handle_image(str(outside), "analyze", engine))
    assert "Blocked by security" in msg.content
    assert engine.calls == []


def test_handle_image_polishes_response(image_file):
    engine = FakeEngine(response="  analysis  \n\n\n\n trailing  ")
    msg = _run(handle_image(str(image_file), "analyze", engine))
    assert "\n\n\n\n" not in msg.content  # blank-line runs collapsed


# ---------------------------------------------------------------------------
# Engine capability check (local fake /props)
# ---------------------------------------------------------------------------


class _PropsHandler(BaseHTTPRequestHandler):
    vision_enabled: bool = True

    def do_GET(self):
        if self.path == "/props":
            val = b"true" if self.vision_enabled else b"false"
            body = b'{"modalities": {"vision": ' + val + b"}}"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *a):
        pass


@pytest.fixture
def props_server():
    server = HTTPServer(("127.0.0.1", 0), _PropsHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def test_supports_images_true(props_server):
    from ultron.core.engine.llama_cpp import LlamaCppEngine

    engine = LlamaCppEngine(base_url=props_server)
    assert _run(engine.supports_images()) is True


def test_supports_images_false(props_server, monkeypatch):
    from ultron.core.engine.llama_cpp import LlamaCppEngine

    monkeypatch.setattr(_PropsHandler, "vision_enabled", False)
    engine = LlamaCppEngine(base_url=props_server)
    assert _run(engine.supports_images()) is False


def test_supports_images_unreachable_is_unknown():
    from ultron.core.engine.llama_cpp import LlamaCppEngine

    engine = LlamaCppEngine(base_url="http://127.0.0.1:1")  # nothing listening
    assert _run(engine.supports_images()) is None



# ---------------------------------------------------------------------------
# Agent flow
# ---------------------------------------------------------------------------


def test_agent_analyzes_image(tmp_path, monkeypatch):
    from ultron.core.tools import paths

    monkeypatch.setattr(paths, "ALLOWED_BASE_DIR", tmp_path)
    p = tmp_path / "chart.png"
    p.write_bytes(base64.b64decode(PNG_B64))
    monkeypatch.chdir(tmp_path)

    agent = SimpleAgent(FakeEngine(response="A steady upward trend."))
    msg = _run(agent.run("analyze this chart.png"))
    assert "A steady upward trend." in msg.content
    assert msg.pending_action is None
