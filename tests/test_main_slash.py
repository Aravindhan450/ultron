"""
Unit tests for the `/agent` slash command in the chat CLI.

The interactive questionary picker is stubbed out; everything else runs
against the real ``handle_slash_command`` code path, including the factory
call that builds the replacement agent.
"""

import asyncio
from types import SimpleNamespace

from ultron.core.agents.react import ReActAgent
from ultron.core.agents.simple import SimpleAgent
from ultron.main import handle_slash_command


class FakeConsole:
    """Minimal console stand-in that records printed messages."""

    def __init__(self):
        self.messages = []

    def print(self, *args, **kwargs):
        self.messages.append(" ".join(str(a) for a in args))


def _run(coro):
    return asyncio.run(coro)


def _make_agent():
    # The /agent path never touches the engine, so None is fine here.
    return SimpleAgent(None)


def _patch_picker(monkeypatch, answer):
    """Routes questionary.select.ask_async() to return *answer*."""
    import questionary

    async def fake_ask_async(self=None):
        return answer

    monkeypatch.setattr(
        questionary,
        "select",
        lambda *a, **k: SimpleNamespace(ask_async=fake_ask_async),
    )


def test_switch_agent_inline():
    console = FakeConsole()
    session = SimpleNamespace()
    handled, should_exit = _run(
        handle_slash_command("/agent react", console, [], agent=_make_agent(), session=session)
    )
    assert handled is True
    assert should_exit is False
    assert isinstance(session.next_agent, ReActAgent)
    assert session.active_agent_type == "react"
    # The switch is queued, not applied inside the handler — history survives.
    assert not hasattr(session, "reloaded_agent")


def test_switch_agent_via_picker(monkeypatch):
    _patch_picker(monkeypatch, "react")
    console = FakeConsole()
    session = SimpleNamespace()
    handled, _ = _run(
        handle_slash_command("/agent", console, [], agent=_make_agent(), session=session)
    )
    assert handled is True
    assert isinstance(session.next_agent, ReActAgent)


def test_cancelled_picker_does_not_switch(monkeypatch):
    _patch_picker(monkeypatch, None)
    console = FakeConsole()
    session = SimpleNamespace()
    handled, _ = _run(
        handle_slash_command("/agent", console, [], agent=_make_agent(), session=session)
    )
    assert handled is True
    assert not hasattr(session, "next_agent")


def test_unknown_agent_type_rejected():
    console = FakeConsole()
    session = SimpleNamespace()
    handled, _ = _run(
        handle_slash_command("/agent llama9", console, [], agent=_make_agent(), session=session)
    )
    assert handled is True
    assert not hasattr(session, "next_agent")
    assert any("Unknown agent type" in m for m in console.messages)


def test_agent_prefix_requires_space():
    # /agentx is not a slash command we own — it must fall through to the
    # generic "unknown command" path instead of being parsed as /agent.
    console = FakeConsole()
    handled, _ = _run(
        handle_slash_command("/agentx", console, [], agent=_make_agent(), session=SimpleNamespace())
    )
    assert handled is True  # still not sent to the AI
    assert any("Unknown command" in m for m in console.messages)


# ---------------------------------------------------------------------------
# /memory
# ---------------------------------------------------------------------------


def test_memory_command_shows_graph_status():
    console = FakeConsole()
    handled, should_exit = _run(handle_slash_command("/memory", console, []))
    assert handled is True
    assert should_exit is False
    assert any("Memory graph" in m for m in console.messages)


# ---------------------------------------------------------------------------
# /security
# ---------------------------------------------------------------------------


def _security_fixture(monkeypatch):
    """
    Pins the shared boundary + settings to a known mode and stubs the .env
    write, returning a dict of what would have been written.
    """
    import ultron.core.agents.security as agent_security
    from ultron.core.config import settings
    from ultron.security import SecurityBoundary

    old_mode = settings.security_mode
    old_boundary = agent_security._boundary
    monkeypatch.setattr(agent_security, "_boundary", SecurityBoundary(mode="interactive"))
    written = {}

    def fake_update_env_file(key, value):
        written[key] = value

    monkeypatch.setattr("ultron.core.config.update_env_file", fake_update_env_file)

    def restore():
        settings.security_mode = old_mode
        agent_security._boundary = old_boundary

    monkeypatch.setattr(settings, "security_mode", old_mode)  # ensure teardown restore
    return written, restore


def test_security_switch_inline(monkeypatch):
    import os

    from ultron.core.agents import security as agent_security
    from ultron.core.config import settings

    written, restore = _security_fixture(monkeypatch)
    try:
        console = FakeConsole()
        handled, _ = _run(handle_slash_command("/security strict", console, []))
        assert handled is True
        assert settings.security_mode == "strict"
        assert agent_security.get_security().mode == "strict"
        assert written.get("ULTRON_SECURITY_MODE") == "strict"
        assert os.environ.get("ULTRON_SECURITY_MODE") == "strict"
        assert any("Switch anytime" in m for m in console.messages)  # status rendered
    finally:
        restore()
        os.environ.pop("ULTRON_SECURITY_MODE", None)


def test_security_switch_via_picker(monkeypatch):
    from ultron.core.agents import security as agent_security
    from ultron.core.config import settings

    written, restore = _security_fixture(monkeypatch)
    try:
        _patch_picker(monkeypatch, "permissive")
        console = FakeConsole()
        handled, _ = _run(handle_slash_command("/security", console, []))
        assert handled is True
        assert settings.security_mode == "permissive"
        assert agent_security.get_security().mode == "permissive"
        assert written.get("ULTRON_SECURITY_MODE") == "permissive"
    finally:
        restore()


def test_security_cancelled_picker_does_not_switch(monkeypatch):
    from ultron.core.config import settings

    written, restore = _security_fixture(monkeypatch)
    try:
        _patch_picker(monkeypatch, None)
        console = FakeConsole()
        handled, _ = _run(handle_slash_command("/security", console, []))
        assert handled is True
        assert settings.security_mode == "interactive"  # unchanged
        assert written == {}  # nothing persisted
    finally:
        restore()


def test_security_unknown_mode_rejected(monkeypatch):
    from ultron.core.config import settings

    written, restore = _security_fixture(monkeypatch)
    try:
        console = FakeConsole()
        handled, _ = _run(handle_slash_command("/security paranoid", console, []))
        assert handled is True
        assert settings.security_mode == "interactive"  # unchanged
        assert any("Unknown security mode" in m for m in console.messages)
        assert written == {}
    finally:
        restore()


def test_security_bare_shows_status(monkeypatch):
    _patch_picker(monkeypatch, None)
    console = FakeConsole()
    handled, _ = _run(handle_slash_command("/security", console, []))
    assert handled is True
    # The status panel + hint are rendered even when the picker is cancelled.
    assert any("Switch anytime" in m for m in console.messages)


# ---------------------------------------------------------------------------
# /model tests
# ---------------------------------------------------------------------------


class FakeModelEngine:
    def __init__(self, active_model="qwen2.5-coder-7b-instruct-q4_k_m.gguf", base_url="http://127.0.0.1:8080"):
        self.active_model = active_model
        self.default_model = active_model
        self.base_url = base_url

    async def get_active_model(self):
        return self.active_model

    async def list_models(self):
        return [self.active_model] if self.active_model else []

    def set_model(self, name):
        self.default_model = name


def test_model_status_display_syncs_with_server():
    console = FakeConsole()
    engine = FakeModelEngine(active_model="qwen2.5-coder-7b-instruct-q4_k_m.gguf")
    agent = SimpleAgent(engine=engine)
    session = SimpleNamespace(active_model="stale-old-model")

    handled, should_exit = _run(
        handle_slash_command("/model", console, [], agent=agent, session=session)
    )
    assert handled is True
    assert should_exit is False
    assert session.active_model == "qwen2.5-coder-7b-instruct-q4_k_m.gguf"
    assert engine.default_model == "qwen2.5-coder-7b-instruct-q4_k_m.gguf"
    assert any("Active Server Model:" in m for m in console.messages)
    assert any("qwen2.5-coder-7b-instruct-q4_k_m.gguf" in m for m in console.messages)
    assert any("llama.cpp" in m for m in console.messages)


def test_model_matching_request_accepted():
    console = FakeConsole()
    engine = FakeModelEngine(active_model="qwen2.5-coder-7b-instruct-q4_k_m.gguf")
    agent = SimpleAgent(engine=engine)
    session = SimpleNamespace(active_model="qwen2.5-coder-7b-instruct-q4_k_m.gguf")

    handled, _ = _run(
        handle_slash_command("/model qwen2.5-coder-7b-instruct-q4_k_m.gguf", console, [], agent=agent, session=session)
    )
    assert handled is True
    assert session.active_model == "qwen2.5-coder-7b-instruct-q4_k_m.gguf"
    assert any("matches currently loaded server model" in m for m in console.messages)


def test_model_switch_to_unloaded_model_rejected():
    console = FakeConsole()
    engine = FakeModelEngine(active_model="qwen2.5-coder-7b-instruct-q4_k_m.gguf")
    agent = SimpleAgent(engine=engine)
    session = SimpleNamespace(active_model="qwen2.5-coder-7b-instruct-q4_k_m.gguf")

    handled, _ = _run(
        handle_slash_command("/model llama-3-8b.gguf", console, [], agent=agent, session=session)
    )
    assert handled is True
    # State MUST NOT be updated to the unloaded model
    assert session.active_model == "qwen2.5-coder-7b-instruct-q4_k_m.gguf"
    assert engine.default_model == "qwen2.5-coder-7b-instruct-q4_k_m.gguf"
    assert any("Dynamic model switching is not supported" in m for m in console.messages)
    assert any("llama-server -m /path/to/llama-3-8b.gguf" in m for m in console.messages)


def test_model_server_offline_falls_back_gracefully():
    from ultron.core.config import settings

    console = FakeConsole()
    engine = FakeModelEngine(active_model=None)
    agent = SimpleAgent(engine=engine)
    session = SimpleNamespace(active_model=settings.model)

    handled, _ = _run(
        handle_slash_command("/model", console, [], agent=agent, session=session)
    )
    assert handled is True
    assert any("Model & Backend Status" in m for m in console.messages)
    assert not any("Ollama" in m for m in console.messages)

