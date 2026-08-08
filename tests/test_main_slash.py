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
