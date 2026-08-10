"""
Tests for the terminal-resize reflow of the whole chat screen
(``ultron.ui.responsive.ResizeReflow``).

Every renderable printed during a chat session is recorded; on resize the
whole recorded screen is re-rendered at the terminal's live width — but only
while the chat prompt's app is running (never mid-processing or inside a
questionary prompt), and never while a rich ``Live`` (the thinking spinner)
is drawing.  prompt_toolkit's ``get_app_or_none`` and ``in_terminal`` are
stubbed so the watcher logic is exercised deterministically.
"""

import asyncio
from contextlib import asynccontextmanager

import pytest

from ultron.ui import responsive


class _FakeApp:
    """Stand-in for a prompt_toolkit Application (compared by identity)."""


class _FakeConsole:
    """Minimal console stand-in: settable width, recorded prints/clears."""

    def __init__(self, width: int = 100):
        self.width = width
        self.cleared = 0
        self.printed: list[object] = []
        self._live_stack: list[object] = []

    def clear(self) -> None:
        self.cleared += 1

    def print(self, *args, **kwargs) -> None:  # pragma: no cover - spy only
        self.printed.append(args)


@pytest.fixture
def chat_app():
    """Return an app identity to pass into ResizeReflow(app=...)."""
    return _FakeApp()


@pytest.fixture
def chat_app_running(monkeypatch, chat_app):
    """Make get_app_or_none return *chat_app* so reflow is allowed."""
    monkeypatch.setattr(responsive, "get_app_or_none", lambda: chat_app)
    return chat_app


@pytest.fixture
def fake_in_terminal(monkeypatch):
    @asynccontextmanager
    async def _noop_in_terminal(*args, **kwargs):
        yield

    monkeypatch.setattr(responsive, "in_terminal", _noop_in_terminal)


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# add() / rebuild() — rebuild blocks
# ---------------------------------------------------------------------------


def test_add_runs_printer_now_and_registers_rebuild_block():
    console = _FakeConsole(width=80)
    reflow = responsive.ResizeReflow(console)
    calls: list[int] = []
    reflow.add(lambda: calls.append(console.width))
    assert calls == [80]  # runs immediately
    assert [kind for kind, _ in reflow._blocks] == ["rebuild"]
    assert reflow._last_width == 80


def test_add_does_not_record_block_prints():
    console = _FakeConsole(width=80)
    reflow = responsive.ResizeReflow(console)
    reflow.add(lambda: console.print("banner"))
    # The block's own output is covered by the rebuild block, not duplicated.
    assert [kind for kind, _ in reflow._blocks] == ["rebuild"]


def test_reset_drops_print_blocks_but_keeps_rebuild_blocks():
    console = _FakeConsole(width=80)
    reflow = responsive.ResizeReflow(console)
    reflow.add(lambda: console.print("banner"))
    reflow._blocks.append(("print", (("box",), {})))  # simulate a response
    reflow.reset()
    assert [kind for kind, _ in reflow._blocks] == ["rebuild"]
    assert reflow._last_width == 80


def test_rebuild_reruns_rebuild_blocks_now_without_recording():
    console = _FakeConsole(width=80)
    reflow = responsive.ResizeReflow(console)
    calls: list[int] = []
    reflow.add(lambda: calls.append(console.width))
    console.width = 50
    reflow.rebuild()
    assert calls == [80, 50]  # re-run at the current width
    assert [kind for kind, _ in reflow._blocks] == ["rebuild"]
    assert reflow._last_width == 50


# ---------------------------------------------------------------------------
# recording / restore
# ---------------------------------------------------------------------------


def test_start_wraps_print_records_and_stop_restores():
    async def scenario():
        console = _FakeConsole(width=100)
        reflow = responsive.ResizeReflow(console)
        reflow.add(lambda: None)
        reflow.start()
        console.print("while recording")
        reflow.stop()
        console.print("after stop")
        return reflow, console

    reflow, console = _run(scenario())
    kinds = [kind for kind, _ in reflow._blocks]
    # Prints while started are recorded into the transcript...
    assert kinds[0] == "rebuild"
    assert kinds.count("print") == 1
    # ...but after stop() the console is restored: nothing more is recorded.
    assert [kind for kind, _ in reflow._blocks] == ["rebuild", "print"]
    # Both prints still reached the console itself.
    assert console.printed == [("while recording",), ("after stop",)]


def test_live_spinner_prints_are_not_recorded():
    async def scenario():
        console = _FakeConsole(width=100)
        reflow = responsive.ResizeReflow(console)
        reflow.add(lambda: None)
        reflow.start()
        console._live_stack.append(object())  # rich Live / Status active
        console.print("spinner frame")
        console._live_stack.pop()
        console.print("real output")
        reflow.stop()
        return reflow

    reflow = _run(scenario())
    kinds = [kind for kind, _ in reflow._blocks]
    assert kinds.count("print") == 1  # only the real output


# ---------------------------------------------------------------------------
# reflow()
# ---------------------------------------------------------------------------


def test_reflow_replays_all_blocks_in_order(fake_in_terminal):
    async def scenario():
        console = _FakeConsole(width=100)
        reflow = responsive.ResizeReflow(console)
        calls: list[tuple[str, int]] = []
        reflow.add(lambda: calls.append(("rebuild", console.width)))
        reflow.start()
        console.print("box one")
        console.print("box two")
        console.width = 60
        await reflow._reflow()
        reflow.stop()
        return calls, console, reflow

    calls, console, reflow = _run(scenario())
    # Rebuild block re-ran at the new width, then both prints replayed.
    assert calls == [("rebuild", 100), ("rebuild", 60)]
    assert console.cleared == 1
    assert console.printed[-2:] == [("box one",), ("box two",)]
    # The replay did not re-record itself.
    kinds = [kind for kind, _ in reflow._blocks]
    assert kinds.count("print") == 2


# ---------------------------------------------------------------------------
# watcher loop
# ---------------------------------------------------------------------------


def test_watcher_reflows_when_width_changes(chat_app_running, fake_in_terminal):
    async def scenario():
        console = _FakeConsole(width=100)
        reflow = responsive.ResizeReflow(
            console, app=chat_app_running, poll_interval=0.01
        )
        calls: list[int] = []
        reflow.add(lambda: calls.append(console.width))
        reflow.start()
        await asyncio.sleep(0.03)  # no width change yet -> no reflow
        console.width = 60
        await asyncio.sleep(0.03)  # width changed -> reflow
        reflow.stop()
        return calls, console

    calls, console = _run(scenario())
    assert calls == [100, 60]  # initial print + one reflow at new width
    assert console.cleared == 1


def test_watcher_ignores_same_width(chat_app_running, fake_in_terminal):
    async def scenario():
        console = _FakeConsole(width=100)
        reflow = responsive.ResizeReflow(
            console, app=chat_app_running, poll_interval=0.01
        )
        calls: list[int] = []
        reflow.add(lambda: calls.append(console.width))
        reflow.start()
        await asyncio.sleep(0.06)  # multiple ticks, width unchanged
        reflow.stop()
        return calls, console

    calls, console = _run(scenario())
    assert calls == [100]  # no reflow without a width change
    assert console.cleared == 0


def test_watcher_skips_when_other_app_running(chat_app, fake_in_terminal, monkeypatch):
    # A different prompt_toolkit app (e.g. a questionary picker) is live.
    monkeypatch.setattr(responsive, "get_app_or_none", lambda: _FakeApp())

    async def scenario():
        console = _FakeConsole(width=100)
        reflow = responsive.ResizeReflow(console, app=chat_app, poll_interval=0.01)
        calls: list[int] = []
        reflow.add(lambda: calls.append(console.width))
        reflow.start()
        await asyncio.sleep(0.02)
        console.width = 60
        await asyncio.sleep(0.03)
        reflow.stop()
        return calls, console

    calls, console = _run(scenario())
    assert calls == [100]  # never reflows under a foreign app
    assert console.cleared == 0


def test_watcher_skips_when_no_app_running(monkeypatch, fake_in_terminal):
    monkeypatch.setattr(responsive, "get_app_or_none", lambda: None)

    async def scenario():
        console = _FakeConsole(width=100)
        reflow = responsive.ResizeReflow(console, poll_interval=0.01)
        calls: list[int] = []
        reflow.add(lambda: calls.append(console.width))
        reflow.start()
        await asyncio.sleep(0.02)
        console.width = 60
        await asyncio.sleep(0.03)
        reflow.stop()
        return calls

    calls = _run(scenario())
    assert calls == [100]  # mid-processing: no app -> no reflow


def test_stop_ends_watcher(chat_app_running, fake_in_terminal):
    async def scenario():
        console = _FakeConsole(width=100)
        reflow = responsive.ResizeReflow(
            console, app=chat_app_running, poll_interval=0.01
        )
        calls: list[int] = []
        reflow.add(lambda: calls.append(console.width))
        reflow.start()
        reflow.stop()
        await asyncio.sleep(0.02)
        console.width = 50
        await asyncio.sleep(0.03)
        return calls

    calls = _run(scenario())
    assert calls == [100]
