"""
ultron.ui.responsive
~~~~~~~~~~~~~~~~~~~~

Terminal-resize reflow for the chat screen.

Rich prints every panel (banner, response boxes, tool outputs, slash-command
tables) into the terminal's scrollback. Unlike prompt_toolkit's own prompt
area — which re-renders on every resize — scrollback content keeps the width
it was drawn at, so a panel printed at 120 columns stays 120 columns wide and
its borders wrap into garbage once the window is shrunk.

``ResizeReflow`` fixes this for the whole conversation: while the chat prompt
is the active application, it watches the terminal width and, when it changes,
re-renders the entire recorded screen at the new width. Every ``console.print``
made during the session is recorded as a replayable block; rich re-renders
those renderables at the console's live width, so re-running them produces a
perfectly reflowed screen. The redraw runs inside prompt_toolkit's
``in_terminal`` context, so the live prompt is suspended, the screen cleared
and rewritten, and the prompt re-rendered below it — the exact mechanism
``print_formatted_text`` uses to print above a running prompt.

Guards:

- The watcher only reflows while the **chat** application (the identity passed
  in) is the running prompt_toolkit app. Interactive side-apps such as
  ``questionary`` prompts are never disturbed.
- While a rich ``Live`` is active (the *Thinking...* spinner) no prints are
  recorded, so spinner frames never pollute the transcript.
- The screen is never cleared while the agent is processing (the chat app is
  not running then), so mid-processing reflow cannot fight the spinner.

The transcript is held in memory for the session — a deliberate tradeoff:
capping it would stop older content from reflowing. It is dropped on
``/clear`` (``reset``) and on process exit.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from prompt_toolkit.application.current import get_app_or_none
from prompt_toolkit.application.run_in_terminal import in_terminal
from rich.console import Console

from ultron.core.logging import get_logger
from ultron.ui.theme import console as _default_console

logger = get_logger("ultron.ui.responsive")

# Block kinds stored in the transcript.
_PRINT = "print"
_REBUILD = "rebuild"


class ResizeReflow:
    """
    Records every renderable printed to *console* during a chat session and,
    on terminal resize, re-renders the whole recorded screen at the live
    width — so every response box, tool output and table follows the window.

    Usage
    -----
    >>> region = ResizeReflow(console, app=session.app)
    >>> region.add(lambda: print_banner(model))     # rebuild block (banner)
    >>> region.start()                              # begin watching resize
    >>> ...
    >>> region.stop()                               # restore + cancel watcher
    """

    def __init__(
        self,
        console: Console = _default_console,
        *,
        app: Any = None,
        poll_interval: float = 0.2,
    ) -> None:
        self._console = console
        self._app = app
        self._poll_interval = poll_interval
        self._blocks: list[tuple[str, Any]] = []
        self._last_width: int | None = None
        self._task: asyncio.Task[None] | None = None
        self._started = False
        self._installed = False
        self._recording = True
        self._reflowing = False
        self._orig_print: Callable[..., Any] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, printer: Callable[[], None]) -> None:
        """Register a rebuild block and run it now.

        Rebuild blocks re-execute a full render helper (e.g. ``print_banner``)
        on reflow, so width-dependent layout decisions such as the banner's
        side/stack/text mode are re-evaluated. Prints emitted while the block
        runs are not recorded individually (the block covers them).
        """
        self._blocks.append((_REBUILD, printer))
        self._recording = False
        try:
            printer()
        finally:
            self._recording = True
        if self._last_width is None:
            self._last_width = self._console.width

    def rebuild(self) -> None:
        """Re-run every rebuild block now, without clearing the screen.

        Used when live state a rebuild block reads changes outside of a
        resize (e.g. the active model shown in the banner): the block is
        already on screen, so re-running it refreshes its content in place
        instead of recording a second copy into the transcript.
        """
        self._recording = False
        try:
            for kind, payload in self._blocks:
                if kind == _REBUILD:
                    payload()
        finally:
            self._recording = True
        self._last_width = self._console.width

    def reset(self) -> None:
        """Drop every recorded print block, keeping the rebuild blocks.

        Used by ``/clear``: the screen keeps what it shows, but a later
        resize replays only the rebuild blocks (e.g. the banner) plus the
        new conversation — never the wiped one.
        """
        self._blocks = [b for b in self._blocks if b[0] == _REBUILD]
        self._last_width = self._console.width

    def start(self) -> None:
        """Wrap ``console.print`` (recording) and begin watching. Idempotent."""
        if self._started:
            return
        self._started = True
        self._install()
        self._task = asyncio.get_running_loop().create_task(self._watch())

    def stop(self) -> None:
        """Restore ``console.print`` and cancel the watcher."""
        self._restore()
        if self._task is not None:
            self._task.cancel()
            self._task = None
        self._started = False

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _install(self) -> None:
        if self._installed:
            return
        self._installed = True
        self._orig_print = self._console.print

        def recorded_print(*args: Any, **kwargs: Any) -> Any:
            # Skip recording while a rich Live is active (the thinking
            # spinner re-renders via console.print internally) and while we
            # are replaying the transcript ourselves.
            if (
                self._recording
                and not self._reflowing
                and not getattr(self._console, "_live_stack", ())
            ):
                self._blocks.append((_PRINT, (args, kwargs)))
            return self._orig_print(*args, **kwargs)

        # rich's Console has no __slots__, so an instance attribute cleanly
        # shadows the bound method for this console only.
        self._console.print = recorded_print  # type: ignore[method-assign]

    def _restore(self) -> None:
        if self._installed and self._orig_print is not None:
            self._console.print = self._orig_print  # type: ignore[method-assign]
        self._installed = False

    def _is_chat_app_active(self) -> bool:
        app = get_app_or_none()
        if self._app is None:
            return app is not None
        return app is self._app

    async def _watch(self) -> None:
        try:
            while True:
                await asyncio.sleep(self._poll_interval)
                if not self._blocks:
                    continue
                # Only reflow while the chat prompt itself is on screen.
                # During questionary prompts or mid-processing the app is a
                # different (or absent) application — touching the screen then
                # would corrupt it.
                if not self._is_chat_app_active():
                    continue
                width = self._console.width
                if width == self._last_width:
                    continue
                try:
                    await self._reflow()
                except Exception:  # reflow must never crash the chat
                    logger.debug("reflow skipped after resize", exc_info=True)
                    # Keep the old width so the next tick retries; a failed
                    # reflow must not be marked as handled.
                    continue
                self._last_width = width
        except asyncio.CancelledError:
            pass

    async def _reflow(self) -> None:
        self._reflowing = True
        try:
            async with in_terminal():
                self._console.clear()
                for kind, payload in self._blocks:
                    if kind == _REBUILD:
                        payload()
                    else:
                        args, kwargs = payload
                        self._console.print(*args, **kwargs)
        finally:
            self._reflowing = False
