"""
ultron.ui.session
~~~~~~~~~~~~~~~~~

Live, terminal-width-aware chat session built on prompt_toolkit.

The chat prompt renders with a persistent bottom toolbar that reflects the
current model, agent, directory, security mode and agent status.  Because
prompt_toolkit re-renders the screen on every keystroke *and* on every
terminal resize, the toolbar re-flows automatically when the window is
resized: wide terminals get the full status row (cwd, security chip, key
hints) while narrow terminals get a compact row that still fits.

All of the building blocks are pure functions of the terminal ``width`` so
they are trivial to unit-test and safe to call from anywhere.
"""

from __future__ import annotations

from collections.abc import Callable
from html import escape

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML, AnyFormattedText

from ultron.core.state import CLIState
from ultron.ui.theme import console

try:  # wcwidth is a runtime dependency of prompt_toolkit
    from wcwidth import wcswidth

    def _display_width(text: str) -> int:
        width = wcswidth(text)
        return width if width >= 0 else len(text)

except ImportError:  # pragma: no cover - unreachable while prompt_toolkit is installed

    def _display_width(text: str) -> int:  # type: ignore[no-redef]
        return len(text)


# ---------------------------------------------------------------------------
# Pure width-adaptive builders
# ---------------------------------------------------------------------------


def _status_style(status: str) -> str:
    """Map an agent status string to a toolbar colour."""
    if status == "Ready":
        return "fg:ansigreen"
    if status.startswith(("Thinking", "Executing")):
        return "fg:ansiyellow"
    return "fg:ansicyan"


def _security_chip_style(security_mode: str) -> str:
    """Map a security mode to a toolbar colour (green/yellow/red traffic light)."""
    return {
        "permissive": "fg:ansigreen",
        "interactive": "fg:ansiyellow",
        "strict": "fg:ansired",
    }.get(security_mode, "fg:ansicyan")


def build_toolbar(
    state: CLIState,
    agent_tag: str,
    security_mode: str,
    width: int,
) -> list[tuple[str, str]]:
    """
    Build the live bottom-toolbar fragments for the given terminal width.

    Segments are dropped progressively as the terminal narrows so the row
    never overflows: the status dot, model and agent always render; the
    decorative wordmark chip needs >= 44 columns, the cwd >= 100, the
    security chip >= 88, and the right-aligned key hints appear only when
    they fit.
    """
    segments: list[tuple[str, str]] = []

    def seg(style: str, text: str) -> None:
        segments.append((style, text))

    if width >= 44:
        seg("bold fg:ansiyellow", " ⚡ Ultron ")
    if width >= 56:
        seg(f"bold {_status_style(state.status)}", f"● {state.status} ")
    else:
        seg(f"bold {_status_style(state.status)}", "● ")
    seg("fg:ansicyan bold", f"{state.active_model} ")
    seg("fg:ansimagenta bold", f"{agent_tag} ")

    if width >= 100:
        seg("fg:ansibrightblack", f"{state.current_dir} ")
    if width >= 88 and security_mode:
        seg(f"{_security_chip_style(security_mode)} bold", f"🔒 {security_mode} ")

    fragments: list[tuple[str, str]] = []
    for index, (style, text) in enumerate(segments):
        if index:
            fragments.append(("fg:gray", "│ "))
        fragments.append((style, text))

    hints = " Esc cancel · /help · Ctrl+D quit "
    content_width = sum(_display_width(text) for _, text in fragments)
    if width >= 70:
        pad = width - content_width - _display_width(hints)
        # At least one space of separation, otherwise skip the hints entirely
        # (a pad of exactly 0 would push the row one cell past the width).
        if pad >= 1:
            fragments.append(("fg:gray", " " * pad))
            fragments.append(("fg:ansibrightblack", hints))

    return fragments


def build_rprompt(security_mode: str, width: int) -> list[tuple[str, str]]:
    """
    Right-side chip on the input line.

    Hidden entirely on very narrow terminals so it never crowds the prompt.
    """
    if width < 56 or not security_mode:
        return []
    return [(f"{_security_chip_style(security_mode)} bold", f"🔒 {security_mode}")]


def build_prompt_html(model: str, agent_tag: str, width: int) -> str:
    """
    Build the input-prompt markup.

    Wide terminals get the model/agent context chips; very narrow terminals
    fall back to a bare arrow so the input line never crowds the screen.
    """
    if width < 46:
        return "<b><style fg='#ffcc00'>❯</style></b> "
    return (
        "<b><style fg='#ffcc00'>❯</style></b> "
        f"<b><style fg='#ff9500'>{escape(model)}</style></b>"
        f"<style fg='#6a6a6a'>/{escape(agent_tag)}</style> "
        "<ansiblue>You</ansiblue> <style fg='#6a6a6a'>›</style> "
    )


# ---------------------------------------------------------------------------
# ChatSession — live prompt wrapper
# ---------------------------------------------------------------------------


class ChatSession:
    """
    prompt_toolkit-backed chat prompt with a live, width-adaptive toolbar.

    The toolbar and rprompt are callables, so prompt_toolkit re-invokes them
    on every render — including on terminal resize — which is what makes the
    interface respond to window changes without any extra bookkeeping.
    """

    def __init__(
        self,
        state: CLIState,
        *,
        session: PromptSession | None = None,
        agent_tag: Callable[[], str],
        security_mode: Callable[[], str] | None = None,
    ) -> None:
        self.state = state
        self.session = session if session is not None else PromptSession()
        self._agent_tag = agent_tag
        self._security_mode = security_mode or (lambda: "")

    def _toolbar(self) -> AnyFormattedText:
        return build_toolbar(
            self.state,
            self._agent_tag(),
            self._security_mode(),
            width=console.width,
        )

    def _rprompt(self) -> AnyFormattedText:
        return build_rprompt(self._security_mode(), width=console.width)

    async def prompt_async(self) -> str:
        return await self.session.prompt_async(
            HTML(build_prompt_html(self.state.active_model, self._agent_tag(), width=console.width)),
            bottom_toolbar=self._toolbar,
            rprompt=self._rprompt,
        )
