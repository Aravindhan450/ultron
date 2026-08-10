"""
ultron.ui.session
~~~~~~~~~~~~~~~~~

Live, terminal-width-aware chat session built on prompt_toolkit.

The chat prompt renders standard input prompts without a persistent bottom toolbar.

All of the building blocks are pure functions of the terminal ``width`` so
they are trivial to unit-test and safe to call from anywhere.
"""

from __future__ import annotations

from collections.abc import Callable

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML, AnyFormattedText

from ultron.core.state import CLIState
from ultron.ui.theme import BLUE, FAINT, GREEN, MUTED, RED, YELLOW, console

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
        return f"fg:{GREEN}"
    if status.startswith(("Thinking", "Executing")):
        return f"fg:{YELLOW}"
    return f"fg:{BLUE}"


def _security_chip_style(security_mode: str) -> str:
    """Map a security mode to a toolbar colour (green/yellow/red traffic light)."""
    return {
        "permissive": f"fg:{GREEN}",
        "interactive": f"fg:{YELLOW}",
        "strict": f"fg:{RED}",
    }.get(security_mode, f"fg:{BLUE}")


def build_toolbar(
    state: CLIState,
    agent_tag: str,
    security_mode: str,
    width: int,
) -> list[tuple[str, str]]:
    """
    Build the live bottom-toolbar fragments for the given terminal width.

Segments are dropped progressively as the terminal narrows so the row
never overflows: the status dot always renders (its text needs >= 44
columns), the model and agent always render, the cwd needs >= 100, the
security chip >= 88, and the right-aligned key hints appear only when
they fit.
    """
    segments: list[tuple[str, str]] = []

    def seg(style: str, text: str) -> None:
        segments.append((style, text))

    if width >= 44:
        seg(f"bold {_status_style(state.status)}", f"● {state.status} ")
    else:
        seg(f"bold {_status_style(state.status)}", "● ")
    seg(f"fg:{MUTED}", f"{state.active_model} ")
    seg(f"fg:{FAINT}", f"{agent_tag} ")

    if width >= 100:
        seg(f"fg:{MUTED}", f"{state.current_dir} ")
    if width >= 88 and security_mode:
        seg(f"{_security_chip_style(security_mode)} bold", f"🔒 {security_mode} ")

    fragments: list[tuple[str, str]] = []
    for index, (style, text) in enumerate(segments):
        if index:
            fragments.append((f"fg:{FAINT}", "│ "))
        fragments.append((style, text))

    hints = " Esc cancel · /help · Ctrl+D quit "
    content_width = sum(_display_width(text) for _, text in fragments)
    if width >= 70:
        pad = width - content_width - _display_width(hints)
        # At least one space of separation, otherwise skip the hints entirely
        # (a pad of exactly 0 would push the row one cell past the width).
        if pad >= 1:
            fragments.append(("fg:gray", " " * pad))
            fragments.append((f"fg:{FAINT}", hints))

    return fragments


def build_prompt_html(model: str, agent_tag: str, width: int) -> str:
    """
    Build the input-prompt markup.

    Claude Code style: the prompt line is just the ``❯`` marker — model and
    agent context live in the bottom toolbar, never crowding the input line
    at any width.
    """
    return f"<b><style fg='{MUTED}'>❯</style></b> "


# ---------------------------------------------------------------------------
# ChatSession — live prompt wrapper
# ---------------------------------------------------------------------------


class ChatSession:
    """
    prompt_toolkit-backed chat prompt with a live, width-adaptive toolbar.

    The toolbar is a callable, so prompt_toolkit re-invokes it on every
    render — including on terminal resize — which is what makes the
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

    def _clear_input_line(self, text: str) -> None:
        """
        Erase the just-submitted input line(s) from the screen.

        After ``prompt_async`` returns, prompt_toolkit leaves the submitted
        text visible on the input line (with the bottom toolbar below it)
        until the next render.  The chat loop echoes the user's message into
        the transcript right after, so without clearing here the message
        would appear twice on screen — Claude Code consumes the input line
        on submit and shows the message once in the conversation.

        The toolbar row is erased too (it re-renders on the next prompt), so
        the geometry is correct whether or not the terminal supports CPR.
        """
        import shutil
        import sys

        width = shutil.get_terminal_size((80, 24)).columns
        if width <= 0:
            width = console.width
        input_rows = max(1, (2 + _display_width(text) + width - 1) // width)
        rows = input_rows  # input row(s) (no bottom toolbar)
        out = ["\r"]
        out.append(f"\x1b[{rows}A")  # up to the top of the prompt area
        for _ in range(rows):
            out.append("\x1b[2K")  # erase whole line
            out.append("\x1b[1B")  # down one
        out.append(f"\x1b[{rows}A")  # back up to where the echo will land
        try:
            sys.stdout.write("".join(out))
            sys.stdout.flush()
        except OSError:  # cosmetic op — never let a terminal glitch crash the chat
            pass

    async def prompt_async(self) -> str:
        text = await self.session.prompt_async(
            HTML(build_prompt_html(self.state.active_model, self._agent_tag(), width=console.width)),
        )
        # The echo below must not duplicate the still-visible input line.
        self._clear_input_line(text)
        return text
