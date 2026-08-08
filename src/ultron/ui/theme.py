"""
ultron.ui.theme
~~~~~~~~~~~~~~~

All Rich-based rendering helpers live here so the rest of the codebase
can import a single, consistent UI surface instead of scattering inline
Panel/Rule/Text calls across modules.

Palette
-------
  Accent / borders   : orange3  (#ff8700)
  Headers / titles   : bold #ffcc00
  Labels / dim text  : dim / grey50
  Success            : bold green
  Error              : bold red
  Info / neutral     : bold cyan
  User input label   : bold cyan
"""

from __future__ import annotations

from typing import Any

from rich import box as richbox
from rich.columns import Columns
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

# ---------------------------------------------------------------------------
# Shared console — import this everywhere instead of creating a new Console()
# ---------------------------------------------------------------------------
console = Console()

# Minimum safe terminal width: below this we skip expansion to avoid wrapping
_MIN_WIDTH = 40


def _term_width() -> int:
    """Return the current terminal width (Rich re-measures on every access)."""
    return console.width


def _safe_expand(width: int | None = None) -> bool:
    """Return True when the terminal is wide enough to expand panels."""
    return (width if width is not None else _term_width()) >= _MIN_WIDTH


def _panel_padding(width: int | None = None) -> tuple[int, int]:
    """
    Return (vertical, horizontal) panel padding.

    Narrow terminals get zero horizontal padding so content keeps more of the
    viewport; wide terminals keep the comfortable breathing room.
    """
    return (0, 0) if (width if width is not None else _term_width()) < 60 else (0, 2)


def _banner_mode(width: int) -> str:
    """
    Pick a banner layout for the terminal width.

    - "side":  ASCII logo beside the session details (wide terminals)
    - "stack": ASCII logo above the details (medium terminals)
    - "text":  single-line wordmark, no ASCII art (very narrow terminals)
    """
    if width >= 116:
        return "side"
    if width >= 46:
        return "stack"
    return "text"


# ---------------------------------------------------------------------------
# UI — static helper class grouping all render primitives
# ---------------------------------------------------------------------------

class UI:
    """
    Static render helpers for the Ultron CLI theme.

    Usage
    -----
    >>> from ultron.ui.theme import UI, console
    >>> UI.render_action_card(
    ...     title="Confirmation Required",
    ...     action="Create new file",
    ...     target="notes.txt",
    ...     preview="Hello world",
    ... )
    """

    # ------------------------------------------------------------------
    # Prompts / Headers
    # ------------------------------------------------------------------

    @staticmethod
    def render_user_prompt(text: str) -> None:
        """Render the user prompt line cleanly with matching Antigravity style."""
        console.print(f"[bold #ffcc00]>[/bold #ffcc00] {text}")

    # ------------------------------------------------------------------
    # Dividers / structure
    # ------------------------------------------------------------------

    @staticmethod
    def render_divider(title: str = "", *, style: str = "#ff8700") -> None:
        """Render a full-width Rule divider with an optional title."""
        if title:
            console.print(Rule(f"[bold {style}]{title}[/bold {style}]", style=style, align="right"))
        else:
            console.print(Rule(style=style))

    # ------------------------------------------------------------------
    # Status messages
    # ------------------------------------------------------------------

    @staticmethod
    def render_status(message: str, status: str = "info") -> None:
        """
        Print a one-line status message styled by *status*.
        """
        _style_map = {
            "info":    "bold cyan",
            "success": "bold green",
            "error":   "bold red",
            "warning": "bold yellow",
        }
        style = _style_map.get(status, "white")
        prefix_map = {
            "success": "✓",
            "error":   "✗",
            "warning": "!",
        }
        prefix = prefix_map.get(status, "")
        if prefix:
            console.print(f"[{style}]{prefix}[/{style}]  {message}")
        else:
            console.print(f"[{style}]{message}[/{style}]")

    # ------------------------------------------------------------------
    # Ultron AI response panel
    # ------------------------------------------------------------------

    @staticmethod
    def render_response(content: str) -> None:
        """
        Render the main Ultron AI response inside a clean 4-sided ROUNDED panel.
        Supports Markdown parsing so LLM outputs and code snippets render properly.
        """
        renderable = Markdown(content) if isinstance(content, str) else content

        console.print(
            Panel(
                renderable,
                title="[bold black on #ff8700] ULTRON [/bold black on #ff8700]",
                title_align="center",
                border_style="#ff8700",
                box=richbox.ROUNDED,
                expand=_safe_expand(),
                padding=_panel_padding(),
            )
        )
        console.print()

    # ------------------------------------------------------------------
    # Tool Calls / Memory Logs (Claude Code Style)
    # ------------------------------------------------------------------

    @staticmethod
    def render_tool_execution(tool_name: str, output: Any) -> None:
        """
        Render tool execution outputs (like memory retrieval) inside a minimal dim box.
        """
        body = Text(overflow="fold")
        body.append(f"Executed tool '{tool_name}':\n\n", style="bold #ffcc00")
        body.append(f"{output}", style="dim white")

        console.print(
            Panel(
                body,
                border_style="grey50",
                box=richbox.SQUARE,
                expand=_safe_expand(),
                padding=_panel_padding(),
            )
        )

    # ------------------------------------------------------------------
    # Confirmation / action cards
    # ------------------------------------------------------------------

    @staticmethod
    def render_action_card(
        title: str,
        action: str,
        target: str = "",
        preview: str = "",
    ) -> None:
        """
        Render a confirmation action card with structured label rows.
        """
        body = Text(overflow="fold")

        body.append(" Action  ", style="bold #ff9500")
        body.append("│ ", style="dim")
        body.append(f"{action}\n", style="white")

        if target:
            body.append(" Target  ", style="bold #ffcc00")
            body.append("│ ", style="dim")
            body.append(f"{target}\n", style="bold white")

        if preview:
            body.append(" Preview ", style="bold dim")
            body.append("│ ", style="dim")
            body.append(f"{preview}", style="italic grey50")

        console.print(
            Panel(
                body,
                title=f"[bold #ffcc00]⚡ {title}[/bold #ffcc00]",
                title_align="left",
                subtitle="[dim]answer below[/dim]",
                subtitle_align="right",
                border_style="#ff8700",
                box=richbox.ROUNDED,
                expand=_safe_expand(),
                padding=_panel_padding(),
            )
        )

    # ------------------------------------------------------------------
    # Error / fallback panel
    # ------------------------------------------------------------------

    @staticmethod
    def render_error(message: str, title: str = "Error") -> None:
        """Render a red error panel."""
        console.print(
            Panel(
                f"[bold red]{message}[/bold red]",
                title=f"[bold red]✗ {title}[/bold red]",
                title_align="left",
                border_style="red",
                box=richbox.HEAVY,
                expand=_safe_expand(),
                padding=_panel_padding(),
            )
        )
        console.print()

    # ------------------------------------------------------------------
    # Startup helpers (Full Logo Antigravity Side-by-Side Layout)
    # ------------------------------------------------------------------

    @staticmethod
    def render_banner(model: str, cwd_short: str) -> None:
        """
        Header banner that adapts to the terminal width:

        - wide terminals: the full ULTRON ASCII wordmark beside the session
          details (Rich Columns, side-by-side)
        - medium terminals: the wordmark stacked above the details
        - very narrow terminals: a compact single-line wordmark with no ASCII
          art, so nothing ever wraps or overflows the viewport
        """
        from ultron import __version__

        mode = _banner_mode(_term_width())

        logo_lines = [
            "██╗   ██╗██╗  ████████╗██████╗  ██████╗ ███╗   ██╗",
            "██║   ██║██║  ╚══██╔══╝██╔══██╗██╔═══██╗████╗  ██║",
            "██║   ██║██║     ██║   ██████╔╝██║   ██║██╔██╗ ██║",
            "██║   ██║██║     ██║   ██╔══██╗██║   ██║██║╚██╗██║",
            "╚██████╔╝███████╗██║   ██║  ██║╚██████╔╝██║ ╚████║",
            " ╚═════╝ ╚══════╝╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝",
        ]
        colors = ["#ffcc00", "#ff9500", "#ff5500", "#ff5500", "#ff2200", "#cc0000"]

        logo_text = Text()
        for line, color in zip(logo_lines, colors):
            logo_text.append(f"{line}\n", style=f"bold {color}")

        # Meta Context Details Stack
        info_text = Text()
        info_text.append(f"\nUltron CLI v{__version__}\n", style="bold #ffcc00")
        info_text.append(f"Model: {model}\n", style="bold white")
        info_text.append(f"Directory: {cwd_short}", style="dim white")

        console.print()
        if mode == "side":
            # Render full logo and details side-by-side
            console.print(Columns([logo_text, info_text], padding=(0, 2), equal=False))
        elif mode == "stack":
            console.print(logo_text)
            console.print(info_text)
        else:
            console.print(
                f"[bold #ffcc00]⚡ ULTRON AI[/] v{__version__}  "
                f"[dim]Model:[/] [bold #ff9500]{model}[/]  "
                f"[dim]Dir:[/] [bold cyan]{cwd_short}[/]"
            )
        console.print()

    @staticmethod
    def render_status_bar(model: str, cwd_short: str = "") -> None:
        """
        Renders a compact single-line status bar.  On narrow terminals the
        directory segment is dropped so the line never wraps.
        """
        from ultron import __version__

        width = _term_width()
        line = f"[bold cyan]⚡ Ultron CLI v{__version__}[/] | [dim]Model:[/] [bold magenta]{model}[/]"
        if width >= 90:
            dir_str = cwd_short or "~"
            line += f" | [dim]Dir:[/] [bold yellow]{dir_str}[/]"
        console.print(line)

    # ------------------------------------------------------------------
    # Help table
    # ------------------------------------------------------------------

    @staticmethod
    def render_help_table() -> None:
        """Print a formatted table of available slash commands."""
        from rich.table import Table

        table = Table(
            title="[bold orange3]Available Commands[/bold orange3]",
            show_header=True,
            header_style="bold #ff9500",
            box=richbox.ROUNDED,
            border_style="orange3",
            expand=_safe_expand(),
        )
        table.add_column("Command", style="bold #ffcc00", no_wrap=True)
        table.add_column("Description", style="white")

        table.add_row("/help",          "Show this help table.")
        table.add_row("/model",         "Select an available LLM model interactively.")
        table.add_row("/memory",        "Show the knowledge-graph memory stats and stored triples.")
        table.add_row("/agent",         "Switch agent type (simple or react); use /agent <type> to skip the picker.")
        table.add_row("/security",      "Show security mode + tier policy; use /security <permissive|interactive|strict> to switch.")
        table.add_row("/clear",         "Reset conversation history to start fresh.")
        table.add_row("/reload",        "Best-effort code reload without restarting (may not catch all changes; restart if unsure).")
        table.add_row("/exit, /quit",   "Exit the chat session.")
        table.add_row("Esc",            "Cancel current task execution while model is thinking.")

        console.print(table)
        console.print()