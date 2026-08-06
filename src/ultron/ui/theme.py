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
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text
from rich.columns import Columns
from rich.markdown import Markdown
from rich import box as richbox

# ---------------------------------------------------------------------------
# Shared console — import this everywhere instead of creating a new Console()
# ---------------------------------------------------------------------------
console = Console()

# Minimum safe terminal width: below this we skip expansion to avoid wrapping
_MIN_WIDTH = 40


def _safe_expand() -> bool:
    """Return True when the terminal is wide enough to expand panels."""
    return console.width >= _MIN_WIDTH


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
                padding=(0, 2),
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
                padding=(0, 1),
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
                padding=(0, 1),
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
                padding=(0, 2),
            )
        )
        console.print()

    # ------------------------------------------------------------------
    # Startup helpers (Full Logo Antigravity Side-by-Side Layout)
    # ------------------------------------------------------------------

    @staticmethod
    def render_banner(model: str, cwd_short: str) -> None:
        """
        Side-by-side header using Rich Columns with the complete ULTRON wordmark logo.
        Automatically scales on narrow terminals without banner line wrapping bugs.
        """
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
        info_text.append("\nUltron CLI v1.0.0\n", style="bold #ffcc00")
        info_text.append(f"Model: {model}\n", style="bold white")
        info_text.append(f"Directory: {cwd_short}", style="dim white")

        console.print()
        # Render full logo and details side-by-side
        console.print(Columns([logo_text, info_text], padding=(0, 2), equal=False))
        console.print()

    @staticmethod
    def render_status_bar(model: str, cwd_short: str = "") -> None:
        """
        Renders a compact single-line status bar displaying current version, active model, and directory.
        """
        dir_str = cwd_short or "~"
        console.print(
            f"[bold cyan]⚡ Ultron CLI v1.0.0[/] | [dim]Model:[/] [bold magenta]{model}[/] | [dim]Dir:[/] [bold yellow]{dir_str}[/]"
        )

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
        table.add_row("/clear",         "Reset conversation history to start fresh.")
        table.add_row("/exit, /quit",   "Exit the chat session.")
        table.add_row("Esc",            "Cancel current task execution while model is thinking.")

        console.print(table)
        console.print()