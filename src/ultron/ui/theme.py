"""
ultron.ui.theme
~~~~~~~~~~~~~~~

All Rich-based rendering helpers live here so the rest of the codebase
can import a single, consistent UI surface instead of scattering inline
Panel/Rule/Text calls across modules.

The visual language is deliberately minimal, in the spirit of Claude Code's
terminal UI: the conversation flows as quiet text, meta information is muted
gray, and color is used sparingly — a molten-orange brand accent, a status
dot, and semantic green/red/yellow for outcomes.  Assistant replies render
inside a thin orange-bordered panel, and the startup banner carries the full
flame gradient.  On terminal resize the whole screen re-renders at the live
width (via ``ultron.ui.responsive``), so every renderable here is
width-adaptive.

Palette (GitHub-dark inspired + Ultron molten flame)
-----------------------------------------------------
  Text            : #e6edf3  near-white body text
  Muted / meta    : #8b949e  gray — labels and secondary info
  Faint           : #5f6b7a  barely-visible hints
  Accent (sparse) : #FB6C00  orange — the primary brand colour
  Flame gradient  : #E73F1E → #FB6C00 → #F9B637 → #FFDD9C (logo)
  Success         : #3fb950
  Error           : #f85149
  Warning         : #d29922
  Info            : #58a6ff  soft blue
"""

from __future__ import annotations

from typing import Any

from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

# ---------------------------------------------------------------------------
# Shared console — import this everywhere instead of creating a new Console()
# ---------------------------------------------------------------------------
# highlight=False: rich's automatic syntax highlighting would paint fragments
# of plain strings (numbers, paths, URLs inside model names / versions / cwd)
# in unrelated colours — e.g. "gemma4:e4b" became "gemm" + green "a4:e4b".
# The theme controls every colour explicitly, so auto-highlighting is off.
console = Console(highlight=False)

# Claude Code-inspired neutrals + Ultron molten-flame brand palette
TEXT = "#e6edf3"     # near-white body text
MUTED = "#8b949e"    # dim gray — meta information
FAINT = "#5f6b7a"    # barely-visible hints
GREEN = "#3fb950"
RED = "#f85149"
YELLOW = "#d29922"
BLUE = "#58a6ff"

# The four Ultron brand colours — the molten flame gradient.
EMBER = "#E73F1E"    # red-orange — deepest flame
ORANGE = "#FB6C00"   # orange — the primary brand accent
AMBER = "#F9B637"    # amber — mid gradient
GOLD = "#FFDD9C"     # pale gold — lightest flame tip

# Primary brand accent — sparse highlights (replaces the old warm amber).
ACCENT = ORANGE

_PREFIX = {"success": "✓", "error": "✗", "warning": "!"}
_STATUS_COLOR = {"info": MUTED, "success": GREEN, "error": RED, "warning": YELLOW}


def _term_width() -> int:
    """Current terminal width, falling back to a sane default."""
    width = console.width
    return width if width and width > 0 else 100


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
        """
        Render the user's message with the Claude Code ``❯`` marker.

        The text is markup-escaped so user input containing ``[`` / ``]`` can
        never be parsed as Rich markup.
        """
        console.print(UI.user_prompt_markup(text))

    @staticmethod
    def user_prompt_markup(text: str) -> str:
        """
        Rich markup for a user-message line (``❯ text``), markup-escaped.

        Extracted so the resize-reflow transcript can record the exact same
        line that ``render_user_prompt`` prints.
        """
        return f"[{MUTED}]❯[/{MUTED}] {escape(text)}"

    # ------------------------------------------------------------------
    # Dividers / structure
    # ------------------------------------------------------------------

    @staticmethod
    def render_divider(title: str = "", *, style: str = FAINT) -> None:
        """Render a full-width Rule divider with an optional muted title."""
        if title:
            console.print(Rule(f"[dim {MUTED}]{title}[/dim {MUTED}]", style=style, align="right"))
        else:
            console.print(Rule(style=style))

    # ------------------------------------------------------------------
    # Status messages
    # ------------------------------------------------------------------

    @staticmethod
    def render_status(message: str, status: str = "info") -> None:
        """
        Print a one-line status message styled by *status*.

        Success / error / warning get a colored ✓ / ✗ / ! prefix; plain
        status (info, goodbye) renders as muted text.
        """
        prefix = _PREFIX.get(status)
        color = _STATUS_COLOR.get(status, TEXT)
        if prefix:
            console.print(f"[{color}]{prefix}[/{color}]  {message}")
        else:
            console.print(f"[{color}]{message}[/{color}]")

    # ------------------------------------------------------------------
    # Assistant response — plain flowing text, no box
    # ------------------------------------------------------------------

    @staticmethod
    def render_response(content: str) -> None:
        """
        Render the assistant reply inside a quiet boxed panel.

        A thin #FB6C00 (orange) border marks the response — the brand flame
        colour — while the content itself renders as plain Markdown.  The
        header carries only the ``ULTRON`` wordmark (caps, matching the
        logo): the model, version and directory are established once in the
        startup banner, so repeating them on every reply would be noise.
        The panel re-flows with the window (the resize reflow re-renders the
        recorded Panel at the live width on every resize).
        """
        renderable = Markdown(content) if isinstance(content, str) else content
        console.print(
            Panel(
                renderable,
                box=box.ROUNDED,
                border_style=ORANGE,
                padding=(0, 1),
                title=f"[bold {ACCENT}]ULTRON[/bold {ACCENT}]",
                title_align="center",
            )
        )
        console.print()

    # ------------------------------------------------------------------
    # Tool calls — compact chip lines
    # ------------------------------------------------------------------

    @staticmethod
    def render_tool_execution(tool_name: str, output: Any) -> None:
        """
        Render a tool call as a compact Claude Code-style chip line, with the
        output below in muted gray — no box.
        """
        console.print(f"[{ACCENT}]✻[/{ACCENT}] [bold {TEXT}]{tool_name}[/bold {TEXT}]")
        body = str(output)
        if body:
            console.print(f"[{MUTED}]{body}[/{MUTED}]")
        console.print()

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
        Render a pending-action confirmation as clean lines: a faint title,
        the ``✻ action — target`` chip, and an optional faint preview.

        Deliberately quiet: the decision prompt (questionary) is the focal
        point, not a loud bordered card.
        """
        if title:
            console.print(f"[{FAINT}]{title}[/{FAINT}]")
        line = f"[{ACCENT}]✻[/{ACCENT}] [bold {TEXT}]{action}[/bold {TEXT}]"
        if target:
            line += f"  [{MUTED}]—[/{MUTED}] {target}"
        console.print(line)
        if preview:
            console.print(f"  [{FAINT}]{preview}[/{FAINT}]")
        console.print()

    # ------------------------------------------------------------------
    # Error / fallback
    # ------------------------------------------------------------------

    @staticmethod
    def render_error(message: str, title: str = "Error") -> None:
        """Render an error as a single red line — no heavy panel."""
        if title and title != "Error":
            console.print(f"[{RED}]✗[/{RED}] [bold {RED}]{title}:[/bold {RED}] {message}")
        else:
            console.print(f"[{RED}]✗[/{RED}] {message}")
        console.print()

    # ------------------------------------------------------------------
    # Startup header — ASCII logo, adaptive to terminal width
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
        # Molten-flame gradient — the four brand colours across the six logo
        # lines, each colour used twice so the transition reads smooth:
        # red-orange, red-orange, orange, orange, amber, pale gold.
        logo_colors = [EMBER, EMBER, ORANGE, ORANGE, AMBER, GOLD]

        logo_text = Text()
        for line, color in zip(logo_lines, logo_colors):
            logo_text.append(f"{line}\n", style=f"bold {color}")

        info_text = Text()
        info_text.append(f"\n⚡ Ultron AI  v{__version__}\n", style=f"bold {ACCENT}")
        info_text.append(f"Model: {model}\n", style=TEXT)
        info_text.append(f"Dir: {cwd_short}", style=MUTED)

        console.print()
        if mode == "side":
            console.print(Columns([logo_text, info_text], padding=(0, 2), equal=False))
        elif mode == "stack":
            console.print(logo_text)
            console.print(info_text)
        else:
            console.print(
                f"[bold {ACCENT}]⚡ Ultron AI[/bold {ACCENT}]  "
                f"[dim {FAINT}]v{__version__}[/dim {FAINT}]"
            )
            console.print(f"[{MUTED}]Model:[/{MUTED}] {model}   [{MUTED}]Dir:[/{MUTED}] {cwd_short}")
        console.print()

    # ------------------------------------------------------------------
    # Help — borderless aligned list
    # ------------------------------------------------------------------

    @staticmethod
    def render_help_table() -> None:
        """Print the available slash commands as a clean borderless list."""
        # expand=True so the list spans exactly the console width — uniform
        # with every other panel in the chat screen (and re-flows on resize).
        table = Table(show_header=False, box=None, expand=True, padding=(0, 2))
        table.add_column(style=f"bold {ACCENT}", no_wrap=True)
        table.add_column(style=TEXT)

        rows = [
            ("/help", "Show this help table."),
            ("/model", "Select an available LLM model interactively."),
            ("/memory", "Show the knowledge-graph memory stats and stored triples."),
            ("/agent", "Switch agent type (simple or react); use /agent <type> to skip the picker."),
            ("/security", "Show security mode + tier policy; use /security <permissive|interactive|strict> to switch."),
            ("/clear", "Reset conversation history to start fresh."),
            ("/reload", "Best-effort code reload without restarting (may not catch all changes; restart if unsure)."),
            ("/exit, /quit", "Exit the chat session."),
            ("Esc", "Cancel current task execution while model is thinking."),
        ]

        console.print(f"[bold {ACCENT}]Available Commands[/bold {ACCENT}]")
        for command, description in rows:
            table.add_row(command, description)
        console.print(table)
        console.print()
