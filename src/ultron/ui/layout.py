"""
ultron.ui.layout
~~~~~~~~~~~~~~~~

Reactive Layout components for Rich Live TUI.
"""

from rich import box
from rich.console import RenderableType
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ultron.core.state import CLIState


def build_header(state: CLIState) -> RenderableType:
    """
    Dynamically compiles top header banner using state.
    Uses overflow='ellipsis' and no_wrap=True on header table cells so window
    minimization or resizing never breaks box borders or splits ASCII lines.
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

    logo_text = Text(no_wrap=True, overflow="ellipsis")
    for line, color in zip(logo_lines, colors):
        logo_text.append(f"{line}\n", style=f"bold {color}")

    # Meta Context Details
    info_table = Table.grid(padding=(0, 1), expand=True)
    info_table.add_column(justify="left", no_wrap=True, overflow="ellipsis")
    
    info_table.add_row(Text(f"ULTRON AI {state.version}", style="bold #ffcc00", no_wrap=True, overflow="ellipsis"))
    
    model_text = Text(no_wrap=True, overflow="ellipsis")
    model_text.append("Model: ", style="dim white")
    model_text.append(state.active_model, style="bold #ff9500")
    info_table.add_row(model_text)

    dir_text = Text(no_wrap=True, overflow="ellipsis")
    dir_text.append("Directory: ", style="dim white")
    dir_text.append(state.current_dir, style="bold cyan")
    info_table.add_row(dir_text)

    status_text = Text(no_wrap=True, overflow="ellipsis")
    status_text.append("Status: ", style="dim white")
    status_text.append(state.status, style="bold green" if state.status == "Ready" else "bold yellow")
    info_table.add_row(status_text)

    # Grid table combining Logo & Info
    grid = Table.grid(expand=True)
    grid.add_column(ratio=2, no_wrap=True, overflow="ellipsis")
    grid.add_column(ratio=1, no_wrap=True, overflow="ellipsis")
    grid.add_row(logo_text, info_table)

    return Panel(
        grid,
        box=box.ROUNDED,
        border_style="#ff8700",
        subtitle="[dim]Type '/help' for commands | '/exit' to quit[/dim]",
        subtitle_align="right"
    )

def build_status_bar(state: CLIState) -> RenderableType:
    """
    Renders the bottom status / prompt guidelines bar.
    """
    bar_text = Text(no_wrap=True, overflow="ellipsis")
    bar_text.append("⚡ ", style="bold #ffcc00")
    bar_text.append(f"Model: {state.active_model}", style="bold #ff9500")
    bar_text.append(" | ", style="dim white")
    bar_text.append(f"Status: {state.status}", style="bold white")
    bar_text.append(" | ", style="dim white")
    bar_text.append("Press Esc to cancel active task", style="dim grey50")

    return Panel(
        bar_text,
        box=box.HORIZONTALS,
        border_style="grey35"
    )

def build_layout(state: CLIState, chat_renderable: RenderableType = None) -> Layout:
    """
    Constructs a Rich Layout with three distinct viewport regions:
      - Header Panel (Fixed Top, ratio=10 or size fixed)
      - Chat Viewport (Flex Center)
      - Status / Input Bar (Fixed Bottom)
    """
    layout = Layout()
    layout.split(
        Layout(name="header", size=9),
        Layout(name="body", ratio=1),
        Layout(name="footer", size=3),
    )

    layout["header"].update(build_header(state))
    if chat_renderable is not None:
        layout["body"].update(chat_renderable)
    else:
        layout["body"].update(Panel("Ultron CLI Ready.", box=box.SIMPLE))
    layout["footer"].update(build_status_bar(state))

    return layout
