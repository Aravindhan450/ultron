import asyncio
import typer
from ultron import __version__
from ultron.core.config import settings
from ultron.core.logging import get_logger

logger = get_logger("ultron.cli")

app = typer.Typer(name="ultron", help="Ultron AI Assistant CLI")

def version_callback(value: bool):
    if value:
        typer.echo(f"ultron version: {__version__}")
        raise typer.Exit()

@app.callback()
def main(
    version: bool = typer.Option(
        None,
        "--version",
        "-v",
        help="Show the version and exit.",
        callback=version_callback,
        is_eager=True,
    ),
):
    """
    Ultron AI Assistant - Local-first JARVIS-style system.
    """
    pass

@app.command()
def run():
    """
    Start the Ultron AI assistant.
    """
    logger.info(f"Starting [bold]Ultron AI Assistant[/bold] v{__version__}...")
    
    # Generate configuration summary excluding secrets and print-filtered parameters
    exclude_keys = {"log_level", "data_dir", "memory_backend"}
    config_details = [
        f"[blue]{key}[/blue]={value}"
        for key, value in settings.model_dump(mode="json").items()
        if key not in exclude_keys and not any(secret_word in key.lower() for secret_word in ["key", "token", "secret", "password"])
    ]
    logger.info(f"Active configuration: {', '.join(config_details)}")
    logger.info("Ultron is running. Press Ctrl+C to stop.")
    
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Ultron is stopping. Goodbye.")

def print_startup_banner(console):
    """
    Prints a polished CLI startup banner including an ASCII art logo,
    active AI model configuration, current working directory, and a divider line.
    """
    from pathlib import Path
    from rich.rule import Rule
    from rich.text import Text
    from ultron.core.config import settings

    # Shorten CWD to use ~ if under home folder
    cwd = Path.cwd()
    try:
        short_cwd = f"~/{cwd.relative_to(Path.home())}" if cwd.is_relative_to(Path.home()) else str(cwd)
    except AttributeError:
        # Fallback for Python < 3.9 if needed
        try:
            short_cwd = f"~/{cwd.relative_to(Path.home())}"
        except ValueError:
            short_cwd = str(cwd)

    # Gradient-styled ASCII art banner for ULTRON (red to orange to purple gradient)
    logo_lines = [
        "█    █  █    ████████ ██████  ██████  █    █",
        "█    █  █       ██    █    █  █    █  ██   █",
        "█    █  █       ██    ██████  █    █  █ █  █",
        "█    █  █       ██    █  █    █    █  █  █ █",
        " ████   ██████  ██    █   █   ██████  █   ██",
    ]

    colors = ["#ff3333", "#ff6600", "#ff9900", "#cc33ff", "#9933ff"]
    
    console.print()
    for line, color in zip(logo_lines, colors):
        console.print(Text(line, style=f"bold {color}"))
    console.print()

    # Active info block
    console.print(f" [bold magenta]Model:[/bold magenta] [cyan]{settings.model}[/cyan]")
    console.print(f" [bold magenta]Directory:[/bold magenta] [dim]{short_cwd}[/dim]")
    console.print(Rule(style="dim cyan"))
    console.print("[dim]Type 'exit' or 'quit' to end the chat.[/dim]\n")

def print_help_table(console):
    """
    Prints a formatted table of available slash commands.
    """
    from rich.table import Table

    table = Table(title="Available Commands", show_header=True, header_style="bold magenta")
    table.add_column("Command", style="cyan", width=12)
    table.add_column("Description", style="white")

    table.add_row("/help", "Show this help table of available commands.")
    table.add_row("/clear", "Reset conversation history to start fresh.")
    table.add_row("/exit, /quit", "Exit the chat session.")

    console.print(table)
    console.print()

def handle_slash_command(cmd: str, console, history: list) -> tuple[bool, bool]:
    """
    Handles slash commands entered by the user.

    Returns:
        tuple[bool, bool]: (handled, should_exit)
        - handled: True if input was a slash command (prevents sending to AI).
        - should_exit: True if the slash command requested exiting the chat loop.
    """
    clean_cmd = cmd.strip().lower()
    
    if clean_cmd in ("/exit", "/quit"):
        return True, True

    if clean_cmd == "/help":
        print_help_table(console)
        return True, False

    if clean_cmd == "/clear":
        from ultron.core.types import ChatMessage, Role
        history.clear()
        history.append(ChatMessage(role=Role.SYSTEM, content="You are Ultron, a helpful local AI assistant."))
        console.print("[bold green]Conversation history cleared.[/bold green]\n")
        return True, False

    console.print(f"[bold yellow]Unknown command:[/bold yellow] [cyan]{cmd}[/cyan]. Type [bold cyan]/help[/bold cyan] to see available commands.\n")
    return True, False

async def async_chat():
    """
    Asynchronous runner for the interactive chat session.
    """
    from ultron.core.agents import get_agent
    from rich.prompt import Prompt
    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    print_startup_banner(console)
    
    try:
        agent = get_agent("simple")
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}")
        console.print(f"[bold red]Initialization Error[/bold red]: {e}")
        return

    logger.debug("Initializing chat session with SimpleAgent...")
    
    from ultron.core.types import ChatMessage, Role, truncate_history

    # Initialize history with a single system prompt message
    history: list[ChatMessage] = [
        ChatMessage(role=Role.SYSTEM, content="You are Ultron, a helpful local AI assistant.")
    ]
    
    while True:
        try:
            user_input = Prompt.ask("[bold blue]You[/bold blue]")
            trimmed_input = user_input.strip()

            if not trimmed_input:
                continue

            # Plain exit/quit fallback or slash command handling
            if trimmed_input.lower() in ("exit", "quit"):
                console.print("[bold yellow]Goodbye.[/bold yellow]")
                break

            if trimmed_input.startswith("/"):
                handled, should_exit = handle_slash_command(trimmed_input, console, history)
                if should_exit:
                    console.print("[bold yellow]Goodbye.[/bold yellow]")
                    break
                if handled:
                    continue
                
            # Truncate history before passing to agent
            truncated_history = truncate_history(history, max_messages=10)
            
            with console.status("[dim]Thinking...[/dim]"):
                response_msg = await agent.run(user_input, truncated_history)
                
            # If the response requests interactive user confirmation via pending_action
            if response_msg.pending_action:
                import questionary
                from ultron.core.tools.registry import get_tool

                action = response_msg.pending_action
                if action.action_type == "run_command":
                    console.print(Panel(
                        f"Action: Execute terminal command\n[bold yellow]{action.target}[/bold yellow]",
                        title="Confirmation Required",
                        border_style="yellow",
                        padding=(0, 1)
                    ))
                elif action.action_type == "overwrite_file":
                    console.print(Panel(
                        f"Action: Overwrite existing file\nTarget: [bold yellow]{action.target}[/bold yellow]",
                        title="Confirmation Required",
                        border_style="yellow",
                        padding=(0, 1)
                    ))

                choice = await questionary.select(
                    "Do you want to allow this action?",
                    choices=["Yes, allow", "No, don't allow"]
                ).ask_async()

                if choice == "Yes, allow":
                    if action.action_type == "run_command":
                        run_cmd_func = get_tool("run_command")
                        result = run_cmd_func(action.target) if run_cmd_func else "Error: Tool 'run_command' not found."
                    elif action.action_type == "overwrite_file":
                        write_func = get_tool("write_file")
                        result = write_func(action.target, action.content or "", overwrite=True) if write_func else "Error: Tool 'write_file' not found."
                    
                    response_msg = ChatMessage(role=Role.ASSISTANT, content=result)
                else:
                    response_msg = ChatMessage(role=Role.ASSISTANT, content="Action cancelled by user.")

            # Render Ultron AI response panel
            console.print(Panel(response_msg.content, title="Ultron", border_style="bold #ff6600", padding=(0, 1)))
            console.print()
            
            # Append both the user message and assistant response to full history
            history.append(ChatMessage(role=Role.USER, content=user_input))
            history.append(response_msg)
            
        except KeyboardInterrupt:
            console.print("\n[bold yellow]Chat interrupted. Goodbye.[/bold yellow]")
            break
        except Exception as e:
            logger.error(f"Error during chat execution: {e}")
            console.print(f"[bold red]Error[/bold red]: {e}\n")

@app.command()
def chat():
    """
    Start an interactive text chat session with Ultron.
    """
    asyncio.run(async_chat())

@app.command()
def logs(
    lines: int = typer.Option(50, "--lines", "-n", help="Number of last log lines to show."),
    follow: bool = typer.Option(False, "--follow", "-f", help="Follow log output in real-time."),
):
    """
    View or stream the Ultron log file.
    """
    import time
    from ultron.core.logging import LOG_FILE

    if not LOG_FILE.exists():
        typer.echo(f"No log file found at {LOG_FILE}.")
        raise typer.Exit()

    if follow:
        typer.echo(f"Following logs from {LOG_FILE} (Ctrl+C to exit)...")
        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                # Go to the end of the file
                f.seek(0, 2)
                while True:
                    line = f.readline()
                    if not line:
                        time.sleep(0.1)
                        continue
                    typer.echo(line, nl=False)
        except KeyboardInterrupt:
            raise typer.Exit()
    else:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            all_lines = f.readlines()
            for line in all_lines[-lines:]:
                typer.echo(line, nl=False)

if __name__ == "__main__":
    app()
