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

    # Block-letter ULTRON wordmark — spacing preserved exactly as specified.
    logo_lines = [
        "     ██╗   ██╗██╗  ████████╗██████╗  ██████╗ ███╗   ██╗",
        "     ██║   ██║██║  ╚══██╔══╝██╔══██╗██╔═══██╗████╗  ██║",
        "     ██║   ██║██║     ██║   ██████╔╝██║   ██║██╔██╗ ██║",
        "     ██║   ██║██║     ██║   ██╔══██╗██║   ██║██║╚██╗██║",
        "     ╚██████╔╝███████╗██║   ██║  ██║╚██████╔╝██║ ╚████║",
        "      ╚═════╝ ╚══════╝╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝",
    ]

    colors = ["#ffcc00", "#ff9500", "#ff5500", "#ff5500", "#ff2200", "#cc0000"]

    console.print()
    for line, color in zip(logo_lines, colors):
        console.print(Text(line, style=f"bold {color}"))
    console.print()

    # Active info block
    console.print(f" [bold #ff9500]Model:[/bold #ff9500] [bold #ffcc00]{settings.model}[/bold #ffcc00]")
    console.print(f" [bold #ff9500]Directory:[/bold #ff9500] [dim]{short_cwd}[/dim]")
    console.print(Rule(style="#ff5500"))
    console.print("[dim]Type '/exit' or '/quit' to end the chat.[/dim]\n")


def print_help_table(console):
    """
    Prints a formatted table of available slash commands.
    """
    from rich.table import Table

    table = Table(title="Available Commands", show_header=True, header_style="bold #ff9500")
    table.add_column("Command", style="bold #ffcc00", width=12)
    table.add_column("Description", style="white")

    table.add_row("/help", "Show this help table of available commands.")
    table.add_row("/model", "Select an available LLM model interactively.")
    table.add_row("/clear", "Reset conversation history to start fresh.")
    table.add_row("/exit, /quit", "Exit the chat session.")

    console.print(table)
    console.print()

async def handle_slash_command(cmd: str, console, history: list, agent=None) -> tuple[bool, bool]:
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

    if clean_cmd == "/model":
        import questionary
        from ultron.core.config import settings

        models: list[str] = []
        if agent and hasattr(agent, "engine") and hasattr(agent.engine, "list_models"):
            models = await agent.engine.list_models()

        if not models:
            console.print("[bold red]No local models found or failed to fetch model list from Ollama.[/bold red]\n")
            return True, False

        selected_model = await questionary.select(
            "Select model:",
            choices=models,
            default=settings.model if settings.model in models else models[0],
        ).ask_async()

        if selected_model:
            settings.model = selected_model
            if agent and hasattr(agent, "engine"):
                agent.engine.default_model = selected_model
            console.print(f"[bold green]Selected model:[/bold green] [cyan]{selected_model}[/cyan]\n")

        return True, False

    if clean_cmd == "/clear":
        from ultron.core.types import ChatMessage, Role
        history.clear()
        history.append(ChatMessage(role=Role.SYSTEM, content="You are Ultron, a helpful local AI assistant."))
        console.print("[bold green]Conversation history cleared.[/bold green]\n")
        return True, False

    console.print(f"[bold yellow]Unknown command:[/bold yellow] [cyan]{cmd}[/cyan]. Type [bold cyan]/help[/bold cyan] to see available commands.\n")
    return True, False

def summarize_pytest_output(raw_output: str) -> str:
    """
    Parses raw pytest -v output and summarizes test counts and failed test names.
    If parsing fails, returns raw_output as fallback.
    """
    import re
    try:
        passed_count = len(re.findall(r'\bPASSED\b', raw_output))
        failed_tests = re.findall(r'^(.*?)\s+FAILED\b', raw_output, flags=re.MULTILINE)
        failed_count = len(failed_tests)

        if passed_count == 0 and failed_count == 0:
            return raw_output

        if failed_count == 0:
            return f"All {passed_count} tests passed! ✅\n\n[dim]# Note: Run again with --raw to see full output[/dim]"

        summary_lines = [f"Test Results: {passed_count} passed, {failed_count} failed\n\nFailed tests:"]
        for test in failed_tests:
            # Clean up test path/name
            clean_test = test.strip().split()[-1] if test.strip() else test.strip()
            summary_lines.append(f"  - {clean_test}")

        summary_lines.append("\n[dim]# Note: Run again with --raw to see full output[/dim]")
        return "\n".join(summary_lines)
    except Exception:
        return raw_output

def summarize_lint_output(raw_output: str) -> str:
    """
    Parses ruff's --output-format=concise output and groups issues by filename.

    Concise format is one line per issue:
      src/foo.py:12:4: F401 'os' imported but unused

    We parse each line with a regex, bucket issues by file, and return a
    tidy grouped summary.  This avoids dumping a wall of text at the user
    for large codebases while still preserving every issue's line number
    and error code.

    Falls back to raw_output if anything unexpected happens during parsing.
    """
    import re
    try:
        # ruff exits 0 with no output or an "All checks passed" line on success
        if not raw_output or not raw_output.strip():
            return "No issues found! ✅"
        if "All checks passed" in raw_output:
            return "No issues found! ✅"

        # Match ruff concise format: file:line:col: CODE message
        issue_pattern = re.compile(
            r'^(?P<file>[^:]+):(?P<line>\d+):\d+:\s+(?P<code>\S+)\s+(?P<msg>.+)$',
            re.MULTILINE
        )
        matches = list(issue_pattern.finditer(raw_output))

        if not matches:
            # Output exists but doesn't match the expected format — show raw
            return raw_output

        # Group issues by filename, preserving order of first appearance
        files: dict[str, list[str]] = {}
        for m in matches:
            fname = m.group("file")
            entry = f"  Line {m.group('line')}: {m.group('code')} {m.group('msg')}"
            files.setdefault(fname, []).append(entry)

        total_issues = len(matches)
        total_files = len(files)
        header = f"Lint Results: {total_issues} issue{'s' if total_issues != 1 else ''} found across {total_files} file{'s' if total_files != 1 else ''}"

        summary_lines = [header]
        for fname, issues in files.items():
            summary_lines.append(f"\n{fname}:")
            summary_lines.extend(issues)

        return "\n".join(summary_lines)
    except Exception:
        return raw_output

async def async_chat():
    """
    Asynchronous runner for the interactive chat session.
    """
    from ultron.core.agents import get_agent
    from rich.console import Console
    from rich.panel import Panel
    from prompt_toolkit import PromptSession
    from prompt_toolkit.formatted_text import HTML

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

    session = PromptSession()

    while True:
        try:
            # Use prompt_toolkit session for full readline/arrow key support
            user_input = await session.prompt_async(HTML("<b><blue>You</blue></b>: "))
            trimmed_input = user_input.strip()

            if not trimmed_input:
                continue

            # Plain exit/quit fallback or slash command handling
            if trimmed_input.lower() in ("/exit", "/quit", "exit", "quit"):
                console.print("[bold yellow]Goodbye.[/bold yellow]")
                break

            if trimmed_input.startswith("/"):
                handled, should_exit = await handle_slash_command(trimmed_input, console, history, agent=agent)
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

                # Show a confirmation panel describing the action BEFORE asking.
                # Each action type gets its own panel text so the user knows
                # exactly what will happen if they choose "Yes, allow".
                if action.action_type == "run_command":
                    console.print(Panel(
                        f"Action: Execute terminal command\n[bold #ffcc00]{action.target}[/bold #ffcc00]",
                        title="Confirmation Required",
                        border_style="#ff9500",
                        padding=(0, 1)
                    ))
                elif action.action_type == "read_file":
                    # Reading exposes file contents — confirm before proceeding
                    console.print(Panel(
                        f"Action: Read file\nTarget: [bold #ffcc00]{action.target}[/bold #ffcc00]",
                        title="Confirmation Required",
                        border_style="#ff9500",
                        padding=(0, 1)
                    ))
                elif action.action_type == "write_file":
                    # Creating a new file — show a content preview so the user
                    # can verify what will be written before committing
                    preview = (action.content or "")[:100]
                    preview_str = preview + ("…" if len(action.content or "") > 100 else "")
                    console.print(Panel(
                        f"Action: Create new file\n"
                        f"Target: [bold #ffcc00]{action.target}[/bold #ffcc00]\n"
                        f"Content preview: [dim]{preview_str}[/dim]",
                        title="Confirmation Required",
                        border_style="#ff9500",
                        padding=(0, 1)
                    ))
                elif action.action_type == "overwrite_file":
                    console.print(Panel(
                        f"Action: Overwrite existing file\nTarget: [bold #ffcc00]{action.target}[/bold #ffcc00]",
                        title="Confirmation Required",
                        border_style="#ff9500",
                        padding=(0, 1)
                    ))

                choice = await questionary.select(
                    "Do you want to allow this action?",
                    choices=["Yes, allow", "No, don't allow"]
                ).ask_async()

                if choice == "Yes, allow":
                    if action.action_type == "run_command":
                        if action.target.startswith("http_request:"):
                            # Parse target format: "http_request:<METHOD>:<URL>[:<BODY>]"
                            parts = action.target.split(":", 3)
                            method = parts[1]
                            url = parts[2]
                            body = parts[3] if len(parts) > 3 else None
                            http_tool = get_tool("make_http_request")
                            result = http_tool(method, url, body) if http_tool else "Error: Tool 'make_http_request' not found."
                        else:
                            run_cmd_func = get_tool("run_command")
                            result = run_cmd_func(action.target) if run_cmd_func else "Error: Tool 'run_command' not found."

                            # Route output through the appropriate summarizer based on the
                            # confirmed command.  Each summarizer converts raw terminal noise
                            # into a clean, human-readable panel — consistent pattern for all.
                            if action.target.startswith("pytest"):
                                result = summarize_pytest_output(result)
                            elif action.target.startswith("ruff check"):
                                result = summarize_lint_output(result)
                            # git log, git commit, custom commands, etc. pass through unchanged

                    elif action.action_type == "read_file":
                        # Execute the actual file read now that the user has confirmed
                        read_func = get_tool("read_file")
                        raw = read_func(action.target) if read_func else "Error: Tool 'read_file' not found."
                        if str(raw).startswith("Error"):
                            result = f"Sorry, I could not find or read the file '{action.target}'."
                        else:
                            result = f"Here are the contents of '{action.target}':\n\n{raw}"

                    elif action.action_type == "write_file":
                        # Execute the new-file write now that the user has confirmed
                        write_func = get_tool("write_file")
                        result = write_func(action.target, action.content or "", overwrite=False) if write_func else "Error: Tool 'write_file' not found."

                    elif action.action_type == "overwrite_file":
                        write_func = get_tool("write_file")
                        result = write_func(action.target, action.content or "", overwrite=True) if write_func else "Error: Tool 'write_file' not found."

                    response_msg = ChatMessage(role=Role.ASSISTANT, content=result)
                else:
                    response_msg = ChatMessage(role=Role.ASSISTANT, content="Action cancelled by user.")

            # Render Ultron AI response panel
            console.print(Panel(response_msg.content, title="Ultron", border_style="bold #ff5500", padding=(0, 1)))
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
