import asyncio

import typer

from ultron import __version__
from ultron.core.config import settings
from ultron.core.intelligence.prompt_assembly import build_response_guidance
from ultron.core.logging import get_logger
from ultron.ui.theme import (
    ACCENT,
    BLUE,
    FAINT,
    GREEN,
    MUTED,
    RED,
    TEXT,
    UI,
    YELLOW,
    console,
)

# Base system prompt: identity + the shared response-style guidance.
_BASE_SYSTEM_PROMPT = "You are Ultron, a helpful local AI assistant.\n\n" + build_response_guidance().rstrip()

logger = get_logger("ultron.cli")

app = typer.Typer(name="ultron", help="Ultron AI Assistant CLI")

# Valid security modes for the /security slash command.
SECURITY_MODES = ("permissive", "interactive", "strict")

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

@app.command()
def run():
    """
    Start the Ultron AI assistant.
    """
    logger.info(f"Starting [bold]Ultron AI Assistant[/bold] v{__version__}...")
    
    # Generate configuration summary excluding secrets and print-filtered parameters
    exclude_keys = {"log_level", "data_dir", "memory_backend"}
    config_details = [
        f"[{BLUE}]{key}[/{BLUE}]={value}"
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

def _short_cwd() -> str:
    """Return CWD with home folder replaced by ~ for compact display."""
    from pathlib import Path
    cwd = Path.cwd()
    try:
        return f"~/{cwd.relative_to(Path.home())}" if cwd.is_relative_to(Path.home()) else str(cwd)
    except (AttributeError, ValueError):
        try:
            return f"~/{cwd.relative_to(Path.home())}"
        except ValueError:
            return str(cwd)


def print_banner(model_name: str) -> None:
    """
    Draws the Molten Core ASCII logo banner and metadata dynamically
    evaluating the given model_name.
    """
    from ultron.ui.theme import UI
    UI.render_banner(model=model_name, cwd_short=_short_cwd())


def print_help_table(console):
    """
    Prints a formatted table of available slash commands.
    """
    from ultron.ui.theme import UI
    UI.render_help_table()


def render_security_status(console, mode: str) -> None:
    """
    Prints the active security mode and how each risk tier maps to a decision
    under that mode, plus the guardrail summary.
    """
    from ultron.security import Decision, RiskTier, SecurityBoundary

    policy = SecurityBoundary(mode=mode)

    console.print(f"[bold {ACCENT}]🔒 Security mode:[/bold {ACCENT}] {mode}")
    for tier in RiskTier:
        decision = policy.decide(tier)
        note = {
            Decision.ALLOW: "runs automatically",
            Decision.CONFIRM: "requires your approval",
            Decision.DENY: "hard-blocked",
        }[decision]
        if tier == RiskTier.MEDIUM:
            note += " (reserved — no action maps to it yet)"
        console.print(
            f"  [{MUTED}]{tier.value:<9}[/{MUTED}] "
            f"[bold {TEXT}]{decision.value:<9}[/bold {TEXT}] {note}"
        )

    console.print(
        f"[{MUTED}]Guardrails always hard-block leaked credentials, unsafe URLs, and "
        f"path escapes. "
        f"Switch anytime with /security <{'|'.join(SECURITY_MODES)}>.[/{MUTED}]"
    )


async def handle_slash_command(
    cmd: str,
    console,
    history: list,
    agent=None,
    session=None,
    state=None,
    reflow=None,
) -> tuple[bool, bool]:
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

        from ultron.core.config import settings, update_env_file

        models: list[str] = []
        if agent and hasattr(agent, "engine") and hasattr(agent.engine, "list_models"):
            models = await agent.engine.list_models()

        if not models:
            console.print(f"[bold {RED}]No local models found or failed to fetch model list from Ollama.[/bold {RED}]\n")
            return True, False

        old_model = state.active_model if state else (getattr(session, "active_model", settings.model) if session else settings.model)

        selected_model = await questionary.select(
            "Select model:",
            choices=models,
            default=old_model if old_model in models else models[0],
        ).ask_async()

        if selected_model:
            import os
            if state:
                state.active_model = selected_model
            if session:
                session.active_model = selected_model
            settings.model = selected_model
            os.environ["ULTRON_MODEL"] = selected_model
            if agent and hasattr(agent, "engine"):
                if hasattr(agent.engine, "set_model"):
                    agent.engine.set_model(selected_model)
                else:
                    agent.engine.default_model = selected_model
                    agent.engine.model = selected_model
            update_env_file("ULTRON_MODEL", selected_model)
            # Refresh the banner in place: the resize reflow replays it from a
            # rebuild block that reads the current model, so re-running that
            # block shows the new model without wiping the conversation.
            if reflow is not None:
                reflow.rebuild()

            console.print(
                f"[{GREEN}]✓[/{GREEN}] Switched model  "
                f"[{MUTED}]{old_model}[/{MUTED}] → [bold {ACCENT}]{selected_model}[/bold {ACCENT}]"
            )

        return True, False

    if clean_cmd == "/agent" or clean_cmd.startswith("/agent "):
        import questionary

        from ultron.core.agents import SUPPORTED_AGENTS, get_agent

        # /agent <type> switches directly; a bare /agent shows a picker
        # (pre-selecting the current agent when known).
        requested = clean_cmd[len("/agent"):].strip()
        if requested:
            selected = requested
        else:
            current_type = getattr(session, "active_agent_type", None)
            selected = await questionary.select(
                "Select agent type:",
                choices=list(SUPPORTED_AGENTS),
                default=current_type if current_type in SUPPORTED_AGENTS else None,
            ).ask_async()

        if not selected:
            return True, False  # user cancelled the picker

        if selected not in SUPPORTED_AGENTS:
            console.print(
                f"[bold {RED}]Unknown agent type:[/bold {RED}] [{ACCENT}]{selected}[/{ACCENT}]. "
                f"Available: {', '.join(SUPPORTED_AGENTS)}.\n"
            )
            return True, False

        try:
            # Reuse the live engine so the current model (and any other live
            # engine state) carries over exactly; the factory still chooses the
            # class from the single SUPPORTED_AGENTS contract.
            new_agent = get_agent(selected, engine=getattr(agent, "engine", None))
        except ValueError as exc:
            console.print(f"[bold {RED}]✗ Failed to switch agent:[/bold {RED}] {exc}\n")
            return True, False

        old_type = getattr(session, "active_agent_type", type(agent).__name__ if agent else "none")

        # Queue the new agent for the chat loop, which swaps it in on the next
        # iteration (same hand-off pattern as /reload). History is preserved.
        if session is not None:
            session.next_agent = new_agent
            session.active_agent_type = selected
            console.print(
                f"[{GREEN}]✓[/{GREEN}] Switched agent  "
                f"[{MUTED}]{old_type}[/{MUTED}] → [bold {ACCENT}]{selected}[/bold {ACCENT}]"
            )
        else:
            console.print(
                f"[bold {YELLOW}]⚠ Switched agent, but no active session to apply it to.[/bold {YELLOW}]\n"
            )
        return True, False

    if clean_cmd == "/security" or clean_cmd.startswith("/security "):
        import os

        import questionary

        from ultron.core.agents.security import get_security
        from ultron.core.config import settings, update_env_file

        current = getattr(get_security(), "mode", None) or settings.security_mode

        # Show the current status up front so the user sees what they have
        # before deciding whether to change it.
        render_security_status(console, current)

        requested = clean_cmd[len("/security"):].strip()
        if requested:
            selected = requested
            if selected not in SECURITY_MODES:
                console.print(
                    f"[bold {RED}]Unknown security mode:[/bold {RED}] [{ACCENT}]{selected}[/{ACCENT}]. "
                    "Available: "
                    f"{', '.join(SECURITY_MODES)}.\n"
                )
                return True, False
        else:
            selected = await questionary.select(
                "Security mode:",
                choices=list(SECURITY_MODES),
                default=current if current in SECURITY_MODES else None,
            ).ask_async()
            if not selected:
                return True, False  # cancelled — current mode stays

        if selected != current:
            # Apply the switch live: the shared boundary the agents use, the
            # settings object, the process env, and the .env file on disk.
            get_security().mode = selected
            settings.security_mode = selected
            os.environ["ULTRON_SECURITY_MODE"] = selected
            update_env_file("ULTRON_SECURITY_MODE", selected)

            console.print(
                f"[{GREEN}]✓[/{GREEN}] Switched security mode  "
                f"[{MUTED}]{current}[/{MUTED}] → [bold {ACCENT}]{selected}[/bold {ACCENT}]"
            )
            console.print(f"[{MUTED}]persisted to ULTRON_SECURITY_MODE in .env[/{MUTED}]")
            render_security_status(console, selected)

        return True, False

    if clean_cmd == "/memory" or clean_cmd.startswith("/memory "):
        from rich.table import Table

        from ultron.core.tools.memory import graph

        sub = clean_cmd[len("/memory"):].strip()

        # /memory clear — drop every graph edge (corrects bad memories).
        if sub == "clear":
            console.print(graph.clear_all_triples())
            console.print()
            return True, False

        # /memory remove <subject> <predicate> <object> — drop one edge.
        if sub.startswith("remove "):
            parts = sub[len("remove "):].split()
            if len(parts) >= 3:
                subject = parts[0]
                object = parts[-1]
                predicate = " ".join(parts[1:-1])
                console.print(graph.remove_triple(subject, predicate, object))
            else:
                console.print(
                    f"[bold {YELLOW}]Usage: /memory remove <subject> <predicate> <object>[/bold {YELLOW}]\n"
                    f"[{FAINT}]e.g. /memory remove Paris is the capital of France[/{FAINT}]\n"
                )
            console.print()
            return True, False

        stats = graph.get_graph_stats()
        triples = graph.get_all_triples()

        console.print(
            f"[bold {ACCENT}]🧠 Memory graph[/bold {ACCENT}] — "
            f"[{TEXT}]{stats['entities']}[/] entities · "
            f"[{TEXT}]{stats['triples']}[/] triples · "
            f"[{MUTED}]{stats['facts']}[/] flat facts"
        )

        table = Table(box=None, show_header=False, expand=True, padding=(0, 2))
        table.add_column(style=f"bold {ACCENT}")
        table.add_column(style=TEXT)
        table.add_row("Entities (nodes)", str(stats["entities"]))
        table.add_row("Triples (edges)", str(stats["triples"]))
        table.add_row("Flat facts", str(stats["facts"]))
        console.print(table)
        if triples:
            console.print(f"[{MUTED}]Stored knowledge:[/{MUTED}]")
            for edge in triples:
                console.print(f"  [{MUTED}]-[/{MUTED}] {edge}")
        else:
            console.print(
                f"[{MUTED}]No triples stored yet — try 'remember that Paris is the capital of France'.[/{MUTED}]"
            )
        console.print()
        return True, False

    if clean_cmd == "/clear":
        from ultron.core.types import ChatMessage, Role
        history.clear()
        if reflow is not None:
            # The resize reflow replays the recorded transcript; /clear must
            # drop it too, or a resize would resurrect the wiped conversation.
            reflow.reset()
        history.append(ChatMessage(role=Role.SYSTEM, content=_BASE_SYSTEM_PROMPT))
        console.print(f"[{GREEN}]✓[/{GREEN}] Conversation history cleared.")
        return True, False

    if clean_cmd == "/reload":
        # Plain-English note: This performs a "soft reload" by re-importing core modules
        # and creating a fresh agent instance from scratch. In Python, importlib.reload()
        # does not update existing object instances in-place; recreating the agent ensures
        # predictable behavior without lingering old code references.
        import importlib

        from ultron.core.types import ChatMessage, Role

        try:
            # Reload core modules in dependency order
            import ultron.core.agents
            import ultron.core.agents.simple
            import ultron.core.tools.builtin.command_runner
            import ultron.core.tools.builtin.file_reader
            import ultron.core.tools.builtin.file_writer
            import ultron.core.tools.builtin.http_client
            import ultron.core.tools.builtin.web_search
            import ultron.core.tools.memory.sqlite
            import ultron.core.tools.registry

            importlib.reload(ultron.core.tools.builtin.file_reader)
            importlib.reload(ultron.core.tools.builtin.file_writer)
            importlib.reload(ultron.core.tools.builtin.command_runner)
            importlib.reload(ultron.core.tools.builtin.http_client)
            importlib.reload(ultron.core.tools.builtin.web_search)
            importlib.reload(ultron.core.tools.memory.sqlite)
            importlib.reload(ultron.core.tools.registry)
            importlib.reload(ultron.core.agents.simple)
            agents_mod = importlib.reload(ultron.core.agents)

            # Re-create fresh agent instance from reloaded agents module
            fresh_agent = agents_mod.get_agent("simple")

            # Update caller's agent reference if applicable
            if agent is not None:
                # If caller passed a mutable dictionary or object wrapper, or if agent is used as global
                pass  # Caller will receive fresh_agent via return or outer scope reference

            # Reset history back to initial SYSTEM message
            history.clear()
            history.append(ChatMessage(role=Role.SYSTEM, content=_BASE_SYSTEM_PROMPT))

            console.print(
                f"[bold {GREEN}]✓ Reloaded — fresh agent and tools loaded. Conversation history reset.[/bold {GREEN}]\n"
                f"[{FAINT}]Note: reload is best-effort. If your changes don't appear, exit (/exit) and restart 'ultron chat' for a guaranteed clean reload.[/{FAINT}]\n"
            )
            # Store fresh agent reference on session object or return indication if needed
            if session is not None:
                session.reloaded_agent = fresh_agent

        except Exception as exc:  # noqa: BLE001 — reloads arbitrary user modules
            console.print(f"[bold {RED}]✗ Failed to reload modules:[/bold {RED}] {exc}\n[{FAINT}]Continuing with existing working agent session.[/{FAINT}]\n")

        return True, False

    console.print(f"[bold {YELLOW}]Unknown command:[/bold {YELLOW}] [{ACCENT}]{cmd}[/{ACCENT}]. Type [bold {ACCENT}]/help[/bold {ACCENT}] to see available commands.\n")
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
            return f"All {passed_count} tests passed! ✅\n\n[{FAINT}]# Note: Run again with --raw to see full output[/{FAINT}]"

        summary_lines = [f"Test Results: {passed_count} passed, {failed_count} failed\n\nFailed tests:"]
        for test in failed_tests:
            # Clean up test path/name
            clean_test = test.strip().split()[-1] if test.strip() else test.strip()
            summary_lines.append(f"  - {clean_test}")

        summary_lines.append(f"\n[{FAINT}]# Note: Run again with --raw to see full output[/{FAINT}]")
        return "\n".join(summary_lines)
    except (IndexError, TypeError, ValueError):
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
    except (IndexError, TypeError, ValueError):
        return raw_output

async def async_chat(agent_type: str = "simple"):
    """
    Asynchronous runner for the interactive chat session.

    The prompt is a live, terminal-width-aware ChatSession: a persistent
    bottom toolbar (model / agent / status / security mode) re-flows
    automatically when the window is resized, and all output rendering is
    width-adaptive.
    """
    from prompt_toolkit import PromptSession

    from ultron.core.agents import get_agent
    from ultron.core.state import CLIState
    from ultron.ui.session import ChatSession

    state = CLIState(
        active_model=settings.model,
        current_dir=_short_cwd(),
        version=f"v{__version__}",
        status="Ready",
    )

    def _security_mode() -> str:
        """
        Live security mode for the toolbar; falls back to settings.

        This runs on every prompt render (keystroke / resize), so a failure
        here must never take the whole chat UI down with it.
        """
        try:
            from ultron.core.agents.security import get_security
            return getattr(get_security(), "mode", None) or settings.security_mode
        except (ImportError, AttributeError, OSError, ValueError):
            return settings.security_mode

    session = PromptSession()
    session.active_model = state.active_model
    session.active_agent_type = agent_type

    chat_ui = ChatSession(
        state=state,
        session=session,
        agent_tag=lambda: getattr(session, "active_agent_type", agent_type),
        security_mode=_security_mode,
    )

    try:
        agent = get_agent(agent_type)
    except (ValueError, OSError) as e:
        logger.error(f"Failed to initialize agent: {e}")
        console.print(f"[bold {RED}]Initialization Error[/bold {RED}]: {e}")
        return

    logger.debug("Initializing chat session with SimpleAgent...")

    from ultron.core.types import ChatMessage, Role, truncate_history

    history: list[ChatMessage] = [
        ChatMessage(role=Role.SYSTEM, content=_BASE_SYSTEM_PROMPT)
    ]

    # Record every renderable printed during the session and, on terminal
    # resize while the chat prompt is live, re-render the whole conversation
    # at the window's new width — so every response box, tool panel and table
    # follows the resize, not just the startup banner.
    from ultron.ui.responsive import ResizeReflow

    reflow = ResizeReflow(console, app=session.app)
    reflow.add(lambda: print_banner(state.active_model))
    reflow.start()

    while True:
        try:
            state.status = "Ready"

            user_input = await chat_ui.prompt_async()
            trimmed_input = user_input.strip()

            if not trimmed_input:
                continue

            if trimmed_input.lower() in ("/exit", "/quit", "exit", "quit"):
                reflow.stop()
                UI.render_status("Goodbye.", status="info")
                break

            # Echo the user's message into the transcript (Claude Code style), so
            # the conversation reads as a natural exchange. Recorded by the resize
            # reflow like every other printed line.
            UI.render_user_prompt(trimmed_input)

            if trimmed_input.startswith("/"):
                handled, should_exit = await handle_slash_command(
                    trimmed_input,
                    console,
                    history,
                    agent=agent,
                    session=session,
                    state=state,
                    reflow=reflow,
                )
                if hasattr(session, "reloaded_agent") and session.reloaded_agent is not None:
                    agent = session.reloaded_agent
                    session.reloaded_agent = None
                # /agent queues a fresh instance here; swap it in for the next turn.
                if getattr(session, "next_agent", None) is not None:
                    agent = session.next_agent
                    session.next_agent = None
                if should_exit:
                    reflow.stop()
                    UI.render_status("Goodbye.", status="info")
                    break
                if handled:
                    continue

            truncated_history = truncate_history(history, max_messages=10)

            import threading

            def _listen_for_esc(cancel_evt: threading.Event):
                import select
                import sys
                import termios
                import tty
                if not sys.stdin.isatty():
                    return
                try:
                    fd = sys.stdin.fileno()
                    old_settings = termios.tcgetattr(fd)
                except (OSError, ValueError):
                    return
                try:
                    tty.setcbreak(fd)
                    while not cancel_evt.is_set():
                        r, _, _ = select.select([fd], [], [], 0.05)
                        if r:
                            ch = sys.stdin.read(1)
                            if ch == "\x1b":
                                r_next, _, _ = select.select([fd], [], [], 0.05)
                                if not r_next:
                                    cancel_evt.set()
                                    break
                                else:
                                    sys.stdin.read(10)
                except (OSError, ValueError) as exc:
                    logger.debug("ESC listener failed: %s", exc)
                finally:
                    try:
                        termios.tcflush(fd, termios.TCIFLUSH)
                    except (OSError, ValueError) as exc:
                        logger.debug("termios tcflush failed: %s", exc)
                    try:
                        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                    except (OSError, ValueError) as exc:
                        logger.debug("termios tcsetattr failed: %s", exc)

            cancel_event = threading.Event()
            esc_thread = threading.Thread(target=_listen_for_esc, args=(cancel_event,), daemon=True)
            esc_thread.start()

            state.status = "Thinking..."
            agent_task = asyncio.create_task(agent.run(trimmed_input, truncated_history))

            with console.status(f"[{FAINT}]Thinking... (Press Esc to cancel)[/{FAINT}]"):
                while not agent_task.done():
                    if cancel_event.is_set():
                        agent_task.cancel()
                        break
                    await asyncio.sleep(0.05)

            cancel_event.set()
            esc_thread.join(timeout=0.2)

            if agent_task.cancelled():
                state.status = "Ready"
                UI.render_status("Task execution cancelled by Esc key.", status="warning")
                continue

            try:
                response_msg = await agent_task
            except asyncio.CancelledError:
                state.status = "Ready"
                UI.render_status("Task execution cancelled by Esc key.", status="warning")
                continue

            if response_msg.pending_action:
                import questionary

                from ultron.core.tools.registry import get_tool

                action = response_msg.pending_action
                state.status = f"Executing Tool: {action.action_type}"

                if action.action_type == "run_command":
                    UI.render_action_card(
                        title="Confirmation Required",
                        action="Execute terminal command",
                        target=action.target,
                    )
                elif action.action_type == "read_file":
                    UI.render_action_card(
                        title="Confirmation Required",
                        action="Read file",
                        target=action.target,
                    )
                elif action.action_type == "write_file":
                    preview = (action.content or "")[:120]
                    if len(action.content or "") > 120:
                        preview += "…"
                    UI.render_action_card(
                        title="Confirmation Required",
                        action="Create new file",
                        target=action.target,
                        preview=preview,
                    )
                elif action.action_type == "overwrite_file":
                    UI.render_action_card(
                        title="Confirmation Required",
                        action="Overwrite existing file",
                        target=action.target,
                    )
                elif action.action_type == "web_search":
                    UI.render_action_card(
                        title="Confirmation Required",
                        action="Search the web",
                        target=action.target,
                    )
                elif action.action_type == "fetch_page":
                    UI.render_action_card(
                        title="Confirmation Required",
                        action="Fetch web page",
                        target=action.target,
                    )
                elif action.action_type == "db_query":
                    # Note: db_query represents a higher-risk action type because non-SELECT queries
                    # modify or delete database state, so explicit interactive confirmation is required.
                    UI.render_action_card(
                        title="Database Warning: Confirmation Required",
                        action="Execute database query",
                        target=action.target,
                    )
                elif action.action_type == "execute_plan":
                    # Proactive plan approval: the full step + permission
                    # preview was shown by the agent; here the user approves
                    # the whole chain in one card instead of being prompted
                    # per step mid-execution.
                    import json as _json

                    steps = _json.loads(action.target or "[]")
                    UI.render_action_card(
                        title="Plan Approval Required",
                        action=f"Execute {len(steps)}-step plan",
                        # The full plan is shown in the preview; dumping the raw
                        # JSON into the chip line would wreck the layout.
                        target="",
                        preview=(action.content or "")[:500],
                    )
                else:
                    UI.render_action_card(
                        title="Confirmation Required",
                        action=action.action_type,
                        target=action.target or "",
                    )

                choice = await questionary.select(
                    "Do you want to allow this action?",
                    choices=["Yes, allow", "No, don't allow"],
                ).ask_async()

                result: str = "Error: Unknown action type."

                if choice == "Yes, allow":
                    if action.action_type == "run_command":
                        if action.target.startswith("http_request:"):
                            parts = action.target.split(":", 3)
                            method = parts[1]
                            url = parts[2]
                            body = parts[3] if len(parts) > 3 else None
                            http_tool = get_tool("make_http_request")
                            result = http_tool(method, url, body) if http_tool else "Error: Tool 'make_http_request' not found."
                        else:
                            run_cmd_func = get_tool("run_command")
                            result = run_cmd_func(action.target) if run_cmd_func else "Error: Tool 'run_command' not found."

                            if action.target.startswith("pytest"):
                                result = summarize_pytest_output(result)
                            elif action.target.startswith("ruff check"):
                                result = summarize_lint_output(result)

                    elif action.action_type == "read_file":
                        read_func = get_tool("read_file")
                        raw = read_func(action.target) if read_func else "Error: Tool 'read_file' not found."
                        if str(raw).startswith("Error"):
                            result = f"Sorry, I could not find or read the file '{action.target}'."
                        else:
                            result = f"Here are the contents of '{action.target}':\n\n{raw}"

                    elif action.action_type == "write_file":
                        write_func = get_tool("write_file")
                        result = write_func(action.target, action.content or "", overwrite=False) if write_func else "Error: Tool 'write_file' not found."

                    elif action.action_type == "overwrite_file":
                        write_func = get_tool("write_file")
                        result = write_func(action.target, action.content or "", overwrite=True) if write_func else "Error: Tool 'write_file' not found."

                    elif action.action_type == "web_search":
                        search_func = get_tool("search_web")
                        result = search_func(action.target) if search_func else "Error: Tool 'search_web' not found."

                    elif action.action_type == "fetch_page":
                        fetch_func = get_tool("fetch_page_text")
                        result = fetch_func(action.target) if fetch_func else "Error: Tool 'fetch_page_text' not found."

                    elif action.action_type == "run_parallel":
                        commands = [c for c in (action.target or "").splitlines() if c.strip()]
                        parallel_func = get_tool("run_parallel")
                        result = parallel_func(commands) if parallel_func else "Error: Tool 'run_parallel' not found."

                    elif action.action_type == "db_query":
                        query_func = get_tool("run_query")
                        result = query_func(action.target) if query_func else "Error: Tool 'run_query' not found."

                    elif action.action_type == "execute_plan":
                        import json as _json

                        from ultron.core.agents.simple import execute_plan

                        steps = _json.loads(action.target or "[]")
                        results = await execute_plan(steps)
                        result = "\n".join(results)

                    else:
                        result = f"Error: Unrecognised action type '{action.action_type}'."

                else:
                    result = "Action cancelled by user."

                response_msg = ChatMessage(role=Role.ASSISTANT, content=result)

            state.status = "Ready"

            import re
            tool_match = re.match(
                r"^Executed tool '(?:\[[^']*\])?([^'\]]+)(?:\[/[^']*\])?':\n\n(.*)$",
                response_msg.content,
                re.DOTALL,
            )
            if tool_match:
                # Every print is recorded by the resize reflow, so tool
                # executions and plain responses reflow automatically.
                UI.render_tool_execution(tool_match.group(1), tool_match.group(2))
            else:
                UI.render_response(response_msg.content)

            history.append(ChatMessage(role=Role.USER, content=trimmed_input))
            history.append(ChatMessage(role=Role.ASSISTANT, content=response_msg.content))

        except KeyboardInterrupt:
            reflow.stop()
            UI.render_status("Chat interrupted. Goodbye.", status="warning")
            break
        except Exception as e:  # noqa: BLE001 — CLI boundary: never crash the chat
            logger.error(f"Error during chat execution: {e}")
            UI.render_error(str(e), title="Runtime Error")

@app.command()
def chat(
    agent: str = typer.Option(
        "simple",
        "--agent",
        "-a",
        help="Agent type to use (simple or react).",
    ),
):
    """
    Start an interactive text chat session with Ultron.
    """
    asyncio.run(async_chat(agent_type=agent))

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
