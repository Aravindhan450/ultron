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

async def async_chat():
    """
    Asynchronous runner for the interactive chat session.
    """
    from ultron.core.agents import get_agent
    from rich.prompt import Prompt
    from rich.console import Console

    console = Console()
    
    try:
        agent = get_agent("simple")
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}")
        console.print(f"[bold red]Initialization Error[/bold red]: {e}")
        return

    logger.info("Initializing chat session with SimpleAgent...")
    console.print("\n[bold green]Ultron Chat Ready. Type 'exit' or 'quit' to end the chat.[/bold green]\n")
    
    history = []
    
    while True:
        try:
            user_input = Prompt.ask("[bold blue]You[/bold blue]")
            if user_input.strip().lower() in ("exit", "quit"):
                console.print("[bold yellow]Goodbye.[/bold yellow]")
                break
                
            if not user_input.strip():
                continue
                
            with console.status("[dim]Thinking...[/dim]"):
                response = await agent.run(user_input, history)
                
            console.print(f"[bold green]Ultron[/bold green]: {response}\n")
            
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": response})
            
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

if __name__ == "__main__":
    app()
