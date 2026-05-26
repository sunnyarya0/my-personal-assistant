"""
main.py
=======
Personal AI Agent Assistant — CLI Entrypoint

Usage:
    python main.py --help
    python main.py chat "Summarise the files in my Desktop"
    python main.py ping
    python main.py config show
    python main.py serve
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

# Make sure the project root is on the Python path when running as a script
sys.path.insert(0, str(Path(__file__).parent))

from core.config import settings
from core.llm_client import get_llm_client, ping_llm
from core.registry import registry
from core.agent import Agent

# ---------------------------------------------------------------------------
# App & console setup
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="assistant",
    help="🤖 Personal AI Agent Assistant — your autonomous productivity partner.",
    add_completion=False,
    rich_markup_mode="rich",
)

console = Console()

BANNER = """
[bold cyan]╔══════════════════════════════════════════╗[/bold cyan]
[bold cyan]║   🤖  Personal AI Agent Assistant        ║[/bold cyan]
[bold cyan]║   Autonomous. Extensible. Yours.         ║[/bold cyan]
[bold cyan]╚══════════════════════════════════════════╝[/bold cyan]
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_banner() -> None:
    console.print(BANNER)


def _abort(message: str) -> None:
    console.print(f"[bold red]Error:[/bold red] {message}")
    raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# CLI Commands
# ---------------------------------------------------------------------------

@app.command()
def ping() -> None:
    """
    🔌 Check if the agent can connect to the configured LLM provider.
    """
    _print_banner()
    try:
        settings.validate_api_key()
    except ValueError as exc:
        _abort(str(exc))

    console.print(f"Pinging [cyan]{settings.llm_provider}[/cyan] (model: {settings.llm_model})...")

    if ping_llm():
        console.print("[bold green]Success![/bold green] Connection established.")
    else:
        _abort("Ping failed. Check your API key and network connection.")


@app.command()
def chat(
    message: str = typer.Argument(..., help="The message or goal to send to the assistant."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show reasoning steps live."),
    max_iterations: int = typer.Option(None, "--max-iter", "-n", help="Override max iterations."),
) -> None:
    """
    💬 Send a goal to the agent and let it reason and act autonomously.

    Example:
        python main.py chat "What files are in my workspace?"
        python main.py chat --verbose "Search the web for Python tips"
    """
    _print_banner()

    # Validate config
    try:
        settings.validate_api_key()
    except ValueError as exc:
        _abort(str(exc))

    # Auto-discover skills
    discovered = registry.discover_skills()
    if discovered and verbose:
        console.print(f"[dim]Loaded {discovered} skill file(s)[/dim]")

    n_tools = len(registry)
    console.print(
        f"[dim]Provider: {settings.llm_provider} | Model: {settings.llm_model} "
        f"| Tools: {n_tools}[/dim]\n"
    )

    # Run agent
    agent = Agent(
        verbose=verbose,
        max_iterations=max_iterations or settings.max_iterations,
    )

    with console.status("[cyan]Agent is working...[/cyan]", spinner="dots") as status:
        if verbose:
            status.stop()
        result = agent.run(message)

    # Display result (verbose already printed the answer inline)
    if not verbose:
        console.print(Panel(
            result.final_answer,
            title=f"[bold green]{settings.agent_name}[/bold green]",
            subtitle=f"[dim]{result.iterations_used} step(s) | {result.total_elapsed_s:.1f}s[/dim]",
            border_style="green",
            padding=(1, 2),
        ))

    if not result.success:
        raise typer.Exit(code=1)





# ---------------------------------------------------------------------------
# Memory Subcommands
# ---------------------------------------------------------------------------

app_memory = typer.Typer(help="🧠 Manage the agent's long-term semantic memory.")
app.add_typer(app_memory, name="memory")


@app_memory.command("list")
def memory_list(limit: int = typer.Option(50, help="Max entries to list")) -> None:
    """List recent memories stored by the agent."""
    from core.memory import get_memory_store
    mem = get_memory_store()
    entries = mem.list_all(limit=limit)

    if not entries:
        console.print("[yellow]Memory is empty.[/yellow]")
        return

    console.print(f"[bold cyan]Long-Term Memory ({mem.count()} total)[/bold cyan]\n")
    for e in entries:
        tags = f" [{', '.join(e.tags)}]" if e.tags else ""
        date = e.timestamp[:10]
        console.print(f"[dim]{e.id[:8]} | {date}{tags}[/dim]")
        console.print(f"{e.content}\n")


@app_memory.command("search")
def memory_search(
    query: str = typer.Argument(..., help="What to search for"),
    top_k: int = typer.Option(5, help="Number of results to return"),
) -> None:
    """Search the memory for relevant information."""
    from core.memory import get_memory_store
    mem = get_memory_store()
    results = mem.search(query, top_k=top_k)

    if not results:
        console.print("[yellow]No relevant memories found.[/yellow]")
        return

    console.print(f"[bold cyan]Search Results for:[/bold cyan] {query}\n")
    for r in results:
        tags = f" [{', '.join(r.entry.tags)}]" if r.entry.tags else ""
        console.print(f"[dim]Score: {r.score:.2f} | {r.entry.timestamp[:10]}{tags}[/dim]")
        console.print(f"{r.content}\n")


@app_memory.command("clear")
def memory_clear(
    force: bool = typer.Option(False, "--force", "-f", help="Bypass confirmation prompt"),
) -> None:
    """Wipe all memories (cannot be undone)."""
    if not force:
        confirm = typer.confirm("Are you sure you want to delete ALL memories? This cannot be undone.")
        if not confirm:
            console.print("[yellow]Aborted.[/yellow]")
            return

    from core.memory import get_memory_store
    mem = get_memory_store()
    count = mem.clear(confirm=True)
    console.print(f"[bold green]Cleared {count} memories.[/bold green]")


# ---------------------------------------------------------------------------
# Skills subcommand group
# ---------------------------------------------------------------------------

app_skills = typer.Typer(help="🔌 Inspect and manage registered agent skills.")
app.add_typer(app_skills, name="skills")


@app_skills.command("list")
def skills_list() -> None:
    """List all auto-discovered agent skills and tools."""
    # Discover skills
    registry.discover_skills()

    tools = registry.get_all_tools()
    if not tools:
        console.print("[yellow]No skills registered.[/yellow]")
        return

    console.print(f"[bold cyan]Registered Agent Skills ({len(tools)} total)[/bold cyan]\n")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Skill Name", style="cyan", no_wrap=True)
    table.add_column("Safety Mode", style="green", no_wrap=True)
    table.add_column("Description", style="white")

    for t in tools:
        type_str = "[red]⚠️ destructive[/red]" if t.destructive else "[green]✅ safe[/green]"
        table.add_row(t.name, type_str, t.description)

    console.print(table)
    console.print()


# ---------------------------------------------------------------------------
# Config subcommand group
# ---------------------------------------------------------------------------

config_app = typer.Typer(help="⚙️  View and manage configuration.")
app.add_typer(config_app, name="config")


@config_app.command("show")
def config_show() -> None:
    """
    Show the current active configuration (secrets are masked).

    Example:
        python main.py config show
    """
    _print_banner()

    table = Table(title="Active Configuration", show_header=True, header_style="bold magenta")
    table.add_column("Setting", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")

    def mask(value: str) -> str:
        if len(value) > 6:
            return value[:4] + "•" * (len(value) - 6) + value[-2:]
        return "•" * len(value) if value else "[dim](not set)[/dim]"

    table.add_row("Provider", settings.llm_provider)
    table.add_row("Model", settings.llm_model)
    table.add_row("API Key", mask(settings.api_key))
    table.add_row("Agent Name", settings.agent_name)
    table.add_row("Safe Mode", "✅ enabled" if settings.safe_mode else "⚠️  disabled")
    table.add_row("Max Iterations", str(settings.max_iterations))
    table.add_row("Workspace Dir", settings.workspace_dir)
    table.add_row("Log Dir", settings.log_dir)
    table.add_row("UI Port", str(settings.ui_port))

    console.print(table)


# ---------------------------------------------------------------------------
# Serve command
# ---------------------------------------------------------------------------

@app.command()
def serve(
    port: Optional[int] = typer.Option(None, "--port", "-p", help="Port to run the web UI on."),
) -> None:
    """
    🌐 Start the web UI dashboard and open your browser automatically.

    Example:
        python main.py serve --port 8080
    """
    import uvicorn
    import webbrowser

    _print_banner()
    _port = port or settings.ui_port
    
    url = f"http://127.0.0.1:{_port}"
    console.print(f"Starting server on [bold]{url}[/bold]...")

    if settings.auto_open_browser:
        console.print(f"Opening browser at [bold]{url}[/bold]...")
        webbrowser.open(url)

    try:
        uvicorn.run("UI.server:app", host="127.0.0.1", port=_port, reload=False, log_level="info")
    except Exception as exc:
        _abort(f"Could not start server: {exc}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
