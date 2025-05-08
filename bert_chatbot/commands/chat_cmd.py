"""Implementation of the chat command."""

import os
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel

from bert_chatbot.commands.common import CacheFile, DatabasePath, SimilarityThreshold, get_database_from_env
from bert_chatbot.core.chatbot import SemanticChatBot

app = typer.Typer(
    help="Interactive chat with the semantic chatbot.",
    context_settings={"help_option_names": ["--help", "-h"]},
)
console = Console()

NoCacheOption = Annotated[
    bool, typer.Option("--no-cache", help="Skip loading from cache, generate vectors from scratch")
]


@app.command()
def interactive(
    database: DatabasePath = None,  # None will use env var via get_database_from_env
    similarity_threshold: SimilarityThreshold = 0.3,
    cache_file: CacheFile = "vector_cache.pkl",
    no_cache: NoCacheOption = False,
) -> None:
    """Start an interactive chat session with the semantic chatbot.

    You can ask questions and get answers based on semantic similarity.
    Type 'exit', 'quit', or 'q' to end the chat.
    Type 'debug' to toggle debug mode which shows similarity scores.
    """
    console.print(Panel.fit("BERT Semantic ChatBot", title="Welcome"))
    console.print("[bold]Loading database and preparing vectors...[/bold]")

    # If database is None, try to get it from environment variable
    if database is None:
        database = get_database_from_env()
        if database is None:
            console.print(
                "[bold red]Error: No database path provided and BERT_CHATBOT_DATABASE "
                "environment variable not set.[/bold red]"
            )
            raise typer.Exit(1)

    # Validate that the database file exists
    if not os.path.exists(database):
        console.print(f"[bold red]Error: Database file '{database}' does not exist.[/bold red]")
        raise typer.Exit(1)

    # If no_cache is True, remove the cache file if it exists
    if no_cache and os.path.exists(cache_file):
        try:
            os.remove(cache_file)
            console.print(f"[yellow]Cache file '{cache_file}' removed as requested.[/yellow]")
        except Exception as e:
            console.print(f"[yellow]Warning: Could not remove cache file: {e}[/yellow]")

    # Initialize the chatbot
    chatbot = SemanticChatBot(database, similarity_threshold, cache_file)
    debug_mode = False

    console.print("[green]Ready! Type your questions or 'exit' to quit.[/green]")

    # Main interaction loop
    while True:
        # Get user input
        query = console.input("\n[bold blue]You:[/bold blue] ")

        # Check for exit commands
        if query.lower() in ("exit", "quit", "q"):
            console.print("[bold]Thank you for using BERT Semantic ChatBot![/bold]")
            break

        # Check for debug toggle
        if query.lower() == "debug":
            debug_mode = not debug_mode
            console.print(f"[yellow]Debug mode {'enabled' if debug_mode else 'disabled'}[/yellow]")
            continue

        # Skip empty queries
        if not query.strip():
            continue

        # Get response from chatbot
        response = chatbot.find_answer(query)

        # Show debug information if enabled
        if debug_mode:
            matched = f"[bold]Matched question:[/bold] {response['matched_question']}"
            similarity = f"[bold]Similarity score:[/bold] {response['similarity']:.4f}"
            threshold = f"[bold]Threshold:[/bold] {similarity_threshold}"

            debug_panel = Panel.fit(
                f"{matched}\n{similarity}\n{threshold}",
                title="Debug Info",
                border_style="yellow",
            )
            console.print(debug_panel)

        # Print the answer
        if response["above_threshold"]:
            console.print(f"\n[bold green]Bot:[/bold green] {response['answer']}")
        else:
            console.print(f"\n[bold red]Bot:[/bold red] {response['answer']}")


DebugOption = Annotated[bool, typer.Option("--debug", help="Show debug information")]


@app.command()
def ask(
    database: DatabasePath = None,  # None will use env var via get_database_from_env
    question: str = typer.Argument(..., help="Question to ask the chatbot"),
    similarity_threshold: SimilarityThreshold = 0.3,
    cache_file: CacheFile = "vector_cache.pkl",
    debug: DebugOption = False,
    no_cache: NoCacheOption = False,
) -> None:
    """Ask a single question to the chatbot and get an answer.

    The chatbot will find the most semantically similar question in the database
    and return the corresponding answer.
    """
    console.print("[bold]Loading database and preparing vectors...[/bold]")

    # If database is None, try to get it from environment variable
    if database is None:
        database = get_database_from_env()
        if database is None:
            console.print(
                "[bold red]Error: No database path provided and BERT_CHATBOT_DATABASE "
                "environment variable not set.[/bold red]"
            )
            raise typer.Exit(1)

    # Validate that the database file exists
    if not os.path.exists(database):
        console.print(f"[bold red]Error: Database file '{database}' does not exist.[/bold red]")
        raise typer.Exit(1)

    # If no_cache is True, remove the cache file if it exists
    if no_cache and os.path.exists(cache_file):
        try:
            os.remove(cache_file)
            console.print(f"[yellow]Cache file '{cache_file}' removed as requested.[/yellow]")
        except Exception as e:
            console.print(f"[yellow]Warning: Could not remove cache file: {e}[/yellow]")

    # Initialize the chatbot
    chatbot = SemanticChatBot(database, similarity_threshold, cache_file)

    # Get response from chatbot
    response = chatbot.find_answer(question)

    # Show debug information if enabled
    if debug:
        matched = f"[bold]Matched question:[/bold] {response['matched_question']}"
        similarity = f"[bold]Similarity score:[/bold] {response['similarity']:.4f}"
        threshold = f"[bold]Threshold:[/bold] {similarity_threshold}"
        console.print(f"[yellow]{matched}[/yellow]")
        console.print(f"[yellow]{similarity}[/yellow]")
        console.print(f"[yellow]{threshold}[/yellow]")

    # Print the answer
    if response["above_threshold"]:
        console.print(f"[bold green]Answer:[/bold green] {response['answer']}")
    else:
        console.print(f"[bold red]Answer:[/bold red] {response['answer']}")
