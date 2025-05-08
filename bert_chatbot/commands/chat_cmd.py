"""Implementation of the chat command."""

import typer
from rich.console import Console
from rich.panel import Panel

from bert_chatbot.commands.common import CacheFile, DatabasePath, SimilarityThreshold
from bert_chatbot.core.chatbot import SemanticChatBot

app = typer.Typer(
    help="Interactive chat with the semantic chatbot.",
    context_settings={"help_option_names": ["--help", "-h"]},
)
console = Console()


@app.command()
def interactive(
    database: DatabasePath,
    similarity_threshold: SimilarityThreshold = 0.3,
    cache_file: CacheFile = "vector_cache.pkl",
) -> None:
    """Start an interactive chat session with the semantic chatbot.

    You can ask questions and get answers based on semantic similarity.
    Type 'exit', 'quit', or 'q' to end the chat.
    Type 'debug' to toggle debug mode which shows similarity scores.
    """
    console.print(Panel.fit("BERT Semantic ChatBot", title="Welcome"))
    console.print("[bold]Loading database and preparing vectors...[/bold]")

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

        # Style the answer based on whether it's above threshold
        style = "green" if response["above_threshold"] else "red"

        # Print the answer
        console.print(f"\n[bold {style}]Bot:[/bold {style}] {response['answer']}")
