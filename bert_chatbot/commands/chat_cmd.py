"""Implementation of the chat command."""

import os
from typing import Annotated, Any

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
    similarity_threshold: SimilarityThreshold = 0.5,
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

    # Удаляем инициализацию chatbot здесь, так как функция ask будет создавать свой экземпляр.
    debug_mode = False

    console.print("[green]Ready! Type your questions or 'exit' to quit.[/green]")

    # Main interaction loop
    while True:
        try:
            question = console.input("[bold magenta]You:[/bold magenta] ")
            if question.lower() == "exit":
                console.print("[bold yellow]Exiting chatbot. Goodbye![/bold yellow]")
                break
            if question.lower() == "debug_on":
                debug_mode = True
                console.print("[cyan]Debug mode ON.[/cyan]")
                continue
            if question.lower() == "debug_off":
                debug_mode = False
                console.print("[cyan]Debug mode OFF.[/cyan]")
                continue

            # Call the ask function to handle the response logic and printing
            # Pass the current debug_mode state to the ask function
            ask(
                question=question,
                database=database, # type: ignore
                similarity_threshold=similarity_threshold,
                cache_file=cache_file,
                debug=debug_mode, # Pass the dynamic debug_mode state
                no_cache=no_cache,
                show_top=0 # Not showing top_matches by default in interactive
            )

        except KeyboardInterrupt:
            console.print("\n[bold yellow]Exiting chatbot. Goodbye![/bold yellow]")
            break
        except Exception as e:
            console.print(f"[bold red]An error occurred:[/bold red] {e}")
            if debug_mode:
                # traceback импортируется и используется только здесь
                import traceback 
                console.print_exception(show_locals=True)


DebugOption = Annotated[bool, typer.Option("--debug", help="Show debug information")]


@app.command()
def ask(
    database: DatabasePath = None,  # None will use env var via get_database_from_env
    question: str = typer.Argument(..., help="Question to ask the chatbot"),
    similarity_threshold: SimilarityThreshold = 0.5,
    cache_file: CacheFile = "vector_cache.pkl",
    debug: DebugOption = False,
    no_cache: NoCacheOption = False,
    show_top: int = 0, # Default to 0, meaning don't show by default
) -> dict[str, Any]:
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

    # Initialize the chatbot
    bot_instance = SemanticChatBot(
        database_path=str(database), # type: ignore
        similarity_threshold=similarity_threshold,
        cache_file=cache_file,
    )
    response = bot_instance.find_answer(question)

    if no_cache and os.path.exists(cache_file): # Moved cache removal after bot init for clarity, though functionally similar
        try:
            os.remove(cache_file)
        except OSError:
            pass

    # Show debug information if enabled
    if debug:
        matched = f"[bold]Matched question:[/bold] {response['matched_question']}"
        similarity = f"[bold]Similarity score:[/bold] {response['similarity']:.4f}"
        threshold = f"[bold]Threshold:[/bold] {similarity_threshold}"
        unmatched_terms = response.get("unmatched_query_terms")
        clarification = response.get("clarification_prefix", "")

        debug_output = f"{matched}\n{similarity}\n{threshold}"
        if unmatched_terms:
            debug_output += f"\n[bold]Unmatched query terms:[/bold] {unmatched_terms}"
        if clarification:
            # Already printed for interactive, but good for single 'ask' if we decide to show it there too
            # For now, interactive prints it before main answer.
            pass # console.print(f"[italic yellow]Bot clarification: {clarification}[/italic yellow]")

        if response.get("source"):
            debug_output += f"\n[bold]Источник:[/bold] {response['source']}"
        debug_panel = Panel(
            debug_output,
            title="[bold magenta]Debug Info[/bold magenta]",
            expand=False,
            border_style="magenta"
        )
        console.print(debug_panel)

    # Print the answer
    # Let's ensure clarification is printed if not in debug and present.
    if not debug:
        clarification = response.get("clarification_prefix", "")
        if clarification:
            console.print(f"\n[italic yellow]Bot clarification: {clarification}[/italic yellow]")

    # Display logic for 'ask' command
    if response.get("match_type") == "oov_with_related_questions" and response.get("relevant_questions_list"):
        console.print(f"\n[bold green]Bot:[/bold green] {response.get('answer', '')}") # Restored "Bot:"
        questions_to_show = response.get("relevant_questions_list", [])
        if questions_to_show:
            console.print("[bold cyan]Найденные вопросы по теме:[/bold cyan]")
            for idx, q_text in enumerate(questions_to_show):
                console.print(f"  {idx + 1}. {q_text}")
        console.print("[italic]Вы можете уточнить ваш запрос, выбрав один из этих вопросов.[/italic]")
    elif response["above_threshold"]:
        console.print(f"\n[bold green]Bot:[/bold green] {response['answer']}") # Restored "Bot:"
        if response.get("source"):
            console.print(f"[dim]Источник:[/dim] [italic blue]{response['source']}[/italic blue]")
        if response.get("score") is not None:
            console.print(f"[dim]Соответствие:[/dim] [bold cyan]{response['score']:.2f}[/bold cyan]")
    elif not response.get("relevant_questions_list"):
        console.print(f"\n[bold red]Bot:[/bold red] {response['answer']}") # Restored "Bot:"

    console.print("\n") # Add a newline for better separation

    return response # Return the full response for potential further use if needed
