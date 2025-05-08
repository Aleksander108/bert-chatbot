"""Implementation for saying hi."""

import typer
from rich.console import Console

from hello.commands.common import Name

app = typer.Typer(
    help="Say hi to someone.",
    context_settings={"help_option_names": ["--help", "-h"]},
)
console = Console()


@app.command()
def person(name: Name) -> None:
    """Say hi to a person."""
    console.print(f"Hi, {name}!")


@app.command()
def world() -> None:
    """Say hi to the world."""
    console.print("Hello, World!")
