"""Command-line interface for the hello package."""

import typer

from hello.commands import hi_cmd

app = typer.Typer(
    help="Hello CLI tool.",
    no_args_is_help=True,
    context_settings={"help_option_names": ["--help", "-h"]},
)

app.add_typer(hi_cmd.app, name="hi", help="Say hi.")

if __name__ == "__main__":
    app()
