"""Common CLI utilities and argument types."""

from typing import Annotated

import typer

Name = Annotated[str, typer.Argument(help="Name to greet")]
