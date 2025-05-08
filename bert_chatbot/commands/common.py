"""Common CLI utilities and argument types."""

from typing import Annotated

import typer

DatabasePath = Annotated[
    str,
    typer.Argument(
        help="Path to the Excel database file with questions and answers",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
]

SimilarityThreshold = Annotated[
    float,
    typer.Option(
        "--threshold",
        "-t",
        help="Minimum similarity score to consider a match (0-1)",
        min=0.0,
        max=1.0,
        show_default=True,
    ),
]

CacheFile = Annotated[
    str,
    typer.Option(
        "--cache-file",
        "-c",
        help="File to cache the vectorized questions",
        show_default=True,
    ),
] 