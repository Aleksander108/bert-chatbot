"""Common CLI utilities and argument types."""

import os
from typing import Annotated, Optional

import typer

def get_database_from_env() -> Optional[str]:
    """Get database path from environment variable if it exists."""
    return os.environ.get("BERT_CHATBOT_DATABASE")

DatabasePath = Annotated[
    Optional[str],
    typer.Argument(
        help="Path to the Excel database file with questions and answers. Can also be set via BERT_CHATBOT_DATABASE environment variable.",
        exists=False,  # We'll check existence manually
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
