"""Stub for sentence_transformers library."""

from collections.abc import Sequence
from typing import Any

import numpy as np


class SentenceTransformer:
    """Stub implementation of SentenceTransformer for testing."""

    def __init__(self, model_name: str) -> None:
        """Initialize a stub model.

        Args:
            model_name: Name of the model
        """
        self.model_name = model_name

    def encode(
        self,
        sentences: str | list[str] | Sequence[str],
        convert_to_numpy: bool = True,
        *args: Any,
        **kwargs: Any
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        """Return random embeddings of shape (n_sentences, 384).

        Args:
            sentences: Input texts to encode
            convert_to_numpy: Whether to convert the output to numpy arrays
            args: Additional positional arguments
            kwargs: Additional keyword arguments

        Returns:
            Array with embeddings
        """
        # Convert to list if it's a single string
        if isinstance(sentences, str):
            sentences = [sentences]
        
        n_sentences = len(sentences)
        # Return random embeddings with shape (n_sentences, 384)
        # Using numpy's newer random generator API
        rng = np.random.default_rng()
        return rng.standard_normal((n_sentences, 384)) 