"""Core implementation of the semantic chatbot."""

import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

if TYPE_CHECKING:
    import numpy.typing as npt
    from scipy.sparse import csr_matrix, spmatrix


class SemanticChatBot:
    """A chatbot that finds answers using semantic similarity with TF-IDF vectorization."""

    def __init__(
        self,
        database_path: str,
        similarity_threshold: float = 0.3,
        cache_file: str = "vector_cache.pkl",
    ) -> None:
        """Initialize the chatbot with the database and vectorizer.

        Args:
            database_path: Path to the Excel database file
            similarity_threshold: Minimum similarity score to consider a match (0-1)
            cache_file: File to cache the vectorized questions

        """
        self.database_path = database_path
        self.similarity_threshold = similarity_threshold
        self.cache_file = cache_file
        self.vectorizer = TfidfVectorizer()

        # These will be set in _load_or_generate_vectors
        self.question_vectors: spmatrix

        # Load database and prepare vectors
        self.df = pd.read_excel(database_path)

        # Handle both uppercase and lowercase column names
        question_col = self._get_column_case_insensitive("вопрос")
        answer_col = self._get_column_case_insensitive("ответ")

        self.questions: list[str] = self.df[question_col].tolist()
        self.answers: list[str] = self.df[answer_col].tolist()

        # Load or generate vectors
        self._load_or_generate_vectors()

    def _get_column_case_insensitive(self, column_name: str) -> str:
        """Get the actual column name regardless of case.

        Args:
            column_name: The column name to find (case-insensitive)

        Returns:
            The actual column name in the dataframe

        Raises:
            KeyError: If no matching column is found

        """
        for col in self.df.columns:
            if col.lower() == column_name.lower():
                return col
        msg = f"Column '{column_name}' not found (case-insensitive)"
        raise KeyError(msg)

    def _load_or_generate_vectors(self) -> None:
        """Load cached vectors and vectorizer if available and valid, otherwise generate new ones."""
        # Check if cached vectors exist and are newer than database
        cache_path = Path(self.cache_file)
        db_path = Path(self.database_path)

        if cache_path.exists() and cache_path.stat().st_mtime > db_path.stat().st_mtime:
            try:
                with cache_path.open("rb") as f:
                    # Note: Only load pickles from trusted sources
                    cache_data: dict[str, Any] = pickle.load(f)

                # Verify cached questions match current questions
                if cache_data["questions"] == self.questions:
                    self.question_vectors = cast("spmatrix", cache_data["vectors"])
                    self.vectorizer = cast("TfidfVectorizer", cache_data["vectorizer"])
                    return
            except Exception:
                # If cache is invalid or any error occurs, regenerate vectors
                pass

        # Generate new vectors
        self.question_vectors = cast("spmatrix", self.vectorizer.fit_transform(self.questions))

        # Cache the vectors and vectorizer
        cache_data: dict[str, Any] = {
            "questions": self.questions,
            "vectors": self.question_vectors,
            "vectorizer": self.vectorizer,
        }
        with Path(self.cache_file).open("wb") as f:
            pickle.dump(cache_data, f)

    def find_answer(self, query: str) -> dict[str, Any]:
        """Find the most similar question and its answer.

        Args:
            query: The user question to find a match for

        Returns:
            Dictionary with the answer, similarity score, and matched question

        """
        # Vectorize the query
        query_vector = cast("csr_matrix", self.vectorizer.transform([query]))

        # Calculate similarity with all questions
        similarities = cast("npt.NDArray[np.float64]", cosine_similarity(query_vector, self.question_vectors).flatten())

        # Find the most similar question
        max_sim_idx = np.argmax(similarities)
        max_similarity = float(similarities[max_sim_idx])

        # Prepare the response
        return {
            "answer": self.answers[max_sim_idx]
            if max_similarity >= self.similarity_threshold
            else "Sorry, I couldn't find a relevant answer.",
            "similarity": max_similarity,
            "matched_question": self.questions[max_sim_idx],
            "above_threshold": max_similarity >= self.similarity_threshold,
        }
