"""Test configuration and fixtures."""

import os
import tempfile
from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from bert_chatbot.core.chatbot import SemanticChatBot


@pytest.fixture
def sample_data() -> dict[str, list[str]]:
    """Create sample data for testing."""
    return {
        "вопрос": ["How are you?", "What is your name?", "What is the weather like today?"],
        "ответ": ["I am fine, thank you!", "My name is BERT ChatBot.", "I cannot check the weather."],
    }


@pytest.fixture
def sample_database_path(sample_data: dict[str, list[str]]) -> Generator[str]:
    """Create a temporary sample database for testing."""
    # Create a temporary Excel file with sample data
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as temp_file:
        temp_path = temp_file.name

    # Write to Excel file
    pd.DataFrame(sample_data).to_excel(temp_path, index=False)

    try:
        yield temp_path
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.unlink(temp_path)


@pytest.fixture
def mock_vectorizer() -> MagicMock:
    """Create a mock TF-IDF vectorizer for testing."""
    mock = MagicMock(spec=TfidfVectorizer)
    # Configure the mock to return simple vectors
    mock.fit_transform.return_value = "mock_vectors"
    mock.transform.return_value = "mock_query_vector"
    return mock


@pytest.fixture
def mock_cosine_similarity() -> MagicMock:
    """Create a mock for cosine_similarity function."""
    mock = MagicMock()
    # Return similarity scores that will result in the first question being the best match
    mock.return_value = [[0.95, 0.5, 0.3]]
    return mock


@pytest.fixture
def sample_chatbot(
    sample_database_path: str, mock_vectorizer: MagicMock, mock_cosine_similarity: MagicMock
) -> SemanticChatBot:
    """Create a sample chatbot instance for testing with mocked components."""
    with patch("bert_chatbot.core.chatbot.TfidfVectorizer", return_value=mock_vectorizer):
        with patch("bert_chatbot.core.chatbot.cosine_similarity", mock_cosine_similarity):
            with patch("bert_chatbot.core.chatbot.pickle.dump"):  # Mock pickle to avoid writing to disk
                return SemanticChatBot(
                    database_path=sample_database_path,
                    similarity_threshold=0.3,
                    cache_file=":memory:",  # In-memory cache for tests
                )
