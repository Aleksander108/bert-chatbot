"""Tests for the SemanticChatBot class."""

import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from bert_chatbot.core.chatbot import SemanticChatBot
from tests.helpers import DEFAULT_QUESTION, get_expected_response


def test_chatbot_initialization(sample_database_path: str) -> None:
    """Test that the chatbot initializes correctly."""
    with patch("bert_chatbot.core.chatbot.TfidfVectorizer"), patch("bert_chatbot.core.chatbot.pickle.dump"):
        chatbot = SemanticChatBot(sample_database_path)

    # Check that the chatbot loads the database
    assert len(chatbot.questions) == 3
    assert len(chatbot.answers) == 3


@patch("bert_chatbot.core.chatbot.cosine_similarity")
def test_find_answer_exact_match(mock_cosine_similarity: MagicMock, sample_chatbot: SemanticChatBot) -> None:
    """Test finding an answer for an exact match."""
    # Configure the mock to return high similarity for the first question
    similarities = np.array([[0.95, 0.5, 0.3]])
    mock_cosine_similarity.return_value = similarities

    response = sample_chatbot.find_answer(DEFAULT_QUESTION)

    assert response["answer"] == get_expected_response(DEFAULT_QUESTION)
    assert response["above_threshold"]
    assert response["matched_question"] == "How are you?"  # First question in the sample data
    assert response["similarity"] == 0.95


@patch("bert_chatbot.core.chatbot.cosine_similarity")
def test_find_answer_similar_match(mock_cosine_similarity: MagicMock, sample_chatbot: SemanticChatBot) -> None:
    """Test finding an answer for a similar but not exact query."""
    # Configure the mock to return medium similarity for the first question
    similarities = np.array([[0.6, 0.3, 0.2]])
    mock_cosine_similarity.return_value = similarities

    query = "How are you doing today?"
    response = sample_chatbot.find_answer(query)

    # Should match with the first question in our mock data
    assert response["matched_question"] == "How are you?"
    assert response["above_threshold"]
    assert 0.3 <= response["similarity"] <= 1.0


@patch("bert_chatbot.core.chatbot.cosine_similarity")
def test_find_answer_no_match(mock_cosine_similarity: MagicMock, sample_chatbot: SemanticChatBot) -> None:
    """Test behavior when no good match is found."""
    # Set a high threshold to force no matches
    sample_chatbot.similarity_threshold = 0.95

    # Configure the mock to return low similarity for all questions
    similarities = np.array([[0.2, 0.1, 0.05]])
    mock_cosine_similarity.return_value = similarities

    query = "Something completely unrelated to the database"
    response = sample_chatbot.find_answer(query)

    assert not response["above_threshold"]
    assert "Sorry, I couldn't find a relevant answer." in response["answer"]
    assert response["similarity"] == 0.2  # The highest similarity from our mock


def test_vector_caching(sample_database_path: str) -> None:
    """Test that vectors are cached correctly."""
    with patch("pickle.dump") as mock_dump:
        with patch("bert_chatbot.core.chatbot.TfidfVectorizer"):
            SemanticChatBot(sample_database_path, cache_file="test_cache.pkl")
        # Should call pickle.dump to cache vectors
        mock_dump.assert_called_once()


def test_loading_cached_vectors(sample_database_path: str) -> None:
    """Test loading vectors from cache if cache exists and is valid."""
    # Create a cache file manually
    cache_file = "test_cache_load.pkl"
    mock_questions = ["How are you?", "What is your name?", "What is the weather like today?"]
    mock_vectors = "mock_vectors"
    mock_vectorizer = "mock_vectorizer_instance"  # Use string instead of MagicMock
    cache_data = {
        "questions": mock_questions,
        "vectors": mock_vectors,
        "vectorizer": mock_vectorizer,  # Add vectorizer to cache data
    }

    cache_path = Path(cache_file)
    with cache_path.open("wb") as f:
        pickle.dump(cache_data, f)

    # Ensure the cache file is newer than the database
    db_path = Path(sample_database_path)
    # Use os.utime with the modified times
    new_mtime = db_path.stat().st_mtime + 10
    cache_path.touch(exist_ok=True)
    # Note: We still need to use os.utime here as Path doesn't have a direct equivalent
    import os

    os.utime(cache_file, (new_mtime, new_mtime))

    # Create a mock vectorizer
    mock_vectorizer = MagicMock()

    # Create a chatbot with patched components
    with (
        patch("bert_chatbot.core.chatbot.TfidfVectorizer", return_value=mock_vectorizer),
        patch("bert_chatbot.core.chatbot.pickle.dump"),
    ):  # Patch pickle.dump to prevent pickling MagicMock
        chatbot = SemanticChatBot(sample_database_path, cache_file=cache_file)

        # Check that vectors were loaded from cache
        assert chatbot.question_vectors == mock_vectors

        # Check that fit_transform was not called
        mock_vectorizer.fit_transform.assert_not_called()

    # Clean up
    if cache_path.exists():
        cache_path.unlink()
