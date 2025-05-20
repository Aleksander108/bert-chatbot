"""Tests for the SemanticChatBot class."""

import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np  # type: ignore[import]
# Removed unused pytest import but left comment for future maintainers
# pytest is needed for fixtures via conftest.py

from bert_chatbot.core.chatbot import SemanticChatBot
from tests.helpers import DEFAULT_QUESTION, get_expected_response


def test_chatbot_initialization(sample_database_path: str) -> None:
    """Test that the chatbot initializes correctly."""
    with patch("bert_chatbot.core.chatbot.SentenceTransformer"), patch("bert_chatbot.core.chatbot.pickle.dump"):
        chatbot = SemanticChatBot(sample_database_path)

    # Check that the chatbot loads the database
    assert len(chatbot.questions) == 3
    assert len(chatbot.answers) == 3


@patch.object(SemanticChatBot, "_check_topic_match", return_value=0.5)
@patch.object(SemanticChatBot, "_calculate_keyword_match", return_value=0.5)
@patch("bert_chatbot.core.chatbot.cosine_similarity")
def test_find_answer_exact_match(
    mock_cosine_similarity: MagicMock, 
    mock_keyword: MagicMock, 
    mock_topic: MagicMock, 
    sample_chatbot: SemanticChatBot
) -> None:
    """Test finding an answer for an exact match."""
    similarities: np.ndarray = np.array([[0.95, 0.5, 0.3]])  # type: ignore[assignment]
    mock_cosine_similarity.return_value = similarities
    
    # Override the combine score calculation
    with patch.object(sample_chatbot, "_calculate_keyword_match", return_value=0.95), \
         patch.object(sample_chatbot, "_check_topic_match", return_value=0.95):
        response = sample_chatbot.find_answer("How are you?")
    
    assert response["above_threshold"] is True
    assert response["similarity"] >= 0.85
    assert response["answer"] == "I am fine, thank you!"
    assert response["matched_question"] == "How are you?"


@patch.object(SemanticChatBot, "_check_topic_match", return_value=0.5)
@patch.object(SemanticChatBot, "_calculate_keyword_match", return_value=0.5)
@patch("bert_chatbot.core.chatbot.cosine_similarity")
def test_find_answer_similar_match(mock_cosine_similarity: MagicMock, mock_keyword: MagicMock, mock_topic: MagicMock, sample_chatbot: SemanticChatBot) -> None:
    """Test finding an answer based on semantic similarity."""
    similarities = np.array([[0.6, 0.8, 0.5]])
    mock_cosine_similarity.return_value = similarities
    
    # This should match the second question with 0.8 similarity
    response = sample_chatbot.find_answer("Tell me your name")
    
    # Check for the pattern without using Cyrillic characters
    assert "информации" in response["answer"]
    assert response["above_threshold"] is True
    assert response["similarity"] > 0.5


@patch.object(SemanticChatBot, "_check_topic_match", return_value=0.2)
@patch.object(SemanticChatBot, "_calculate_keyword_match", return_value=0.2)
@patch("bert_chatbot.core.chatbot.cosine_similarity")
def test_find_answer_no_match(mock_cosine_similarity: MagicMock, mock_keyword: MagicMock, mock_topic: MagicMock, sample_chatbot: SemanticChatBot) -> None:
    """Test finding an answer when no good match exists."""
    original_threshold = sample_chatbot.similarity_threshold
    sample_chatbot.similarity_threshold = 0.95
    
    query = "Something completely unrelated to the database"
    response = sample_chatbot.find_answer(query)
    
    sample_chatbot.similarity_threshold = original_threshold
    
    # Verify the response shows unknown words message
    assert "Извините" in response["answer"]
    assert "информации" in response["answer"]
    assert "Попробуйте переформулировать запрос" in response["answer"]
    assert not response["above_threshold"]


@patch("bert_chatbot.core.chatbot.pickle.load")
@patch("bert_chatbot.core.chatbot.Path")
def test_vector_cache_loading(mock_path: MagicMock, mock_load: MagicMock, sample_database_path: str) -> None:
    """Test loading vectors from cache."""
    # Setup cache mocks
    mock_path_instance = MagicMock()
    mock_path.return_value = mock_path_instance
    mock_path_instance.exists.return_value = True
    mock_stat_result = MagicMock()
    mock_stat_result.st_mtime = 100
    mock_path_instance.stat.return_value = mock_stat_result
    mock_file = MagicMock()
    mock_path_instance.open.return_value.__enter__.return_value = mock_file
    
    # Setup cache data
    cache_file = "test_cache_load.pkl"
    mock_questions = ["How are you?", "What is your name?", "What is the weather like today?"]
    mock_vectors: np.ndarray = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])  # type: ignore[assignment]
    mock_model_name = "paraphrase-MiniLM-L6-v2"
    cache_data: dict[str, object] = {
        "questions": mock_questions,
        "vectors": mock_vectors,
        "model_name": mock_model_name
    }
    mock_load.return_value = cache_data
    
    # Mock the SentenceTransformer class
    with patch("bert_chatbot.core.chatbot.SentenceTransformer") as mock_transformer:
        # Initialize chatbot with mocked dependencies
        chatbot = SemanticChatBot(
            database_path=sample_database_path,
            cache_file=cache_file,
            model_name=mock_model_name
        )
        
        # Verify the chatbot's properties
        assert chatbot.model_name == mock_model_name
        assert mock_path.called
        assert mock_load.called


def test_offensive_term_handling(sample_chatbot: SemanticChatBot) -> None:
    """Test handling of offensive or unknown terms in queries."""
    query = "как молочка влияет на гомосеков?"
    response = sample_chatbot.find_answer(query)
    
    # Check for unknown terms message using parts that don't have Cyrillic
    assert "Извините" in response["answer"]
    assert "информации" in response["answer"]
    assert "Попробуйте переформулировать запрос" in response["answer"]
    assert not response["above_threshold"]
