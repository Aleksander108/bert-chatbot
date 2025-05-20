"""Tests for chat command functionality."""

import os
import sys
from typing import Any
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from bert_chatbot.commands.chat_cmd import interactive
from tests.helpers import DEFAULT_QUESTION, get_expected_response

runner = CliRunner(mix_stderr=False)


@patch("sys.exit")
@patch("builtins.input", return_value="exit")
@patch("bert_chatbot.commands.chat_cmd.SemanticChatBot")
@patch("os.path.exists", return_value=True)
def test_interactive_chat_init(
    mock_exists: MagicMock,  # type: ignore[type-arg]
    mock_chatbot_class: MagicMock,  # type: ignore[type-arg]
    mock_input: MagicMock,  # type: ignore[type-arg]
    mock_exit: MagicMock,  # type: ignore[type-arg]
    sample_database_path: str
) -> None:
    """Test chat command initialization with database path by calling the function directly."""
    # Setup mock chatbot
    mock_chatbot = MagicMock()
    mock_chatbot_class.return_value = mock_chatbot
    
    # Call the interactive chat function with a database path
    interactive(database=sample_database_path, similarity_threshold=0.7)
    
    # Check that the chatbot was initialized with the correct parameters
    mock_chatbot_class.assert_called_once_with(sample_database_path, 0.7, "vector_cache.pkl")


@patch("sys.exit")
@patch("builtins.input", return_value="exit")
@patch("bert_chatbot.commands.chat_cmd.SemanticChatBot")
@patch("os.path.exists", return_value=True)
def test_interactive_chat_with_options(
    mock_exists: MagicMock,  # type: ignore[type-arg]
    mock_chatbot_class: MagicMock,  # type: ignore[type-arg]
    mock_input: MagicMock,  # type: ignore[type-arg]
    mock_exit: MagicMock,  # type: ignore[type-arg]
    sample_database_path: str
) -> None:
    """Test chat command with custom options."""
    # Setup mock chatbot
    mock_chatbot = MagicMock()
    mock_chatbot_class.return_value = mock_chatbot
    
    # Call the interactive chat function with custom threshold
    interactive(database=sample_database_path, similarity_threshold=0.9)
    
    # Check that the chatbot was initialized with the correct parameters
    mock_chatbot_class.assert_called_once_with(sample_database_path, 0.9, "vector_cache.pkl")


@patch("sys.exit")
@patch("builtins.input", return_value="exit")
@patch("bert_chatbot.commands.chat_cmd.SemanticChatBot")
@patch("os.path.exists", return_value=True)
def test_interactive_chat_with_env_var(
    mock_exists: MagicMock,  # type: ignore[type-arg]
    mock_chatbot_class: MagicMock,  # type: ignore[type-arg]
    mock_input: MagicMock,  # type: ignore[type-arg]
    mock_exit: MagicMock  # type: ignore[type-arg]
) -> None:
    """Test chat command using database path from environment variable."""
    # Setup environment variable
    os.environ["BERT_CHATBOT_DATABASE"] = "test_env_var.xlsx"
    
    try:
        # Setup mock chatbot
        mock_chatbot = MagicMock()
        mock_chatbot_class.return_value = mock_chatbot
        
        # Call the interactive chat function without a database path
        interactive(database=None)
        
        # Check that the chatbot was initialized with the path from environment variable
        mock_chatbot_class.assert_called_once_with("test_env_var.xlsx", 0.5, "vector_cache.pkl")
    finally:
        # Clean up environment
        del os.environ["BERT_CHATBOT_DATABASE"]


@patch("os.path.exists", return_value=True)
@patch("sys.exit")
@patch("bert_chatbot.commands.chat_cmd.console")
@patch("bert_chatbot.commands.chat_cmd.SemanticChatBot")
def test_interactive_chat_question_answer(
    mock_chatbot_class: MagicMock,  # type: ignore[type-arg]
    mock_console: MagicMock,  # type: ignore[type-arg]
    mock_exit: MagicMock,  # type: ignore[type-arg]
    mock_exists: MagicMock,  # type: ignore[type-arg]
    sample_database_path: str
) -> None:
    """Test asking a question and getting the answer."""
    # Setup input sequence: first a question, then exit
    question = "How are you?"
    expected_answer = "I am fine, thank you!"
    
    with patch("builtins.input", side_effect=[question, "exit"]):
        # Setup mock chatbot
        mock_chatbot = MagicMock()
        mock_chatbot.find_answer.return_value = {
            "answer": expected_answer,
            "similarity": 0.95,
            "matched_question": question,
            "above_threshold": True
        }
        mock_chatbot_class.return_value = mock_chatbot
        
        # Call the interactive chat function
        interactive(database=sample_database_path)
        
        # Check that the chatbot was asked the right question
        mock_chatbot.find_answer.assert_called_with(question)
        
        # Check that the answer was printed with Rich console
        # It will contain the answer somewhere in the calls, not necessarily exact match
        assert any(expected_answer in str(call) for call in mock_console.print.call_args_list)
