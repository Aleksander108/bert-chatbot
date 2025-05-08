"""Tests for chat command functionality."""

import os
import sys
from typing import Any
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from bert_chatbot.commands.chat_cmd import interactive
from tests.helpers import DEFAULT_QUESTION, get_expected_response

runner = CliRunner(mix_stderr=False)


@patch("bert_chatbot.commands.chat_cmd.SemanticChatBot")
@patch("bert_chatbot.commands.chat_cmd.console.input", return_value="exit")
@patch.object(sys, "exit")
@patch("bert_chatbot.commands.chat_cmd.os.path.exists", return_value=True)
def test_interactive_chat_init(
    mock_exists: Any, mock_exit: Any, mock_input: Any, mock_chatbot_class: Any, sample_database_path: str
) -> None:
    """Test chat command initialization with database path by calling the function directly."""
    # Call the function directly instead of through the Typer app
    interactive(database=sample_database_path)

    # Check the chatbot was initialized with the right parameters
    mock_chatbot_class.assert_called_once_with(
        sample_database_path,
        0.3,  # default threshold
        "vector_cache.pkl",  # default cache file
    )


@patch("bert_chatbot.commands.chat_cmd.SemanticChatBot")
@patch("bert_chatbot.commands.chat_cmd.console.input", return_value="exit")
@patch.object(sys, "exit")
@patch("bert_chatbot.commands.chat_cmd.os.path.exists", return_value=True)
def test_interactive_chat_with_options(
    mock_exists: Any, mock_exit: Any, mock_input: Any, mock_chatbot_class: Any, sample_database_path: str
) -> None:
    """Test chat command with custom threshold and cache file."""
    # Call the function directly with custom parameters
    interactive(database=sample_database_path, similarity_threshold=0.5, cache_file="custom_cache.pkl")

    # Check the chatbot was initialized with the right parameters
    mock_chatbot_class.assert_called_once_with(
        sample_database_path,
        0.5,  # custom threshold
        "custom_cache.pkl",  # custom cache file
    )


@patch("bert_chatbot.commands.chat_cmd.SemanticChatBot")
@patch("bert_chatbot.commands.chat_cmd.console.input", return_value="exit")
@patch.object(sys, "exit")
@patch("bert_chatbot.commands.chat_cmd.os.path.exists", return_value=True)
def test_interactive_chat_with_env_var(
    mock_exists: Any, mock_exit: Any, mock_input: Any, mock_chatbot_class: Any
) -> None:
    """Test chat command using database path from environment variable."""
    test_db_path = "/path/from/env/database.xlsx"
    
    # Set the environment variable
    with patch.dict(os.environ, {"BERT_CHATBOT_DATABASE": test_db_path}, clear=True):
        # Call the function without providing a database path
        interactive()

    # Check the chatbot was initialized with the path from environment variable
    mock_chatbot_class.assert_called_once_with(
        test_db_path,
        0.3,  # default threshold
        "vector_cache.pkl",  # default cache file
    )


@patch("bert_chatbot.commands.chat_cmd.os.path.exists", return_value=True)
@patch.object(sys, "exit")
@patch("bert_chatbot.commands.chat_cmd.console.print")
@patch("bert_chatbot.commands.chat_cmd.SemanticChatBot")
def test_interactive_chat_question_answer(
    mock_chatbot_class: Any, mock_print: Any, mock_exit: Any, mock_exists: Any, sample_database_path: str
) -> None:
    """Test asking a question and getting the answer."""
    # Create a mock chatbot instance
    mock_chatbot = MagicMock()
    # Configure the mock chatbot's find_answer method to return a known response
    mock_chatbot.find_answer.return_value = {
        "answer": get_expected_response(DEFAULT_QUESTION),
        "similarity": 0.95,
        "matched_question": DEFAULT_QUESTION,
        "above_threshold": True,
    }
    # Configure the mock class to return our mock chatbot instance
    mock_chatbot_class.return_value = mock_chatbot

    # Mock console.input to ask a question and then exit
    with patch("bert_chatbot.commands.chat_cmd.console.input", side_effect=[DEFAULT_QUESTION, "exit"]):
        # Call the function directly
        interactive(database=sample_database_path)

    # Check the chatbot's find_answer method was called with the right query
    mock_chatbot.find_answer.assert_called_once_with(DEFAULT_QUESTION)

    # Check that the answer was printed (at least one call should include the answer)
    answer_printed = False
    for call_args in mock_print.call_args_list:
        if get_expected_response(DEFAULT_QUESTION) in str(call_args):
            answer_printed = True
            break

    assert answer_printed, "Expected answer was not printed"
