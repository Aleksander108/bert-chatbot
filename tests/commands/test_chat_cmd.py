"""Tests for chat command functionality."""

import sys
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from bert_chatbot.commands.chat_cmd import interactive
from tests.helpers import DEFAULT_QUESTION, get_expected_response

runner = CliRunner(mix_stderr=False)


@patch("bert_chatbot.commands.chat_cmd.SemanticChatBot")
@patch("bert_chatbot.commands.chat_cmd.console.input", return_value="exit")
@patch.object(sys, "exit")
def test_interactive_chat_init(mock_exit, mock_input, mock_chatbot_class, sample_database_path) -> None:
    """Test chat command initialization with database path by calling the function directly."""
    # Call the function directly instead of through the Typer app
    interactive(sample_database_path)

    # Check the chatbot was initialized with the right parameters
    mock_chatbot_class.assert_called_once_with(
        sample_database_path,
        0.3,  # default threshold
        "vector_cache.pkl",  # default cache file
    )


@patch("bert_chatbot.commands.chat_cmd.SemanticChatBot")
@patch("bert_chatbot.commands.chat_cmd.console.input", return_value="exit")
@patch.object(sys, "exit")
def test_interactive_chat_with_options(mock_exit, mock_input, mock_chatbot_class, sample_database_path) -> None:
    """Test chat command with custom threshold and cache file."""
    # Call the function directly with custom parameters
    interactive(sample_database_path, 0.5, "custom_cache.pkl")

    # Check the chatbot was initialized with the right parameters
    mock_chatbot_class.assert_called_once_with(
        sample_database_path,
        0.5,  # custom threshold
        "custom_cache.pkl",  # custom cache file
    )


@patch("bert_chatbot.commands.chat_cmd.SemanticChatBot")
@patch("bert_chatbot.commands.chat_cmd.console.print")
@patch.object(sys, "exit")
def test_interactive_chat_question_answer(mock_exit, mock_print, mock_chatbot_class, sample_database_path) -> None:
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
        interactive(sample_database_path)

    # Check the chatbot's find_answer method was called with the right query
    mock_chatbot.find_answer.assert_called_once_with(DEFAULT_QUESTION)

    # Check that the answer was printed (at least one call should include the answer)
    answer_printed = False
    for call_args in mock_print.call_args_list:
        if get_expected_response(DEFAULT_QUESTION) in str(call_args):
            answer_printed = True
            break

    assert answer_printed, "Expected answer was not printed"
