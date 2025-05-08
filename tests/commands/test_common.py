"""Tests for common CLI utilities and argument types."""

import os
from unittest.mock import patch

from bert_chatbot.commands.common import get_database_from_env


def test_get_database_from_env_returns_none_when_not_set():
    """Test that get_database_from_env returns None when the environment variable is not set."""
    with patch.dict(os.environ, {}, clear=True):
        assert get_database_from_env() is None


def test_get_database_from_env_returns_path_when_set():
    """Test that get_database_from_env returns the path when the environment variable is set."""
    test_path = "/path/to/database.xlsx"
    with patch.dict(os.environ, {"BERT_CHATBOT_DATABASE": test_path}, clear=True):
        assert get_database_from_env() == test_path 