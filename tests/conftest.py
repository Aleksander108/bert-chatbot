"""Test configuration and fixtures."""

import tempfile
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from bert_chatbot.core.chatbot import SemanticChatBot


@pytest.fixture
def sample_data() -> dict[str, list[str]]:
    """Create sample data for testing."""
    return {
        "вопрос": ["How are you?", "What is your name?", "What is the weather like today?"],
        "ответ": ["I am fine, thank you!", "My name is BERT ChatBot.", "I cannot check the weather."],
    }


@pytest.fixture
def sample_database_path() -> Generator[str, None, None]:
    """Generate a temporary Excel file for testing."""
    temp_file = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False)
    temp_path = temp_file.name
    temp_file.close()
    
    # Sample test data
    sample_data = {
        "вопрос": ["How are you?", "What is your name?", "What is the weather like today?"],
        "ответ": ["I am fine, thank you!", "My name is BERT ChatBot.", "I cannot check the weather."],
    }
    
    # Write to Excel file
    pd.DataFrame(sample_data).to_excel(temp_path, index=False)  # type: ignore[call-overload]
    
    try:
        yield temp_path
    finally:
        Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def sample_chatbot(sample_database_path: str) -> SemanticChatBot:
    """Create a SemanticChatBot instance with test data and mocked components."""
    # Create mock for SentenceTransformer and its methods
    with patch("bert_chatbot.core.chatbot.SentenceTransformer") as mock_transformer:
        # Configure the mock encode method
        mock_encode = MagicMock()
        mock_encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
        mock_transformer.return_value.encode = mock_encode
        
        # Create the chatbot with the mock and test data
        chatbot = SemanticChatBot(
            database_path=sample_database_path,
            similarity_threshold=0.6,
            keyword_match_threshold=0.3
        )
        
        return chatbot
