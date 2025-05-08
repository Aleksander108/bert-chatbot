"""Utility functions and constants for tests."""

# Test data constants
TEST_QUESTIONS = [
    "How are you?",
    "What is your name?",
    "What is the weather like today?",
    "Help me with my homework",
    "Tell me a joke",
]

DEFAULT_QUESTION = "How are you?"


# Utility functions for testing
def get_expected_response(question: str) -> str:
    """Return the expected response for a test question."""
    responses = {
        "How are you?": "I am fine, thank you!",
        "What is your name?": "My name is BERT ChatBot.",
        "What is the weather like today?": "I cannot check the weather.",
        "Help me with my homework": "Sorry, I couldn't find a relevant answer.",
        "Tell me a joke": "Sorry, I couldn't find a relevant answer.",
    }
    return responses.get(question, "Sorry, I couldn't find a relevant answer.")
