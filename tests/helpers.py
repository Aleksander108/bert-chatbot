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
        "Help me with my homework": "Извините, я не нашел подходящего ответа на ваш вопрос.",
        "Tell me a joke": "Извините, я не нашел подходящего ответа на ваш вопрос.",
    }
    return responses.get(question, "Извините, я не нашел подходящего ответа на ваш вопрос.")
