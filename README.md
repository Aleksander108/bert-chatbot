# BERT Semantic ChatBot

A semantic chatbot that provides answers to user queries by finding the most semantically similar questions in an Excel database. The bot uses TF-IDF (Term Frequency-Inverse Document Frequency) vectorization to convert questions into numerical vectors and employs cosine similarity to find the closest match in the database.

## Features

- Semantic matching using TF-IDF vectorization
- Caching system for faster subsequent runs
- Interactive command-line interface
- Debug mode to show similarity scores and matched questions
- Configurable similarity threshold

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/bert-chatbot.git
cd bert-chatbot

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package and dependencies
pip install -e .
```

## Usage

The bot requires an Excel database with two columns:
- `вопрос` (question): Contains the user query
- `ответ` (answer): Contains the corresponding answer

### Interactive Chat

```bash
# Start the interactive chat with a database
bert-chatbot chat interactive path/to/database.xlsx

# With custom similarity threshold
bert-chatbot chat interactive path/to/database.xlsx --threshold 0.4

# With custom cache file
bert-chatbot chat interactive path/to/database.xlsx --cache-file my_cache.pkl
```

### Commands

Once in the interactive chat, you can use these commands:
- Type any question to get an answer
- Type `debug` to toggle debug mode (shows similarity scores)
- Type `exit`, `quit`, or `q` to end the chat

## How It Works

1. **TF-IDF Vectorization**: Converts text questions into numerical vectors
2. **Cosine Similarity**: Measures the similarity between the user query and all questions in the database
3. **Threshold Filtering**: Returns the most similar answer if it's above the configured threshold

## Future Improvements

- Advanced NLP models like BERT for better semantic understanding
- Multi-language support
- REST API interface
- Web or desktop user interface
- Database management tools
- Analytics for tracking common questions