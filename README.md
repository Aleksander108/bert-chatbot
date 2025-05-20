# BERT Semantic ChatBot

A semantic chatbot that provides answers to user queries by finding the most semantically similar questions in an Excel database. The bot uses TF-IDF (Term Frequency-Inverse Document Frequency) vectorization to convert questions into numerical vectors and employs cosine similarity to find the closest match in the database.

## Features

- Semantic matching using TF-IDF vectorization
- Caching system for faster subsequent runs
- Interactive command-line interface
- Single question mode via command line
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

# Using environment variable for database path (instead of command line argument)
export BERT_CHATBOT_DATABASE=/path/to/database.xlsx
bert-chatbot chat interactive
```

### Single Question Mode

```bash
# Ask a single question directly from the command line
bert-chatbot chat ask path/to/database.xlsx "What is the capital of France?"

# With debug information
bert-chatbot chat ask path/to/database.xlsx "What is the capital of France?" --debug

# With custom similarity threshold
bert-chatbot chat ask path/to/database.xlsx "What is the capital of France?" --threshold 0.4

# Using environment variable for database path
export BERT_CHATBOT_DATABASE=/path/to/database.xlsx
bert-chatbot chat ask "What is the capital of France?"
```

### Web Interface (FastAPI)

To use the web interface:

1.  **Install dependencies** (if you haven't already, ensure `uv` syncs with `pyproject.toml`):
    ```bash
    uv sync
    ```

2.  **Run the FastAPI application** from the project root directory:
    ```bash
    python main.py
    ```
    Alternatively, using uvicorn directly (useful for development with auto-reload):
    ```bash
    uvicorn main:app --reload
    ```

3.  **Open your browser** and go to:
    ```
    http://127.0.0.1:8000
    ```
    The `main.py` script will attempt to load `test_data_production.xlsx` from the root directory by default. You can change the `DATABASE_PATH` variable in `main.py` if your data file is located elsewhere or named differently.

### Commands in Interactive Mode

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