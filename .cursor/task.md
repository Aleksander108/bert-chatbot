# Task

Given my current project starter, transform (adopt) the project to implement the following.
My old project name was py-hello.
My new project name is going to be bert-chatbot.

# Semantic Chat Bot with Vector-Based Search

## Project Overview

This project implements a semantic chatbot that provides answers to user queries by finding the most semantically similar questions in an Excel database. The bot uses TF-IDF (Term Frequency-Inverse Document Frequency) vectorization to convert questions into numerical vectors and employs cosine similarity to find the closest match in the database.

## Architecture

The system consists of the following components:

1. **Data Storage**: An Excel database with two columns:
   - `вопрос` (question): Contains the user query
   - `ответ` (answer): Contains the corresponding answer

2. **Vector Generation**: TF-IDF vectorization converts text questions into numerical vectors

3. **Similarity Matching**: Cosine similarity measures determine the closest match between a user query and the database questions

4. **Caching System**: Vector embeddings are cached to disk for faster subsequent runs

## Implementation Details

### Core Components

- **SemanticChatBot Class**: The main class handling all functionality
- **TfidfVectorizer**: Converts text to vector representations
- **Pandas**: Handles Excel data processing
- **Numpy & Scikit-learn**: Perform vector operations and similarity calculations

### Data Flow

1. **Initialization**:
   - Load Excel database with questions and answers
   - Initialize TF-IDF vectorizer
   - Generate or load cached vectors

2. **Query Processing**:
   - Convert user query to TF-IDF vector
   - Calculate cosine similarity with all questions in database
   - Find the most similar question based on similarity score
   - Return corresponding answer if above threshold

3. **Vector Caching**:
   - Check if cached vectors exist and are newer than the database
   - Load cached vectors if valid
   - Generate new vectors if needed
   - Save generated vectors to disk

## Technical Decisions

### Why TF-IDF Vectorization?

TF-IDF was chosen for its balance of simplicity and effectiveness. It captures the importance of words in a document relative to a corpus, allowing for semantic matching without requiring deep learning models or external dependencies.

Advantages:

- Fast computation
- No external model downloads required
- Effective for domain-specific vocabulary
- Language-agnostic (works with multiple languages)

### Similarity Threshold

The system uses a configurable similarity threshold (default: 0.3) to determine if a match is relevant. This allows for tuning the trade-off between:

- Higher threshold: More precise matches but may miss similar questions
- Lower threshold: More flexible matching but may return less relevant answers

### Caching Mechanism

Vectors are cached to disk to improve performance on subsequent runs. The cache validation ensures that:

- If the database changes, new vectors are generated
- If questions match the cached version, cached vectors are reused

## Usage

The bot can be used in interactive mode where users type questions and receive answers:

```bash
python chatbot.py path/to/database.xlsx
```

Key features:

- Debug mode shows similarity scores and matched questions
- Ability to toggle debug mode with 'debug' command
- Exit with 'exit', 'quit', or 'q' commands

## Future Improvements

Potential enhancements for the system:

1. **Advanced NLP**: Integration with more sophisticated NLP models like BERT/GPT for better semantic understanding
2. **Multi-language Support**: Enhanced language processing for non-Russian languages
3. **API Interface**: REST API for integration with other systems
4. **UI Frontend**: Web or desktop interface for non-technical users
5. **Database Management**: Admin interface for maintaining the Q&A database
6. **Analytics**: Tracking of common questions and answer effectiveness
