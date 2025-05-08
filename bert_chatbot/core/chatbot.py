"""Core implementation of the semantic chatbot."""

import os
import pickle
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SemanticChatBot:
    """A chatbot that finds answers using semantic similarity with TF-IDF vectorization."""

    def __init__(
        self,
        database_path: str,
        similarity_threshold: float = 0.3,
        cache_file: str = "vector_cache.pkl",
    ) -> None:
        """Initialize the chatbot with the database and vectorizer.

        Args:
            database_path: Path to the Excel database file
            similarity_threshold: Minimum similarity score to consider a match (0-1)
            cache_file: File to cache the vectorized questions
        """
        self.database_path = database_path
        self.similarity_threshold = similarity_threshold
        self.cache_file = cache_file
        self.vectorizer = TfidfVectorizer()
        
        # Load database and prepare vectors
        self.df = pd.read_excel(database_path)
        self.questions = self.df["вопрос"].tolist()
        self.answers = self.df["ответ"].tolist()
        
        # Load or generate vectors
        self.question_vectors = self._load_or_generate_vectors()
    
    def _load_or_generate_vectors(self) -> Any:
        """Load cached vectors if available and valid, otherwise generate new ones.
        
        Returns:
            The TF-IDF vectors for all questions in the database
        """
        # Check if cached vectors exist and are newer than database
        if os.path.exists(self.cache_file) and os.path.getmtime(self.cache_file) > os.path.getmtime(self.database_path):
            try:
                with open(self.cache_file, "rb") as f:
                    cache_data = pickle.load(f)
                
                # Verify cached questions match current questions
                if cache_data["questions"] == self.questions:
                    return cache_data["vectors"]
            except (pickle.UnpicklingError, KeyError):
                # If cache is invalid, regenerate vectors
                pass
        
        # Generate new vectors
        vectors = self.vectorizer.fit_transform(self.questions)
        
        # Cache the vectors
        cache_data = {
            "questions": self.questions,
            "vectors": vectors
        }
        with open(self.cache_file, "wb") as f:
            pickle.dump(cache_data, f)
        
        return vectors
    
    def find_answer(self, query: str, debug: bool = False) -> Dict[str, Any]:
        """Find the most similar question and its answer.
        
        Args:
            query: The user question to find a match for
            debug: Whether to include debug information in the response
            
        Returns:
            Dictionary with the answer, similarity score, and matched question
        """
        # Vectorize the query
        query_vector = self.vectorizer.transform([query])
        
        # Calculate similarity with all questions
        similarities = cosine_similarity(query_vector, self.question_vectors).flatten()
        
        # Find the most similar question
        max_sim_idx = np.argmax(similarities)
        max_similarity = similarities[max_sim_idx]
        
        # Prepare the response
        response = {
            "answer": self.answers[max_sim_idx] if max_similarity >= self.similarity_threshold else "Sorry, I couldn't find a relevant answer.",
            "similarity": float(max_similarity),
            "matched_question": self.questions[max_sim_idx],
            "above_threshold": max_similarity >= self.similarity_threshold
        }
        
        return response 