"""B2 Retrieval-Augmented Personalization (RAG) Baseline implementation."""

from __future__ import annotations

import re
from typing import List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.data.synthetic import Interaction


class RetrievalBaseline:
    """B2 Baseline: Retrieves top-k historical interactions based on query similarity."""

    def __init__(self, top_k: int = 3, max_tokens: int = 256) -> None:
        self.top_k = top_k
        self.max_tokens = max_tokens

    def retrieve(self, query: str, history: List[Interaction]) -> List[Interaction]:
        """Retrieve the top-k most similar historical interactions using TF-IDF cosine similarity."""
        if not history:
            return []

        corpus = [f"{h.query} {h.context}" for h in history]
        try:
            vectorizer = TfidfVectorizer(stop_words="english")
            tfidf_matrix = vectorizer.fit_transform(corpus)
            query_vec = vectorizer.transform([query])
            scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
            ranked_indices = scores.argsort()[::-1][: self.top_k]
            return [history[idx] for idx in ranked_indices]
        except Exception:
            # Fallback to recent history if vectorizer cannot fit (e.g. empty vocab)
            return history[-self.top_k :]

    def format_prompt(self, query: str, retrieved: List[Interaction]) -> str:
        """Format retrieved interactions into few-shot in-context demonstrations with token budget control."""
        demonstrations = []
        for idx, item in enumerate(retrieved, 1):
            demo = (
                f"--- Example Past Interaction {idx} (Context: {item.context}) ---\n"
                f"User: {item.query}\n"
                f"Assistant: {item.user_response}\n"
            )
            demonstrations.append(demo)

        combined_demos = "\n".join(demonstrations)
        
        # Approximate token budget matching (1 word ~ 1.3 tokens)
        words = combined_demos.split()
        max_words = int(self.max_tokens / 1.3)
        if len(words) > max_words:
            combined_demos = " ".join(words[:max_words]) + "\n...[truncated to match token budget]..."

        return (
            f"<|system|>\nYou are an AI assistant. Below are past examples of how the user expects interactions to be handled:\n"
            f"{combined_demos}\n"
            f"Now respond to the user's current query in the same manner.\n"
            f"<|user|>\n{query}\n"
            f"<|assistant|>\n"
        )
