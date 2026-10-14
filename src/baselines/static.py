"""B1 Static Profile Baseline implementation."""

from __future__ import annotations

from typing import List, Optional

from src.data.synthetic import Interaction
from src.preferences.contextual import StaticProfileExtractor
from src.preferences.extractor import PreferenceCache, PreferenceExtractionResult


class StaticBaseline:
    """B1 Baseline: Uses a static user profile extracted once from interaction history."""

    def __init__(self, cache: Optional[PreferenceCache] = None, model: Optional[Any] = None) -> None:
        self.extractor = StaticProfileExtractor(cache=cache, model=model)

    def extract_profile(self, user_id: str, history: List[Interaction], max_rules: int = 5) -> PreferenceExtractionResult:
        """Extract static profile once across all user interactions."""
        return self.extractor.extract(user_id=user_id, history=history, max_rules=max_rules)

    def format_prompt(self, query: str, profile_result: PreferenceExtractionResult, max_rules: int = 5, max_tokens: Optional[int] = None) -> str:
        """Format the steering prompt for the main generator."""
        rules_text = profile_result.to_steering_prompt(max_rules=max_rules, max_tokens=max_tokens)
        return (
            f"<|system|>\nYou are an AI assistant. Adhere strictly to the user's general behavioral profile:\n"
            f"{rules_text}\n"
            f"<|user|>\n{query}\n"
            f"<|assistant|>\n"
        )
