"""Structured preference schema, base extractor, and persistent caching."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from src.data.synthetic import Interaction

logger = logging.getLogger(__name__)


class InferredPreferenceRule(BaseModel):
    """Structured representation of an inferred preference rule."""
    rule_id: str
    preference: str = Field(description="Operational natural language preference statement")
    target_aspect: str = Field(description="Aspect category, e.g. verbosity, structure, tone, code_style, salutation")
    context_condition: str = Field(default="general", description="Context under which this preference applies")
    supporting_evidence_ids: List[str] = Field(default_factory=list, description="IDs of historical interactions exhibiting this preference")
    confidence: float = Field(default=0.8, description="Uncalibrated confidence estimate between 0.0 and 1.0")
    conflict_detected: bool = Field(default=False, description="Whether opposing evidence exists in other contexts")
    conflict_details: Optional[str] = Field(default=None, description="Explanation of the contextual conflict if detected")
    recency_weight: float = Field(default=1.0, description="Temporal recency weight score")


class PreferenceExtractionResult(BaseModel):
    """Container for preference extraction results for a user interaction."""
    user_id: str
    query_context: Optional[str]
    method: str
    rules: List[InferredPreferenceRule]
    raw_response: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_steering_prompt(self, max_rules: int = 5) -> str:
        """Render rules into natural language instructions for the main generator."""
        if not self.rules:
            return ""
        
        selected = self.rules[:max_rules]
        lines = ["User Personalization Preferences to adhere to strictly:"]
        for idx, r in enumerate(selected, 1):
            ctx_str = f" [Applicable in: {r.context_condition}]" if r.context_condition != "general" else ""
            lines.append(f"{idx}. {r.preference}{ctx_str} (Confidence: {r.confidence:.2f})")
        return "\n".join(lines)


class PreferenceCache:
    """Persistent on-disk cache for preference extraction results."""
    def __init__(self, cache_dir: str = ".cache/preferences") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _make_key(self, user_id: str, method: str, query: str, history_len: int, extra: str = "") -> str:
        raw = f"{user_id}::{method}::{query}::{history_len}::{extra}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, user_id: str, method: str, query: str, history_len: int, extra: str = "") -> Optional[PreferenceExtractionResult]:
        key = self._make_key(user_id, method, query, history_len, extra)
        p = self.cache_dir / f"{key}.json"
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return PreferenceExtractionResult(**data)
            except Exception as e:
                logger.warning("Cache read failed for key %s: %e", key, e)
        return None

    def set(self, user_id: str, method: str, query: str, history_len: int, result: PreferenceExtractionResult, extra: str = "") -> None:
        key = self._make_key(user_id, method, query, history_len, extra)
        p = self.cache_dir / f"{key}.json"
        try:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(result.model_dump(), f, indent=2)
        except Exception as e:
            logger.warning("Cache write failed for key %s: %e", key, e)
