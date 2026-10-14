"""Generation orchestrator connecting preference extractors, baseline formatters, and the frozen main LLM."""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from src.baselines.retrieval import RetrievalBaseline
from src.baselines.static import StaticBaseline
from src.data.synthetic import Interaction
from src.models.loader import ModelManager
from src.preferences.contextual import ContextAwarePreferenceExtractor, DynamicTaskExtractor
from src.preferences.extractor import InferredPreferenceRule, PreferenceCache
from src.utils.memory import track_execution_resources

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    user_id: str
    query: str
    context: str
    baseline_id: str
    prompt: str
    response_text: str
    prompt_tokens: int
    generated_tokens: int
    extraction_latency_sec: float
    generation_latency_sec: float
    peak_vram_mb: float
    inferred_rules: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class PersonalizationPipeline:
    """End-to-end personalization pipeline supporting all experimental baselines B0-B4."""

    def __init__(
        self,
        model_manager: ModelManager,
        cache_dir: str = ".cache/preferences",
        max_context_tokens: int = 256,
        temporal_half_life: float = 10.0,
    ) -> None:
        self.model_manager = model_manager
        self.cache = PreferenceCache(cache_dir=cache_dir)
        self.max_context_tokens = max_context_tokens

        # Baseline handlers
        self.b1_handler = StaticBaseline(cache=self.cache)
        self.b2_handler = RetrievalBaseline(top_k=3, max_tokens=max_context_tokens)
        self.b3_handler = DynamicTaskExtractor(cache=self.cache)
        self.b4_handler = ContextAwarePreferenceExtractor(cache=self.cache, temporal_half_life=temporal_half_life)

    def run_baseline(
        self,
        baseline_id: str,
        user_id: str,
        query: str,
        context: str,
        history: List[Interaction],
        max_rules: int = 5,
    ) -> GenerationResult:
        """Run a single baseline evaluation on a target query."""
        extraction_time = 0.0
        inferred_rules: List[InferredPreferenceRule] = []
        retrieved_ids: List[str] = []
        meta: Dict[str, Any] = {}

        t0 = time.perf_counter()

        if baseline_id == "B0":
            # Zero-shot, unpersonalized
            prompt = (
                f"<|system|>\nYou are a helpful and accurate AI assistant.\n"
                f"<|user|>\n{query}\n"
                f"<|assistant|>\n"
            )

        elif baseline_id == "B1":
            # Static preference profile
            profile_res = self.b1_handler.extract_profile(user_id=user_id, history=history)
            inferred_rules = profile_res.rules
            prompt = self.b1_handler.format_prompt(query, profile_res, max_rules=max_rules)

        elif baseline_id == "B2":
            # Retrieval-augmented personalization (RAG)
            retrieved = self.b2_handler.retrieve(query=query, history=history)
            retrieved_ids = [h.interaction_id for h in retrieved]
            prompt = self.b2_handler.format_prompt(query, retrieved)
            meta["retrieved_ids"] = retrieved_ids

        elif baseline_id == "B3":
            # Dynamic task-only extraction
            task_res = self.b3_handler.extract(user_id=user_id, query=query, history=history, max_rules=max_rules)
            inferred_rules = task_res.rules
            rules_prompt = task_res.to_steering_prompt(max_rules=max_rules)
            prompt = (
                f"<|system|>\nYou are an AI assistant. Comply with the user's inferred preferences for this task:\n"
                f"{rules_prompt}\n"
                f"<|user|>\n{query}\n"
                f"<|assistant|>\n"
            )

        elif baseline_id == "B4":
            # Context-Aware Preference Learning (CAPL)
            capl_res = self.b4_handler.extract(user_id=user_id, query=query, history=history, max_rules=max_rules)
            inferred_rules = capl_res.rules
            rules_prompt = capl_res.to_steering_prompt(max_rules=max_rules)
            meta.update(capl_res.metadata)
            prompt = (
                f"<|system|>\nYou are an AI assistant. Strictly adhere to these context-scoped user preferences:\n"
                f"{rules_prompt}\n"
                f"<|user|>\n{query}\n"
                f"<|assistant|>\n"
            )
        else:
            raise ValueError(f"Unknown baseline ID: {baseline_id}")

        extraction_time = time.perf_counter() - t0

        # Execute generation on the frozen main generator with resource tracking
        with track_execution_resources() as resource_stats:
            response_text = self.model_manager.generate_response(prompt=prompt, temperature=0.0)

        # Token count estimations
        prompt_tokens = len(prompt.split()) * 4 // 3
        generated_tokens = len(response_text.split()) * 4 // 3

        return GenerationResult(
            user_id=user_id,
            query=query,
            context=context,
            baseline_id=baseline_id,
            prompt=prompt,
            response_text=response_text,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            extraction_latency_sec=round(extraction_time, 4),
            generation_latency_sec=resource_stats["elapsed_sec"],
            peak_vram_mb=resource_stats["peak_vram_mb"],
            inferred_rules=[r.model_dump() for r in inferred_rules],
            metadata=meta,
        )
