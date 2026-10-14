"""Implementation of Static (B1), Dynamic (B3), and Context-Aware (B4) preference extraction algorithms."""

from __future__ import annotations

import json
import logging
import math
import re
from typing import Any, Dict, List, Optional

from src.data.synthetic import Interaction
from src.preferences.extractor import (
    InferredPreferenceRule,
    PreferenceCache,
    PreferenceExtractionResult,
)

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Heuristic Feature Extractor for Historical Interactions
# -----------------------------------------------------------------------------
def analyze_interaction_features(interaction: Interaction) -> Dict[str, Any]:
    """Analyze stylistic and behavioral features from an interaction's response text."""
    text = interaction.user_response.lower()
    words = text.split()
    word_count = len(words)

    has_bullets = bool(re.search(r"^\s*[-*•]\s+", interaction.user_response, re.MULTILINE))
    has_numbered = bool(re.search(r"^\s*\d+\.\s+", interaction.user_response, re.MULTILINE))
    has_code = "```" in interaction.user_response or "def " in interaction.user_response
    has_greeting = bool(re.search(r"^(hello|hi|greetings|hey there)\b", text, re.MULTILINE))
    has_signoff = bool(re.search(r"\b(sincerely|cheers|best regards|engineering lead)\b", text, re.MULTILINE))

    # Formality / Tone heuristics
    is_formal = ("sincerely" in text) or ("strategic" in text) or ("audit" in text) or ("align" in text)
    is_academic = ("analyzing" in text) or ("trade-offs" in text) or ("nomenclature" in text)

    return {
        "word_count": word_count,
        "is_terse": word_count < 100,
        "is_elaborate": word_count >= 100,
        "has_bullets": has_bullets,
        "has_numbered": has_numbered,
        "has_code": has_code,
        "has_salutation": has_greeting or has_signoff,
        "is_formal": is_formal,
        "is_academic": is_academic,
    }


def classify_query_context(query: str) -> str:
    """Classify the contextual domain of a user query."""
    q_low = query.lower()
    if any(k in q_low for k in ["database", "locking", "memory", "paxos", "raft", "circuit", "microservice", "latency", "python", "code", "bug"]):
        return "technical_engineering"
    if any(k in q_low for k in ["cost", "cloud migration", "risk", "board", "executive", "vendor", "roadmap", "strategy", "slo"]):
        return "executive_briefing"
    if any(k in q_low for k in ["explain", "binary search", "memoization", "transformer", "gradient", "big-o", "intuition", "tutoring"]):
        return "educational_tutoring"
    return "casual_chat"


# -----------------------------------------------------------------------------
# B1: Static Preference Profile Extractor
# -----------------------------------------------------------------------------
class StaticProfileExtractor:
    """B1 Baseline: Extracts a single static preference profile across all interactions."""

    def __init__(self, cache: Optional[PreferenceCache] = None, model: Any = None) -> None:
        self.cache = cache
        self.model = model

    def extract(self, user_id: str, history: List[Interaction], max_rules: int = 5) -> PreferenceExtractionResult:
        if self.cache:
            hit = self.cache.get(user_id, "B1_static", "", len(history))
            if hit:
                return hit

        if not history:
            return PreferenceExtractionResult(
                user_id=user_id,
                query_context="general",
                method="B1_static",
                rules=[],
                raw_response="No history available.",
            )

        # Aggregate across entire history regardless of context
        features = [analyze_interaction_features(h) for h in history]
        total = len(features)
        
        terse_count = sum(1 for f in features if f["is_terse"])
        bullet_count = sum(1 for f in features if f["has_bullets"])
        numbered_count = sum(1 for f in features if f["has_numbered"])
        code_count = sum(1 for f in features if f["has_code"])
        salutation_count = sum(1 for f in features if f["has_salutation"])

        rules: List[InferredPreferenceRule] = []
        rule_idx = 0

        # Global verbosity rule
        rule_idx += 1
        is_terse = terse_count >= total / 2
        rules.append(InferredPreferenceRule(
            rule_id=f"rule_b1_{rule_idx}",
            preference="Keep responses concise and under 100 words." if is_terse else "Provide detailed, comprehensive explanations.",
            target_aspect="verbosity",
            context_condition="general",
            supporting_evidence_ids=[h.interaction_id for h, f in zip(history, features) if f["is_terse"] == is_terse][:3],
            confidence=round(max(terse_count, total - terse_count) / total, 2),
            conflict_detected=False,
        ))

        # Global structure rule
        rule_idx += 1
        if bullet_count > total / 3:
            rules.append(InferredPreferenceRule(
                rule_id=f"rule_b1_{rule_idx}",
                preference="Format content in bullet points.",
                target_aspect="structure",
                context_condition="general",
                supporting_evidence_ids=[h.interaction_id for h, f in zip(history, features) if f["has_bullets"]][:3],
                confidence=round(bullet_count / total, 2),
            ))
        elif numbered_count > total / 3:
            rules.append(InferredPreferenceRule(
                rule_id=f"rule_b1_{rule_idx}",
                preference="Organize responses in numbered sequential steps.",
                target_aspect="structure",
                context_condition="general",
                supporting_evidence_ids=[h.interaction_id for h, f in zip(history, features) if f["has_numbered"]][:3],
                confidence=round(numbered_count / total, 2),
            ))

        # Global salutation rule
        rule_idx += 1
        has_salut = salutation_count > total / 2
        rules.append(InferredPreferenceRule(
            rule_id=f"rule_b1_{rule_idx}",
            preference="Include polite salutations and sign-offs." if has_salut else "Never include greetings or sign-offs.",
            target_aspect="salutation",
            context_condition="general",
            supporting_evidence_ids=[h.interaction_id for h, f in zip(history, features) if f["has_salutation"] == has_salut][:3],
            confidence=round(max(salutation_count, total - salutation_count) / total, 2),
        ))

        result = PreferenceExtractionResult(
            user_id=user_id,
            query_context="general",
            method="B1_static",
            rules=rules[:max_rules],
            raw_response=f"Aggregated static profile across {total} historical interactions.",
        )

        if self.cache:
            self.cache.set(user_id, "B1_static", "", len(history), result)

        return result


# -----------------------------------------------------------------------------
# B3: Dynamic Task-Only Preference Extractor
# -----------------------------------------------------------------------------
class DynamicTaskExtractor:
    """B3 Baseline: Extracts task-specific preferences without contextual scoping or conflict resolution."""

    def __init__(self, cache: Optional[PreferenceCache] = None, model: Any = None) -> None:
        self.cache = cache
        self.model = model

    def extract(self, user_id: str, query: str, history: List[Interaction], max_rules: int = 5) -> PreferenceExtractionResult:
        if self.cache:
            hit = self.cache.get(user_id, "B3_dynamic", query, len(history))
            if hit:
                return hit

        if not history:
            return PreferenceExtractionResult(
                user_id=user_id,
                query_context=None,
                method="B3_dynamic",
                rules=[],
                raw_response="No history available.",
            )

        # Inspect historical interactions with task keyword matching, but WITHOUT context separation or temporal decay
        q_tokens = set(re.findall(r"\w+", query.lower()))
        scored_history = []
        for h in history:
            h_tokens = set(re.findall(r"\w+", h.query.lower()))
            overlap = len(q_tokens.intersection(h_tokens))
            scored_history.append((overlap, h))

        scored_history.sort(key=lambda x: x[0], reverse=True)
        top_interactions = [h for _, h in scored_history[:5]]

        features = [analyze_interaction_features(h) for h in top_interactions]
        total = len(features)

        rules: List[InferredPreferenceRule] = []
        # Task verbosity
        terse_count = sum(1 for f in features if f["is_terse"])
        is_terse = terse_count >= total / 2
        rules.append(InferredPreferenceRule(
            rule_id="rule_b3_1",
            preference=f"For tasks like '{query[:30]}...', keep answers under 100 words." if is_terse else f"For tasks like '{query[:30]}...', provide elaborate technical reasoning.",
            target_aspect="verbosity",
            context_condition="task_specific",
            supporting_evidence_ids=[h.interaction_id for h in top_interactions[:2]],
            confidence=0.75,
            conflict_detected=False,
        ))

        # Task structure
        bullet_count = sum(1 for f in features if f["has_bullets"])
        if bullet_count > total / 2:
            rules.append(InferredPreferenceRule(
                rule_id="rule_b3_2",
                preference="Format the response in bulleted points for quick scanning.",
                target_aspect="structure",
                context_condition="task_specific",
                supporting_evidence_ids=[h.interaction_id for h, f in zip(top_interactions, features) if f["has_bullets"]][:2],
                confidence=0.75,
            ))

        result = PreferenceExtractionResult(
            user_id=user_id,
            query_context=None,
            method="B3_dynamic",
            rules=rules[:max_rules],
            raw_response=f"Dynamically extracted task rules for query: {query}",
        )

        if self.cache:
            self.cache.set(user_id, "B3_dynamic", query, len(history), result)

        return result


# -----------------------------------------------------------------------------
# B4: Context-Aware Preference Learning (CAPL - Proposed Method)
# -----------------------------------------------------------------------------
class ContextAwarePreferenceExtractor:
    """B4 Proposed: Context decomposition, temporal weighting, conflict resolution, and structured rules."""

    def __init__(self, cache: Optional[PreferenceCache] = None, model: Any = None, temporal_half_life: float = 10.0) -> None:
        self.cache = cache
        self.model = model
        self.temporal_half_life = temporal_half_life

    def extract(self, user_id: str, query: str, history: List[Interaction], max_rules: int = 5) -> PreferenceExtractionResult:
        if self.cache:
            hit = self.cache.get(user_id, "B4_contextual", query, len(history))
            if hit:
                return hit

        if not history:
            return PreferenceExtractionResult(
                user_id=user_id,
                query_context="general",
                method="B4_contextual",
                rules=[],
                raw_response="No history available.",
            )

        # 1. Context Decomposition
        detected_context = classify_query_context(query)
        current_step = max(h.step for h in history) + 1

        # 2. Context Partitioning & Temporal Weighting
        matching_interactions: List[tuple[Interaction, float, Dict[str, Any]]] = []
        divergent_interactions: List[tuple[Interaction, float, Dict[str, Any]]] = []

        for h in history:
            # Half-life temporal decay
            delta_t = max(0, current_step - h.step)
            w_recency = math.exp(-math.log(2) * (delta_t / self.temporal_half_life))
            feat = analyze_interaction_features(h)

            if h.context == detected_context:
                matching_interactions.append((h, w_recency, feat))
            else:
                divergent_interactions.append((h, w_recency, feat))

        # 3. Detect Conflicts Between Contexts
        conflict_detected = False
        conflict_msg = None

        if matching_interactions and divergent_interactions:
            match_terse_weight = sum(w for _, w, f in matching_interactions if f["is_terse"])
            match_total_weight = sum(w for _, w, _ in matching_interactions)
            match_terse_ratio = match_terse_weight / max(match_total_weight, 1e-6)

            divergent_terse_weight = sum(w for _, w, f in divergent_interactions if f["is_terse"])
            divergent_total_weight = sum(w for _, w, _ in divergent_interactions)
            divergent_terse_ratio = divergent_terse_weight / max(divergent_total_weight, 1e-6)

            if abs(match_terse_ratio - divergent_terse_ratio) > 0.4:
                conflict_detected = True
                conflict_msg = f"User prefers {'terse' if match_terse_ratio > 0.5 else 'elaborate'} in '{detected_context}', but behaves with opposing verbosity in other contexts."

        # 4. Context-Scoped Preference Synthesis
        rules: List[InferredPreferenceRule] = []
        rule_idx = 0

        target_pool = matching_interactions if matching_interactions else [(h, 1.0, analyze_interaction_features(h)) for h in history[-5:]]
        total_w = sum(w for _, w, _ in target_pool)

        # Verbosity Aspect
        rule_idx += 1
        terse_w = sum(w for _, w, f in target_pool if f["is_terse"])
        is_terse = (terse_w / max(total_w, 1e-6)) >= 0.5
        conf_verb = round(max(terse_w, total_w - terse_w) / max(total_w, 1e-6), 2)
        rules.append(InferredPreferenceRule(
            rule_id=f"rule_b4_{rule_idx}",
            preference=f"Under {detected_context}, keep responses strictly under 100 words with zero unnecessary preamble." if is_terse else f"Under {detected_context}, provide comprehensive architectural depth with thorough trade-off analysis.",
            target_aspect="verbosity",
            context_condition=detected_context,
            supporting_evidence_ids=[h.interaction_id for h, _, f in target_pool if f["is_terse"] == is_terse][:3],
            confidence=conf_verb,
            conflict_detected=conflict_detected,
            conflict_details=conflict_msg,
            recency_weight=round(sum(w for _, w, f in target_pool if f["is_terse"] == is_terse) / max(len(target_pool), 1), 2),
        ))

        # Structure Aspect
        bullet_w = sum(w for _, w, f in target_pool if f["has_bullets"])
        numbered_w = sum(w for _, w, f in target_pool if f["has_numbered"])
        rule_idx += 1
        if bullet_w > total_w * 0.35:
            rules.append(InferredPreferenceRule(
                rule_id=f"rule_b4_{rule_idx}",
                preference=f"Under {detected_context}, present all recommendations in crisp bullet points.",
                target_aspect="structure",
                context_condition=detected_context,
                supporting_evidence_ids=[h.interaction_id for h, _, f in target_pool if f["has_bullets"]][:3],
                confidence=round(bullet_w / max(total_w, 1e-6), 2),
                conflict_detected=conflict_detected,
            ))
        elif numbered_w > total_w * 0.35:
            rules.append(InferredPreferenceRule(
                rule_id=f"rule_b4_{rule_idx}",
                preference=f"Under {detected_context}, structure analysis in strictly numbered actionable steps.",
                target_aspect="structure",
                context_condition=detected_context,
                supporting_evidence_ids=[h.interaction_id for h, _, f in target_pool if f["has_numbered"]][:3],
                confidence=round(numbered_w / max(total_w, 1e-6), 2),
                conflict_detected=conflict_detected,
            ))

        # Salutation Aspect
        rule_idx += 1
        salut_w = sum(w for _, w, f in target_pool if f["has_salutation"])
        has_salut = (salut_w / max(total_w, 1e-6)) >= 0.5
        conf_salut = round(max(salut_w, total_w - salut_w) / max(total_w, 1e-6), 2)
        rules.append(InferredPreferenceRule(
            rule_id=f"rule_b4_{rule_idx}",
            preference=f"Under {detected_context}, include formal executive greeting and sign-off." if has_salut else f"Under {detected_context}, never include greetings or sign-offs; jump immediately into the answer.",
            target_aspect="salutation",
            context_condition=detected_context,
            supporting_evidence_ids=[h.interaction_id for h, _, f in target_pool if f["has_salutation"] == has_salut][:3],
            confidence=conf_salut,
            conflict_detected=conflict_detected,
            recency_weight=1.0,
        ))

        # Code Style Aspect
        code_w = sum(w for _, w, f in target_pool if f["has_code"])
        if code_w > total_w * 0.4:
            rule_idx += 1
            rules.append(InferredPreferenceRule(
                rule_id=f"rule_b4_{rule_idx}",
                preference=f"Under {detected_context}, provide executable Python code implementations.",
                target_aspect="code_style",
                context_condition=detected_context,
                supporting_evidence_ids=[h.interaction_id for h, _, f in target_pool if f["has_code"]][:3],
                confidence=round(code_w / max(total_w, 1e-6), 2),
            ))
        elif detected_context == "executive_briefing":
            rule_idx += 1
            rules.append(InferredPreferenceRule(
                rule_id=f"rule_b4_{rule_idx}",
                preference="Under executive_briefing, avoid raw code snippets and focus on business trade-offs.",
                target_aspect="code_style",
                context_condition="executive_briefing",
                supporting_evidence_ids=[],
                confidence=0.9,
            ))

        result = PreferenceExtractionResult(
            user_id=user_id,
            query_context=detected_context,
            method="B4_contextual",
            rules=rules[:max_rules],
            raw_response=f"Context-aware synthesis for context '{detected_context}' with conflict_detected={conflict_detected}.",
            metadata={"conflict_detected": conflict_detected, "context": detected_context},
        )

        if self.cache:
            self.cache.set(user_id, "B4_contextual", query, len(history), result)

        return result
