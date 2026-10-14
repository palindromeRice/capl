"""Unit tests for preference schemas, caching, and extraction algorithms B1, B3, B4."""

import tempfile
from pathlib import Path

from src.data.synthetic import generate_synthetic_dataset
from src.preferences.contextual import (
    ContextAwarePreferenceExtractor,
    DynamicTaskExtractor,
    StaticProfileExtractor,
    analyze_interaction_features,
    classify_query_context,
)
from src.preferences.extractor import (
    InferredPreferenceRule,
    PreferenceCache,
    PreferenceExtractionResult,
)


def test_inferred_rule_pydantic_validation() -> None:
    rule = InferredPreferenceRule(
        rule_id="test_1",
        preference="Keep responses concise and under 100 words.",
        target_aspect="verbosity",
        context_condition="technical_engineering",
        supporting_evidence_ids=["int_001", "int_002"],
        confidence=0.88,
        conflict_detected=True,
    )
    assert rule.rule_id == "test_1"
    assert rule.confidence == 0.88
    assert rule.conflict_detected is True

    result = PreferenceExtractionResult(
        user_id="user_001",
        query_context="technical_engineering",
        method="B4_contextual",
        rules=[rule],
        raw_response="test response",
    )
    prompt = result.to_steering_prompt()
    assert "User Personalization Preferences" in prompt
    assert "under 100 words" in prompt


def test_preference_cache() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = PreferenceCache(cache_dir=tmpdir)
        rule = InferredPreferenceRule(
            rule_id="r1",
            preference="Prefers bullet points",
            target_aspect="structure",
        )
        res = PreferenceExtractionResult(
            user_id="u1",
            query_context="general",
            method="B1_static",
            rules=[rule],
            raw_response="cached raw",
        )
        cache.set("u1", "B1_static", "test query", 10, res)

        cached_hit = cache.get("u1", "B1_static", "test query", 10)
        assert cached_hit is not None
        assert cached_hit.rules[0].preference == "Prefers bullet points"

        # Check miss
        miss = cache.get("u1", "B1_static", "different query", 10)
        assert miss is None


def test_b1_static_extractor() -> None:
    users = generate_synthetic_dataset(num_users=1, interactions_per_user=6, seed=42)
    user = users[0]
    extractor = StaticProfileExtractor()
    result = extractor.extract(user.user_id, user.history)
    assert len(result.rules) > 0
    assert result.method == "B1_static"


def test_b3_dynamic_extractor() -> None:
    users = generate_synthetic_dataset(num_users=1, interactions_per_user=6, seed=42)
    user = users[0]
    extractor = DynamicTaskExtractor()
    query = "How to optimize database queries?"
    result = extractor.extract(user.user_id, query, user.history)
    assert len(result.rules) > 0
    assert result.method == "B3_dynamic"


def test_b4_context_aware_extractor() -> None:
    users = generate_synthetic_dataset(num_users=1, interactions_per_user=8, seed=42)
    user = users[0]
    extractor = ContextAwarePreferenceExtractor(temporal_half_life=5.0)

    # Technical query
    tech_query = "Explain database locking algorithms"
    tech_res = extractor.extract(user.user_id, tech_query, user.history)
    assert tech_res.query_context == "technical_engineering"
    assert len(tech_res.rules) > 0

    # Executive query
    exec_query = "Provide infrastructure cost projections for board review"
    exec_res = extractor.extract(user.user_id, exec_query, user.history)
    assert exec_res.query_context == "executive_briefing"
    assert len(exec_res.rules) > 0
