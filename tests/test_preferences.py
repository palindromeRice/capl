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


def test_parse_preference_rules_json_robustness() -> None:
    from src.preferences.contextual import parse_preference_rules_json

    # 1. Markdown code block JSON
    raw_md = (
        "Here are the extracted rules:\n```json\n[\n"
        '  {"rule_id": "r1", "preference": "Use bullet points", "target_aspect": "structure", "confidence": 0.9}\n'
        "]\n```"
    )
    rules_md = parse_preference_rules_json(raw_md)
    assert len(rules_md) == 1
    assert rules_md[0].preference == "Use bullet points"

    # 2. Bare JSON array with missing fields
    raw_bare = '[{"preference": "Concise answers", "target_aspect": "verbosity"}]'
    rules_bare = parse_preference_rules_json(raw_bare)
    assert len(rules_bare) == 1
    assert rules_bare[0].target_aspect == "verbosity"
    assert rules_bare[0].confidence == 0.8  # default

    # 3. Malformed syntax
    raw_invalid = "Not a json string at all"
    rules_inv = parse_preference_rules_json(raw_invalid)
    assert rules_inv == []


def test_llm_driven_preference_extraction() -> None:
    from src.models.loader import ModelConfig, ModelManager

    users = generate_synthetic_dataset(num_users=1, interactions_per_user=6, seed=42)
    user = users[0]

    pref_cfg = ModelConfig(model_id="Qwen/Qwen3.5-2B")
    main_cfg = ModelConfig(model_id="Qwen/Qwen3.5-9B")
    mm = ModelManager(pref_config=pref_cfg, main_config=main_cfg, mock_mode=True)

    # B4 with LLM model
    b4_extractor = ContextAwarePreferenceExtractor(model=mm)
    res_b4 = b4_extractor.extract(user.user_id, "Explain database locking algorithms", user.history)
    assert len(res_b4.rules) > 0
    assert res_b4.query_context == "technical_engineering"

    # B1 with LLM model
    b1_extractor = StaticProfileExtractor(model=mm)
    res_b1 = b1_extractor.extract(user.user_id, user.history)
    assert len(res_b1.rules) > 0

    # B3 with LLM model
    b3_extractor = DynamicTaskExtractor(model=mm)
    res_b3 = b3_extractor.extract(user.user_id, "Explain database locking algorithms", user.history)
    assert len(res_b3.rules) > 0


def test_token_budget_matching_in_prompt() -> None:
    rule_1 = InferredPreferenceRule(rule_id="r1", preference="A" * 100, target_aspect="verbosity")
    rule_2 = InferredPreferenceRule(rule_id="r2", preference="B" * 100, target_aspect="structure")
    rule_3 = InferredPreferenceRule(rule_id="r3", preference="C" * 100, target_aspect="salutation")

    res = PreferenceExtractionResult(
        user_id="u1",
        query_context="general",
        method="B4_contextual",
        rules=[rule_1, rule_2, rule_3],
        raw_response="",
    )

    full_prompt = res.to_steering_prompt(max_rules=3)
    bounded_prompt = res.to_steering_prompt(max_rules=3, max_tokens=15)
    assert len(bounded_prompt) < len(full_prompt)
