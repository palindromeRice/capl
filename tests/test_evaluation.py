"""Unit tests for inference metrics, deterministic checks, and pairwise judging."""

from src.data.synthetic import GroundTruthRule
from src.evaluation.metrics import (
    BlindedPairwiseJudge,
    compute_confidence_interval,
    evaluate_deterministic_satisfaction,
    evaluate_preference_inference,
)


def test_confidence_interval_calculation() -> None:
    vals = [0.8, 0.85, 0.9, 0.95, 1.0]
    mean, low, high = compute_confidence_interval(vals, confidence=0.95)
    assert 0.89 <= mean <= 0.91
    assert low < mean < high


def test_evaluate_deterministic_satisfaction() -> None:
    # 1. Bullets and max words
    response = "- Step 1: Optimize memory.\n- Step 2: Use cache."
    criteria = {"max_word_count": 50, "required_structure": "bullets"}
    score, checks = evaluate_deterministic_satisfaction(response, criteria)
    assert score == 1.0
    assert checks["verbosity_satisfied"] is True
    assert checks["structure_satisfied"] is True

    # 2. Forbidden phrases
    response_with_salut = "Hello, here is the answer."
    crit_no_hello = {"forbidden_phrases": ["hello", "hi"]}
    score, checks = evaluate_deterministic_satisfaction(response_with_salut, crit_no_hello)
    assert score == 0.0
    assert checks["salutation_satisfied"] is False


def test_blinded_pairwise_judge() -> None:
    judge = BlindedPairwiseJudge(seed=42)
    resp_good = "- Point 1: Fast.\n- Point 2: Reliable."
    resp_bad = "Hello, here is a very long unstructured paragraph without any bullet points."
    criteria = {"forbidden_phrases": ["hello"], "required_structure": "bullets"}

    result = judge.compare(
        query="Test query",
        context="technical_engineering",
        criteria=criteria,
        response_a=resp_good,
        response_b=resp_bad,
        label_a="GoodModel",
        label_b="BadModel",
    )
    assert result["winner"] == "GoodModel"


def test_evaluate_preference_inference() -> None:
    gt_rules = [
        GroundTruthRule(
            rule_id="r1",
            target_aspect="verbosity",
            context_condition="technical_engineering",
            preference_statement="Keep responses concise and under 100 words.",
        ),
        GroundTruthRule(
            rule_id="r2",
            target_aspect="structure",
            context_condition="technical_engineering",
            preference_statement="Format content in bullet points.",
        ),
    ]

    inferred = [
        {"target_aspect": "verbosity", "context_condition": "technical_engineering", "preference": "Keep responses concise"},
        {"target_aspect": "structure", "context_condition": "technical_engineering", "preference": "Use bullet points"},
    ]

    res = evaluate_preference_inference(inferred, gt_rules, target_context="technical_engineering")
    assert res.precision > 0.8
    assert res.recall > 0.8
    assert res.f1 > 0.8
    assert res.context_accuracy == 1.0
