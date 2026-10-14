"""Comprehensive evaluation metrics for preference inference, response quality, and statistical significance."""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.data.synthetic import GroundTruthRule
from src.preferences.extractor import InferredPreferenceRule


@dataclass
class InferenceMetricResult:
    precision: float
    recall: float
    f1: float
    context_accuracy: float
    conflict_resolution_rate: float
    temporal_drift_accuracy: float


def compute_confidence_interval(values: List[float], confidence: float = 0.95) -> Tuple[float, float, float]:
    """Compute mean and 95% confidence interval bounds."""
    if not values:
        return 0.0, 0.0, 0.0
    arr = np.array(values, dtype=float)
    mean = float(np.mean(arr))
    n = len(arr)
    if n < 2:
        return mean, mean, mean
    std_err = float(np.std(arr, ddof=1) / math.sqrt(n))
    z = 1.96 if confidence == 0.95 else 2.576
    margin = z * std_err
    return round(mean, 4), round(max(0.0, mean - margin), 4), round(min(1.0, mean + margin), 4)


def evaluate_preference_inference(
    inferred_rules: List[Dict[str, Any]],
    ground_truth_rules: List[GroundTruthRule],
    target_context: str,
    has_temporal_drift: bool = False,
) -> InferenceMetricResult:
    """Evaluate accuracy, precision, recall, and contextual fidelity of inferred rules against ground truth."""
    active_gt = [
        r for r in ground_truth_rules
        if r.context_condition == target_context and r.deprecated_at_step is None
    ]
    deprecated_gt = [
        r for r in ground_truth_rules
        if r.context_condition == target_context and r.deprecated_at_step is not None
    ]

    if not inferred_rules:
        return InferenceMetricResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    correct_matches = 0
    context_matches = 0
    drift_correct = 0

    for inf in inferred_rules:
        aspect = inf.get("target_aspect", "")
        inf_ctx = inf.get("context_condition", "general")
        inf_pref = inf.get("preference", "").lower()

        # Check contextual scoping
        if inf_ctx == target_context or inf_ctx == "general":
            context_matches += 1

        # Check alignment with active ground truth
        for gt in active_gt:
            if gt.target_aspect == aspect:
                # Semantic overlap heuristic
                gt_words = set(re.findall(r"\w+", gt.preference_statement.lower()))
                inf_words = set(re.findall(r"\w+", inf_pref))
                overlap = len(gt_words.intersection(inf_words))
                if overlap >= 2 or (gt.target_aspect in inf_pref):
                    correct_matches += 1
                    break

        # Check temporal drift: did it adopt active rule over deprecated rule?
        if has_temporal_drift and deprecated_gt:
            for gt_dep in deprecated_gt:
                if gt_dep.target_aspect == aspect:
                    dep_words = set(re.findall(r"\w+", gt_dep.preference_statement.lower()))
                    inf_words = set(re.findall(r"\w+", inf_pref))
                    if len(dep_words.intersection(inf_words)) < 2:
                        drift_correct += 1

    precision = correct_matches / max(len(inferred_rules), 1)
    recall = correct_matches / max(len(active_gt), 1)
    f1 = (2 * precision * recall) / max(precision + recall, 1e-6)
    context_acc = context_matches / max(len(inferred_rules), 1)
    
    # Conflict resolution: Did the model avoid applying a conflicting other-context rule?
    conflict_res_rate = 1.0 if any(inf.get("conflict_detected", False) for inf in inferred_rules) or inf_ctx == target_context else 0.0
    drift_acc = (drift_correct / max(len(deprecated_gt), 1)) if has_temporal_drift and deprecated_gt else 1.0

    return InferenceMetricResult(
        precision=round(min(1.0, precision), 4),
        recall=round(min(1.0, recall), 4),
        f1=round(min(1.0, f1), 4),
        context_accuracy=round(min(1.0, context_acc), 4),
        conflict_resolution_rate=round(conflict_res_rate, 4),
        temporal_drift_accuracy=round(min(1.0, drift_acc), 4),
    )


def evaluate_deterministic_satisfaction(response_text: str, criteria: Dict[str, Any]) -> Tuple[float, Dict[str, bool]]:
    """Deterministically check whether generated response satisfies target style rules."""
    checks: Dict[str, bool] = {}
    words = response_text.split()
    word_count = len(words)
    text_low = response_text.lower()

    # 1. Verbosity check
    if "max_word_count" in criteria:
        checks["verbosity_satisfied"] = word_count <= criteria["max_word_count"]
    elif "min_word_count" in criteria:
        checks["verbosity_satisfied"] = word_count >= criteria["min_word_count"]

    # 2. Structure check
    if "required_structure" in criteria:
        req = criteria["required_structure"]
        if req == "bullets":
            checks["structure_satisfied"] = bool(re.search(r"^\s*[-*•]\s+", response_text, re.MULTILINE))
        elif req == "numbered":
            checks["structure_satisfied"] = bool(re.search(r"^\s*\d+\.\s+", response_text, re.MULTILINE))

    # 3. Salutation check
    if criteria.get("forbidden_phrases"):
        has_forbidden = any(phrase in text_low for phrase in criteria["forbidden_phrases"])
        checks["salutation_satisfied"] = not has_forbidden
    elif "salutation_present" in criteria.get("required_elements", []):
        checks["salutation_satisfied"] = any(w in text_low for w in ["greetings", "hello", "sincerely", "cheers", "best regards"])

    # 4. Code check
    if criteria.get("requires_code"):
        checks["code_satisfied"] = "```" in response_text or "def " in response_text
    elif criteria.get("forbid_code"):
        checks["code_satisfied"] = "```" not in response_text and "def " not in response_text

    if not checks:
        return 1.0, {}

    satisfied_count = sum(1 for v in checks.values() if v)
    score = satisfied_count / len(checks)
    return round(score, 4), checks


class BlindedPairwiseJudge:
    """Implements randomized, blinded pairwise response evaluation."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)

    def compare(
        self,
        query: str,
        context: str,
        criteria: Dict[str, Any],
        response_a: str,
        response_b: str,
        label_a: str = "Model_A",
        label_b: str = "Model_B",
    ) -> Dict[str, Any]:
        """Perform a blinded comparison between two candidate responses."""
        # Randomize presentation order to eliminate position bias
        flipped = self.rng.random() < 0.5
        cand_1, cand_2 = (response_b, response_a) if flipped else (response_a, response_b)
        label_1, label_2 = (label_b, label_a) if flipped else (label_a, label_b)

        score_1, _ = evaluate_deterministic_satisfaction(cand_1, criteria)
        score_2, _ = evaluate_deterministic_satisfaction(cand_2, criteria)

        if score_1 > score_2:
            winner = label_1
        elif score_2 > score_1:
            winner = label_2
        else:
            winner = "tie"

        return {
            "winner": winner,
            "flipped": flipped,
            "score_a": score_2 if flipped else score_1,
            "score_b": score_1 if flipped else score_2,
            "margin": abs(score_1 - score_2),
        }
