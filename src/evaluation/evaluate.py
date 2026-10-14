"""Evaluation engine orchestrating multi-baseline benchmark runs, metrics aggregation, and reporting."""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from src.data.synthetic import UserHistory, load_dataset_from_disk
from src.evaluation.metrics import (
    BlindedPairwiseJudge,
    compute_confidence_interval,
    evaluate_deterministic_satisfaction,
    evaluate_preference_inference,
)
from src.generation.generator import GenerationResult, PersonalizationPipeline

logger = logging.getLogger(__name__)


class BenchmarkEvaluator:
    """Evaluates personalization pipelines across all test samples and baselines."""

    def __init__(
        self,
        pipeline: PersonalizationPipeline,
        output_dir: str = "results",
        seed: int = 42,
    ) -> None:
        self.pipeline = pipeline
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.judge = BlindedPairwiseJudge(seed=seed)

    def evaluate_users(
        self,
        users: List[UserHistory],
        baselines: List[str] = ["B0", "B1", "B2", "B3", "B4"],
        max_samples_per_user: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run full evaluation suite across users and baselines."""
        all_generation_results: List[GenerationResult] = []
        raw_eval_records: List[Dict[str, Any]] = []

        baseline_satisfaction: Dict[str, List[float]] = {b: [] for b in baselines}
        baseline_latency: Dict[str, List[float]] = {b: [] for b in baselines}
        baseline_tokens: Dict[str, List[float]] = {b: [] for b in baselines}
        baseline_vram: Dict[str, List[float]] = {b: [] for b in baselines}

        inference_metrics: Dict[str, Dict[str, List[float]]] = {
            b: {"precision": [], "recall": [], "f1": [], "context_acc": [], "conflict_rate": [], "drift_acc": []}
            for b in ["B1", "B3", "B4"]
        }

        pairwise_wins: Dict[str, int] = {f"B4_vs_{b}": 0 for b in baselines if b != "B4"}
        pairwise_losses: Dict[str, int] = {f"B4_vs_{b}": 0 for b in baselines if b != "B4"}
        pairwise_ties: Dict[str, int] = {f"B4_vs_{b}": 0 for b in baselines if b != "B4"}

        total_eval_samples = 0

        for user in users:
            samples = user.test_samples[:max_samples_per_user] if max_samples_per_user else user.test_samples
            has_temporal_drift = any(r.deprecated_at_step is not None for r in user.all_rules)

            for sample in samples:
                total_eval_samples += 1
                sample_results: Dict[str, GenerationResult] = {}

                for b in baselines:
                    gen_res = self.pipeline.run_baseline(
                        baseline_id=b,
                        user_id=user.user_id,
                        query=sample.query,
                        context=sample.context,
                        history=user.history,
                    )
                    sample_results[b] = gen_res
                    all_generation_results.append(gen_res)

                    # Deterministic satisfaction check
                    sat_score, checks = evaluate_deterministic_satisfaction(gen_res.response_text, sample.deterministic_criteria)
                    baseline_satisfaction[b].append(sat_score)
                    baseline_latency[b].append(gen_res.extraction_latency_sec + gen_res.generation_latency_sec)
                    baseline_tokens[b].append(gen_res.prompt_tokens + gen_res.generated_tokens)
                    baseline_vram[b].append(gen_res.peak_vram_mb)

                    # Preference inference evaluation for rule-based methods
                    if b in inference_metrics:
                        inf_res = evaluate_preference_inference(
                            inferred_rules=gen_res.inferred_rules,
                            ground_truth_rules=user.all_rules,
                            target_context=sample.context,
                            has_temporal_drift=has_temporal_drift,
                        )
                        inference_metrics[b]["precision"].append(inf_res.precision)
                        inference_metrics[b]["recall"].append(inf_res.recall)
                        inference_metrics[b]["f1"].append(inf_res.f1)
                        inference_metrics[b]["context_acc"].append(inf_res.context_accuracy)
                        inference_metrics[b]["conflict_rate"].append(inf_res.conflict_resolution_rate)
                        inference_metrics[b]["drift_acc"].append(inf_res.temporal_drift_accuracy)

                    raw_eval_records.append({
                        "user_id": user.user_id,
                        "sample_id": sample.sample_id,
                        "context": sample.context,
                        "baseline_id": b,
                        "satisfaction_score": sat_score,
                        "prompt_tokens": gen_res.prompt_tokens,
                        "generated_tokens": gen_res.generated_tokens,
                        "latency_sec": gen_res.extraction_latency_sec + gen_res.generation_latency_sec,
                        "peak_vram_mb": gen_res.peak_vram_mb,
                        "checks": checks,
                    })

                # Blinded pairwise evaluation: B4 vs each other baseline
                if "B4" in sample_results:
                    resp_b4 = sample_results["B4"].response_text
                    for b in baselines:
                        if b == "B4":
                            continue
                        cmp_res = self.judge.compare(
                            query=sample.query,
                            context=sample.context,
                            criteria=sample.deterministic_criteria,
                            response_a=resp_b4,
                            response_b=sample_results[b].response_text,
                            label_a="B4",
                            label_b=b,
                        )
                        pair_key = f"B4_vs_{b}"
                        if cmp_res["winner"] == "B4":
                            pairwise_wins[pair_key] += 1
                        elif cmp_res["winner"] == b:
                            pairwise_losses[pair_key] += 1
                        else:
                            pairwise_ties[pair_key] += 1

        # Aggregate summary statistics with 95% Confidence Intervals
        summary: Dict[str, Any] = {
            "num_users": len(users),
            "total_samples": total_eval_samples,
            "baselines": {},
            "inference_metrics": {},
            "pairwise_win_rates": {},
        }

        for b in baselines:
            m_sat, low_sat, high_sat = compute_confidence_interval(baseline_satisfaction[b])
            m_lat, low_lat, high_lat = compute_confidence_interval(baseline_latency[b])
            m_tok, low_tok, high_tok = compute_confidence_interval(baseline_tokens[b])
            m_vram, _, _ = compute_confidence_interval(baseline_vram[b])

            summary["baselines"][b] = {
                "satisfaction_mean": m_sat,
                "satisfaction_ci95": [low_sat, high_sat],
                "latency_sec_mean": m_lat,
                "latency_ci95": [low_lat, high_lat],
                "total_tokens_mean": m_tok,
                "tokens_ci95": [low_tok, high_tok],
                "peak_vram_mb": m_vram,
            }

        for b in inference_metrics:
            summary["inference_metrics"][b] = {
                metric: compute_confidence_interval(vals)[0]
                for metric, vals in inference_metrics[b].items()
            }

        for pair_key in pairwise_wins:
            total_matches = pairwise_wins[pair_key] + pairwise_losses[pair_key] + pairwise_ties[pair_key]
            win_rate = (pairwise_wins[pair_key] / total_matches) if total_matches > 0 else 0.0
            summary["pairwise_win_rates"][pair_key] = {
                "wins": pairwise_wins[pair_key],
                "losses": pairwise_losses[pair_key],
                "ties": pairwise_ties[pair_key],
                "win_rate": round(win_rate, 4),
            }

        # Save to disk
        self._export_results(summary, raw_eval_records)
        self._generate_plots(summary)

        return summary

    def _export_results(self, summary: Dict[str, Any], raw_records: List[Dict[str, Any]]) -> None:
        """Export summary JSON and raw CSV records."""
        with open(self.output_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        if raw_records:
            keys = ["user_id", "sample_id", "context", "baseline_id", "satisfaction_score", "prompt_tokens", "generated_tokens", "latency_sec", "peak_vram_mb"]
            with open(self.output_dir / "raw_eval_records.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(raw_records)

    def _generate_plots(self, summary: Dict[str, Any]) -> None:
        """Render comparison plots."""
        try:
            baselines = list(summary["baselines"].keys())
            satisfaction = [summary["baselines"][b]["satisfaction_mean"] for b in baselines]
            errors = [
                summary["baselines"][b]["satisfaction_mean"] - summary["baselines"][b]["satisfaction_ci95"][0]
                for b in baselines
            ]

            fig, ax = plt.subplots(figsize=(8, 5))
            colors = ["#7f7f7f", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
            ax.bar(baselines, satisfaction, yerr=errors, capsize=5, color=colors[:len(baselines)], alpha=0.85)
            ax.set_ylabel("Preference Rule Satisfaction Rate (0 - 1)")
            ax.set_title("Benchmark Comparison Across Baselines (with 95% CI)")
            ax.set_ylim(0.0, 1.1)
            ax.grid(axis="y", linestyle="--", alpha=0.5)

            plot_path = self.output_dir / "baseline_satisfaction_comparison.png"
            plt.tight_layout()
            plt.savefig(plot_path, dpi=200)
            plt.close()
            logger.info("Saved benchmark plot to %s", plot_path)
        except Exception as e:
            logger.warning("Could not generate plots: %s", e)
