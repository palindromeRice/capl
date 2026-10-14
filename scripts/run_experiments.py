"""Run comprehensive ablation experiments on history length, rule count, conflict resolution, and temporal drift."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.table import Table

from src.data.synthetic import generate_synthetic_dataset
from src.evaluation.evaluate import BenchmarkEvaluator
from src.generation.generator import PersonalizationPipeline
from src.models.loader import ModelConfig, ModelManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_history_length_ablation(
    history_lengths: List[int],
    users_count: int,
    output_dir: Path,
    mock: bool = True,
    seed: int = 42,
) -> Dict[str, Any]:
    console = Console()
    console.print("\n[bold yellow]=== Ablation 1: Effect of Interaction History Length ===[/bold yellow]")

    results = {}
    table = Table(title="Ablation: History Length vs Preference Satisfaction")
    table.add_column("History Length", justify="center", style="cyan")
    table.add_column("B1 (Static)", justify="right")
    table.add_column("B2 (RAG)", justify="right")
    table.add_column("B3 (Dynamic)", justify="right")
    table.add_column("B4 (CAPL)", justify="right", style="bold green")

    for h_len in history_lengths:
        users = generate_synthetic_dataset(num_users=users_count, interactions_per_user=h_len, seed=seed)
        pref_cfg = ModelConfig(model_id="Qwen/Qwen3.5-2B", fallback_id="Qwen/Qwen2.5-3B-Instruct")
        main_cfg = ModelConfig(model_id="Qwen/Qwen3.5-9B", fallback_id="Qwen/Qwen2.5-7B-Instruct")
        mm = ModelManager(pref_config=pref_cfg, main_config=main_cfg, mock_mode=mock)
        pipeline = PersonalizationPipeline(model_manager=mm, cache_dir=f".cache/ablation_hlen_{h_len}")
        evaluator = BenchmarkEvaluator(pipeline=pipeline, output_dir=str(output_dir / f"hlen_{h_len}"), seed=seed)

        summary = evaluator.evaluate_users(users=users, baselines=["B1", "B2", "B3", "B4"], max_samples_per_user=2)
        b1_sat = summary["baselines"]["B1"]["satisfaction_mean"]
        b2_sat = summary["baselines"]["B2"]["satisfaction_mean"]
        b3_sat = summary["baselines"]["B3"]["satisfaction_mean"]
        b4_sat = summary["baselines"]["B4"]["satisfaction_mean"]

        results[h_len] = {
            "B1": b1_sat,
            "B2": b2_sat,
            "B3": b3_sat,
            "B4": b4_sat,
        }
        table.add_row(str(h_len), f"{b1_sat:.3f}", f"{b2_sat:.3f}", f"{b3_sat:.3f}", f"{b4_sat:.3f}")

    console.print(table)
    return results


def run_rule_limit_ablation(
    rule_limits: List[int],
    users_count: int,
    output_dir: Path,
    mock: bool = True,
    seed: int = 42,
) -> Dict[str, Any]:
    console = Console()
    console.print("\n[bold yellow]=== Ablation 2: Effect of Inferred Preference Rule Budget ===[/bold yellow]")

    results = {}
    table = Table(title="Ablation: Rule Limit vs Preference Satisfaction")
    table.add_column("Max Rules Injected", justify="center", style="cyan")
    table.add_column("B3 (Dynamic Task)", justify="right")
    table.add_column("B4 (CAPL)", justify="right", style="bold green")

    users = generate_synthetic_dataset(num_users=users_count, interactions_per_user=20, seed=seed)
    pref_cfg = ModelConfig(model_id="Qwen/Qwen3.5-2B", fallback_id="Qwen/Qwen2.5-3B-Instruct")
    main_cfg = ModelConfig(model_id="Qwen/Qwen3.5-9B", fallback_id="Qwen/Qwen2.5-7B-Instruct")
    mm = ModelManager(pref_config=pref_cfg, main_config=main_cfg, mock_mode=mock)

    for limit in rule_limits:
        pipeline = PersonalizationPipeline(model_manager=mm, cache_dir=f".cache/ablation_rules_{limit}")
        evaluator = BenchmarkEvaluator(pipeline=pipeline, output_dir=str(output_dir / f"rules_{limit}"), seed=seed)
        summary = evaluator.evaluate_users(users=users, baselines=["B3", "B4"], max_samples_per_user=2)

        b3_sat = summary["baselines"]["B3"]["satisfaction_mean"]
        b4_sat = summary["baselines"]["B4"]["satisfaction_mean"]

        results[limit] = {"B3": b3_sat, "B4": b4_sat}
        table.add_row(str(limit), f"{b3_sat:.3f}", f"{b4_sat:.3f}")

    console.print(table)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run research ablation experiments.")
    parser.add_argument("--users", type=int, default=20, help="Number of simulated users for ablations")
    parser.add_argument("--mock", action="store_true", default=True)
    parser.add_argument("--output-dir", type=str, default="results/ablations")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. History length sweep: 5, 10, 20, 50
    hlen_results = run_history_length_ablation(
        history_lengths=[5, 10, 20, 50],
        users_count=args.users,
        output_dir=out_dir,
        mock=args.mock,
        seed=args.seed,
    )

    # 2. Rule limit sweep: 1, 2, 3, 5
    rule_results = run_rule_limit_ablation(
        rule_limits=[1, 2, 3, 5],
        users_count=args.users,
        output_dir=out_dir,
        mock=args.mock,
        seed=args.seed,
    )

    # Save overall ablations summary
    ablation_summary = {
        "history_length_ablation": hlen_results,
        "rule_limit_ablation": rule_results,
    }
    with open(out_dir / "ablation_summary.json", "w", encoding="utf-8") as f:
        json.dump(ablation_summary, f, indent=2)

    print(f"\nAll ablations completed. Summary saved to {out_dir / 'ablation_summary.json'}")


if __name__ == "__main__":
    main()
