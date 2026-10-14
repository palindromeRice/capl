"""Run primary experimental baselines B0 to B4 and export research results."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run experimental baselines B0 to B4.")
    parser.add_argument("--users", type=int, default=50, help="Number of simulated users (pilot: 50-100)")
    parser.add_argument("--interactions", type=int, default=20, help="Interactions per user")
    parser.add_argument("--mock", action="store_true", default=True, help="Use mock generator for fast reproducible verification")
    parser.add_argument("--live", dest="mock", action="store_false", help="Load real quantized models on GPU")
    parser.add_argument("--output-dir", type=str, default="results/baselines")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--drift-ratio", type=float, default=0.25)

    args = parser.parse_args()
    console = Console()

    console.print(f"[bold cyan]Running Primary Baselines Experiment[/bold cyan] (Users: {args.users}, Interactions: {args.interactions}, Mock: {args.mock})")

    # 1. Generate controlled benchmark dataset
    users = generate_synthetic_dataset(
        num_users=args.users,
        interactions_per_user=args.interactions,
        temporal_drift_ratio=args.drift_ratio,
        seed=args.seed,
    )

    # 2. Configure models
    pref_cfg = ModelConfig(model_id="Qwen/Qwen3.5-2B", fallback_id="Qwen/Qwen2.5-3B-Instruct")
    main_cfg = ModelConfig(model_id="Qwen/Qwen3.5-9B", fallback_id="Qwen/Qwen2.5-7B-Instruct")
    mm = ModelManager(pref_config=pref_cfg, main_config=main_cfg, mock_mode=args.mock)

    # 3. Initialize pipeline & evaluator
    pipeline = PersonalizationPipeline(model_manager=mm, cache_dir=".cache/preferences")
    evaluator = BenchmarkEvaluator(pipeline=pipeline, output_dir=args.output_dir, seed=args.seed)

    # 4. Run evaluation across B0 - B4
    summary = evaluator.evaluate_users(
        users=users,
        baselines=["B0", "B1", "B2", "B3", "B4"],
    )

    # 5. Render results table
    table = Table(title="Primary Baselines Evaluation Results (with 95% Confidence Intervals)")
    table.add_column("Baseline ID", style="cyan", justify="center")
    table.add_column("Description", style="white")
    table.add_column("Rule Satisfaction", justify="right")
    table.add_column("95% CI", justify="center")
    table.add_column("Avg Latency (s)", justify="right")
    table.add_column("Total Tokens", justify="right")
    table.add_column("Peak VRAM (MB)", justify="right")

    baseline_names = {
        "B0": "Zero-Shot (No Personalization)",
        "B1": "Static User Profile",
        "B2": "Retrieval Personalization (RAG)",
        "B3": "Dynamic Task-Only Extraction",
        "B4": "Context-Aware Preference Learning (CAPL)",
    }

    for b, meta in summary["baselines"].items():
        table.add_row(
            b,
            baseline_names.get(b, b),
            f"{meta['satisfaction_mean']:.3f}",
            f"[{meta['satisfaction_ci95'][0]:.3f}, {meta['satisfaction_ci95'][1]:.3f}]",
            f"{meta['latency_sec_mean']:.3f}",
            f"{meta['total_tokens_mean']:.0f}",
            f"{meta['peak_vram_mb']:.1f}",
        )

    console.print(table)

    # Pairwise Table
    pair_table = Table(title="Blinded Pairwise Head-to-Head Win Rates (B4 vs Baselines)")
    pair_table.add_column("Matchup", style="magenta")
    pair_table.add_column("B4 Wins", justify="right")
    pair_table.add_column("Opponent Wins", justify="right")
    pair_table.add_column("Ties", justify="right")
    pair_table.add_column("B4 Win Rate", justify="right", style="bold green")

    for match, data in summary["pairwise_win_rates"].items():
        pair_table.add_row(
            match,
            str(data["wins"]),
            str(data["losses"]),
            str(data["ties"]),
            f"{data['win_rate'] * 100:.1f}%",
        )

    console.print(pair_table)
    console.print(f"[bold green]Results successfully exported to {args.output_dir}/summary.json and raw_eval_records.csv[/bold green]")


if __name__ == "__main__":
    main()
