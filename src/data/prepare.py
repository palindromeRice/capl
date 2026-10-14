"""Data preparation pipeline supporting synthetic benchmark generation and public dataset adapters."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

from src.data.synthetic import UserHistory, generate_synthetic_dataset, save_dataset_to_disk

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def prepare_synthetic_benchmark(
    num_users: int = 50,
    interactions_per_user: int = 20,
    temporal_drift_ratio: float = 0.25,
    seed: int = 42,
    output_path: str = "data/synthetic_pilot.json",
) -> List[UserHistory]:
    """Generate and save the controlled synthetic benchmark."""
    logger.info("Generating synthetic dataset with %d users, %d interactions/user, seed=%d...", num_users, interactions_per_user, seed)
    dataset = generate_synthetic_dataset(
        num_users=num_users,
        interactions_per_user=interactions_per_user,
        temporal_drift_ratio=temporal_drift_ratio,
        seed=seed,
    )
    save_dataset_to_disk(dataset, output_path)
    logger.info("Saved synthetic benchmark to %s (Total users: %d)", output_path, len(dataset))
    return dataset


def prepare_public_lamp_adapter(output_path: str = "data/lamp_subset.json") -> bool:
    """Attempt to load a public personalization dataset (LaMP) if accessible."""
    try:
        from datasets import load_dataset
        logger.info("Attempting to fetch public LaMP dataset from Hugging Face...")
        ds = load_dataset("haotiansun014/LaMP_raw", "LaMP_2", split="train[:100]")
        logger.info("Successfully fetched %d records from LaMP_2.", len(ds))
        # Note: In real setup, format into UserHistory structure
        return True
    except Exception as e:
        logger.warning("Could not fetch remote LaMP dataset (%s). Using synthetic benchmark for controlled evaluation.", e)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare datasets for preference learning experiments.")
    parser.add_argument("--dataset", choices=["synthetic", "lamp"], default="synthetic")
    parser.add_argument("--users", type=int, default=50)
    parser.add_argument("--interactions", type=int, default=20)
    parser.add_argument("--drift-ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="data/synthetic_pilot.json")

    args = parser.parse_args()

    if args.dataset == "synthetic":
        prepare_synthetic_benchmark(
            num_users=args.users,
            interactions_per_user=args.interactions,
            temporal_drift_ratio=args.drift_ratio,
            seed=args.seed,
            output_path=args.output,
        )
    elif args.dataset == "lamp":
        success = prepare_public_lamp_adapter(args.output)
        if not success:
            logger.info("Falling back to synthetic dataset generation.")
            prepare_synthetic_benchmark(output_path=args.output)


if __name__ == "__main__":
    main()
