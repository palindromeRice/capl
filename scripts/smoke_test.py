import logging
import sys
from pathlib import Path

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.synthetic import generate_synthetic_dataset
from src.evaluation.evaluate import BenchmarkEvaluator
from src.generation.generator import PersonalizationPipeline
from src.models.loader import ModelConfig, ModelManager
from src.utils.memory import detect_hardware, flush_memory

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_smoke_test() -> bool:
    logger.info("=== STEP 1: Detecting Hardware & Memory ===")
    hw = detect_hardware()
    logger.info("Detected Device: %s, Compute: %s, VRAM: %.1f MB", hw.device_name, hw.cuda_compute_capability, hw.total_memory_mb)
    flush_memory()

    logger.info("=== STEP 2: Generating Synthetic Benchmark Pilot ===")
    users = generate_synthetic_dataset(num_users=2, interactions_per_user=6, temporal_drift_ratio=0.5, seed=42)
    assert len(users) == 2, "Expected 2 users generated"
    assert len(users[0].history) == 6, "Expected 6 interactions in history"
    assert len(users[0].test_samples) > 0, "Expected held-out test samples"
    logger.info("Generated %d users successfully. Sample user has %d historical interactions.", len(users), len(users[0].history))

    logger.info("=== STEP 3: Initializing Model Manager (Mock Mode) ===")
    pref_cfg = ModelConfig(model_id="Qwen/Qwen3.5-2B", fallback_id="Qwen/Qwen2.5-3B-Instruct")
    main_cfg = ModelConfig(model_id="Qwen/Qwen3.5-9B", fallback_id="Qwen/Qwen2.5-7B-Instruct")
    mm = ModelManager(pref_config=pref_cfg, main_config=main_cfg, mock_mode=True)
    mm.load_preference_agent()
    mm.load_main_generator()

    logger.info("=== STEP 4: Initializing Personalization Pipeline ===")
    pipeline = PersonalizationPipeline(model_manager=mm, cache_dir=".cache/smoke_test")

    logger.info("=== STEP 5: Running Benchmark Evaluator on B0-B4 ===")
    evaluator = BenchmarkEvaluator(pipeline=pipeline, output_dir="results/smoke_test", seed=42)
    summary = evaluator.evaluate_users(users=users, baselines=["B0", "B1", "B2", "B3", "B4"], max_samples_per_user=2)

    logger.info("=== STEP 6: Validating Results ===")
    assert "baselines" in summary, "Missing baselines in summary"
    for b in ["B0", "B1", "B2", "B3", "B4"]:
        assert b in summary["baselines"], f"Baseline {b} not found in summary"
        sat_mean = summary["baselines"][b]["satisfaction_mean"]
        logger.info("Baseline %s -> Satisfaction Rate: %.2f (CI: %s)", b, sat_mean, summary["baselines"][b]["satisfaction_ci95"])

    b4_sat = summary["baselines"]["B4"]["satisfaction_mean"]
    b0_sat = summary["baselines"]["B0"]["satisfaction_mean"]
    assert b4_sat >= b0_sat, f"Proposed B4 satisfaction ({b4_sat}) should be >= B0 ({b0_sat})"

    logger.info("=== SMOKE TEST PASSED SUCCESSFULLY ===")
    return True


if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)
