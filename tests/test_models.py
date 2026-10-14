"""Unit tests for hardware detection and mock model manager."""

from src.models.loader import ModelConfig, ModelManager
from src.utils.memory import detect_hardware, flush_memory, track_execution_resources


def test_hardware_detection() -> None:
    hw = detect_hardware()
    assert hw.device_type in ["cuda", "mps", "cpu"]
    assert isinstance(hw.total_memory_mb, float)
    flush_memory()


def test_mock_model_generation() -> None:
    pref_cfg = ModelConfig(model_id="Qwen/Qwen3.5-2B")
    main_cfg = ModelConfig(model_id="Qwen/Qwen3.5-9B")
    mm = ModelManager(pref_config=pref_cfg, main_config=main_cfg, mock_mode=True)
    mm.load_preference_agent()
    mm.load_main_generator()

    with track_execution_resources() as stats:
        resp = mm.generate_response("Test prompt with bullet points and under 100 words")

    assert len(resp) > 0
    assert stats["elapsed_sec"] >= 0.0
