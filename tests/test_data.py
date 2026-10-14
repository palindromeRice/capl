"""Unit tests for synthetic dataset generator and serialization."""

import tempfile
from pathlib import Path

from src.data.synthetic import (
    AspectCategory,
    generate_synthetic_dataset,
    load_dataset_from_disk,
    save_dataset_to_disk,
)


def test_generate_synthetic_dataset() -> None:
    users = generate_synthetic_dataset(num_users=5, interactions_per_user=10, temporal_drift_ratio=0.4, seed=123)
    assert len(users) == 5

    user = users[0]
    assert user.user_id == "user_000"
    assert len(user.history) == 10
    assert len(user.test_samples) > 0
    assert len(user.all_rules) > 0

    # Test context diversity
    contexts = {h.context for h in user.history}
    assert len(contexts) >= 2, "Expected multiple contexts in user history"

    # Test test sample separation (no leakage)
    history_queries = {h.query for h in user.history}
    for s in user.test_samples:
        assert s.query not in history_queries, "Evaluation query leaked into history!"


def test_dataset_serialization() -> None:
    users = generate_synthetic_dataset(num_users=2, interactions_per_user=4, seed=42)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_file = Path(tmpdir) / "test_data.json"
        save_dataset_to_disk(users, str(tmp_file))
        assert tmp_file.exists()

        loaded_users = load_dataset_from_disk(str(tmp_file))
        assert len(loaded_users) == 2
        assert loaded_users[0].user_id == users[0].user_id
        assert len(loaded_users[0].history) == len(users[0].history)
        assert len(loaded_users[0].all_rules) == len(users[0].all_rules)
