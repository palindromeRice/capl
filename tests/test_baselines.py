"""Unit tests for baseline formatters and retrieval."""

from src.baselines.retrieval import RetrievalBaseline
from src.baselines.static import StaticBaseline
from src.data.synthetic import generate_synthetic_dataset


def test_static_baseline_formatting() -> None:
    users = generate_synthetic_dataset(num_users=1, interactions_per_user=4, seed=42)
    user = users[0]
    b1 = StaticBaseline()
    profile = b1.extract_profile(user.user_id, user.history)
    prompt = b1.format_prompt("Test query", profile)

    assert "<|system|>" in prompt
    assert "<|user|>\nTest query" in prompt
    assert "<|assistant|>" in prompt


def test_retrieval_baseline() -> None:
    users = generate_synthetic_dataset(num_users=1, interactions_per_user=10, seed=42)
    user = users[0]
    b2 = RetrievalBaseline(top_k=3, max_tokens=256)

    retrieved = b2.retrieve("database query locking", user.history)
    assert len(retrieved) <= 3
    assert len(retrieved) > 0

    prompt = b2.format_prompt("database query locking", retrieved)
    assert "Example Past Interaction" in prompt
    assert "<|user|>\ndatabase query locking" in prompt
