"""Tests for project configuration."""

from production_iris.config import ModelConfig


def test_default_model_config() -> None:
    """Check the default configuration."""
    config = ModelConfig()

    assert config.target_column == "target"
    assert config.test_size == 0.2
    assert config.random_state == 42
    assert config.n_estimators == 100
