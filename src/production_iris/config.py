"""Project configuration."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for Iris model training."""

    target_column: str = "target"
    test_size: float = 0.2
    random_state: int = 42
    n_estimators: int = 100
