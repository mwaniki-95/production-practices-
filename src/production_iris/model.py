"""Model training, evaluation and prediction."""

import logging

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from production_iris.config import ModelConfig
from production_iris.preprocessing import (
    FEATURE_COLUMNS,
    preprocess_data,
)

logger = logging.getLogger(__name__)


def train_model(
    df: pd.DataFrame,
    config: ModelConfig | None = None,
) -> tuple[Pipeline, dict[str, float]]:
    """Train a classifier and return the fitted pipeline and metrics."""
    if config is None:
        config = ModelConfig()

    df = preprocess_data(df)

    x = df[FEATURE_COLUMNS]
    y = df[config.target_column]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=config.n_estimators,
                    random_state=config.random_state,
                ),
            ),
        ]
    )

    logger.info("Starting model training.")
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)

    metrics = {"accuracy": float(accuracy)}

    logger.info("Model training completed. Accuracy: %.4f", accuracy)

    return model, metrics


def predict(
    model: Pipeline,
    observations: pd.DataFrame,
) -> list[int]:
    """Predict Iris class labels for new observations."""
    missing = set(FEATURE_COLUMNS) - set(observations.columns)

    if missing:
        raise ValueError(f"Missing required features: {sorted(missing)}")

    if observations.empty:
        raise ValueError("Input observations cannot be empty.")

    if observations[FEATURE_COLUMNS].isnull().any().any():
        raise ValueError("Features cannot contain missing values.")

    predictions = model.predict(observations[FEATURE_COLUMNS])
    return [int(label) for label in predictions]
