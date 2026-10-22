"""Tests for model training and prediction."""

import pytest
from sklearn.pipeline import Pipeline

from production_iris.data import load_iris_data
from production_iris.model import predict, train_model


def test_train_model_returns_pipeline_and_metrics() -> None:
    """Check that training returns a fitted pipeline and metrics."""
    df = load_iris_data()

    model, metrics = train_model(df)

    assert isinstance(model, Pipeline)
    assert "accuracy" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_predict_returns_one_label_per_observation() -> None:
    """Check predictions for new observations."""
    df = load_iris_data()
    model, _ = train_model(df)

    features = df[
        [
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)",
        ]
    ].iloc[:3]

    predictions = predict(model, features)

    assert len(predictions) == 3
    assert all(label in (0, 1, 2) for label in predictions)


def test_train_model_rejects_empty_data() -> None:
    """Check that training rejects an empty DataFrame."""
    df = load_iris_data().iloc[0:0]

    with pytest.raises(ValueError, match="cannot be empty"):
        train_model(df)
