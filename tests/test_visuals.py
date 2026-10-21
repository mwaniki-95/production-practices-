"""Tests for Iris visualization functions."""

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from production_iris.data import load_iris_data
from production_iris.visualization.comparisons import plot_categorical_features
from production_iris.visualization.correlations import plot_correlation_matrix
from production_iris.visualization.distributions import (
    plot_feature_distributions,
)


@pytest.fixture
def iris_df() -> pd.DataFrame:
    """Load the actual Iris dataset for visualization tests."""
    return load_iris_data()


def test_plot_feature_distributions(iris_df: pd.DataFrame) -> None:
    """Test that feature distribution plots return a Figure."""
    figure = plot_feature_distributions(iris_df)

    assert isinstance(figure, plt.Figure)

    plt.close(figure)


def test_plot_correlation_matrix(iris_df: pd.DataFrame) -> None:
    """Test that the correlation matrix returns a Figure."""
    figure = plot_correlation_matrix(iris_df)

    assert isinstance(figure, plt.Figure)

    plt.close(figure)


def test_plot_categorical_features(iris_df: pd.DataFrame) -> None:
    """Test that categorical feature plots return a Figure."""
    figure = plot_categorical_features(iris_df)

    assert isinstance(figure, plt.Figure)

    plt.close(figure)
