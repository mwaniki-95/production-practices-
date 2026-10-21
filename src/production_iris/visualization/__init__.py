"""Visualization functions for Iris dataset."""

from production_iris.visualization.comparisons import plot_categorical_features
from production_iris.visualization.correlations import plot_correlation_matrix
from production_iris.visualization.distributions import (
    plot_feature_distributions,
)

__all__ = [
    "plot_categorical_features",
    "plot_correlation_matrix",
    "plot_feature_distributions",
]
