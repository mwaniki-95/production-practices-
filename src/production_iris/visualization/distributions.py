"""Visualization functions for feature distributions."""

import matplotlib.pyplot as plt
import pandas as pd


def plot_feature_distributions(
    df: pd.DataFrame,
) -> plt.Figure:
    """Plot histograms for all numerical features."""

    features = df.select_dtypes(include="number")

    if features.empty:
        raise ValueError("DataFrame contains no numerical features.")

    axes = features.hist(figsize=(10, 8), bins=15)
    figure = axes[0, 0].get_figure()
    figure.suptitle("Feature Distributions")
    figure.tight_layout()

    return figure
