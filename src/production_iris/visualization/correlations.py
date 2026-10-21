"""Visualization functions for feature correlations."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_correlation_matrix(
    df: pd.DataFrame,
) -> plt.Figure:
    """Plot a correlation heatmap for numerical features."""

    features = df.select_dtypes(include="number")

    if features.shape[1] < 2:
        raise ValueError("At least two numerical columns are required.")

    correlation_matrix = features.corr()

    figure, axis = plt.subplots(figsize=(9, 7))

    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        ax=axis,
    )

    axis.set_title("Feature Correlation Matrix")
    figure.tight_layout()

    return figure
