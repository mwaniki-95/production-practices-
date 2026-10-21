"""Visualization functions for categorical features."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_categorical_features(
    df: pd.DataFrame,
) -> plt.Figure:
    """Plot category counts for string and categorical columns."""

    categorical_columns = df.select_dtypes(
        include=["object", "string", "category"]
    ).columns

    if len(categorical_columns) == 0:
        raise ValueError("DataFrame contains no categorical columns.")

    figure, axes = plt.subplots(
        nrows=len(categorical_columns),
        ncols=1,
        figsize=(8, 4 * len(categorical_columns)),
        squeeze=False,
    )

    for index, column in enumerate(categorical_columns):
        sns.countplot(
            data=df,
            x=column,
            order=df[column].value_counts().index,
            ax=axes[index, 0],
        )
        axes[index, 0].set_title(f"Distribution of {column}")
        axes[index, 0].tick_params(axis="x", rotation=30)

    figure.tight_layout()
    return figure
