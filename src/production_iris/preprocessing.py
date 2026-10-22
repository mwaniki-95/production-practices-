"""Data preprocessing utilities."""

import pandas as pd

FEATURE_COLUMNS = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate the Iris dataset."""
    if df.empty:
        raise ValueError("Input DataFrame cannot be empty.")

    required_columns = FEATURE_COLUMNS + ["target"]
    missing = set(required_columns) - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    result = df.copy()

    # Remove duplicate observations.
    result = result.drop_duplicates().copy()

    # Fill missing feature values with their respective medians.
    result[FEATURE_COLUMNS] = result[FEATURE_COLUMNS].fillna(
        result[FEATURE_COLUMNS].median()
    )

    # The target must not be missing.
    if result["target"].isnull().any():
        raise ValueError("Target column contains missing values.")

    # Validate target labels.
    if not result["target"].isin([0, 1, 2]).all():
        raise ValueError("Target contains invalid class labels.")

    # Confirm that feature values are now complete.
    if result[FEATURE_COLUMNS].isnull().any().any():
        raise ValueError("Feature columns still contain missing values.")

    return result
