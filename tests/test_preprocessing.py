"""Tests for preprocessing functions."""

from production_iris.data import load_iris_data
from production_iris.preprocessing import preprocess_data


def test_preprocess_data() -> None:
    """Tests the preprocessing function functionality"""

    df = load_iris_data()

    # Introduce some missing values and duplicates for testing
    df.loc[0, "sepal_length"] = None
    df.loc[1] = df.loc[2]  # Duplicate row

    preprocessed_df = preprocess_data(df)

    assert preprocessed_df.isnull().sum().sum() == 0  # No missing values
    assert preprocessed_df.duplicated().sum() == 0  # No duplicates
