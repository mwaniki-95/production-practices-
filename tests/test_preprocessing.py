"""Tests for the preprocessing module."""

from production_iris.data import load_iris_data
from production_iris.preprocessing import preprocess_data


def test_preprocess_data() -> None:
    """Test that preprocessing handles missing values and duplicates."""
    df = load_iris_data()

    # Introduce a missing value.
    df.loc[0, "sepal length (cm)"] = None

    # Introduce a duplicate row.
    df.loc[1] = df.loc[2]

    preprocessed_df = preprocess_data(df)

    # Verify that missing values have been handled.
    assert preprocessed_df.isnull().sum().sum() == 0

    # Verify that duplicate rows have been removed.
    assert not preprocessed_df.duplicated().any()

    # Verify that required columns are present.
    assert "target" in preprocessed_df.columns
    assert "species" in preprocessed_df.columns
