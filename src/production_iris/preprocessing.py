"""Preprocessing functions for the production Iris dataset."""

import pandas as pd


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove missing values and duplicates from the Iris dataset."""
    df = df.dropna().drop_duplicates().copy()

    return df
