"""Exploratory data analysis for the Iris dataset."""

import pandas as pd


def exploratory_data_analysis(df: pd.DataFrame) -> dict[str, object]:
    """Return summary information about the Iris DataFrame.

    Raises:
        ValueError: If the 'species' column is missing.
    """
    if "species" not in df.columns:
        raise ValueError("DataFrame must contain a 'species' column")

    features = df.select_dtypes(include="number")

    return {
        "dataframe_size": df.shape,
        "data_types": df.dtypes,
        "column_names": df.columns.tolist(),
        "missing_values": df.isna().sum(),
        "summary_statistics": df.describe(),
        "correlation_matrix": features.corr(),
        "unique_species": df["species"].unique(),
        "species_count": df["species"].value_counts(),
    }
