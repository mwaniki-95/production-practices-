"""Data loading for the Iris dataset."""

import pandas as pd
from sklearn.datasets import load_iris


def load_iris_data() -> pd.DataFrame:
    """Load Iris with 'target' (0, 1, 2) and 'species' columns."""
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["target"] = iris.target
    df["species"] = df["target"].map(dict(enumerate(iris.target_names)))
    return df
