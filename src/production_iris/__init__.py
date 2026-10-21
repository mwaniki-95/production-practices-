"""Production Iris package."""

from .data import load_iris_data
from .eda import exploratory_data_analysis
from .preprocessing import preprocess_data

__all__ = [
    "load_iris_data",
    "exploratory_data_analysis",
    "preprocess_data",
]
