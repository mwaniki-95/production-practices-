"""Production Iris package."""

from .data import load_iris_data
from .eda import exploratory_data_analysis

__all__ = [
    "load_iris_data",
    "exploratory_data_analysis",
]
