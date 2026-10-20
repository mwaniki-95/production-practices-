
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from production_iris.data import load_iris_data
from production_iris.visualization.comparisons import (
    compare_species_distribution,
)


def test_compare_species_distribution_runs_without_error():
    df = load_iris_data()

    try:
        compare_species_distribution(df)
        assert plt.gcf() is not None
    finally:
        plt.close("all")
