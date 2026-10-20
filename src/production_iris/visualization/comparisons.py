"""Visualizations for comparing different aspects of the Iris dataset."""

import matplotlib.pyplot as plt
import seaborn as sns


def compare_species_distribution(df):
    """Compare the distribution of species in the Iris dataset."""

    sns.countplot(x="species", data=df)
    plt.title("Species Distribution")
    plt.xlabel("Species")
    plt.ylabel("Count")
    plt.show()
