from production_iris.data import load_iris_data
from production_iris.eda import exploratory_data_analysis


def test_eda_returns_summary():
    df = load_iris_data()
    summary = exploratory_data_analysis(df)

    assert isinstance(summary, dict)
    assert "dataframe_size" in summary
    assert "data_types" in summary
    assert "column_names" in summary
    assert "missing_values" in summary
    assert "summary_statistics" in summary
    assert "correlation_matrix" in summary
    assert "unique_species" in summary
    assert "species_count" in summary
