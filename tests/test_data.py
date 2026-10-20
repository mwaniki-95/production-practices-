from production_iris.data import load_iris_data


def test_iris_data_loads():
    df = load_iris_data()

    assert df is not None
    assert not df.empty


def test_iris_data_has_species():
    df = load_iris_data()

    assert "species" in df.columns
