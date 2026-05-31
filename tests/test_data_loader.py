from src.data_loader import load_data



def test_load_data():

    df = load_data(
        "data/raw/data.csv"
    )

    assert df is not None

    assert df.shape[0] > 0