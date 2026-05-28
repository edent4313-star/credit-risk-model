import pandas as pd

from src.data_processing import (
    handle_missing_values
)



def test_missing_values():

    df = pd.DataFrame({
        "A": [1, None, 3]
    })

    processed_df = handle_missing_values(df)

    assert processed_df.isnull().sum().sum() == 0