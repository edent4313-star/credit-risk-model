import pandas as pd


from src.data_processing import (
    AggregateFeatures,
    DateFeatures
)

from src.data_processing import (
    handle_missing_values
)



def test_missing_values():

    df = pd.DataFrame({
        "A": [1, None, 3]
    })

    processed_df = handle_missing_values(df)

    assert processed_df.isnull().sum().sum() == 0




def test_aggregate_features():

    df = pd.DataFrame({

        "CustomerId": [1, 1, 2],

        "Amount": [100, 200, 300],

        "TransactionId": [1, 2, 3]
    })

    transformed = AggregateFeatures().transform(df)

    assert "Total_Transaction_Amount" in transformed.columns


def test_date_features():

    df = pd.DataFrame({

        "TransactionStartTime": [
            "2025-01-01 10:00:00"
        ]
    })

    transformed = DateFeatures().transform(df)

    assert "Transaction_Hour" in transformed.columns