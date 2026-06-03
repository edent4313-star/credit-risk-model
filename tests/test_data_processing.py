import pandas as pd
import pytest
from src.data_processing import load_and_preprocess_data, split_dataset


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


<<<<<<< HEAD
 

    
def dummy_fraud_data(tmp_path):
    df = pd.DataFrame({
        'TransactionId': ['TX01', 'TX02', 'TX03', 'TX04', 'TX05'],
        'ProviderId': ['P1', 'P2', 'P1', 'P2', 'P1'],
        'Amount': [10, 20, 15, 40, 50],
        'FraudResult': [0, 0, 1, 0, 1]
    })
    csv_file = tmp_path / "dummy_data.csv"
    df.to_csv(csv_file, index=False)
    return str(csv_file)

def test_load_and_preprocess_drops_identifiers(dummy_fraud_data):
    X, y = load_and_preprocess_data(dummy_fraud_data)
    assert 'TransactionId' not in X.columns
    assert 'FraudResult' not in X.columns
    assert len(y) == 5

def test_split_dataset_dimensions(dummy_fraud_data):
    X, y = load_and_preprocess_data(dummy_fraud_data)
    X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=0.4, random_state=42)
    assert len(X_train) == 3
    assert len(X_test) == 2

    import pandas as pd

from src.data_processing import (
    AggregateFeatures,
    DateFeatures
)
=======
>>>>>>> 5190f6fdda8d40e777d0d40888b64700cda81401


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