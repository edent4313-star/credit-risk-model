import pandas as pd

from src.target_engineering import (
    RFMTransformer,
    HighRiskLabeler
)


def test_rfm_creation():

    df = pd.DataFrame({

        "CustomerId":
        ["A", "A", "B"],

        "TransactionId":
        [1, 2, 3],

        "Amount":
        [100, 200, 50],

        "TransactionStartTime":
        [
            "2024-01-01",
            "2024-01-02",
            "2024-01-05"
        ]
    })

    rfm = (
        RFMTransformer()
        .fit_transform(df)
    )

    assert "Recency" in rfm.columns
    assert "Frequency" in rfm.columns
    assert "Monetary" in rfm.columns


def test_high_risk_label():

    rfm = pd.DataFrame({

        "CustomerId":
        ["A", "B", "C"],

        "Recency":
        [10, 100, 50],

        "Frequency":
        [20, 1, 5],

        "Monetary":
        [5000, 50, 1000]
    })

    labeler = HighRiskLabeler()

    result = (
        labeler.fit_transform(rfm)
    )

    assert "is_high_risk" in result.columns