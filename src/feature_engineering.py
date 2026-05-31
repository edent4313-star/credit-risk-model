import pandas as pd
from parso import python


def create_customer_features(df):
    """
    Create customer-level aggregate features.
    """

    customer_features = (
        df.groupby("CustomerId")
        .agg(
            Total_Transaction_Amount=(
                "Amount",
                "sum"
            ),
            Average_Transaction_Amount=(
                "Amount",
                "mean"
            ),
            Transaction_Count=(
                "TransactionId",
                "count"
            ),
            Std_Transaction_Amount=(
                "Amount",
                "std"
            ),
            Max_Transaction_Amount=(
                "Amount",
                "max"
            ),
            Min_Transaction_Amount=(
                "Amount",
                "min"
            )
        )
        .reset_index()
    )

    return customer_features


def extract_time_features(df):

    df["TransactionStartTime"] = pd.to_datetime(
        df["TransactionStartTime"]
    )

    df["Transaction_Hour"] = (
        df["TransactionStartTime"].dt.hour
    )

    df["Transaction_Day"] = (
        df["TransactionStartTime"].dt.day
    )

    df["Transaction_Month"] = (
        df["TransactionStartTime"].dt.month
    )

    df["Transaction_Year"] = (
        df["TransactionStartTime"].dt.year
    )

    return df


