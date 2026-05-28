import pandas as pd



def dataset_overview(df):
    """
    Display dataset overview.
    """

    print("Dataset Shape:")
    print(df.shape)

    print("\nData Types:")
    print(df.dtypes)

    print("\nSummary Statistics:")
    print(df.describe())



def check_duplicates(df):
    """
    Check duplicate records.
    """

    return df.duplicated().sum()