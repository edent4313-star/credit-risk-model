import pandas as pd

from config import logger


def get_missing_values(df):
    """
    Calculate missing values.
    """

    try:
        missing_values = df.isnull().sum()

        missing_percentage = (
            df.isnull().mean() * 100
        )

        missing_df = pd.DataFrame({
            "Missing Count": missing_values,
            "Missing Percentage": missing_percentage
        }).sort_values(
            by="Missing Percentage",
            ascending=False
        )

        logger.info("Missing value analysis completed")

        return missing_df

    except Exception as e:
        logger.error(f"Error calculating missing values: {e}")
        raise



def handle_missing_values(df):
    """
    Fill numerical missing values with median.
    """

    try:
        numerical_cols = df.select_dtypes(
            include="number"
        ).columns

        for col in numerical_cols:
            df[col] = df[col].fillna(
                df[col].median()
            )

        logger.info("Missing values handled successfully")

        return df

    except Exception as e:
        logger.error(f"Error handling missing values: {e}")
        raise