import pandas as pd
import os

from config import logger


def load_data(filepath):
    """
    Load dataset safely.
    """

    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"File not found: {filepath}"
            )

        df = pd.read_csv(filepath)

        if df.empty:
            raise ValueError(
                "Dataset is empty."
            )

        logger.info("Dataset loaded successfully")

        return df

    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise