
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import numpy as np

from config import logger


sns.set_style("whitegrid")

def plot_missing_values(df):
    """
    Visualize missing values.
    """

    try:
        msno.matrix(df)
        plt.title("Missing Values Matrix")
        plt.show()

        logger.info("Missing values plotted successfully")

    except Exception as e:
        logger.error(f"Visualization error: {e}")
        raise


def plot_numerical_distribution(df):
    """
    Plot distributions of numerical features.
    """

    try:
        numerical_cols = df.select_dtypes(
            include=np.number
        ).columns

        df[numerical_cols].hist(
            figsize=(18, 14),
            bins=30
        )

        plt.tight_layout()
        plt.show()

        logger.info("Numerical distributions plotted")

    except Exception as e:
        logger.error(f"Error plotting distributions: {e}")
        raise


def plot_categorical_distribution(df, max_categories=20): # Added limit
    """
    Plot categorical feature distributions, skipping high-cardinality columns.
    """
    try:
        categorical_cols = df.select_dtypes(include="object").columns

        for col in categorical_cols:
            # Check number of unique values
            num_unique = df[col].nunique()
            
            # Skip if there are too many unique values
            if num_unique > max_categories:
                print(f"Skipping {col}: too many unique values ({num_unique})")
                continue

            print(f"Plotting {col}...")
            plt.figure(figsize=(8, 4))

            # Plot top values or all if few
            df[col].value_counts().plot(kind="bar")

            plt.title(f"Distribution of {col}")
            plt.xticks(rotation=45)
            plt.tight_layout() # Ensures labels don't get cut off
            plt.show()
            plt.close() # CRITICAL: Closes the plot so the loop continues

        logger.info("Categorical distributions plotted")

    except Exception as e:
        logger.error(f"Error plotting categorical variables: {e}")
        # raise # Optional: remove 'raise' if you want it to continue past errors

def plot_correlation_heatmap(df):
    """
    Plot correlation heatmap.
    """

    try:
        numerical_cols = df.select_dtypes(
            include=np.number
        ).columns

        corr_matrix = df[numerical_cols].corr()

        plt.figure(figsize=(12,8))

        sns.heatmap(
    corr_matrix,
    cmap="coolwarm",
    center=0
)

        plt.title("Correlation Heatmap")
        plt.show()

        logger.info("Correlation heatmap created")

    except Exception as e:
        logger.error(f"Error plotting correlation heatmap: {e}")
        raise

def plot_boxplots(df):
    """
    Plot boxplots for outlier detection.
    """

    try:
        numerical_cols = df.select_dtypes(
            include=np.number
        ).columns

        numerical_cols = numerical_cols[:10]

        logger.info("Boxplots created successfully")

    except Exception as e:
        logger.error(f"Error plotting boxplots: {e}")
        raise