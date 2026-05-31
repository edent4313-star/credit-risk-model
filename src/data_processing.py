import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import logger
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer
from xverse.transformer import WOE


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


    from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

# =====================================================
# Aggregate Features
# =====================================================

class AggregateFeatures(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
            return self

    def transform(self, X):

        X = X.copy()

        agg = (
            X.groupby("CustomerId")
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
                )
            )
            .reset_index()
        )

        X = X.merge(
            agg,
            on="CustomerId",
            how="left"
        )

        return X


# =====================================================
# Date Features
# =====================================================

class DateFeatures(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = X.copy()

        X["TransactionStartTime"] = pd.to_datetime(
            X["TransactionStartTime"]
        )

        X["Transaction_Hour"] = (
            X["TransactionStartTime"].dt.hour
        )

        X["Transaction_Day"] = (
            X["TransactionStartTime"].dt.day
        )

        X["Transaction_Month"] = (
            X["TransactionStartTime"].dt.month
        )

        X["Transaction_Year"] = (
            X["TransactionStartTime"].dt.year
        )

        return X


# =====================================================
# WoE Transformer
# =====================================================

class WoETransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y):
        return self

    def transform(self, X):
        return X


# =====================================================
# Pipeline Builder
# =====================================================

def build_pipeline(df):

    numeric_features = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_features = df.select_dtypes(
        include="object"
    ).columns.tolist()

    if "CustomerId" in categorical_features:
        categorical_features.remove(
            "CustomerId"
        )

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "scaler",
                    StandardScaler()
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "encoder",
                    OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_pipeline,
                numeric_features
            ),
            (
                "cat",
                categorical_pipeline,
                categorical_features
            )
        ]
    )

    pipeline = Pipeline(
        steps=[
            (
                "aggregate_features",
                AggregateFeatures()
            ),
            (
                "date_features",
                DateFeatures()
            ),
            (
                "preprocessor",
                preprocessor
            )
        ]
    )

    return pipeline

def calculate_credit_risk_metrics(df_features, target_series, bins=10):
    """
    Calculates Weight of Evidence (WoE) and Information Value (IV)
    to bypass internal xverse package bugs.
    """
    df_features = df_features.reset_index(drop=True)
    target_series = target_series.reset_index(drop=True).astype(int)
    
    transformed_df = pd.DataFrame()
    iv_records = []
    
    for col in df_features.columns:
        try:
            binned_col = pd.qcut(df_features[col], q=bins, duplicates='drop')
        except ValueError:
            binned_col = pd.cut(df_features[col], bins=bins, duplicates='drop')
            
        crosstab = pd.crosstab(binned_col, target_series)
        
        if 0 not in crosstab.columns: crosstab[0] = 0
        if 1 not in crosstab.columns: crosstab[1] = 0
            
        crosstab.columns = ['Good', 'Bad']
        
        crosstab['Good_Dist'] = (crosstab['Good'] + 0.5) / crosstab['Good'].sum()
        crosstab['Bad_Dist'] = (crosstab['Bad'] + 0.5) / crosstab['Bad'].sum()
        
        crosstab['WOE'] = np.log(crosstab['Good_Dist'] / crosstab['Bad_Dist'])
        crosstab['IV_bin'] = (crosstab['Good_Dist'] - crosstab['Bad_Dist']) * crosstab['WOE']
        total_iv = crosstab['IV_bin'].sum()
        
        transformed_df[f'{col}_woe'] = binned_col.map(crosstab['WOE'])
        
        if total_iv < 0.02:
            strength = "Useless"
        elif total_iv < 0.1:
            strength = "Weak"
        elif total_iv < 0.3:
            strength = "Medium"
        elif total_iv < 0.5:
            strength = "Strong"
        else:
            strength = "Suspiciously High (Check for leakage)"
            
        iv_records.append({
            'Variable': col,
            'Information Value (IV)': total_iv,
            'Predictive Power': strength
        })
        
    iv_df = pd.DataFrame(iv_records).sort_values(by='Information Value (IV)', ascending=False)
    
    return transformed_df, iv_df