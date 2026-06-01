import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import logger
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer
from xverse.transformer import WOE

from sklearn.cluster import KMeans




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
def cap_outliers(processed_df, cols, lower_quantile=0.01, upper_quantile=0.99):
    """
    Caps extreme outliers in specified numerical columns using IQR or custom quantiles.
    This prevents extreme values from breaking linear estimators.
    """
    df_copy = processed_df.copy()
    for col in cols:
        if col in df_copy.columns:
            lower_bound = df_copy[col].quantile(lower_quantile)
            upper_bound = df_copy[col].quantile(upper_quantile)
            df_copy[col] = np.clip(df_copy[col], lower_bound, upper_bound)
    return df_copy

def calculate_credit_risk_metrics_2(processed_df, default_col='FraudResult', amount_col='Amount'):
    """
    Calculates foundational Basel II risk parameters (PD, LGD, EAD, EL)
    scaled to the alternative transactional data context.
    """
    metrics_df = processed_df.copy()
    
    # 1. Probability of Default (PD): Simple historical baseline segment mapping
    # (In a real scenario, this comes from your trained scorecard probabilities)
    global_pd = metrics_df[default_col].mean()
    metrics_df['Probability_of_Default'] = np.where(metrics_df[default_col] == 1, 1.0, global_pd)
    
    # 2. Exposure at Default (EAD): Total gross cash exposure at the moment of transaction
    metrics_df['Exposure_at_Default'] = metrics_df[amount_col].abs()
    
    # 3. Loss Given Default (LGD): Assuming a standard recovery rate for alternative unsecured retail data
    # If a default occurs, online retail micro-lending historically recovers less collateral (e.g., 85% loss)
    metrics_df['Loss_Given_Default'] = 0.85
    
    # 4. Expected Loss (EL) Formula: EL = PD * LGD * EAD
    metrics_df['Expected_Loss'] = (
        metrics_df['Probability_of_Default'] * metrics_df['Loss_Given_Default'] * metrics_df['Exposure_at_Default']
    )
    
    return metrics_df

def preprocess_with_woe(processed_df, feature_cols, target_col='FraudResult'):
    """
    Modular preprocessing function that fits a Weight of Evidence (WoE) 
    transformer and returns the transformed features along with the IV summary table.
    """
    # 1. Isolate matrices and cast target variable type
    X = processed_df[feature_cols].copy()
    y = processed_df[target_col].astype(int)
    
    # 2. Fit and execute the transformer engine
    clf = WOE()
    clf.fit(X, y)
    
    # 3. Transform features and clean the Information Value summary layout
    X_transformed = clf.transform(X)
    iv_summary = clf.iv_df
    iv_summary.columns = ['Feature_Name', 'Information_Value']
    iv_summary = iv_summary.sort_values(by='Information_Value', ascending=False).reset_index(drop=True)
    
    return X_transformed, iv_summary

class RFMTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = X.copy()

        X["TransactionStartTime"] = pd.to_datetime(
            X["TransactionStartTime"]
        )

        snapshot_date = (
            X["TransactionStartTime"].max()
            + pd.Timedelta(days=1)
        )

        rfm = (
            X.groupby("CustomerId")
            .agg(
                Recency=(
                    "TransactionStartTime",
                    lambda x:
                    (
                        snapshot_date
                        - x.max()
                    ).days
                ),
                Frequency=(
                    "TransactionId",
                    "count"
                ),
                Monetary=(
                    "Amount",
                    "sum"
                )
            )
            .reset_index()
        )

        return rfm


class HighRiskLabeler(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        n_clusters=3,
        random_state=42
    ):

        self.n_clusters = n_clusters
        self.random_state = random_state

    def fit(self, X, y=None):

        self.scaler = StandardScaler()

        rfm_scaled = self.scaler.fit_transform(
            X[
                [
                    "Recency",
                    "Frequency",
                    "Monetary"
                ]
            ]
        )

        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10
        )

        self.kmeans.fit(rfm_scaled)

        return self

    def transform(self, X):

        X = X.copy()

        rfm_scaled = self.scaler.transform(
            X[
                [
                    "Recency",
                    "Frequency",
                    "Monetary"
                ]
            ]
        )

        X["Cluster"] = (
            self.kmeans.predict(
                rfm_scaled
            )
        )

        cluster_summary = (
            X.groupby("Cluster")
            [
                [
                    "Recency",
                    "Frequency",
                    "Monetary"
                ]
            ]
            .mean()
        )

        high_risk_cluster = (
            cluster_summary
            .sort_values(
                by=[
                    "Frequency",
                    "Monetary"
                ]
            )
            .index[0]
        )

        X["is_high_risk"] = np.where(
            X["Cluster"] == high_risk_cluster,
            1,
            0
        )

        return X


def merge_target(
    original_df,
    rfm_df
):

    return original_df.merge(
        rfm_df[
            [
                "CustomerId",
                "is_high_risk"
            ]
        ],
        on="CustomerId",
        how="left"
    )