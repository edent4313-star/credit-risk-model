import pandas as pd
import mlflow
import mlflow.sklearn

import scipy.sparse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {

        "accuracy":
            accuracy_score(
                y_test,
                predictions
            ),

        "precision":
            precision_score(
                y_test,
                predictions
            ),

        "recall":
            recall_score(
                y_test,
                predictions
            ),

        "f1":
            f1_score(
                y_test,
                predictions
            ),

        "roc_auc":
            roc_auc_score(
                y_test,
                probabilities
            )
    }

    return metrics




import scipy.sparse
from sklearn.model_selection import train_test_split

def train_models(df):
    # DO NOT convert to dense (no .toarray())
    # Sparse matrices don't have .drop(), so use indexing instead.
    
    # Assuming 'is_high_risk' is the last column (index -1)
    # If it's a different column, you must know its index.
    
    # X = all columns except the last one
    X = df[:, :-1]
    # y = the last column
    y = df[:, -1].toarray().ravel() # Flatten to 1D array for labels

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    # Scikit-learn models handle sparse matrices automatically
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(random_state=42)
    
    }

    best_model = None
    best_auc = 0

    mlflow.set_experiment("credit_risk_models")

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)

            metrics = evaluate_model(model, X_test, y_test)

            mlflow.log_params(model.get_params())
            mlflow.log_metrics(metrics)

            # Updated to use 'name' instead of 'artifact_path'
            mlflow.sklearn.log_model(
                sk_model=model,
                name=name
            )

            if metrics["roc_auc"] > best_auc:
                best_auc = metrics["roc_auc"]
                best_model = model
    with mlflow.start_run(run_name="Best_Model"):

        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="best_model",
            registered_model_name="CreditRiskBestModel"
        )

    return best_model, X_test, y_test
def tune_random_forest(
    X_train,
    y_train
):

    param_grid = {

        "n_estimators":
            [100, 200],

        "max_depth":
            [5, 10, 20],

        "min_samples_split":
            [2, 5]
    }

    rf = RandomForestClassifier(
        random_state=42
    )

    grid = GridSearchCV(
        rf,
        param_grid,
        cv=3,
        scoring="roc_auc",
        n_jobs=-1
    )

    grid.fit(
        X_train,
        y_train
    )

import pandas as pd
import mlflow
import mlflow.sklearn

import scipy.sparse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {

        "accuracy":
            accuracy_score(
                y_test,
                predictions
            ),

        "precision":
            precision_score(
                y_test,
                predictions
            ),

        "recall":
            recall_score(
                y_test,
                predictions
            ),

        "f1":
            f1_score(
                y_test,
                predictions
            ),

        "roc_auc":
            roc_auc_score(
                y_test,
                probabilities
            )
    }

    return metrics




import scipy.sparse
from sklearn.model_selection import train_test_split

def train_models(df):
    # DO NOT convert to dense (no .toarray())
    # Sparse matrices don't have .drop(), so use indexing instead.
    
    # Assuming 'is_high_risk' is the last column (index -1)
    # If it's a different column, you must know its index.
    
    # X = all columns except the last one
    X = df[:, :-1]
    # y = the last column
    y = df[:, -1].toarray().ravel() # Flatten to 1D array for labels

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    # Scikit-learn models handle sparse matrices automatically
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(random_state=42)
    
    }

    best_model = None
    best_auc = 0

    mlflow.set_experiment("credit_risk_models")

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)

            metrics = evaluate_model(model, X_test, y_test)

            mlflow.log_params(model.get_params())
            mlflow.log_metrics(metrics)

            # Updated to use 'name' instead of 'artifact_path'
            mlflow.sklearn.log_model(
                sk_model=model,
                name=name
            )

            if metrics["roc_auc"] > best_auc:
                best_auc = metrics["roc_auc"]
                best_model = model
    with mlflow.start_run(run_name="Best_Model"):

        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="best_model",
            registered_model_name="CreditRiskBestModel"
        )

    return best_model, X_test, y_test
def tune_random_forest(
    X_train,
    y_train
):

    param_grid = {

        "n_estimators":
            [100, 200],

        "max_depth":
            [5, 10, 20],

        "min_samples_split":
            [2, 5]
    }

    rf = RandomForestClassifier(
        random_state=42
    )

    grid = GridSearchCV(
        rf,
        param_grid,
        cv=3,
        scoring="roc_auc",
        n_jobs=-1
    )

    grid.fit(
        X_train,
        y_train
    )

    return grid.best_estimator_