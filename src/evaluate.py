from sklearn.metrics import (
    confusion_matrix,
    classification_report
)


'''def generate_report(
    model,
    X_test,
    y_test
):

    predictions = model.predict(
        X_test
    )

    print(
        classification_report(
            y_test,
            predictions
        )
    )

    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )'''
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


def evaluate_model(model, X_test, y_test):
    """
    Evaluate classification model performance.
    """

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(
            y_test,
            predictions
        ),

        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "f1_score": f1_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "roc_auc": roc_auc_score(
            y_test,
            probabilities
        )
    }

    return metrics


def generate_report(
    model,
    X_test,
    y_test
):
    """
    Print classification report and confusion matrix.
    """

    predictions = model.predict(X_test)

    print("=" * 50)
    print("Classification Report")
    print("=" * 50)

    print(
        classification_report(
            y_test,
            predictions
        )
    )

    print("=" * 50)
    print("Confusion Matrix")
    print("=" * 50)

    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )