
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(
            y_test, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_test, y_pred, zero_division=0
        ),
        "F1-score": f1_score(
            y_test, y_pred, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y_test, y_proba)
    }

    return metrics


def compare_models(models, X_test, y_test):

    results = []

    for model_name, model in models.items():

        metrics = evaluate_model(model, X_test, y_test)
        metrics["Model"] = model_name

        results.append(metrics)

    return pd.DataFrame(results).set_index("Model")


def get_confusion_matrix(model, X_test, y_test):

    y_pred = model.predict(X_test)

    return confusion_matrix(y_test, y_pred)


def get_classification_report(model, X_test, y_test):

    y_pred = model.predict(X_test)

    return classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
