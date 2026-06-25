import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


def evaluate_classification_model(y_true, y_pred, y_proba):
    """
    Evaluate a binary classification model.

    Args:
        y_true: Actual target labels.
        y_pred: Predicted class labels.
        y_proba: Predicted probability scores for positive class.

    Returns:
        dict: Evaluation metrics and confusion matrix.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "confusion_matrix": confusion_matrix(y_true, y_pred)
    }

    return metrics


def print_model_report(y_true, y_pred, y_proba):
    """
    Print classification metrics in a readable format.
    """
    metrics = evaluate_classification_model(y_true, y_pred, y_proba)

    print("Model Evaluation Metrics")
    print("------------------------")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-score:  {metrics['f1_score']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")

    print("\nConfusion Matrix:")
    print(metrics["confusion_matrix"])

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    return metrics