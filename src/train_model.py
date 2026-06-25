from pathlib import Path

import mlflow
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.data_loader import load_raw_data
from src.preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    prepare_model_data,
    split_model_data,
)


MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")

MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

LOGISTIC_MODEL_PATH = MODELS_DIR / "real_world_logistic_regression_baseline.joblib"
XGBOOST_MODEL_PATH = MODELS_DIR / "real_world_xgboost_model.joblib"
MODEL_COMPARISON_PATH = REPORTS_DIR / "real_world_model_comparison.csv"
MLFLOW_EXPERIMENT_NAME = "real_world_fraud_detection"


def build_preprocessor() -> ColumnTransformer:
    """
    Build preprocessing pipeline for numeric and categorical features.
    """
    numeric_transformer = StandardScaler()

    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=True,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, NUMERIC_FEATURES),
            ("categorical", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        sparse_threshold=0.3,
    )

    return preprocessor


def build_logistic_regression_pipeline() -> Pipeline:
    """
    Build Logistic Regression baseline pipeline.
    """
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ]
    )

    return pipeline


def build_xgboost_pipeline(scale_pos_weight: float) -> Pipeline:
    """
    Build XGBoost advanced model pipeline.

    scale_pos_weight helps XGBoost handle the severe class imbalance.
    """
    model = XGBClassifier(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ]
    )

    return pipeline


def evaluate_model(model, X_test, y_test, threshold=0.5) -> dict:
    """
    Evaluate a fraud model using classification metrics.
    """
    fraud_probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (fraud_probabilities >= threshold).astype(int)

    cm = confusion_matrix(y_test, predictions)

    metrics = {
        "threshold": threshold,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1_score": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, fraud_probabilities),
        "pr_auc": average_precision_score(y_test, fraud_probabilities),
        "true_negatives": int(cm[0][0]),
        "false_positives": int(cm[0][1]),
        "false_negatives": int(cm[1][0]),
        "true_positives": int(cm[1][1]),
        "confusion_matrix": cm,
        "classification_report": classification_report(
            y_test,
            predictions,
            zero_division=0,
        ),
    }

    return metrics


def print_metrics(model_name: str, metrics: dict):
    """
    Print model metrics cleanly.
    """
    print("=" * 80)
    print(model_name)
    print("=" * 80)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-score:  {metrics['f1_score']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"PR-AUC:    {metrics['pr_auc']:.4f}")
    print()
    print("Confusion Matrix:")
    print(metrics["confusion_matrix"])
    print()
    print("Classification Report:")
    print(metrics["classification_report"])
    print()

def log_mlflow_run(model_name: str, params: dict, metrics: dict, model_path: Path):
    """
    Log model parameters, metrics, and saved model artifact to MLflow.
    """
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(params)

        numeric_metrics = {
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "roc_auc": metrics["roc_auc"],
            "pr_auc": metrics["pr_auc"],
            "false_positives": metrics["false_positives"],
            "false_negatives": metrics["false_negatives"],
            "true_positives": metrics["true_positives"],
            "true_negatives": metrics["true_negatives"],
        }

        mlflow.log_metrics(numeric_metrics)

        if model_path.exists():
            mlflow.log_artifact(str(model_path))


def train_and_compare_models():
    """
    Train Logistic Regression and XGBoost models, evaluate them, and save artifacts.
    """
    print("Loading data...")
    df = load_raw_data()

    print("Preparing model features...")
    X, y = prepare_model_data(df)

    print("Splitting train/test data...")
    X_train, X_test, y_train, y_test = split_model_data(X, y)

    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()
    scale_pos_weight = negative_count / positive_count

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Fraud cases in train: {positive_count}")
    print(f"Non-fraud cases in train: {negative_count}")
    print(f"XGBoost scale_pos_weight: {scale_pos_weight:.2f}")
    print()

    print("Training Logistic Regression baseline...")
    logistic_model = build_logistic_regression_pipeline()
    logistic_model.fit(X_train, y_train)

    logistic_metrics = evaluate_model(logistic_model, X_test, y_test)
    joblib.dump(logistic_model, LOGISTIC_MODEL_PATH)

    log_mlflow_run(
        model_name="Logistic Regression Baseline",
        params={
            "model_type": "LogisticRegression",
            "max_iter": 1000,
            "class_weight": "balanced",
            "threshold": 0.5,
        },
        metrics=logistic_metrics,
        model_path=LOGISTIC_MODEL_PATH,
    )

    print("Training XGBoost model...")
    xgboost_model = build_xgboost_pipeline(scale_pos_weight=scale_pos_weight)
    xgboost_model.fit(X_train, y_train)

    xgboost_metrics = evaluate_model(xgboost_model, X_test, y_test)
    joblib.dump(xgboost_model, XGBOOST_MODEL_PATH)

    log_mlflow_run(
        model_name="XGBoost Advanced Model",
        params={
            "model_type": "XGBClassifier",
            "n_estimators": 250,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "scale_pos_weight": scale_pos_weight,
            "threshold": 0.5,
        },
        metrics=xgboost_metrics,
        model_path=XGBOOST_MODEL_PATH,
    )

    print_metrics("Logistic Regression Baseline", logistic_metrics)
    print_metrics("XGBoost Advanced Model", xgboost_metrics)

    comparison = pd.DataFrame(
        [
            {
                "model": "Logistic Regression",
                "accuracy": logistic_metrics["accuracy"],
                "precision": logistic_metrics["precision"],
                "recall": logistic_metrics["recall"],
                "f1_score": logistic_metrics["f1_score"],
                "roc_auc": logistic_metrics["roc_auc"],
                "pr_auc": logistic_metrics["pr_auc"],
                "false_positives": logistic_metrics["false_positives"],
                "false_negatives": logistic_metrics["false_negatives"],
                "true_positives": logistic_metrics["true_positives"],
                "true_negatives": logistic_metrics["true_negatives"],
            },
            {
                "model": "XGBoost",
                "accuracy": xgboost_metrics["accuracy"],
                "precision": xgboost_metrics["precision"],
                "recall": xgboost_metrics["recall"],
                "f1_score": xgboost_metrics["f1_score"],
                "roc_auc": xgboost_metrics["roc_auc"],
                "pr_auc": xgboost_metrics["pr_auc"],
                "false_positives": xgboost_metrics["false_positives"],
                "false_negatives": xgboost_metrics["false_negatives"],
                "true_positives": xgboost_metrics["true_positives"],
                "true_negatives": xgboost_metrics["true_negatives"],
            },
        ]
    )

    comparison.to_csv(MODEL_COMPARISON_PATH, index=False)

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name="Model Comparison Report"):
        mlflow.log_param("selected_model", "XGBoost")
        mlflow.log_param("baseline_model", "Logistic Regression")
        mlflow.log_param("selection_reason", "Higher recall, higher PR-AUC, fewer false positives")

        mlflow.log_metric(
            "recall_improvement",
            xgboost_metrics["recall"] - logistic_metrics["recall"],
        )

        mlflow.log_metric(
            "precision_improvement",
            xgboost_metrics["precision"] - logistic_metrics["precision"],
        )

        mlflow.log_metric(
            "pr_auc_improvement",
            xgboost_metrics["pr_auc"] - logistic_metrics["pr_auc"],
        )

        mlflow.log_metric(
            "false_positive_reduction",
            logistic_metrics["false_positives"] - xgboost_metrics["false_positives"],
        )

        mlflow.log_metric(
            "false_negative_reduction",
            logistic_metrics["false_negatives"] - xgboost_metrics["false_negatives"],
        )

        mlflow.log_metric("xgboost_recall", xgboost_metrics["recall"])
        mlflow.log_metric("xgboost_precision", xgboost_metrics["precision"])
        mlflow.log_metric("xgboost_pr_auc", xgboost_metrics["pr_auc"])

        mlflow.log_artifact(str(MODEL_COMPARISON_PATH))

    print("Saved artifacts:")
    print(f"Logistic Regression model: {LOGISTIC_MODEL_PATH}")
    print(f"XGBoost model: {XGBOOST_MODEL_PATH}")
    print(f"Model comparison report: {MODEL_COMPARISON_PATH}")

    return logistic_model, xgboost_model, comparison


if __name__ == "__main__":
    train_and_compare_models()