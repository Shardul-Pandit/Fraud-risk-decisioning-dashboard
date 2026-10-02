"""
Feature ablation: how much does the model depend on gender and job?

Trains the same XGBoost pipeline on different feature sets and evaluates
each one two ways:

1. random split   - the same stratified 80/20 split used for the headline
                    metrics. Transactions from one card land in both train
                    and test.
2. card split     - all transactions from a card go to either train or
                    test. This measures how the model does on customers it
                    has never seen, which is the honest test for features
                    that act like a customer ID (job, state, city_pop).

Run with: python -m src.ablation
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from src.data_loader import load_full_data
from src.preprocessing import NUMERIC_FEATURES, TARGET_COLUMN, engineer_features
from src.train_model import build_xgboost_pipeline, evaluate_model


ABLATION_PATH = Path("reports") / "feature_ablation.csv"

FEATURE_SETS = {
    "original (gender + job)": ["category", "gender", "state", "job"],
    "no gender": ["category", "state", "job"],
    "no gender, no job": ["category", "state"],
}


def _fit_and_score(X_train, X_test, y_train, y_test, categorical_features):
    features = NUMERIC_FEATURES + categorical_features
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = build_xgboost_pipeline(
        scale_pos_weight=scale_pos_weight,
        numeric_features=NUMERIC_FEATURES,
        categorical_features=categorical_features,
    )
    model.fit(X_train[features], y_train)

    return evaluate_model(model, X_test[features], y_test)


def run_ablation() -> pd.DataFrame:
    df = engineer_features(load_full_data())
    y = df[TARGET_COLUMN]

    random_split = train_test_split(
        df, y, test_size=0.2, random_state=42, stratify=y
    )

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(df, y, groups=df["cc_num"]))
    card_split = (
        df.iloc[train_idx], df.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]
    )

    rows = []
    for split_name, (X_train, X_test, y_train, y_test) in {
        "random": random_split,
        "by card": card_split,
    }.items():
        for set_name, categorical_features in FEATURE_SETS.items():
            m = _fit_and_score(X_train, X_test, y_train, y_test, categorical_features)
            rows.append(
                {
                    "split": split_name,
                    "feature_set": set_name,
                    "recall": m["recall"],
                    "precision": m["precision"],
                    "pr_auc": m["pr_auc"],
                    "roc_auc": m["roc_auc"],
                    "false_positives": m["false_positives"],
                    "false_negatives": m["false_negatives"],
                    "true_positives": m["true_positives"],
                    "test_rows": len(y_test),
                    "test_fraud": int(y_test.sum()),
                }
            )
            print(rows[-1], flush=True)

    result = pd.DataFrame(rows)
    result.to_csv(ABLATION_PATH, index=False)
    return result


if __name__ == "__main__":
    print(run_ablation().to_string(index=False))
