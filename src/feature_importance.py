from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path("models") / "real_world_xgboost_model.joblib"
REPORTS_DIR = Path("reports")
DETAILED_IMPORTANCE_PATH = REPORTS_DIR / "xgboost_feature_importance_detailed.csv"
GROUPED_IMPORTANCE_PATH = REPORTS_DIR / "xgboost_feature_importance_grouped.csv"


def clean_feature_name(feature_name: str) -> str:
    """
    Clean sklearn ColumnTransformer feature names.
    """
    return (
        feature_name
        .replace("numeric__", "")
        .replace("categorical__", "")
    )


def get_base_feature(clean_name: str) -> str:
    """
    Group one-hot encoded categorical features back to their original column.

    Examples:
    category_misc_net -> category
    state_TX -> state
    job_Cytogeneticist -> job
    """
    categorical_prefixes = ["category_", "gender_", "state_", "job_"]

    for prefix in categorical_prefixes:
        if clean_name.startswith(prefix):
            return prefix.replace("_", "")

    return clean_name


def generate_xgboost_feature_importance():
    """
    Generate detailed and grouped XGBoost feature importance reports.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Train the model first using: python -m src.train_model"
        )

    REPORTS_DIR.mkdir(exist_ok=True)

    pipeline = joblib.load(MODEL_PATH)

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    raw_feature_names = preprocessor.get_feature_names_out()
    feature_names = [clean_feature_name(name) for name in raw_feature_names]

    importances = model.feature_importances_

    detailed_importance = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance": importances,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    detailed_importance["base_feature"] = detailed_importance["feature"].apply(
        get_base_feature
    )

    grouped_importance = (
        detailed_importance
        .groupby("base_feature", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    detailed_importance.to_csv(DETAILED_IMPORTANCE_PATH, index=False)
    grouped_importance.to_csv(GROUPED_IMPORTANCE_PATH, index=False)

    return detailed_importance, grouped_importance


if __name__ == "__main__":
    detailed, grouped = generate_xgboost_feature_importance()

    print("XGBoost feature importance generated successfully.")
    print()
    print("Top 15 detailed features:")
    print(detailed.head(15))
    print()
    print("Grouped feature importance:")
    print(grouped)
    print()
    print(f"Detailed report saved to: {DETAILED_IMPORTANCE_PATH}")
    print(f"Grouped report saved to: {GROUPED_IMPORTANCE_PATH}")