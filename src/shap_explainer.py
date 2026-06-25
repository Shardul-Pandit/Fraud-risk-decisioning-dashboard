from pathlib import Path

import joblib
import pandas as pd
import shap

from src.preprocessing import engineer_features, MODEL_FEATURES
from src.feature_importance import clean_feature_name, get_base_feature


MODEL_PATH = Path("models") / "real_world_xgboost_model.joblib"


def _to_dense_array(matrix):
    """
    Convert sparse matrices to dense arrays for SHAP.
    """
    if hasattr(matrix, "toarray"):
        return matrix.toarray()

    return matrix


def _format_feature_value(value):
    """
    Format feature values for readable output.
    """
    if isinstance(value, float):
        return round(value, 4)

    return value


def generate_local_shap_explanation(
    transactions: pd.DataFrame,
    selected_index: int,
    top_n: int = 6,
) -> dict:
    """
    Generate local SHAP explanations for one selected transaction.

    Args:
        transactions: Raw transaction DataFrame without the target column.
        selected_index: Index of the transaction to explain.
        top_n: Number of top local features to return.

    Returns:
        Dictionary containing transaction-specific SHAP drivers.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Train the model first using: python -m src.train_model"
        )

    if selected_index not in transactions.index:
        raise ValueError(f"selected_index {selected_index} not found in transactions.")

    pipeline = joblib.load(MODEL_PATH)

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    engineered_transactions = engineer_features(transactions)
    selected_features = engineered_transactions.loc[[selected_index], MODEL_FEATURES]

    transformed_features = preprocessor.transform(selected_features)
    transformed_features = _to_dense_array(transformed_features)

    raw_feature_names = preprocessor.get_feature_names_out()
    feature_names = [clean_feature_name(name) for name in raw_feature_names]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(transformed_features)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    if len(shap_values.shape) == 3:
        shap_values = shap_values[:, :, 1]

    local_values = shap_values[0]

    detailed_shap = pd.DataFrame(
        {
            "feature": feature_names,
            "shap_value": local_values,
        }
    )

    detailed_shap["abs_shap_value"] = detailed_shap["shap_value"].abs()
    detailed_shap["base_feature"] = detailed_shap["feature"].apply(get_base_feature)

    grouped_shap = (
        detailed_shap
        .groupby("base_feature", as_index=False)
        .agg(
            shap_value=("shap_value", "sum"),
            abs_shap_value=("abs_shap_value", "sum"),
        )
        .sort_values("abs_shap_value", ascending=False)
        .reset_index(drop=True)
    )

    row_values = selected_features.iloc[0].to_dict()

    grouped_shap["feature_value"] = grouped_shap["base_feature"].apply(
        lambda feature: _format_feature_value(row_values.get(feature, "N/A"))
    )

    grouped_shap["impact"] = grouped_shap["shap_value"].apply(
        lambda value: "Increased fraud risk" if value > 0 else "Reduced fraud risk"
    )

    top_features = grouped_shap.head(top_n)

    risk_increasing = grouped_shap[grouped_shap["shap_value"] > 0].head(top_n)
    risk_reducing = grouped_shap[grouped_shap["shap_value"] < 0].head(top_n)

    return {
        "selected_index": selected_index,
        "top_features": top_features.to_dict(orient="records"),
        "risk_increasing_features": risk_increasing.to_dict(orient="records"),
        "risk_reducing_features": risk_reducing.to_dict(orient="records"),
    }


if __name__ == "__main__":
    from src.data_loader import load_raw_data

    df = load_raw_data()
    transactions = df.drop(columns=["is_fraud"], errors="ignore")

    for index in [0, 1685, 306221]:
        explanation = generate_local_shap_explanation(
            transactions=transactions,
            selected_index=index,
            top_n=6,
        )

        print("=" * 80)
        print(f"Local SHAP Explanation for index {index}")
        print("=" * 80)

        print("Top features:")
        for feature in explanation["top_features"]:
            print(feature)

        print()