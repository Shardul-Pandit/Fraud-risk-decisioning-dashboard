from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from src.preprocessing import engineer_features, MODEL_FEATURES
from src.feature_importance import clean_feature_name, get_base_feature


MODEL_PATH = Path("models") / "real_world_xgboost_model.joblib"


def _format_feature_value(value):
    """
    Format feature values for readable output.
    """
    if isinstance(value, float):
        return round(value, 4)

    return value


FEATURE_LABELS = {
    "amt": "Transaction Amount",
    "category": "Merchant Category",
    "state": "Customer State",
    "transaction_hour": "Transaction Hour",
    "transaction_day_of_week": "Transaction Day of Week",
    "transaction_month": "Transaction Month",
    "is_weekend": "Weekend Transaction",
    "customer_age": "Customer Age",
    "city_pop": "City Population",
    "distance_from_home_km": "Distance From Home",
    "time_since_last_transaction_minutes": "Time Since Last Transaction",
    "distance_from_last_transaction_km": "Distance From Previous Transaction",
    "implied_travel_speed_kmh": "Implied Travel Speed",
}

KM_TO_MILES = 0.621371
NO_PREVIOUS_TRANSACTION = 999999
DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def format_feature_label(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature.replace("_", " ").title())


def format_feature_display_value(feature: str, value) -> str:
    """
    Turn a raw feature value into the text shown to analysts and to the LLM.

    Formatting happens here, in code, so the LLM never has to convert units
    or round anything itself.
    """
    try:
        if feature == "amt":
            return f"${float(value):,.2f}"
        if feature in ("distance_from_home_km", "distance_from_last_transaction_km"):
            return f"{float(value) * KM_TO_MILES:,.1f} miles"
        if feature == "implied_travel_speed_kmh":
            if float(value) >= NO_PREVIOUS_TRANSACTION:
                return "not measurable (same timestamp as previous transaction)"
            return f"{float(value) * KM_TO_MILES:,.1f} mph"
        if feature == "time_since_last_transaction_minutes":
            if float(value) >= NO_PREVIOUS_TRANSACTION:
                return "no earlier transaction on this card"
            return f"{float(value):,.1f} minutes"
        if feature == "transaction_hour":
            return f"{int(value)}:00"
        if feature == "transaction_day_of_week":
            return DAY_NAMES[int(value)]
        if feature == "is_weekend":
            return "yes" if int(value) == 1 else "no"
        if feature == "city_pop":
            return f"{int(value):,}"
        if feature in ("customer_age", "transaction_month"):
            return str(int(value))
    except (TypeError, ValueError, IndexError):
        pass

    return str(value)


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

    # Keep the matrix exactly as the model sees it at prediction time.
    # The pipeline feeds XGBoost a sparse matrix, where the zeros of the
    # one-hot columns are treated as "missing". Converting to a dense array
    # turns them into real zeros, which sends the row down different tree
    # branches, so SHAP would explain a prediction the model never made.
    transformed_features = preprocessor.transform(selected_features)

    raw_feature_names = preprocessor.get_feature_names_out()
    feature_names = [clean_feature_name(name) for name in raw_feature_names]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(transformed_features)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    if len(shap_values.shape) == 3:
        shap_values = shap_values[:, :, 1]

    local_values = np.asarray(shap_values[0], dtype=float)

    # Additivity check: SHAP values plus the base value must add up to the
    # model's own output (in log-odds). If they don't, the explanation does
    # not describe this prediction and must not be shown.
    probability = float(model.predict_proba(transformed_features)[0, 1])
    model_log_odds = float(np.log(probability / (1 - probability)))
    base_value = float(np.ravel(explainer.expected_value)[0])
    shap_log_odds = base_value + float(local_values.sum())

    if abs(shap_log_odds - model_log_odds) > 0.01:
        raise ValueError(
            "SHAP values do not add up to the model output "
            f"({shap_log_odds:.3f} vs {model_log_odds:.3f})."
        )

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

    grouped_shap["label"] = grouped_shap["base_feature"].apply(format_feature_label)
    grouped_shap["display_value"] = grouped_shap.apply(
        lambda row: format_feature_display_value(
            row["base_feature"], row_values.get(row["base_feature"], "N/A")
        ),
        axis=1,
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