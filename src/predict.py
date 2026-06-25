from pathlib import Path

import joblib
import pandas as pd

from src.preprocessing import engineer_features, MODEL_FEATURES


MODEL_PATH = Path("models") / "real_world_xgboost_model.joblib"


def load_model():
    """
    Load the trained real-world fraud detection model pipeline.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Please train the model first using: python -m src.train_model"
        )

    return joblib.load(MODEL_PATH)


def assign_risk_band(fraud_probability: float) -> str:
    """
    Convert fraud probability into a risk band.
    """
    if fraud_probability >= 0.80:
        return "High Risk"
    if fraud_probability >= 0.30:
        return "Medium Risk"
    return "Low Risk"


def assign_recommendation(fraud_probability: float) -> str:
    """
    Convert fraud probability into a business recommendation.
    """
    if fraud_probability >= 0.80:
        return "Deny"
    if fraud_probability >= 0.30:
        return "Manual Review"
    return "Approve"


def build_prediction_response(fraud_probability: float) -> dict:
    """
    Build the standard model response dictionary.
    """
    fraud_probability = float(fraud_probability)

    return {
        "fraud_probability": fraud_probability,
        "risk_band": assign_risk_band(fraud_probability),
        "recommendation": assign_recommendation(fraud_probability),
    }


def predict_from_model_features(model_features: pd.DataFrame) -> dict:
    """
    Predict fraud risk from already-engineered model features.

    Args:
        model_features: DataFrame with exactly one row and MODEL_FEATURES columns.

    Returns:
        Dictionary with fraud probability, risk band, and recommendation.
    """
    if not isinstance(model_features, pd.DataFrame):
        raise TypeError("model_features must be a pandas DataFrame.")

    if len(model_features) != 1:
        raise ValueError("model_features must contain exactly one row.")

    missing_features = [
        col for col in MODEL_FEATURES if col not in model_features.columns
    ]

    if missing_features:
        raise ValueError(f"Missing model features: {missing_features}")

    model = load_model()

    X = model_features[MODEL_FEATURES]
    fraud_probability = model.predict_proba(X)[:, 1][0]

    return build_prediction_response(fraud_probability)


def predict_transaction_risk(transaction: pd.DataFrame) -> dict:
    """
    Predict fraud risk for one raw transaction row.

    This works for single-row prediction, but previous-transaction features
    will use fallback values because only one transaction is available.
    """
    if not isinstance(transaction, pd.DataFrame):
        raise TypeError("transaction must be a pandas DataFrame.")

    if len(transaction) != 1:
        raise ValueError("transaction must contain exactly one row.")

    engineered_transaction = engineer_features(transaction)
    model_features = engineered_transaction[MODEL_FEATURES]

    return predict_from_model_features(model_features)


def predict_transaction_with_context(
    transactions: pd.DataFrame,
    selected_index: int,
) -> dict:
    """
    Predict fraud risk for one transaction using the full transaction dataset
    as context.

    This allows previous-transaction features like:
    - time_since_last_transaction_minutes
    - distance_from_last_transaction_km
    - implied_travel_speed_kmh

    to be calculated correctly before selecting the row.
    """
    if not isinstance(transactions, pd.DataFrame):
        raise TypeError("transactions must be a pandas DataFrame.")

    if selected_index not in transactions.index:
        raise ValueError(f"selected_index {selected_index} not found in transactions.")

    engineered_transactions = engineer_features(transactions)
    selected_features = engineered_transactions.loc[[selected_index], MODEL_FEATURES]

    return predict_from_model_features(selected_features)