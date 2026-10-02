import numpy as np
import pandas as pd
import pytest

from src.input_validator import validate_transaction_dataframe
from src.policy_retriever import POLICY_SECTION_TITLES, get_policy_section
from src.predict import assign_recommendation, assign_risk_band, predict_transaction_with_context
from src.preprocessing import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    clean_merchant_name,
    engineer_features,
    haversine_distance_km,
)
from src.shap_explainer import generate_local_shap_explanation


def test_model_does_not_use_gender_or_job():
    assert "gender" not in MODEL_FEATURES
    assert "job" not in MODEL_FEATURES
    assert len(MODEL_FEATURES) == 13
    assert len(NUMERIC_FEATURES) == 11 and len(CATEGORICAL_FEATURES) == 2


def test_upload_without_gender_or_job_is_valid(transactions):
    is_valid, message, _ = validate_transaction_dataframe(
        transactions.drop(columns=["gender", "job"])
    )
    assert is_valid, message


def test_haversine_known_distance():
    # New York to Los Angeles is about 3,936 km.
    distance = haversine_distance_km(40.7128, -74.0060, 34.0522, -118.2437)
    assert distance == pytest.approx(3936, rel=0.01)


def test_engineered_features_have_no_missing_or_infinite_values(transactions):
    engineered = engineer_features(transactions)[NUMERIC_FEATURES]
    assert not engineered.isnull().any().any()
    assert np.isfinite(engineered.to_numpy(dtype=float)).all()


def test_first_transaction_on_a_card_uses_fallback_values():
    row = pd.DataFrame(
        {
            "trans_date_trans_time": ["2020-06-21 12:14:25"],
            "cc_num": [1],
            "dob": ["1990-01-01"],
            "lat": [40.0], "long": [-74.0], "merch_lat": [40.1], "merch_long": [-74.1],
        }
    )
    engineered = engineer_features(row).iloc[0]
    assert engineered["time_since_last_transaction_minutes"] == 999999
    assert engineered["implied_travel_speed_kmh"] == 0
    assert engineered["customer_age"] == 30


@pytest.mark.parametrize(
    "probability, band, recommendation",
    [
        (0.0, "Low Risk", "Approve"),
        (0.2999, "Low Risk", "Approve"),
        (0.30, "Medium Risk", "Manual Review"),
        (0.7999, "Medium Risk", "Manual Review"),
        (0.80, "High Risk", "Deny"),
        (1.0, "High Risk", "Deny"),
    ],
)
def test_decision_thresholds(probability, band, recommendation):
    assert assign_risk_band(probability) == band
    assert assign_recommendation(probability) == recommendation


@pytest.mark.parametrize(
    "index, band", [(0, "Low Risk"), (1685, "Medium Risk"), (306221, "High Risk")]
)
def test_demo_transactions_land_in_their_bands(transactions, index, band):
    assert predict_transaction_with_context(transactions, index)["risk_band"] == band


@pytest.mark.parametrize("index", [0, 1685, 306221])
def test_shap_explains_every_feature_exactly_once(transactions, index):
    # generate_local_shap_explanation raises if SHAP values do not add up
    # to the model output, so a clean return is the additivity check.
    explanation = generate_local_shap_explanation(transactions, index, top_n=20)
    features = [d["base_feature"] for d in explanation["top_features"]]
    assert sorted(features) == sorted(MODEL_FEATURES)


def test_every_policy_section_is_found_in_the_policy_file():
    for section in POLICY_SECTION_TITLES:
        assert len(get_policy_section(section)) > 40

    with pytest.raises(ValueError):
        get_policy_section("not_a_section")


def test_merchant_prefix_is_stripped_for_display_only(transactions):
    assert clean_merchant_name("fraud_Kiehn Inc") == "Kiehn Inc"
    assert clean_merchant_name("Kiehn Inc") == "Kiehn Inc"
    # The stored data keeps the original value.
    assert transactions["merchant"].str.startswith("fraud_").all()
