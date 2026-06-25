import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


TARGET_COLUMN = "is_fraud"

NUMERIC_FEATURES = [
    "amt",
    "city_pop",
    "transaction_hour",
    "transaction_day_of_week",
    "transaction_month",
    "is_weekend",
    "customer_age",
    "distance_from_home_km",
    "time_since_last_transaction_minutes",
    "distance_from_last_transaction_km",
    "implied_travel_speed_kmh",
]

CATEGORICAL_FEATURES = [
    "category",
    "gender",
    "state",
    "job",
]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def haversine_distance_km(lat1, lon1, lat2, lon2):
    """
    Calculate distance in kilometers between two latitude/longitude points.
    """
    earth_radius_km = 6371

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arcsin(np.sqrt(a))

    return earth_radius_km * c


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create fraud detection features from the real-world-style transaction dataset.

    Important:
    - cc_num is used only to calculate previous-transaction behavior.
    - Raw personal identifiers are not kept as model features.
    """
    df = df.copy()

    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["dob"] = pd.to_datetime(df["dob"])

    df["transaction_hour"] = df["trans_date_trans_time"].dt.hour
    df["transaction_day_of_week"] = df["trans_date_trans_time"].dt.dayofweek
    df["transaction_month"] = df["trans_date_trans_time"].dt.month
    df["is_weekend"] = df["transaction_day_of_week"].isin([5, 6]).astype(int)

    df["customer_age"] = (
        df["trans_date_trans_time"].dt.year - df["dob"].dt.year
    )

    df["distance_from_home_km"] = haversine_distance_km(
        df["lat"],
        df["long"],
        df["merch_lat"],
        df["merch_long"],
    )

    df = df.sort_values(["cc_num", "trans_date_trans_time"])

    df["previous_transaction_time"] = df.groupby("cc_num")[
        "trans_date_trans_time"
    ].shift(1)

    df["previous_merch_lat"] = df.groupby("cc_num")["merch_lat"].shift(1)
    df["previous_merch_long"] = df.groupby("cc_num")["merch_long"].shift(1)

    df["time_since_last_transaction_minutes"] = (
        (
            df["trans_date_trans_time"] - df["previous_transaction_time"]
        ).dt.total_seconds()
        / 60
    )

    df["distance_from_last_transaction_km"] = haversine_distance_km(
        df["previous_merch_lat"],
        df["previous_merch_long"],
        df["merch_lat"],
        df["merch_long"],
    )

    df["implied_travel_speed_kmh"] = (
        df["distance_from_last_transaction_km"]
        / (df["time_since_last_transaction_minutes"] / 60)
    )

    df["time_since_last_transaction_minutes"] = (
        df["time_since_last_transaction_minutes"].fillna(999999)
    )

    df["distance_from_last_transaction_km"] = (
        df["distance_from_last_transaction_km"].fillna(0)
    )

    df["implied_travel_speed_kmh"] = (
        df["implied_travel_speed_kmh"]
        .replace([np.inf, -np.inf], 999999)
        .fillna(0)
    )

    return df


def prepare_model_data(df: pd.DataFrame):
    """
    Prepare X and y for model training.
    """
    df = engineer_features(df)

    X = df[MODEL_FEATURES]
    y = df[TARGET_COLUMN]

    return X, y


def split_model_data(X, y, test_size=0.2, random_state=42):
    """
    Create train/test split while preserving fraud class imbalance.
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )