import pandas as pd

from src.config import ROOT_DIR, REAL_WORLD_DATA_PATH, LEGACY_RAW_DATA_PATH


SAMPLE_DATA_PATH = ROOT_DIR / "data" / "sample" / "fraud_sample.csv"


def load_real_world_data() -> pd.DataFrame:
    """
    Load the bundled real-world-style fraud transaction sample dataset.

    The sample preserves the original row indices (in the "Unnamed: 0" column),
    so demo transaction indices continue to resolve correctly.

    Returns:
        pd.DataFrame: Raw fraud transaction dataset.
    """
    if not SAMPLE_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {SAMPLE_DATA_PATH}. "
            "Please place fraud_sample.csv inside data/sample/."
        )

    df = pd.read_csv(SAMPLE_DATA_PATH, index_col="Unnamed: 0")
    df.index.name = None
    return df


def load_legacy_data() -> pd.DataFrame:
    """
    Load the legacy anonymized Kaggle credit card fraud dataset.

    Returns:
        pd.DataFrame: Raw anonymized fraud dataset.
    """
    if not LEGACY_RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {LEGACY_RAW_DATA_PATH}. "
            "Please place creditcard.csv inside data/raw/."
        )

    df = pd.read_csv(LEGACY_RAW_DATA_PATH)
    return df


def load_raw_data() -> pd.DataFrame:
    """
    Default data loader for the project.

    The project now uses the real-world-style fraud dataset by default.
    """
    return load_real_world_data()