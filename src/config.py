from pathlib import Path

# Project root directory
ROOT_DIR = Path(__file__).resolve().parents[1]

# Data directories
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
DATABASE_DIR = ROOT_DIR / "data" / "database"

# Legacy anonymized dataset
LEGACY_RAW_DATA_PATH = RAW_DATA_DIR / "creditcard.csv"
LEGACY_TARGET_COLUMN = "Class"

# Real-World fraud dataset
REAL_WORLD_DATA_PATH = RAW_DATA_DIR / "fraudTest.csv"
TARGET_COLUMN = "is_fraud"

# Model directory
MODEL_DIR = ROOT_DIR / "models"

# Reports directory
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = ROOT_DIR / "figures"

# Database path
DATABASE_PATH = DATABASE_DIR / "fraud_detection.db"

# Random seed for reproducibility
RANDOM_STATE = 42

