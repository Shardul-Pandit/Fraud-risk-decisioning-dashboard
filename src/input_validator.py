import pandas as pd


TARGET_COLUMN = "is_fraud"

REQUIRED_RAW_COLUMNS = [
    "trans_date_trans_time",
    "cc_num",
    "merchant",
    "category",
    "amt",
    "gender",
    "city",
    "state",
    "zip",
    "lat",
    "long",
    "city_pop",
    "job",
    "dob",
    "trans_num",
    "unix_time",
    "merch_lat",
    "merch_long",
]

OPTIONAL_COLUMNS = [
    "Unnamed: 0",
    "first",
    "last",
    "street",
    TARGET_COLUMN,
]

NUMERIC_COLUMNS = [
    "cc_num",
    "amt",
    "zip",
    "lat",
    "long",
    "city_pop",
    "unix_time",
    "merch_lat",
    "merch_long",
]


def validate_transaction_dataframe(df: pd.DataFrame):
    """
    Validate a real-world-style fraud transaction dataset.

    The uploaded file should match the fraudTest.csv schema used by the project.
    The target column is optional because user-uploaded prediction files may not
    include labels.
    """
    if df is None:
        return False, "No data was provided.", None

    if not isinstance(df, pd.DataFrame):
        return False, "Input must be a pandas DataFrame.", None

    if df.empty:
        return False, "The uploaded file is empty.", None

    cleaned_df = df.copy()

    missing_columns = [
        column for column in REQUIRED_RAW_COLUMNS
        if column not in cleaned_df.columns
    ]

    if missing_columns:
        return (
            False,
            (
                "The uploaded file does not match the expected real-world fraud "
                f"transaction schema. Missing required columns: {missing_columns}. "
                "Expected columns include transaction time, card number, merchant, "
                "category, amount, customer location, merchant location, job, date "
                "of birth, and transaction identifiers."
            ),
            None,
        )

    invalid_numeric_columns = []

    for column in NUMERIC_COLUMNS:
        converted = pd.to_numeric(cleaned_df[column], errors="coerce")

        if converted.isnull().any():
            invalid_numeric_columns.append(column)
        else:
            cleaned_df[column] = converted

    if invalid_numeric_columns:
        return (
            False,
            (
                "The uploaded file contains non-numeric values in columns that "
                f"must be numeric: {invalid_numeric_columns}."
            ),
            None,
        )

    date_columns = ["trans_date_trans_time", "dob"]

    invalid_date_columns = []

    for column in date_columns:
        converted = pd.to_datetime(cleaned_df[column], errors="coerce")

        if converted.isnull().any():
            invalid_date_columns.append(column)
        else:
            cleaned_df[column] = converted

    if invalid_date_columns:
        return (
            False,
            (
                "The uploaded file contains invalid date/time values in columns: "
                f"{invalid_date_columns}."
            ),
            None,
        )

    if cleaned_df[REQUIRED_RAW_COLUMNS].isnull().sum().sum() > 0:
        return (
            False,
            "The uploaded file contains missing values in required columns.",
            None,
        )

    return (
        True,
        "Transaction data is valid and ready for fraud risk scoring.",
        cleaned_df,
    )


def validate_single_transaction(transaction: pd.DataFrame):
    """
    Validate that the input contains exactly one transaction row.
    """
    is_valid, message, cleaned_df = validate_transaction_dataframe(transaction)

    if not is_valid:
        return is_valid, message, cleaned_df

    if len(cleaned_df) != 1:
        return (
            False,
            "A single transaction prediction requires exactly one row.",
            None,
        )

    return True, "Single transaction is valid.", cleaned_df