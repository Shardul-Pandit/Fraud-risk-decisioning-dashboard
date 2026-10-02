# Legacy notebooks

These notebooks are from the first version of this project, which used the
anonymized Kaggle credit card dataset (`creditcard.csv`, 284,807 transactions,
features V1 to V28). The project later moved to the `fraudTest.csv` dataset
described in the main README, because its named features (amount, merchant
category, location, time) allow feature engineering and readable explanations.

They are kept for reference only. Their numbers do not describe the deployed
model, and they are not maintained: they were written when `load_raw_data()`
returned the anonymized dataset (now `load_legacy_data()`). The models they
produced are in `models/legacy/`.
