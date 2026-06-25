from src.data_loader import load_raw_data
from src.input_validator import validate_transaction_dataframe
from src.agents.fraud_decision_workflow import (
    run_fraud_decision_workflow_with_context,
)


DEMO_TRANSACTIONS = {
    "Low-Risk Example": 0,
    "Medium-Risk Example": 1685,
    "High-Risk Example": 306221,
}


def print_workflow_result(label: str, result: dict):
    """
    Print a clean fraud decision workflow result.
    """
    risk_result = result["risk_result"]
    investigation_result = result["investigation_result"]
    policy_result = result["policy_result"]
    final_decision = result["final_decision"]

    print("=" * 80)
    print(label)
    print("=" * 80)

    print(f"Fraud Probability: {risk_result['fraud_probability']:.4f}")
    print(f"Risk Band: {risk_result['risk_band']}")
    print(f"Final Recommendation: {final_decision['final_recommendation']}")
    print(f"Business Action: {final_decision['business_action']}")
    print()

    print("AI Analyst Summary:")
    print(investigation_result["summary"])
    print()

    print("Policy Guidance:")
    print(policy_result["policy_guidance"])
    print()


def main():
    """
    Run demo fraud decision workflow examples.
    """
    print("Loading real-world fraud transaction dataset...")
    df = load_raw_data()

    is_valid, message, clean_df = validate_transaction_dataframe(df)

    if not is_valid:
        print("Dataset validation failed.")
        print(message)
        return

    print(message)
    print(f"Dataset shape: {clean_df.shape}")
    print()

    # Drop label column if present because prediction uses raw transaction features only.
    transactions = clean_df.drop(columns=["is_fraud"], errors="ignore")

    for label, index in DEMO_TRANSACTIONS.items():
        result = run_fraud_decision_workflow_with_context(
            transactions=transactions,
            selected_index=index,
        )

        print_workflow_result(label, result)


if __name__ == "__main__":
    main()