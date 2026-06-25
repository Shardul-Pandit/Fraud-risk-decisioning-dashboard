TOP_EDA_RISK_SIGNALS = [
    "V17",
    "V14",
    "V12",
    "V10",
    "V16",
    "V3",
    "V7"
]


RULE_BASED_INVESTIGATION_TEMPLATE = """
Transaction Risk Investigation Summary

Fraud Probability: {fraud_probability:.4f}
Risk Band: {risk_band}
Initial Recommendation: {recommendation}

Key Risk Signals:
{risk_signals}

Analysis:
{analysis}

Suggested Action:
{suggested_action}
"""