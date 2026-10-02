from src.config import ROOT_DIR


POLICY_PATH = ROOT_DIR / "data" / "policies" / "fraud_policy.md"


def load_fraud_policy() -> str:
    """
    Load the local fraud policy markdown file.

    Returns:
        str: Fraud policy text.
    """
    if not POLICY_PATH.exists():
        raise FileNotFoundError(
            f"Fraud policy file not found at {POLICY_PATH}"
        )

    return POLICY_PATH.read_text(encoding="utf-8")


POLICY_SECTION_TITLES = {
    "low_risk": "Low Risk",
    "medium_risk": "Medium Risk",
    "high_risk": "High Risk",
    "false_positives_and_negatives": "False Positives and False Negatives",
    "analyst_review": "Analyst Review Guidance",
    "escalation": "Escalation Guidance",
}


def get_policy_section(section: str) -> str:
    """
    Return one section of fraud_policy.md by key (see POLICY_SECTION_TITLES).

    The text is read from the policy file, so editing the policy changes
    what the Investigation Agent retrieves without touching code.
    """
    title = POLICY_SECTION_TITLES.get(section)

    if title is None:
        raise ValueError(
            f"Unknown policy section '{section}'. "
            f"Choose one of: {sorted(POLICY_SECTION_TITLES)}"
        )

    lines = []
    in_section = False

    for line in load_fraud_policy().splitlines():
        if line.startswith("#"):
            if in_section:
                break
            in_section = line.lstrip("#").strip() == title
            continue

        if in_section and line.strip() and line.strip() != "---":
            lines.append(line.strip())

    if not lines:
        raise ValueError(f"Policy section '{title}' not found in {POLICY_PATH.name}")

    return " ".join(lines)


def retrieve_policy_section(recommendation: str) -> str:
    """
    Retrieve the most relevant policy section based on the recommendation.

    Args:
        recommendation (str): Approve, Manual Review, or Deny.

    Returns:
        str: Relevant policy guidance.
    """
    policy_text = load_fraud_policy()

    if recommendation == "Approve":
        return (
            "Relevant Policy: Low Risk\n\n"
            "Transactions with fraud probability below 0.30 are considered low risk. "
            "The recommended action is to approve the transaction and continue standard monitoring."
        )

    if recommendation == "Manual Review":
        return (
            "Relevant Policy: Medium Risk\n\n"
            "Transactions with fraud probability greater than or equal to 0.30 and below 0.80 "
            "are considered medium risk. The recommended action is to send the transaction to manual review."
        )

    if recommendation == "Deny":
        return (
            "Relevant Policy: High Risk\n\n"
            "Transactions with fraud probability greater than or equal to 0.80 are considered high risk. "
            "The recommended action is to deny the transaction and escalate the case to fraud operations."
        )

    return (
        "Relevant Policy: General Fraud Risk Policy\n\n"
        "Fraud probability scores should be used with threshold-based decisioning to balance "
        "fraud prevention and customer friction."
    )