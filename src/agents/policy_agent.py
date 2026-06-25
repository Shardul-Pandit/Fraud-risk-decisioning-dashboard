from src.policy_retriever import retrieve_policy_section


class PolicyAgent:
    """
    Agent responsible for retrieving fraud policy guidance
    that supports the model's business recommendation.
    """

    def review_policy(self, risk_result: dict) -> dict:
        """
        Retrieve policy guidance for the risk recommendation.

        Args:
            risk_result (dict): Output from RiskScoringAgent.

        Returns:
            dict: Policy review result.
        """
        recommendation = risk_result["recommendation"]
        fraud_probability = risk_result["fraud_probability"]

        policy_guidance = retrieve_policy_section(recommendation)

        return {
            "agent_name": "Policy Agent",
            "recommendation": recommendation,
            "fraud_probability": fraud_probability,
            "policy_guidance": policy_guidance
        }