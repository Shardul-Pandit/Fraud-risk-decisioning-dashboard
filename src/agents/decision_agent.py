class DecisionAgent:
    """
    Agent responsible for creating the final fraud risk decision.
    """

    def create_final_decision(
        self,
        risk_result: dict,
        investigation_result: dict,
        policy_result: dict,
    ) -> dict:
        """
        Combine model score, investigation summary, and policy guidance into
        a final business decision.
        """
        recommendation = risk_result["recommendation"]
        risk_band = risk_result["risk_band"]
        fraud_probability = risk_result["fraud_probability"]

        investigation_summary = investigation_result["investigation_summary"]
        policy_guidance = policy_result["policy_guidance"]

        business_action = self._map_business_action(recommendation)

        final_report = self._build_final_report(
            final_recommendation=recommendation,
            risk_band=risk_band,
            fraud_probability=fraud_probability,
            business_action=business_action,
            policy_guidance=policy_guidance,
            investigation_summary=investigation_summary,
        )

        return {
            "agent_name": "Decision Agent",
            "final_recommendation": recommendation,
            "risk_band": risk_band,
            "fraud_probability": fraud_probability,
            "business_action": business_action,
            "policy_guidance": policy_guidance,
            "investigation_summary": investigation_summary,
            "final_report": final_report,
        }

    def _map_business_action(self, recommendation: str) -> str:
        """
        Convert a decision category into an operational business action.
        """
        if recommendation == "Deny":
            return "Block transaction and escalate to fraud operations"

        if recommendation == "Manual Review":
            return "Hold transaction for manual fraud analyst review"

        if recommendation == "Approve":
            return "Approve transaction and continue standard monitoring"

        return "Review transaction decision manually"

    def _build_final_report(
        self,
        final_recommendation: str,
        risk_band: str,
        fraud_probability: float,
        business_action: str,
        policy_guidance: str,
        investigation_summary: str,
    ) -> str:
        """
        Build a readable final fraud decision report.
        """
        return (
            "\nFinal Fraud Risk Decision Report\n"
            f"Final Recommendation: {final_recommendation}\n"
            f"Risk Band: {risk_band}\n"
            f"Fraud Probability: {fraud_probability:.4f}\n"
            f"Business Action: {business_action}\n\n"
            f"Policy Support:\n{policy_guidance}\n\n"
            f"Investigation Summary:\n{investigation_summary}\n"
        )