import pandas as pd

from src.predict import predict_transaction_risk, predict_transaction_with_context


class RiskScoringAgent:
    """
    Agent responsible for scoring transaction fraud risk using the trained ML model.
    """

    def score_transaction(self, transaction: pd.DataFrame) -> dict:
        """
        Score a single raw transaction row.

        This works for one-row prediction, but previous-transaction features
        will use fallback values because only one transaction is available.
        """
        prediction = predict_transaction_risk(transaction)

        return {
            "agent_name": "Risk Scoring Agent",
            "fraud_probability": prediction["fraud_probability"],
            "risk_band": prediction["risk_band"],
            "recommendation": prediction["recommendation"],
        }

    def score_transaction_with_context(
        self,
        transactions: pd.DataFrame,
        selected_index: int,
    ) -> dict:
        """
        Score one transaction using the full transaction dataset as context.

        This allows previous transaction features like travel speed and
        time since last transaction to be calculated correctly.
        """
        prediction = predict_transaction_with_context(
            transactions=transactions,
            selected_index=selected_index,
        )

        return {
            "agent_name": "Risk Scoring Agent",
            "fraud_probability": prediction["fraud_probability"],
            "risk_band": prediction["risk_band"],
            "recommendation": prediction["recommendation"],
        }