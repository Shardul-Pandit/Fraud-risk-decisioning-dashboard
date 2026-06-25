import pandas as pd

from src.agents.risk_scoring_agent import RiskScoringAgent
from src.agents.investigation_agent import InvestigationAgent
from src.agents.policy_agent import PolicyAgent
from src.agents.decision_agent import DecisionAgent
from src.shap_explainer import generate_local_shap_explanation


class FraudDecisionWorkflow:
    """
    Multi-agent workflow for fraud risk decisioning.

    Workflow:
    1. Risk Scoring Agent predicts fraud probability.
    2. Investigation Agent creates analyst-style explanation.
    3. Policy Agent retrieves matching policy guidance.
    4. Decision Agent creates final business recommendation.
    """

    def __init__(self):
        self.risk_scoring_agent = RiskScoringAgent()
        self.investigation_agent = InvestigationAgent()
        self.policy_agent = PolicyAgent()
        self.decision_agent = DecisionAgent()

    def run(self, transaction: pd.DataFrame) -> dict:
        """
        Run the workflow for a single raw transaction row.

        This works for one-row prediction, but previous-transaction features
        use fallback values because only one transaction is available.
        """
        risk_result = self.risk_scoring_agent.score_transaction(transaction)

        investigation_result = self.investigation_agent.investigate(
            transaction=transaction,
            risk_result=risk_result,
        )

        policy_result = self.policy_agent.review_policy(risk_result)

        final_decision = self.decision_agent.create_final_decision(
            risk_result=risk_result,
            investigation_result=investigation_result,
            policy_result=policy_result,
        )

        return {
            "workflow_name": "Fraud Decision Workflow",
            "risk_result": risk_result,
            "investigation_result": investigation_result,
            "policy_result": policy_result,
            "final_decision": final_decision,
        }

    def run_with_context(
        self,
        transactions: pd.DataFrame,
        selected_index: int,
    ) -> dict:
        """
        Run the workflow for one selected transaction using the full transaction
        dataset as context.

        This allows previous-transaction features such as time since last
        transaction, distance from previous merchant, and implied travel speed
        to be calculated correctly.
        """
        if not isinstance(transactions, pd.DataFrame):
            raise TypeError("transactions must be a pandas DataFrame.")

        if selected_index not in transactions.index:
            raise ValueError(f"selected_index {selected_index} not found in transactions.")

        selected_transaction = transactions.loc[[selected_index]]

        risk_result = self.risk_scoring_agent.score_transaction_with_context(
            transactions=transactions,
            selected_index=selected_index,
        )

        try:
            shap_explanation = generate_local_shap_explanation(
                transactions=transactions,
                selected_index=selected_index,
                top_n=6,
            )
        except Exception as error:
            shap_explanation = {
            "error": str(error),
            "top_features": [],
                "risk_increasing_features": [],
                "risk_reducing_features": [],
            }

        investigation_result = self.investigation_agent.investigate_with_context(
            transactions=transactions,
            selected_index=selected_index,
            risk_result=risk_result,
        )

        policy_result = self.policy_agent.review_policy(risk_result)

        final_decision = self.decision_agent.create_final_decision(
            risk_result=risk_result,
            investigation_result=investigation_result,
            policy_result=policy_result,
        )

        return {
            "workflow_name": "Fraud Decision Workflow",
            "selected_index": selected_index,
            "selected_transaction": selected_transaction,
            "risk_result": risk_result,
            "investigation_result": investigation_result,
            "policy_result": policy_result,
            "final_decision": final_decision,
            "shap_explanation": shap_explanation,
        }


def run_fraud_decision_workflow(transaction: pd.DataFrame) -> dict:
    """
    Convenience function for running the workflow on a single transaction.
    """
    workflow = FraudDecisionWorkflow()
    return workflow.run(transaction)


def run_fraud_decision_workflow_with_context(
    transactions: pd.DataFrame,
    selected_index: int,
) -> dict:
    """
    Convenience function for running the workflow on a selected transaction
    using full transaction history context.
    """
    workflow = FraudDecisionWorkflow()
    return workflow.run_with_context(
        transactions=transactions,
        selected_index=selected_index,
    )