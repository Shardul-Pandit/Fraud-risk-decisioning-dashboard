"""
Tools the Investigation Agent's LLM can call.

Each tool is a read-only lookup that returns facts already computed by
code: SHAP values, the card's transaction history, and policy text. The
LLM decides which ones to call. It cannot change anything through them.
"""
import pandas as pd

from src.preprocessing import clean_merchant_name
from src.policy_retriever import POLICY_SECTION_TITLES, get_policy_section


MAX_RECENT_TRANSACTIONS = 10
MAX_DRIVERS = 5


TOOL_SPECS = [
    {
        "name": "get_shap_drivers",
        "description": (
            "Return the model's SHAP attributions for this transaction: which "
            "features increased the fraud score and which reduced it, with each "
            "feature's value. Use this to explain why the model scored the "
            "transaction the way it did."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "get_recent_transactions",
        "description": (
            "Return the same card's most recent transactions before this one "
            "(time, merchant, category, amount, minutes before this transaction). "
            "Use this to check for unusual velocity or a change in spending pattern."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": f"How many transactions to return (1 to {MAX_RECENT_TRANSACTIONS}).",
                }
            },
        },
    },
    {
        "name": "get_policy_section",
        "description": (
            "Return one section of the written fraud risk policy. Use the "
            "section matching the transaction's risk band, plus analyst_review "
            "or escalation guidance when relevant."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "section": {
                    "type": "string",
                    "enum": sorted(POLICY_SECTION_TITLES),
                }
            },
            "required": ["section"],
        },
    },
]


class InvestigationToolbox:
    """Binds the tools to one transaction."""

    def __init__(
        self,
        transactions: pd.DataFrame,
        selected_index,
        shap_explanation: dict,
    ):
        self.transactions = transactions
        self.selected_index = selected_index
        self.shap_explanation = shap_explanation or {}

    def functions(self) -> dict:
        return {
            "get_shap_drivers": self.get_shap_drivers,
            "get_recent_transactions": self.get_recent_transactions,
            "get_policy_section": self.get_policy_section,
        }

    def get_shap_drivers(self) -> dict:
        if self.shap_explanation.get("error"):
            return {"error": "SHAP explanation is not available for this transaction."}

        def simplify(drivers):
            return [
                {
                    "feature": driver["label"],
                    "value": driver["display_value"],
                    "effect": driver["impact"].lower(),
                }
                for driver in drivers[:MAX_DRIVERS]
            ]

        return {
            "note": "Listed strongest first.",
            "risk_increasing": simplify(
                self.shap_explanation.get("risk_increasing_features", [])
            ),
            "risk_reducing": simplify(
                self.shap_explanation.get("risk_reducing_features", [])
            ),
        }

    def get_recent_transactions(self, limit: int = 5) -> dict:
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            limit = 5
        limit = max(1, min(limit, MAX_RECENT_TRANSACTIONS))

        current = self.transactions.loc[self.selected_index]
        current_time = pd.to_datetime(current["trans_date_trans_time"])

        times = pd.to_datetime(self.transactions["trans_date_trans_time"])
        earlier = self.transactions[
            (self.transactions["cc_num"] == current["cc_num"]) & (times < current_time)
        ].copy()

        if earlier.empty:
            return {
                "earlier_transactions_available": 0,
                "transactions": [],
                "note": "No earlier transactions for this card in the available data.",
            }

        earlier["_time"] = pd.to_datetime(earlier["trans_date_trans_time"])
        recent = earlier.sort_values("_time", ascending=False).head(limit)

        return {
            "earlier_transactions_available": int(len(earlier)),
            "transactions": [
                {
                    "time": row["_time"].strftime("%Y-%m-%d %H:%M"),
                    "merchant": clean_merchant_name(row["merchant"]),
                    "category": str(row["category"]),
                    "amount": f"${float(row['amt']):,.2f}",
                    "minutes_before_this_transaction": (
                        f"{(current_time - row['_time']).total_seconds() / 60:,.1f}"
                    ),
                }
                for _, row in recent.iterrows()
            ],
        }

    def get_policy_section(self, section: str) -> dict:
        return {"section": section, "text": get_policy_section(section)}
