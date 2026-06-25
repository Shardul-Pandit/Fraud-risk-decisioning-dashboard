import pandas as pd

from src.preprocessing import engineer_features


class InvestigationAgent:
    """
    Agent responsible for generating a fraud analyst-style investigation summary.

    This rule-based version works without an API key.
    Later, this can be upgraded to use an LLM when an API key is available.
    """

    def _km_to_miles(self, km: float) -> float:
        """
        Convert kilometers to miles for explanation text.
    """
        return km * 0.621371

    def investigate(
        self,
        transaction: pd.DataFrame,
        risk_result: dict,
    ) -> dict:
        """
        Generate an analyst-style investigation summary for one transaction.

        Args:
            transaction: DataFrame containing exactly one raw transaction row.
            risk_result: Output from the Risk Scoring Agent.

        Returns:
            Dictionary containing investigation summary and key risk signals.
        """
        if not isinstance(transaction, pd.DataFrame):
            raise TypeError("transaction must be a pandas DataFrame.")

        if len(transaction) != 1:
            raise ValueError("transaction must contain exactly one row.")

        engineered = engineer_features(transaction)
        row = engineered.iloc[0]

        fraud_probability = risk_result["fraud_probability"]
        risk_band = risk_result["risk_band"]
        recommendation = risk_result["recommendation"]

        amount = float(row.get("amt", 0))
        category = row.get("category", "unknown")
        merchant = row.get("merchant", "unknown")
        state = row.get("state", "unknown")
        transaction_hour = int(row.get("transaction_hour", 0))
        customer_age = int(row.get("customer_age", 0))
        distance_from_home = float(row.get("distance_from_home_km", 0))
        time_since_last = float(row.get("time_since_last_transaction_minutes", 0))
        distance_from_last = float(row.get("distance_from_last_transaction_km", 0))
        travel_speed = float(row.get("implied_travel_speed_kmh", 0))

        key_risk_signals = self._build_key_risk_signals(
            amount=amount,
            category=category,
            transaction_hour=transaction_hour,
            distance_from_home=distance_from_home,
            time_since_last=time_since_last,
            distance_from_last=distance_from_last,
            travel_speed=travel_speed,
        )

        summary = self._build_summary(
            fraud_probability=fraud_probability,
            risk_band=risk_band,
            recommendation=recommendation,
            amount=amount,
            category=category,
            merchant=merchant,
            state=state,
            transaction_hour=transaction_hour,
            customer_age=customer_age,
            distance_from_home=distance_from_home,
            time_since_last=time_since_last,
            distance_from_last=distance_from_last,
            travel_speed=travel_speed,
            key_risk_signals=key_risk_signals,
        )

        return {
            "agent_name": "Investigation Agent",
            "summary": summary,
            "investigation_summary": summary,
            "key_risk_signals": key_risk_signals,
            "suggested_action": recommendation,
        }

    def investigate_with_context(
        self,
        transactions: pd.DataFrame,
        selected_index: int,
        risk_result: dict,
    ) -> dict:
        """
        Generate an investigation summary using the full transaction dataset as context.

        This allows previous-transaction features to be calculated correctly.
        """
        if selected_index not in transactions.index:
            raise ValueError(f"selected_index {selected_index} not found in transactions.")

        engineered = engineer_features(transactions)
        selected = engineered.loc[[selected_index]]

        return self.investigate(
            transaction=selected,
            risk_result=risk_result,
        )

    def _build_key_risk_signals(
        self,
        amount: float,
        category: str,
        transaction_hour: int,
        distance_from_home: float,
        time_since_last: float,
        distance_from_last: float,
        travel_speed: float,
    ) -> list:
        """
        Create human-readable risk signals from transaction features.
        """
        signals = []

        if amount >= 500:
            signals.append(f"High transaction amount (${amount:,.2f})")
        elif amount >= 100:
            signals.append(f"Moderate transaction amount (${amount:,.2f})")

        if transaction_hour < 5 or transaction_hour > 22:
            signals.append(f"Unusual transaction hour ({transaction_hour}:00)")

        distance_from_home_miles = self._km_to_miles(distance_from_home)

        if distance_from_home >= 500:
            signals.append(
                f"Merchant is far from customer home location ({distance_from_home_miles:,.1f} miles)"
            )
        elif distance_from_home >= 100:
            signals.append(
                f"Merchant is moderately far from customer home location ({distance_from_home_miles:,.1f} miles)"
            )

        if time_since_last < 60 and distance_from_last >= 500:
            signals.append(
                "Large location change shortly after previous transaction"
            )

        travel_speed_mph = self._km_to_miles(travel_speed)

        if travel_speed >= 900:
            signals.append(
                f"Implied travel speed appears physically implausible ({travel_speed_mph:,.1f} mph)"
            )
        elif travel_speed >= 300:
            signals.append(
                f"High implied travel speed between transactions ({travel_speed_mph:,.1f} mph)"
            )

        high_attention_categories = {
            "shopping_net",
            "misc_net",
            "grocery_pos",
            "gas_transport",
        }

        if category in high_attention_categories:
            signals.append(f"Merchant category requires closer review ({category})")

        if not signals:
            signals.append(
                "No single rule-based red flag dominated this case; the model score reflects the overall transaction profile"
            )

        return signals

    def _build_summary(
        self,
        fraud_probability: float,
        risk_band: str,
        recommendation: str,
        amount: float,
        category: str,
        merchant: str,
        state: str,
        transaction_hour: int,
        customer_age: int,
        distance_from_home: float,
        time_since_last: float,
        distance_from_last: float,
        travel_speed: float,
        key_risk_signals: list,
    ) -> str:
        """
        Build the final analyst-style explanation.
        """
        probability_percent = fraud_probability * 100

        distance_from_home_miles = self._km_to_miles(distance_from_home)
        distance_from_last_miles = self._km_to_miles(distance_from_last)
        travel_speed_mph = self._km_to_miles(travel_speed)

        signal_text = "; ".join(key_risk_signals)

        if (
            len(key_risk_signals) == 1
            and key_risk_signals[0]
            == "No single rule-based red flag dominated this case; the model score reflects the overall transaction profile"
        ):
            if risk_band == "Medium Risk":
                signal_text = (
                    "Although no single rule-based red flag dominated this case, "
                    "the overall transaction profile produced an elevated model risk score "
                    "and should be reviewed manually"
                )
            elif risk_band == "Low Risk":
                signal_text = (
                    "No single rule-based red flag dominated this case, and the overall "
                    "transaction profile remained within the low-risk range"
                )

        if risk_band == "High Risk":
            action_text = (
                "The recommended action is to deny the transaction and escalate it "
                "for fraud analyst review."
            )
        elif risk_band == "Medium Risk":
            action_text = (
                "The recommended action is to send the transaction to manual review "
                "before approval."
            )
        else:
            action_text = (
                "The recommended action is to approve the transaction while continuing "
                "standard monitoring."
            )

        previous_transaction_text = ""

        if time_since_last < 999999:
            if travel_speed >= 300 or (time_since_last < 60 and distance_from_last >= 500):
                previous_transaction_text = (
                    f" Recent transaction history shows {time_since_last:,.1f} minutes since "
                    f"the last transaction, {distance_from_last_miles:,.1f} miles from the "
                    f"previous merchant location, and an implied travel speed of "
                    f"{travel_speed_mph:,.1f} mph."
                )


        summary = (
            f"This transaction was classified as {risk_band} with a fraud probability "
            f"of {probability_percent:.2f}%. The transaction involved a ${amount:,.2f} "
            f"purchase at {merchant} in the {category} category, located in {state}, "
            f"around hour {transaction_hour}:00. The customer age is approximately "
            f"{customer_age}, and the merchant is {distance_from_home_miles:,.1f} miles from "
            f"the customer home location.{previous_transaction_text} Key risk signals: "
            f"{signal_text}. {action_text}"
        )

        return summary