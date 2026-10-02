import json
import re

import pandas as pd

from src.agents.investigation_tools import TOOL_SPECS, InvestigationToolbox
from src.llm_client import (
    DEFAULT_MAX_PER_DAY,
    DEFAULT_MAX_PER_MINUTE,
    DEFAULT_MAX_TOOL_STEPS,
    LLMError,
    RateLimiter,
    SummaryCache,
    build_provider_chain,
    get_setting,
    run_with_failover,
)
from src.preprocessing import clean_merchant_name
from src.prompts import INVESTIGATION_SYSTEM_PROMPT, INVESTIGATION_USER_PROMPT


TEMPLATE_MODE = "Rule-based template"
MAX_SIGNALS = 3

# Shared by every session in the app process.
_summary_cache = SummaryCache()
_minute_limiter = RateLimiter(
    int(get_setting("LLM_MAX_PER_MINUTE", DEFAULT_MAX_PER_MINUTE)), 60
)
_day_limiter = RateLimiter(
    int(get_setting("LLM_MAX_PER_DAY", DEFAULT_MAX_PER_DAY)), 24 * 60 * 60
)

_NUMBER_PATTERN = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _numbers_in(text: str) -> set:
    numbers = set()
    for token in _NUMBER_PATTERN.findall(text):
        try:
            numbers.add(float(token.replace(",", "")))
        except ValueError:
            continue
    return numbers


def find_ungrounded_numbers(summary: str, facts: str) -> list:
    """
    Return numbers that appear in the LLM summary but not in the facts it
    was given (the prompt plus every tool result).

    This is how "the LLM never computes or changes a number" is enforced
    rather than just requested: a summary with an invented or recalculated
    figure is rejected and the next provider or the template is used.
    """
    allowed = _numbers_in(facts)
    return sorted(number for number in _numbers_in(summary) if number not in allowed)


class InvestigationAgent:
    """
    Writes the analyst findings for one scored transaction.

    Two modes, same inputs:
    - LLM mode: the model chooses which lookups to run (SHAP drivers, recent
      card activity, policy text) and writes the findings from the results.
    - Template mode: a deterministic summary built from the same SHAP
      output. Used when no API key is set, the API fails, the rate limit is
      hit, or the LLM's answer fails the number check.

    In both modes the risk signals come from SHAP, so the text can never
    disagree with the model explanation shown next to it. The agent does
    not decide anything: probability, band and decision are inputs.
    """

    def __init__(
        self,
        providers=None,
        cache=None,
        limiters=None,
        max_steps=None,
    ):
        self._providers = providers
        self._cache = cache if cache is not None else _summary_cache
        self._limiters = (
            limiters if limiters is not None else [_minute_limiter, _day_limiter]
        )
        self._max_steps = max_steps or int(
            get_setting("LLM_MAX_TOOL_STEPS", DEFAULT_MAX_TOOL_STEPS)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def investigate(
        self,
        transaction: pd.DataFrame,
        risk_result: dict,
        shap_explanation: dict | None = None,
    ) -> dict:
        """Investigate a single transaction row with no card history."""
        if not isinstance(transaction, pd.DataFrame):
            raise TypeError("transaction must be a pandas DataFrame.")

        if len(transaction) != 1:
            raise ValueError("transaction must contain exactly one row.")

        return self.investigate_with_context(
            transactions=transaction,
            selected_index=transaction.index[0],
            risk_result=risk_result,
            shap_explanation=shap_explanation,
        )

    def investigate_with_context(
        self,
        transactions: pd.DataFrame,
        selected_index,
        risk_result: dict,
        shap_explanation: dict | None = None,
    ) -> dict:
        if selected_index not in transactions.index:
            raise ValueError(f"selected_index {selected_index} not found in transactions.")

        shap_explanation = shap_explanation or {
            "error": "SHAP explanation was not provided.",
            "risk_increasing_features": [],
            "risk_reducing_features": [],
        }
        row = transactions.loc[selected_index]
        facts = self._transaction_facts(row, risk_result)

        key_risk_signals = self._build_key_risk_signals(shap_explanation)
        template_summary = self._build_template_summary(
            facts, risk_result, shap_explanation
        )

        result = {
            "agent_name": "Investigation Agent",
            "summary": template_summary,
            "investigation_summary": template_summary,
            "key_risk_signals": key_risk_signals,
            "suggested_action": risk_result["recommendation"],
            "summary_mode": TEMPLATE_MODE,
            "fallback_reason": None,
            "tool_trace": [],
            "llm_steps": 0,
            "from_cache": False,
        }

        cache_key = (
            str(row.get("trans_num", selected_index)),
            round(float(risk_result["fraud_probability"]), 6),
        )
        cached = self._cache.get(cache_key)

        if cached is not None:
            return {**result, **cached, "from_cache": True}

        providers = (
            self._providers if self._providers is not None else build_provider_chain()
        )

        if not any(provider.available() for provider in providers):
            result["fallback_reason"] = "No LLM API key configured"
            return result

        if not all(limiter.allow() for limiter in self._limiters):
            result["fallback_reason"] = "LLM rate limit reached, try again shortly"
            return result

        toolbox = InvestigationToolbox(transactions, selected_index, shap_explanation)
        prompt = INVESTIGATION_USER_PROMPT.format(**facts)

        def validate(loop_result):
            evidence = prompt + json.dumps(
                [call["result"] for call in loop_result.tool_trace], default=str
            )
            ungrounded = find_ungrounded_numbers(loop_result.text, evidence)
            if ungrounded:
                raise LLMError(
                    f"summary rejected, numbers not found in the evidence: {ungrounded}"
                )

        loop_result, provider, errors = run_with_failover(
            providers=providers,
            system=INVESTIGATION_SYSTEM_PROMPT,
            prompt=prompt,
            tool_specs=TOOL_SPECS,
            tool_functions=toolbox.functions(),
            max_steps=self._max_steps,
            validate=validate,
        )

        if loop_result is None:
            result["fallback_reason"] = "; ".join(errors) or "LLM unavailable"
            return result

        llm_fields = {
            "summary": loop_result.text,
            "investigation_summary": loop_result.text,
            "summary_mode": provider.label,
            "fallback_reason": None,
            "tool_trace": loop_result.tool_trace,
            "llm_steps": loop_result.steps,
        }
        self._cache.set(cache_key, llm_fields)

        return {**result, **llm_fields}

    # ------------------------------------------------------------------
    # Deterministic pieces (shared by both modes)
    # ------------------------------------------------------------------
    def _transaction_facts(self, row: pd.Series, risk_result: dict) -> dict:
        timestamp = pd.to_datetime(row["trans_date_trans_time"])

        return {
            "transaction_id": str(row.get("trans_num", "unknown")),
            "amount": f"${float(row['amt']):,.2f}",
            "merchant": clean_merchant_name(row.get("merchant", "unknown")),
            "category": str(row.get("category", "unknown")),
            "state": str(row.get("state", "unknown")),
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M"),
            "probability": f"{risk_result['fraud_probability'] * 100:.2f}%",
            "risk_band": risk_result["risk_band"],
            "recommendation": risk_result["recommendation"],
        }

    def _describe(self, driver: dict) -> str:
        return f"{driver['label']} ({driver['display_value']})"

    def _build_key_risk_signals(self, shap_explanation: dict) -> list:
        """
        Risk signals are the features SHAP says pushed this score up.
        Nothing here is hardcoded per category or threshold, so a signal
        can only be listed if the model actually treated it as risky.
        """
        if shap_explanation.get("error"):
            return ["Model explanation unavailable for this transaction"]

        increasing = shap_explanation.get("risk_increasing_features", [])[:MAX_SIGNALS]

        if not increasing:
            return ["No feature materially increased the model's risk score"]

        return [f"{self._describe(driver)} increased fraud risk" for driver in increasing]

    def _join(self, items: list) -> str:
        if len(items) <= 1:
            return "".join(items)
        return ", ".join(items[:-1]) + " and " + items[-1]

    def _build_template_summary(
        self,
        facts: dict,
        risk_result: dict,
        shap_explanation: dict,
    ) -> str:
        risk_band = risk_result["risk_band"]

        opening = (
            f"This transaction was classified as {risk_band} with a fraud probability "
            f"of {facts['probability']}. It was a {facts['amount']} purchase at "
            f"{facts['merchant']} in the {facts['category']} category, made by a "
            f"customer in {facts['state']} at {facts['timestamp']}."
        )

        if shap_explanation.get("error"):
            drivers = "A model explanation could not be generated for this transaction."
        else:
            increasing = [
                self._describe(d)
                for d in shap_explanation.get("risk_increasing_features", [])[:MAX_SIGNALS]
            ]
            reducing = [
                self._describe(d)
                for d in shap_explanation.get("risk_reducing_features", [])[:MAX_SIGNALS]
            ]

            up = (
                f"The strongest factors raising the model's score were {self._join(increasing)}."
                if increasing
                else "No factor materially raised the model's score."
            )
            down = (
                f" Factors lowering the score were {self._join(reducing)}."
                if reducing
                else ""
            )
            # Lead with whichever side explains the outcome.
            drivers = (down.strip() + " " + up) if risk_band == "Low Risk" and reducing else up + down

        if risk_band == "High Risk":
            action = (
                "The recommended action is to deny the transaction and escalate it "
                "for fraud analyst review."
            )
        elif risk_band == "Medium Risk":
            action = (
                "The recommended action is to send the transaction to manual review "
                "before approval."
            )
        else:
            action = (
                "The recommended action is to approve the transaction while continuing "
                "standard monitoring."
            )

        return f"{opening} {drivers} {action}"
