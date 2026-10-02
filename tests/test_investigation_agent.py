import pytest

from src.agents.fraud_decision_workflow import FraudDecisionWorkflow
from src.agents.investigation_agent import (
    TEMPLATE_MODE,
    InvestigationAgent,
    find_ungrounded_numbers,
)
from src.llm_client import LLMError, RateLimiter, SummaryCache
from tests.conftest import FakeProvider, text_turn, tool_turn

HIGH_RISK_INDEX = 306221


def make_workflow(providers, limiters=None):
    workflow = FraudDecisionWorkflow()
    workflow.investigation_agent = InvestigationAgent(
        providers=providers,
        cache=SummaryCache(),
        limiters=limiters if limiters is not None else [],
    )
    return workflow


def test_no_key_falls_back_to_template(transactions):
    result = make_workflow([FakeProvider([], has_key=False)]).run_with_context(
        transactions, HIGH_RISK_INDEX
    )
    investigation = result["investigation_result"]

    assert investigation["summary_mode"] == TEMPLATE_MODE
    assert investigation["fallback_reason"] == "No LLM API key configured"
    assert "High Risk" in investigation["summary"]


def test_llm_summary_used_and_tool_calls_traced(transactions):
    provider = FakeProvider(
        [
            tool_turn(
                ("get_shap_drivers", {}),
                ("get_recent_transactions", {"limit": 2}),
                ("get_policy_section", {"section": "high_risk"}),
            ),
            text_turn("The transaction is High Risk and policy says to deny it."),
        ],
        name="Fake LLM",
    )
    investigation = make_workflow([provider]).run_with_context(
        transactions, HIGH_RISK_INDEX
    )["investigation_result"]

    assert investigation["summary_mode"] == "Fake LLM"
    assert investigation["summary"].startswith("The transaction is High Risk")
    assert [c["tool"] for c in investigation["tool_trace"]] == [
        "get_shap_drivers",
        "get_recent_transactions",
        "get_policy_section",
    ]
    recent = investigation["tool_trace"][1]["result"]
    assert len(recent["transactions"]) == 2
    assert "0.80" in investigation["tool_trace"][2]["result"]["text"]


def test_api_failure_falls_back_to_template(transactions):
    provider = FakeProvider([LLMError("503 service unavailable")])
    investigation = make_workflow([provider]).run_with_context(
        transactions, HIGH_RISK_INDEX
    )["investigation_result"]

    assert investigation["summary_mode"] == TEMPLATE_MODE
    assert "503" in investigation["fallback_reason"]


def test_invented_number_is_rejected(transactions):
    provider = FakeProvider([text_turn("Fraud probability is 12.34%, so approve.")])
    investigation = make_workflow([provider]).run_with_context(
        transactions, HIGH_RISK_INDEX
    )["investigation_result"]

    assert investigation["summary_mode"] == TEMPLATE_MODE
    assert "12.34" in investigation["fallback_reason"]


def test_rate_limit_falls_back_and_skips_the_llm(transactions):
    provider = FakeProvider([text_turn("should not be used")])
    limiter = RateLimiter(max_calls=0, per_seconds=60)
    investigation = make_workflow([provider], limiters=[limiter]).run_with_context(
        transactions, HIGH_RISK_INDEX
    )["investigation_result"]

    assert investigation["summary_mode"] == TEMPLATE_MODE
    assert "rate limit" in investigation["fallback_reason"]
    assert provider.calls == []


def test_llm_summary_is_cached_per_transaction(transactions):
    provider = FakeProvider([text_turn("Cached findings.")], name="Fake LLM")
    workflow = make_workflow([provider])

    first = workflow.run_with_context(transactions, HIGH_RISK_INDEX)["investigation_result"]
    second = workflow.run_with_context(transactions, HIGH_RISK_INDEX)["investigation_result"]

    assert len(provider.calls) == 1
    assert first["from_cache"] is False and second["from_cache"] is True
    assert second["summary"] == "Cached findings."


def test_llm_cannot_change_the_decision(transactions):
    provider = FakeProvider([text_turn("I would approve this transaction instead.")])
    result = make_workflow([provider]).run_with_context(transactions, HIGH_RISK_INDEX)

    assert result["final_decision"]["final_recommendation"] == "Deny"
    assert result["risk_result"]["risk_band"] == "High Risk"


@pytest.mark.parametrize("index", [0, 1685, HIGH_RISK_INDEX])
def test_template_signals_agree_with_shap(transactions, index):
    result = make_workflow([]).run_with_context(transactions, index)
    shap_explanation = result["shap_explanation"]
    investigation = result["investigation_result"]

    increasing = {d["label"] for d in shap_explanation["risk_increasing_features"]}
    reducing = {d["label"] for d in shap_explanation["risk_reducing_features"]}

    for signal in investigation["key_risk_signals"]:
        label = signal.split(" (")[0]
        assert label in increasing
        assert label not in reducing


def test_find_ungrounded_numbers():
    facts = "Amount: $1,234.50. Probability: 97.05%. Time 00:42."
    assert find_ungrounded_numbers("It cost $1,234.50 at 97.05% risk.", facts) == []
    assert find_ungrounded_numbers("Roughly 97% risk.", facts) == [97.0]
