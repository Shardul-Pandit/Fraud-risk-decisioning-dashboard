import pytest

from src.llm_client import (
    GeminiProvider,
    LLMError,
    RateLimiter,
    SummaryCache,
    build_provider_chain,
    run_tool_loop,
    run_with_failover,
)
from tests.conftest import FakeProvider, text_turn, tool_turn

SPECS = [{"name": "lookup", "description": "d", "parameters": {"type": "object", "properties": {}}}]


def test_tool_loop_runs_tools_then_returns_text():
    provider = FakeProvider([tool_turn(("lookup", {"x": 2})), text_turn("done")])
    result = run_tool_loop(provider, "sys", "go", SPECS, {"lookup": lambda x: {"double": x * 2}})

    assert result.text == "done"
    assert result.steps == 2
    assert result.tool_trace == [
        {"step": 1, "tool": "lookup", "args": {"x": 2}, "result": {"double": 4}}
    ]
    # The tool result was sent back to the model on the second call.
    assert provider.calls[1]["messages"][-1]["results"][0]["result"] == {"double": 4}


def test_tool_loop_caps_steps_and_withholds_tools_on_last_step():
    provider = FakeProvider(
        [tool_turn(("lookup", {})), tool_turn(("lookup", {})), text_turn("forced answer")]
    )
    result = run_tool_loop(provider, "sys", "go", SPECS, {"lookup": lambda: {}}, max_steps=3)

    assert result.text == "forced answer"
    assert len(provider.calls) == 3
    assert provider.calls[0]["tools"] is not None
    assert provider.calls[2]["tools"] is None


def test_tool_loop_fails_if_model_never_answers():
    provider = FakeProvider([tool_turn(("lookup", {})), tool_turn(("lookup", {}))])
    with pytest.raises(LLMError):
        run_tool_loop(provider, "sys", "go", SPECS, {"lookup": lambda: {}}, max_steps=2)


def test_unknown_tool_and_tool_errors_are_reported_not_raised():
    def broken():
        raise RuntimeError("boom")

    provider = FakeProvider(
        [tool_turn(("missing", {}), ("broken", {})), text_turn("ok")]
    )
    result = run_tool_loop(provider, "sys", "go", SPECS, {"broken": broken})

    assert "Unknown tool" in result.tool_trace[0]["result"]["error"]
    assert "boom" in result.tool_trace[1]["result"]["error"]


def test_failover_moves_to_next_provider_on_error():
    first = FakeProvider([LLMError("429 rate limit")], name="First")
    second = FakeProvider([text_turn("from second")], name="Second")

    result, provider, errors = run_with_failover([first, second], "s", "p", SPECS, {})

    assert result.text == "from second"
    assert provider is second
    assert errors == ["429 rate limit"]


def test_failover_skips_providers_without_a_key():
    no_key = FakeProvider([], name="NoKey", has_key=False)
    result, provider, errors = run_with_failover([no_key], "s", "p", SPECS, {})

    assert result is None and provider is None
    assert "no API key" in errors[0]
    assert no_key.calls == []


def test_failover_survives_unexpected_exceptions():
    crashing = FakeProvider([ValueError("sdk bug")], name="Crashy")
    result, _, errors = run_with_failover([crashing], "s", "p", SPECS, {})

    assert result is None
    assert "sdk bug" in errors[0]


def test_failover_rejects_answers_that_fail_validation():
    def validate(loop_result):
        if "bad" in loop_result.text:
            raise LLMError("rejected")

    first = FakeProvider([text_turn("bad answer")], name="First")
    second = FakeProvider([text_turn("good answer")], name="Second")
    result, provider, errors = run_with_failover(
        [first, second], "s", "p", SPECS, {}, validate=validate
    )

    assert result.text == "good answer" and provider is second
    assert errors == ["rejected"]


def test_rate_limiter_window():
    now = [0.0]
    limiter = RateLimiter(max_calls=2, per_seconds=60, clock=lambda: now[0])

    assert limiter.allow() and limiter.allow()
    assert not limiter.allow()
    now[0] = 61
    assert limiter.allow()


def test_summary_cache_evicts_oldest():
    cache = SummaryCache(max_size=2)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.get("a")
    cache.set("c", 3)

    assert cache.get("b") is None
    assert cache.get("a") == 1 and cache.get("c") == 3


def test_provider_chain_comes_from_config(monkeypatch):
    chain = build_provider_chain("gemini:model-a, gemini:model-b, unknown:x, junk")

    assert [p.model for p in chain] == ["model-a", "model-b"]
    assert all(isinstance(p, GeminiProvider) for p in chain)

    monkeypatch.setenv("LLM_CHAIN", "gemini:from-env")
    assert [p.model for p in build_provider_chain()] == ["from-env"]


def test_gemini_provider_unavailable_without_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert GeminiProvider(model="m").available() is False
    assert GeminiProvider(model="m", api_key="k").available() is True
