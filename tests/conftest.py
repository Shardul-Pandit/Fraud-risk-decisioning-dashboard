import os
from pathlib import Path

import pytest

from src.llm_client import LLMError, LLMResponse, ToolCall

ROOT = Path(__file__).resolve().parents[1]

# Model and report paths in src/ are relative to the project root.
os.chdir(ROOT)


class FakeProvider:
    """
    Stands in for a real LLM. `script` is a list of turns; each turn is an
    LLMResponse, or an Exception to raise. Records what it was asked.
    """

    def __init__(self, script, name="Fake", has_key=True):
        self.script = list(script)
        self.label = name
        self.has_key = has_key
        self.calls = []

    def available(self):
        return self.has_key

    def generate(self, system, messages, tools):
        self.calls.append({"messages": list(messages), "tools": tools})
        turn = self.script.pop(0)
        if isinstance(turn, Exception):
            raise turn
        return turn


def tool_turn(*calls):
    return LLMResponse(tool_calls=[ToolCall(name=n, args=a) for n, a in calls])


def text_turn(text):
    return LLMResponse(text=text)


@pytest.fixture(scope="session")
def transactions():
    from src.data_loader import load_raw_data

    return load_raw_data().drop(columns=["is_fraud"])
