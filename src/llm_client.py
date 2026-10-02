"""
LLM provider layer for the Investigation Agent.

Design goals:
- The rest of the app never imports a vendor SDK. It talks to LLMProvider.
- Providers and model names come from config (LLM_CHAIN), not from code.
- Every failure (no key, API down, rate limit, bad output) falls through to
  the next entry in the chain, and finally to the rule-based template, so
  the app always produces a summary.

Adding a provider later (OpenAI, Claude):
1. Write a class with the same three members as GeminiProvider
   (label, available(), generate()).
2. Register it in PROVIDER_REGISTRY.
3. Add "openai:<model>" to LLM_CHAIN in secrets or .env.
"""
import os
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional at runtime
    pass


DEFAULT_LLM_CHAIN = "gemini:gemini-3.5-flash-lite,gemini:gemini-2.5-flash-lite"
DEFAULT_MAX_TOOL_STEPS = 4
DEFAULT_MAX_PER_MINUTE = 8
DEFAULT_MAX_PER_DAY = 150
REQUEST_TIMEOUT_MS = 20_000


class LLMError(Exception):
    """Raised when a provider cannot produce a usable answer."""


@dataclass
class ToolCall:
    name: str
    args: dict
    id: str | None = None


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list = field(default_factory=list)
    raw: object = None  # provider-native assistant turn, replayed as-is


@dataclass
class ToolLoopResult:
    text: str
    tool_trace: list
    steps: int


def get_setting(name: str, default=None):
    """
    Read a setting from the environment (.env locally), then from Streamlit
    secrets (Streamlit Community Cloud), then fall back to the default.
    """
    value = os.environ.get(name)
    if value:
        return value

    try:
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass

    return default


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------
class GeminiProvider:
    """Google Gemini through the google-genai SDK."""

    provider_name = "Gemini"

    def __init__(self, model: str, api_key: str | None = None):
        self.model = model
        self._api_key = api_key
        self._client = None

    @property
    def label(self) -> str:
        return f"{self.provider_name} ({self.model})"

    def _key(self):
        return self._api_key or get_setting("GEMINI_API_KEY")

    def available(self) -> bool:
        return bool(self._key())

    def _get_client(self):
        if self._client is None:
            from google import genai
            from google.genai import types

            self._client = genai.Client(
                api_key=self._key(),
                http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
            )
        return self._client

    def _to_contents(self, messages: list) -> list:
        from google.genai import types

        contents = []
        for message in messages:
            role = message["role"]

            if role == "user":
                contents.append(
                    types.Content(role="user", parts=[types.Part(text=message["text"])])
                )
            elif role == "assistant":
                # Replay Gemini's own turn untouched. It can carry thought
                # signatures that the API requires on the next request.
                contents.append(message["raw"])
            elif role == "tool":
                contents.append(
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_function_response(
                                name=result["name"],
                                response={"result": result["result"]},
                            )
                            for result in message["results"]
                        ],
                    )
                )
        return contents

    def generate(self, system: str, messages: list, tools: list | None) -> LLMResponse:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.2,
            max_output_tokens=2048,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        )

        if tools:
            config.tools = [
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(
                            name=tool["name"],
                            description=tool["description"],
                            parameters_json_schema=tool["parameters"],
                        )
                        for tool in tools
                    ]
                )
            ]

        try:
            response = self._get_client().models.generate_content(
                model=self.model,
                contents=self._to_contents(messages),
                config=config,
            )
        except Exception as error:
            raise LLMError(f"{self.label}: {type(error).__name__}: {error}") from error

        if not response.candidates or response.candidates[0].content is None:
            raise LLMError(f"{self.label}: empty response")

        content = response.candidates[0].content
        text_parts = []
        tool_calls = []

        for part in content.parts or []:
            if part.function_call is not None:
                tool_calls.append(
                    ToolCall(
                        name=part.function_call.name,
                        args=dict(part.function_call.args or {}),
                        id=part.function_call.id,
                    )
                )
            elif part.text and not getattr(part, "thought", False):
                text_parts.append(part.text)

        return LLMResponse(
            text="".join(text_parts).strip(), tool_calls=tool_calls, raw=content
        )


PROVIDER_REGISTRY = {
    "gemini": GeminiProvider,
}


def build_provider_chain(chain: str | None = None) -> list:
    """
    Turn "gemini:model-a,gemini:model-b" into provider objects, in order.
    Unknown provider names are skipped rather than crashing the app.
    """
    chain = chain or get_setting("LLM_CHAIN", DEFAULT_LLM_CHAIN)
    providers = []

    for entry in chain.split(","):
        entry = entry.strip()
        if not entry or ":" not in entry:
            continue

        provider_name, model = entry.split(":", 1)
        provider_class = PROVIDER_REGISTRY.get(provider_name.strip().lower())

        if provider_class is not None and model.strip():
            providers.append(provider_class(model=model.strip()))

    return providers


# ---------------------------------------------------------------------------
# Tool-calling loop
# ---------------------------------------------------------------------------
def run_tool_loop(
    provider,
    system: str,
    prompt: str,
    tool_specs: list,
    tool_functions: dict,
    max_steps: int = DEFAULT_MAX_TOOL_STEPS,
) -> ToolLoopResult:
    """
    Let the model choose which tools to call, run them, feed results back,
    and stop when it writes its answer.

    max_steps caps the number of model calls. On the last step the tools are
    withheld, which forces the model to answer with what it already has.
    """
    messages = [{"role": "user", "text": prompt}]
    tool_trace = []

    for step in range(1, max_steps + 1):
        is_last_step = step == max_steps
        response = provider.generate(
            system, messages, None if is_last_step else tool_specs
        )

        if response.tool_calls and not is_last_step:
            results = []

            for call in response.tool_calls:
                function = tool_functions.get(call.name)

                if function is None:
                    result = {"error": f"Unknown tool: {call.name}"}
                else:
                    try:
                        result = function(**call.args)
                    except Exception as error:
                        result = {"error": f"{type(error).__name__}: {error}"}

                tool_trace.append(
                    {
                        "step": step,
                        "tool": call.name,
                        "args": call.args,
                        "result": result,
                    }
                )
                results.append({"name": call.name, "id": call.id, "result": result})

            messages.append(
                {"role": "assistant", "tool_calls": response.tool_calls, "raw": response.raw}
            )
            messages.append({"role": "tool", "results": results})
            continue

        if response.text:
            return ToolLoopResult(text=response.text, tool_trace=tool_trace, steps=step)

        raise LLMError(f"{provider.label}: no text in final response")

    raise LLMError(f"{provider.label}: no answer within {max_steps} steps")


def run_with_failover(
    providers: list,
    system: str,
    prompt: str,
    tool_specs: list,
    tool_functions: dict,
    max_steps: int = DEFAULT_MAX_TOOL_STEPS,
    validate=None,
):
    """
    Try each provider in order. Returns (result, provider, errors).

    result is None when every provider failed or none had a key. The caller
    then uses the rule-based template. validate(result) may raise LLMError
    to reject an answer, which also moves on to the next provider.
    """
    errors = []

    for provider in providers:
        if not provider.available():
            errors.append(f"{provider.label}: no API key configured")
            continue

        try:
            result = run_tool_loop(
                provider, system, prompt, tool_specs, tool_functions, max_steps
            )
            if validate is not None:
                validate(result)
            return result, provider, errors
        except LLMError as error:
            errors.append(str(error))
        except Exception as error:  # never let an LLM problem break the app
            errors.append(f"{provider.label}: {type(error).__name__}: {error}")

    return None, None, errors


# ---------------------------------------------------------------------------
# Cost and abuse controls
# ---------------------------------------------------------------------------
class RateLimiter:
    """
    Sliding-window limiter shared by every visitor of the app process.

    The public app runs on a free-tier key, so this protects the quota from
    one visitor (or a bot) clicking the button repeatedly.
    """

    def __init__(self, max_calls: int, per_seconds: float, clock=time.monotonic):
        self.max_calls = max_calls
        self.per_seconds = per_seconds
        self._clock = clock
        self._calls = deque()
        self._lock = threading.Lock()

    def allow(self) -> bool:
        with self._lock:
            now = self._clock()

            while self._calls and now - self._calls[0] >= self.per_seconds:
                self._calls.popleft()

            if len(self._calls) >= self.max_calls:
                return False

            self._calls.append(now)
            return True


class SummaryCache:
    """Small in-memory LRU cache: one LLM summary per transaction."""

    def __init__(self, max_size: int = 500):
        self.max_size = max_size
        self._items = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            if key in self._items:
                self._items.move_to_end(key)
                return self._items[key]
            return None

    def set(self, key, value):
        with self._lock:
            self._items[key] = value
            self._items.move_to_end(key)

            while len(self._items) > self.max_size:
                self._items.popitem(last=False)
