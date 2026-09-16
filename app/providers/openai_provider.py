import json
import re
from typing import Any, Protocol

import httpx

CONNECT_TIMEOUT_SECONDS = 5.0
REQUEST_TIMEOUT_SECONDS = 60.0
MAX_RETRIES = 1
MAX_TOKENS = 2048
_FENCE_RE = re.compile(
    r"^\s*```(?:json|JSON)?\s*\n?(.*?)\n?```\s*$",
    re.DOTALL,
)


class LLMProvider(Protocol):
    """Minimal contract the analyzer depends on."""

    model: str

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        ...


class Embedder(Protocol):
    model: str

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed documents/passages."""
        ...


class OpenAIProvider:
    """Works with any OpenAI-compatible chat endpoint (OpenAI, Groq, ...)."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "",
        temperature: float = 0.2,
    ) -> None:
        from openai import OpenAI

        kwargs: dict[str, Any] = {
            "api_key": api_key,
            "timeout": httpx.Timeout(
                REQUEST_TIMEOUT_SECONDS, connect=CONNECT_TIMEOUT_SECONDS
            ),
            "max_retries": MAX_RETRIES,
        }
        if base_url:
            kwargs["base_url"] = base_url

        self._client = OpenAI(**kwargs)
        self.model = model
        self._temperature = temperature
        self._max_tokens = MAX_TOKENS

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return _parse_chat_completion(response)


def _parse_chat_completion(response: Any) -> dict[str, Any]:
    choices = getattr(response, "choices", None)
    if not choices:
        raise ValueError("LLM returned no choices")

    choice = choices[0]
    finish_reason = getattr(choice, "finish_reason", None)
    if finish_reason == "length":
        raise ValueError("LLM response truncated (finish_reason=length)")

    message = getattr(choice, "message", None)
    content = getattr(message, "content", None) if message is not None else None
    if not isinstance(content, str) or not content.strip():
        raise ValueError("LLM returned empty content")

    content = _strip_code_fences(content)

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("LLM returned invalid JSON") from exc

    if not isinstance(parsed, dict):
        raise ValueError("Model did not return a JSON object")

    return parsed


def _strip_code_fences(content: str) -> str:
    text = content.strip()
    fenced = _FENCE_RE.match(text)
    if fenced:
        return fenced.group(1).strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return text


class OpenAIEmbedder:
    def __init__(self, api_key: str, model: str) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]
