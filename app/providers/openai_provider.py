import json
from typing import Any, Protocol


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

        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url

        self._client = OpenAI(**kwargs)
        self.model = model
        self._temperature = temperature

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=self._temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )

        content = response.choices[0].message.content or "{}"
        parsed = json.loads(content)

        if not isinstance(parsed, dict):
            raise ValueError("Model did not return a JSON object")

        return parsed


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
