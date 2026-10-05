from types import SimpleNamespace

import pytest

from app.providers.openai_provider import (
    CONNECT_TIMEOUT_SECONDS,
    MAX_RETRIES,
    MAX_TOKENS,
    REQUEST_TIMEOUT_SECONDS,
    OpenAIProvider,
    _parse_chat_completion,
)


def _response(*, content: str | None, finish_reason: str | None = "stop", choices=None):
    if choices is not None:
        return SimpleNamespace(choices=choices)
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice])


def test_client_uses_bounded_timeout_and_single_retry():
    provider = OpenAIProvider(api_key="k", model="m", base_url="https://example.com/v1")
    timeout = provider._client.timeout
    assert timeout.connect == CONNECT_TIMEOUT_SECONDS
    assert timeout.read == REQUEST_TIMEOUT_SECONDS
    assert timeout.write == REQUEST_TIMEOUT_SECONDS
    assert provider._client.max_retries == MAX_RETRIES


def test_complete_json_passes_max_tokens():
    provider = OpenAIProvider(api_key="k", model="unit-model")
    captured: dict = {}

    class _Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return _response(content='{"summary": "ok"}')

    provider._client = SimpleNamespace(
        chat=SimpleNamespace(completions=_Completions())
    )

    parsed = provider.complete_json("sys", "user")
    assert parsed == {"summary": "ok"}
    assert captured["max_tokens"] == MAX_TOKENS
    assert captured["model"] == "unit-model"


def test_parse_rejects_missing_choices():
    with pytest.raises(ValueError, match="no choices"):
        _parse_chat_completion(SimpleNamespace(choices=[]))
    with pytest.raises(ValueError, match="no choices"):
        _parse_chat_completion(SimpleNamespace(choices=None))


def test_parse_rejects_truncated_length_finish():
    with pytest.raises(ValueError, match="finish_reason=length"):
        _parse_chat_completion(
            _response(content='{"summary": "partial"}', finish_reason="length")
        )


def test_parse_rejects_empty_or_missing_content():
    with pytest.raises(ValueError, match="empty content"):
        _parse_chat_completion(_response(content=""))
    with pytest.raises(ValueError, match="empty content"):
        _parse_chat_completion(_response(content=None))
    with pytest.raises(ValueError, match="empty content"):
        _parse_chat_completion(
            SimpleNamespace(choices=[SimpleNamespace(message=None, finish_reason="stop")])
        )


def test_parse_rejects_invalid_json_and_non_objects():
    with pytest.raises(ValueError, match="invalid JSON"):
        _parse_chat_completion(_response(content="not-json"))
    with pytest.raises(ValueError, match="JSON object"):
        _parse_chat_completion(_response(content="[1, 2]"))


def test_parse_accepts_json_object():
    parsed = _parse_chat_completion(
        _response(content='{"summary": "ok", "recommendations": []}')
    )
    assert parsed["summary"] == "ok"


def test_parse_strips_markdown_code_fences():
    fenced = """```json
{"summary": "fenced", "recommendations": []}
```"""
    parsed = _parse_chat_completion(_response(content=fenced))
    assert parsed["summary"] == "fenced"

    fenced_plain = "```\n{\"summary\": \"plain\"}\n```"
    assert _parse_chat_completion(_response(content=fenced_plain))["summary"] == "plain"
