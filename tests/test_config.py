import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import app


def test_groq_defaults():
    s = Settings(llm_provider="groq", llm_api_key="gk")
    base_url, model, key = s.resolved_llm()
    assert base_url == "https://api.groq.com/openai/v1"
    assert model == "llama-3.3-70b-versatile"
    assert key == "gk"
    assert s.require_llm_config() == (base_url, model, key)


def test_openai_defaults():
    s = Settings(llm_provider="openai", llm_api_key="ok")
    base_url, model, key = s.resolved_llm()
    assert base_url == ""
    assert model == "gpt-4o-mini"
    assert key == "ok"


def test_explicit_overrides_win():
    s = Settings(
        llm_provider="groq",
        llm_model="llama-3.1-8b-instant",
        llm_base_url="http://localhost:11434/v1",
        llm_api_key="gk",
    )
    base_url, model, key = s.resolved_llm()
    assert base_url == "http://localhost:11434/v1"
    assert model == "llama-3.1-8b-instant"
    assert key == "gk"


def test_openai_embedding_key_is_not_used_for_llm():
    s = Settings(
        llm_provider="groq",
        llm_api_key="gk",
        openai_api_key="sk-openai",
    )
    _, _, key = s.resolved_llm()
    assert key == "gk"


def test_openai_embedding_key_is_not_a_groq_fallback():
    s = Settings(llm_provider="groq", llm_api_key="", openai_api_key="sk-openai")
    _, _, key = s.resolved_llm()
    assert key == ""
    with pytest.raises(ValueError, match="LLM_API_KEY is required"):
        s.require_llm_config()


def test_unknown_provider_fails_clearly():
    s = Settings(llm_provider="anthropic", llm_api_key="x")
    with pytest.raises(ValueError, match="Unsupported LLM_PROVIDER"):
        s.resolved_llm()


def test_blank_provider_fails_clearly():
    s = Settings(llm_provider="  ", llm_api_key="x")
    with pytest.raises(ValueError, match="Unsupported LLM_PROVIDER"):
        s.resolved_llm()


def test_whitespace_llm_api_key_is_rejected():
    s = Settings(llm_provider="groq", llm_api_key="   ")
    with pytest.raises(ValueError, match="LLM_API_KEY is required"):
        s.require_llm_config()


def test_app_starts_when_llm_config_is_valid():
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}


def test_startup_fails_when_llm_config_invalid(monkeypatch):
    class _Invalid:
        def require_llm_config(self) -> tuple[str, str, str]:
            raise ValueError("LLM_API_KEY is required")

    monkeypatch.setattr("app.main.settings", _Invalid())
    with pytest.raises(ValueError, match="LLM_API_KEY is required"):
        with TestClient(app):
            pass
