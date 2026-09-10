from app.config import Settings


def test_groq_defaults():
    s = Settings(llm_provider="groq", groq_api_key="gk")
    base_url, model, key = s.resolved_llm()
    assert base_url == "https://api.groq.com/openai/v1"
    assert model == "llama-3.3-70b-versatile"
    assert key == "gk"


def test_openai_defaults():
    s = Settings(llm_provider="openai", openai_api_key="ok")
    base_url, model, key = s.resolved_llm()
    assert base_url == ""
    assert model == "gpt-4o-mini"
    assert key == "ok"


def test_explicit_overrides_win():
    s = Settings(
        llm_provider="groq",
        llm_model="llama-3.1-8b-instant",
        llm_base_url="http://localhost:11434/v1",
        groq_api_key="gk",
    )
    base_url, model, _ = s.resolved_llm()
    assert base_url == "http://localhost:11434/v1"
    assert model == "llama-3.1-8b-instant"
