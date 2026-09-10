from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

# base_url + default model per known OpenAI-compatible provider
_PROVIDER_DEFAULTS: dict[str, tuple[str, str]] = {
    "openai": ("", "gpt-4o-mini"),
    "groq": ("https://api.groq.com/openai/v1", "llama-3.3-70b-versatile"),
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    port: int = 8000
    ai_service_api_key: str = ""

    # ---- LLM (any OpenAI-compatible endpoint: openai, groq, ...) ----
    llm_provider: str = "groq"
    llm_model: str = ""      # blank -> provider default
    llm_base_url: str = ""   # blank -> provider default
    openai_api_key: str = ""
    groq_api_key: str = ""

    # ---- Embeddings for protocol retrieval ----
    embeddings_provider: str = "local"  # "local" (free, offline) | "openai"
    local_embedding_model: str = "intfloat/multilingual-e5-small"
    openai_embedding_model: str = "text-embedding-3-small"

    # ---- Phase 2 retrieval ----
    protocols_dir: str = "D:/protocolscarvo"
    index_dir: str = "data/index"
    retrieval_top_k: int = 6

    log_level: str = "INFO"

    def resolved_llm(self) -> tuple[str, str, str]:
        """Return (base_url, model, api_key) for the configured provider."""
        provider = self.llm_provider.lower()
        base_default, model_default = _PROVIDER_DEFAULTS.get(provider, ("", ""))
        base_url = self.llm_base_url or base_default
        model = self.llm_model or model_default
        if provider == "groq":
            key = self.groq_api_key or self.openai_api_key
        else:
            key = self.openai_api_key or self.groq_api_key
        return base_url, model, key


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
