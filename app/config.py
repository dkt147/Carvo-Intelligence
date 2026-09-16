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

    # ---- LLM (groq | openai). One key, no cross-provider fallback. ----
    llm_provider: str = "groq"
    llm_api_key: str = ""
    llm_model: str = ""      # blank -> provider default
    llm_base_url: str = ""   # blank -> provider default

    # Embeddings only (EMBEDDINGS_PROVIDER=openai). Never used for the LLM.
    openai_api_key: str = ""

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
        """Return (base_url, model, api_key) for the configured provider.

        Raises ValueError if LLM_PROVIDER is not groq or openai.
        """
        provider = (self.llm_provider or "").strip().lower()
        if provider not in _PROVIDER_DEFAULTS:
            supported = ", ".join(sorted(_PROVIDER_DEFAULTS))
            raise ValueError(
                f"Unsupported LLM_PROVIDER '{self.llm_provider}'. "
                f"Supported providers: {supported}."
            )

        base_default, model_default = _PROVIDER_DEFAULTS[provider]
        base_url = (self.llm_base_url or "").strip() or base_default
        model = (self.llm_model or "").strip() or model_default
        key = (self.llm_api_key or "").strip()
        return base_url, model, key

    def require_llm_config(self) -> tuple[str, str, str]:
        """Fail fast when the process cannot serve analyses (D9 / Q18)."""
        base_url, model, key = self.resolved_llm()
        provider = (self.llm_provider or "").strip().lower()
        if not key:
            raise ValueError(
                "LLM_API_KEY is required. Set it to the API key for "
                f"LLM_PROVIDER={provider}. GROQ_API_KEY is ignored; "
                "OPENAI_API_KEY is used only when EMBEDDINGS_PROVIDER=openai."
            )
        if not model:
            raise ValueError(
                "LLM_MODEL is empty after applying provider defaults. "
                "Set LLM_MODEL or use LLM_PROVIDER=groq or openai."
            )
        return base_url, model, key


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
