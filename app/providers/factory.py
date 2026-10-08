from app.config import settings
from app.providers.openai_provider import Embedder, LLMProvider, OpenAIEmbedder, OpenAIProvider


def build_llm_provider() -> LLMProvider:
    base_url, model, api_key = settings.require_llm_config()
    return OpenAIProvider(api_key=api_key, model=model, base_url=base_url)


def build_embedder() -> Embedder:
    if settings.embeddings_provider.lower() == "openai":
        return OpenAIEmbedder(settings.openai_api_key, settings.openai_embedding_model)

    from app.providers.local_embedder import LocalEmbedder

    return LocalEmbedder(settings.local_embedding_model)
