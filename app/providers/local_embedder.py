"""Free, offline embeddings via fastembed (ONNX - no torch, no API).

Default model `intfloat/multilingual-e5-small` (384-dim) handles Hebrew. fastembed
applies the correct e5 "passage:" / "query:" prefixes for `embed` vs `query_embed`.
The model is downloaded once (~130 MB) and cached under the fastembed cache dir.
"""


class LocalEmbedder:
    def __init__(self, model_name: str = "intfloat/multilingual-e5-small") -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name)
        self.model = model_name

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return [vector.tolist() for vector in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.query_embed(text))).tolist()
