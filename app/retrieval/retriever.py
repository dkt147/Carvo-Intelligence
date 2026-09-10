"""FAISS-backed similarity search over protocol chunks.

The index stores L2-normalized embeddings in an inner-product index, so scores
are cosine similarity in [-1, 1]. Chunk metadata is kept in a parallel JSON file.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from app.providers.openai_provider import Embedder

INDEX_FILE = "index.faiss"
META_FILE = "chunks.json"


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    source: str        # "<book>/<parashah>"
    section: str
    score: float


def normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype("float32")


class Retriever(Protocol):
    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        ...


class FaissRetriever:
    def __init__(self, index, metadata: list[dict], embedder: Embedder) -> None:
        self._index = index
        self._metadata = metadata
        self._embedder = embedder

    @classmethod
    def load(cls, index_dir: str | Path, embedder: Embedder) -> "FaissRetriever":
        import faiss

        directory = Path(index_dir)
        index_path = directory / INDEX_FILE
        meta_path = directory / META_FILE
        if not index_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"No FAISS index in {directory}")

        index = faiss.read_index(str(index_path))
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        return cls(index, metadata, embedder)

    def _embed_query(self, query: str) -> list[float]:
        embed_query = getattr(self._embedder, "embed_query", None)
        if callable(embed_query):
            return embed_query(query)
        return self._embedder.embed([query])[0]

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        if not query.strip() or self._index.ntotal == 0:
            return []

        vector = normalize(np.array([self._embed_query(query)], dtype="float32"))
        scores, indices = self._index.search(vector, min(k, self._index.ntotal))

        results: list[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            meta = self._metadata[idx]
            results.append(
                RetrievedChunk(
                    text=meta["text"],
                    source=meta["source"],
                    section=meta.get("section", ""),
                    score=float(score),
                )
            )
        return results
