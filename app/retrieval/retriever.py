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
MANIFEST_FILE = "manifest.json"


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


def write_manifest(
    index_dir: str | Path,
    *,
    embedding_model: str,
    embeddings_provider: str,
    dimension: int,
    count: int,
) -> None:
    path = Path(index_dir) / MANIFEST_FILE
    path.write_text(
        json.dumps(
            {
                "embedding_model": embedding_model,
                "embeddings_provider": embeddings_provider,
                "dimension": int(dimension),
                "count": int(count),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _embedder_dimension(embedder: Embedder) -> int:
    probe = embedder.embed(["ping"])
    if not probe or not probe[0]:
        raise ValueError("Embedder returned no dimension probe vector")
    return len(probe[0])


def validate_index(
    index,
    metadata: list,
    embedder: Embedder,
    manifest: dict,
) -> None:
    model = str(getattr(embedder, "model", "") or "")
    expected_model = str(manifest.get("embedding_model") or "")
    if not expected_model:
        raise ValueError("manifest.json is missing embedding_model")
    if not model:
        raise ValueError("Embedder has no model identifier to compare with the index")
    if expected_model != model:
        raise ValueError(
            f"Embedding model mismatch: index was built with {expected_model!r}, "
            f"service is using {model!r}. Rebuild the index."
        )

    try:
        expected_dimension = int(manifest.get("dimension"))
        expected_count = int(manifest.get("count"))
    except (TypeError, ValueError) as exc:
        raise ValueError("manifest.json dimension and count must be integers") from exc
    if expected_dimension <= 0 or expected_count <= 0:
        raise ValueError("manifest.json dimension and count must be positive")

    dimension = int(getattr(index, "d", 0) or 0)
    live_dimension = _embedder_dimension(embedder)
    count = int(getattr(index, "ntotal", 0) or 0)
    meta_count = len(metadata) if isinstance(metadata, list) else -1

    if dimension != expected_dimension:
        raise ValueError(
            f"Index dimension {dimension} does not match manifest {expected_dimension}"
        )
    if live_dimension != dimension:
        raise ValueError(
            f"Embedder dimension {live_dimension} does not match index dimension {dimension}"
        )
    if count != expected_count:
        raise ValueError(
            f"Index vector count {count} does not match manifest {expected_count}"
        )
    if meta_count != count:
        raise ValueError(
            f"chunks.json has {meta_count} records but the index has {count} vectors"
        )


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

        manifest_path = directory / MANIFEST_FILE
        if not manifest_path.exists():
            raise ValueError(
                f"Index at {directory} is missing {MANIFEST_FILE}. "
                "Rebuild with: python -m app.ingestion.build_index"
            )

        index = faiss.read_index(str(index_path))
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, list):
            raise ValueError("chunks.json must contain a JSON array")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("manifest.json must contain a JSON object")
        validate_index(index, metadata, embedder, manifest)
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
