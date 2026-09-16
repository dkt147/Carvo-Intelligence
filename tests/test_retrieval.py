import numpy as np

from app.analysis.analyzer import Analyzer
from app.retrieval.retriever import FaissRetriever, normalize
from app.schemas.analysis import AnalysisRequest
from tests.test_analyses import VALID_PAYLOAD, FakeProvider


class HashEmbedder:
    """Deterministic offline embeddings: bag-of-words over a tiny vocab."""

    VOCAB = ["promise", "milestone", "report", "friday", "budget", "trust", "delay", "plan"]
    model = "hash-embedder"

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            low = text.lower()
            out.append([float(low.count(word)) + 0.01 for word in self.VOCAB])
        return out

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]


def _build_retriever(texts: list[str]) -> FaissRetriever:
    import faiss

    embedder = HashEmbedder()
    matrix = normalize(np.array(embedder.embed(texts), dtype="float32"))
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    metadata = [
        {"text": t, "source": f"book/parashah-{i}", "section": "aliyah-1"}
        for i, t in enumerate(texts)
    ]
    return FaissRetriever(index, metadata, embedder)


def test_search_ranks_the_relevant_chunk_first():
    retriever = _build_retriever(
        [
            "A promise was made to deliver the milestone by friday",
            "Notes about the annual budget and trust",
            "Unrelated plan about office seating",
        ]
    )
    hits = retriever.search("the promise milestone friday report", k=2)
    assert hits
    assert hits[0].text.startswith("A promise was made")
    assert hits[0].score >= hits[-1].score


def test_empty_query_returns_nothing():
    retriever = _build_retriever(["some protocol text"])
    assert retriever.search("   ", k=3) == []


def test_analyzer_injects_retrieved_sources_into_metadata():
    retriever = _build_retriever(
        [
            "A promise was made to deliver the milestone by friday",
            "Budget and trust considerations",
        ]
    )
    analyzer = Analyzer(
        FakeProvider({"summary": "ok", "recommendations": []}),
        retriever=retriever,
        top_k=2,
    )
    response = analyzer.analyze(AnalysisRequest.model_validate(VALID_PAYLOAD))

    assert response.status == "COMPLETED"
    assert response.metadata["retrievedCount"] == 2
    assert "book/parashah-0" in response.metadata["retrievedSources"]


def _write_index(directory, texts, embedder=None, manifest_overrides=None):
    import faiss
    import json

    from app.retrieval.retriever import INDEX_FILE, META_FILE, write_manifest

    embedder = embedder or HashEmbedder()
    matrix = normalize(np.array(embedder.embed(texts), dtype="float32"))
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    metadata = [
        {"text": t, "source": f"book/parashah-{i}", "section": "aliyah-1"}
        for i, t in enumerate(texts)
    ]
    faiss.write_index(index, str(directory / INDEX_FILE))
    (directory / META_FILE).write_text(
        json.dumps(metadata, ensure_ascii=False), encoding="utf-8"
    )
    kwargs = {
        "embedding_model": embedder.model,
        "embeddings_provider": "local",
        "dimension": int(matrix.shape[1]),
        "count": int(index.ntotal),
    }
    if manifest_overrides:
        kwargs.update(manifest_overrides)
    write_manifest(directory, **kwargs)
    return embedder


def test_load_validates_manifest_and_round_trips(tmp_path):
    texts = ["A promise was made to deliver the milestone by friday"]
    embedder = _write_index(tmp_path, texts)
    loaded = FaissRetriever.load(tmp_path, embedder)
    hits = loaded.search("promise milestone friday", k=1)
    assert hits
    assert "promise" in hits[0].text.lower()


def test_load_rejects_model_mismatch(tmp_path):
    import pytest

    _write_index(tmp_path, ["promise milestone"], manifest_overrides={"embedding_model": "other-model"})
    with pytest.raises(ValueError, match="Embedding model mismatch"):
        FaissRetriever.load(tmp_path, HashEmbedder())


def test_load_rejects_missing_manifest(tmp_path):
    import pytest
    from app.retrieval.retriever import INDEX_FILE, META_FILE

    (tmp_path / INDEX_FILE).write_bytes(b"not-faiss")
    (tmp_path / META_FILE).write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="missing manifest.json"):
        FaissRetriever.load(tmp_path, HashEmbedder())


def test_load_rejects_dimension_mismatch(tmp_path):
    import pytest

    _write_index(tmp_path, ["promise milestone"], manifest_overrides={"dimension": 16})
    with pytest.raises(ValueError, match="Index dimension"):
        FaissRetriever.load(tmp_path, HashEmbedder())


def test_load_rejects_count_mismatch(tmp_path):
    import pytest

    _write_index(tmp_path, ["promise milestone"], manifest_overrides={"count": 99})
    with pytest.raises(ValueError, match="vector count"):
        FaissRetriever.load(tmp_path, HashEmbedder())
