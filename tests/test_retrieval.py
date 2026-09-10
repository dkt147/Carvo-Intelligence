import numpy as np

from app.analysis.analyzer import Analyzer
from app.retrieval.retriever import FaissRetriever, normalize
from app.schemas.analysis import AnalysisRequest
from tests.test_analyses import VALID_PAYLOAD, FakeProvider


class HashEmbedder:
    """Deterministic offline embeddings: bag-of-words over a tiny vocab."""

    VOCAB = ["promise", "milestone", "report", "friday", "budget", "trust", "delay", "plan"]

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            low = text.lower()
            out.append([float(low.count(word)) + 0.01 for word in self.VOCAB])
        return out


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
