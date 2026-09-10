"""Build the FAISS retrieval index from the protocol .docx files.

    python -m app.ingestion.build_index
    python -m app.ingestion.build_index --protocols-dir D:/protocolscarvo --limit 3

Needs OPENAI_API_KEY (used for embeddings). Output goes to <index-dir>/:
  index.faiss   - inner-product index over L2-normalized embeddings
  chunks.json   - parallel metadata (text, source, section, ...)
"""

import argparse
import json
from pathlib import Path

import numpy as np

from app.config import settings
from app.ingestion.chunker import chunk_document
from app.ingestion.docx_loader import iter_documents
from app.providers.factory import build_embedder
from app.providers.openai_provider import Embedder
from app.retrieval.retriever import INDEX_FILE, META_FILE, normalize

EMBED_BATCH = 64


def _collect_chunks(protocols_dir: str, limit: int | None) -> list[dict]:
    records: list[dict] = []
    for doc_number, doc in enumerate(iter_documents(protocols_dir), start=1):
        if limit is not None and doc_number > limit:
            break
        source = f"{doc.book}/{doc.parashah}" if doc.book else doc.parashah
        for chunk in chunk_document(doc.text):
            records.append(
                {
                    "text": chunk.text,
                    "source": source,
                    "book": doc.book,
                    "parashah": doc.parashah,
                    "section": chunk.section,
                    "chunk_index": chunk.chunk_index,
                }
            )
    return records


def _embed_all(embedder: Embedder, texts: list[str]) -> np.ndarray:
    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBED_BATCH):
        batch = texts[start : start + EMBED_BATCH]
        vectors.extend(embedder.embed(batch))
        print(f"  embedded {min(start + EMBED_BATCH, len(texts))}/{len(texts)}")
    return normalize(np.array(vectors, dtype="float32"))


def build(protocols_dir: str, index_dir: str, limit: int | None = None) -> int:
    import faiss

    print(f"Loading protocols from {protocols_dir} ...")
    records = _collect_chunks(protocols_dir, limit)
    if not records:
        raise SystemExit("No chunks produced - check the protocols directory.")
    print(f"{len(records)} chunks from the documents.")

    embedder = build_embedder()
    print(f"Embedding with {settings.embeddings_provider} ({getattr(embedder, 'model', '?')}) ...")
    matrix = _embed_all(embedder, [r["text"] for r in records])

    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)

    out_dir = Path(index_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / INDEX_FILE))
    (out_dir / META_FILE).write_text(
        json.dumps(records, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote {index.ntotal} vectors to {out_dir}/")
    return index.ntotal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocols-dir", default=settings.protocols_dir)
    parser.add_argument("--index-dir", default=settings.index_dir)
    parser.add_argument("--limit", type=int, default=None, help="max documents")
    args = parser.parse_args()
    build(args.protocols_dir, args.index_dir, args.limit)


if __name__ == "__main__":
    main()
