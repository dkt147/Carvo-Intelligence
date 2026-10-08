"""Startup-time LLM and retrieval resources (Q17)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI

from app.config import Settings
from app.providers.factory import build_embedder, build_llm_provider
from app.retrieval.retriever import INDEX_FILE, MANIFEST_FILE, META_FILE, FaissRetriever, Retriever

logger = logging.getLogger(__name__)


def load_retriever(settings: Settings) -> tuple[Retriever | None, int, bool]:
    """Load the FAISS retriever once.

    Missing index files: return (None, 0, False) without touching the embedder.
    Present but unloadable files: raise so startup fails instead of retrying
    a model download on every request.
    """
    directory = Path(settings.index_dir)
    index_path = directory / INDEX_FILE
    meta_path = directory / META_FILE
    manifest_path = directory / MANIFEST_FILE
    if not index_path.exists() or not meta_path.exists():
        logger.warning(
            "No retrieval index at %s - running without protocol retrieval. "
            "Build one with: python -m app.ingestion.build_index",
            directory,
        )
        return None, 0, False
    if not manifest_path.exists():
        raise RuntimeError(
            f"Retrieval index at {directory} is missing {MANIFEST_FILE}. "
            "Rebuild with: python -m app.ingestion.build_index"
        )

    try:
        embedder = build_embedder()
        retriever = FaissRetriever.load(directory, embedder)
        index_size = int(getattr(retriever._index, "ntotal", 0))
        logger.info(
            "Loaded FAISS retrieval index from %s (%s vectors)",
            directory,
            index_size,
        )
        return retriever, index_size, True
    except Exception as exc:  # noqa: BLE001 - must not be retried per request
        raise RuntimeError(
            f"Failed to load retrieval index from {directory}: {exc}"
        ) from exc


def initialize_resources(app: FastAPI, settings: Settings) -> None:
    try:
        settings.require_llm_config()
    except ValueError as exc:
        logger.error("LLM configuration is invalid: %s", exc)
        raise

    try:
        app.state.llm_provider = build_llm_provider()
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to initialize LLM provider: %s", exc)
        raise

    retriever, index_size, embedder_ready = load_retriever(settings)
    app.state.retriever = retriever
    app.state.index_size = index_size
    app.state.embedder_ready = embedder_ready


def readiness_payload(app: FastAPI) -> dict[str, Any]:
    llm_ok = getattr(app.state, "llm_provider", None) is not None
    retriever = getattr(app.state, "retriever", None)
    index_size = int(getattr(app.state, "index_size", 0) or 0)
    embedder_ready = bool(getattr(app.state, "embedder_ready", False))
    retriever_ok = retriever is not None
    ready = llm_ok and retriever_ok and embedder_ready and index_size > 0
    return {
        "status": "ok" if ready else "not_ready",
        "llm": llm_ok,
        "indexSize": index_size,
        "embedder": embedder_ready,
        "retriever": retriever_ok,
    }
