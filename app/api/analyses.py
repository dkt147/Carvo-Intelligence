import logging
from functools import lru_cache

from fastapi import APIRouter, Depends, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from app.analysis.analyzer import Analyzer
from app.config import settings
from app.providers.factory import build_embedder, build_llm_provider
from app.providers.openai_provider import LLMProvider
from app.retrieval.retriever import FaissRetriever, Retriever
from app.schemas.analysis import AnalysisRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["analyses"])


@lru_cache
def _provider() -> LLMProvider:
    return build_llm_provider()


@lru_cache
def _retriever() -> Retriever | None:
    try:
        retriever = FaissRetriever.load(settings.index_dir, build_embedder())
        logger.info("Loaded FAISS retrieval index from %s", settings.index_dir)
        return retriever
    except FileNotFoundError:
        logger.warning(
            "No retrieval index at %s - running without protocol retrieval. "
            "Build one with: python -m app.ingestion.build_index",
            settings.index_dir,
        )
        return None
    except ImportError:
        logger.warning("Embedder unavailable (missing dependency) - retrieval disabled.")
        return None


def get_analyzer() -> Analyzer:
    return Analyzer(_provider(), _retriever(), top_k=settings.retrieval_top_k)


def _auth_error(authorization: str | None) -> str | None:
    expected = settings.ai_service_api_key
    if not expected:
        return None
    if authorization != f"Bearer {expected}":
        return "Invalid or missing API key"
    return None


@router.post("/analyses")
async def create_analysis(
    payload: AnalysisRequest,
    request: Request,
    analyzer: Analyzer = Depends(get_analyzer),
) -> JSONResponse:
    error = _auth_error(request.headers.get("authorization"))
    if error:
        return JSONResponse(status_code=401, content={"error": error})

    result = await run_in_threadpool(analyzer.analyze, payload)
    return JSONResponse(content=result.model_dump(exclude_none=True))
