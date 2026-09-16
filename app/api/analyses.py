import logging

from fastapi import APIRouter, Depends, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from app.analysis.analyzer import Analyzer
from app.config import settings
from app.schemas.analysis import AnalysisRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["analyses"])


def get_analyzer(request: Request) -> Analyzer:
    return Analyzer(
        request.app.state.llm_provider,
        getattr(request.app.state, "retriever", None),
        top_k=settings.retrieval_top_k,
    )


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
