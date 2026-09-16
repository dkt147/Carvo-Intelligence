import logging

from fastapi import APIRouter, Depends, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from app.analysis.analyzer import Analyzer
from app.api.auth import require_api_key
from app.config import settings
from app.schemas.analysis import AnalysisRequest

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1",
    tags=["analyses"],
    dependencies=[Depends(require_api_key)],
)


def get_analyzer(request: Request) -> Analyzer:
    return Analyzer(
        request.app.state.llm_provider,
        getattr(request.app.state, "retriever", None),
        top_k=settings.retrieval_top_k,
    )


@router.post("/analyses")
async def create_analysis(
    payload: AnalysisRequest,
    analyzer: Analyzer = Depends(get_analyzer),
) -> JSONResponse:
    result = await run_in_threadpool(analyzer.analyze, payload)
    return JSONResponse(content=result.model_dump(exclude_none=True))
