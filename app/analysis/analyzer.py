import logging
from typing import Any

from app.analysis.prompts import (
    SYSTEM_PROMPT,
    build_retrieval_query,
    build_user_prompt,
)
from app.providers.openai_provider import LLMProvider
from app.retrieval.retriever import Retriever
from app.schemas.analysis import AnalysisRequest, AnalysisResponse, Recommendation

logger = logging.getLogger(__name__)

ANALYSIS_VERSION = "phase-2"


class Analyzer:
    def __init__(
        self,
        provider: LLMProvider,
        retriever: Retriever | None = None,
        top_k: int = 6,
    ) -> None:
        self._provider = provider
        self._retriever = retriever
        self._top_k = top_k

    def analyze(self, request: AnalysisRequest) -> AnalysisResponse:
        try:
            retrieved = self._retrieve(request)
            raw = self._provider.complete_json(
                SYSTEM_PROMPT, build_user_prompt(request, retrieved)
            )
            response = self._to_response(request, raw)
            response.metadata["retrievedCount"] = len(retrieved)
            if retrieved:
                response.metadata.setdefault(
                    "retrievedSources",
                    sorted({c.source for c in retrieved}),
                )
            return response
        except Exception as exc:  # noqa: BLE001 - failures must go back on the contract
            logger.exception("analysis failed for requestId=%s", request.requestId)
            return AnalysisResponse(
                requestId=request.requestId,
                status="FAILED",
                error=str(exc) or exc.__class__.__name__,
                metadata=self._base_metadata(),
            )

    def _retrieve(self, request: AnalysisRequest):
        if self._retriever is None:
            return []
        try:
            return self._retriever.search(build_retrieval_query(request), self._top_k)
        except Exception:  # noqa: BLE001 - retrieval is best-effort, never fatal
            logger.exception("retrieval failed for requestId=%s", request.requestId)
            return []

    def _to_response(
        self, request: AnalysisRequest, raw: dict[str, Any]
    ) -> AnalysisResponse:
        recommendations = [
            Recommendation(title=item["title"], description=item["description"])
            for item in raw.get("recommendations", [])
            if isinstance(item, dict)
            and isinstance(item.get("title"), str)
            and isinstance(item.get("description"), str)
        ]

        metadata = self._base_metadata()
        model_metadata = raw.get("metadata")
        if isinstance(model_metadata, dict):
            metadata.update(model_metadata)

        protocol_version_id = raw.get("protocolVersionId")
        if not isinstance(protocol_version_id, str):
            protocol_version_id = None

        return AnalysisResponse(
            requestId=request.requestId,
            status="COMPLETED",
            summary=_as_str(raw.get("summary")),
            reasoning=_as_str(raw.get("reasoning")),
            protocolVersionId=protocol_version_id,
            recommendations=recommendations,
            metadata=metadata,
        )

    def _base_metadata(self) -> dict[str, Any]:
        return {
            "analysisVersion": ANALYSIS_VERSION,
            "model": getattr(self._provider, "model", "unknown"),
        }


def _as_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None
