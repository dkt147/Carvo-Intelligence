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
PUBLIC_ERROR = "Analysis provider failed"
_TRUSTED_METADATA_KEYS = frozenset(
    {"analysisVersion", "model", "retrievedCount", "retrievedSources"}
)


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
                response.metadata["retrievedSources"] = sorted(
                    {c.source for c in retrieved}
                )
            return response
        except Exception as exc:  # noqa: BLE001 - failures must go back on the contract
            logger.exception("analysis failed for requestId=%s", request.requestId)
            return AnalysisResponse(
                requestId=request.requestId,
                status="FAILED",
                error=_public_error(exc),
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
        summary = _as_str(raw.get("summary"))
        if not summary:
            raise ValueError("Analysis produced no summary")

        recommendations = [
            Recommendation(title=item["title"], description=item["description"])
            for item in raw.get("recommendations", [])
            if isinstance(item, dict)
            and isinstance(item.get("title"), str)
            and isinstance(item.get("description"), str)
        ]

        metadata: dict[str, Any] = {}
        model_metadata = raw.get("metadata")
        if isinstance(model_metadata, dict):
            metadata.update(
                {
                    key: value
                    for key, value in model_metadata.items()
                    if key not in _TRUSTED_METADATA_KEYS
                }
            )
        metadata.update(self._base_metadata())

        protocol_version_id = raw.get("protocolVersionId")
        if not isinstance(protocol_version_id, str):
            protocol_version_id = None

        return AnalysisResponse(
            requestId=request.requestId,
            status="COMPLETED",
            summary=summary,
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


def _public_error(exc: BaseException) -> str:
    text = str(exc).strip()
    lowered = text.lower()
    if "http://" in lowered or "https://" in lowered:
        return PUBLIC_ERROR
    if isinstance(exc, ValueError) and text:
        return text[:200]
    return PUBLIC_ERROR
