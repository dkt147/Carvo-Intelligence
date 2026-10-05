"""Request/response contract shared with the CARVO backend.

Field names are camelCase on purpose: they must match the JSON the backend
sends (`src/modules/ai/ai.types.ts`) and expects back (`AiAnalysisResponse`).
"""

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_ID_LEN = 128
MAX_TITLE_LEN = 500
MAX_STATUS_LEN = 64
MAX_DESCRIPTION_LEN = 8_000
MAX_PROTOCOL_CONTENT_LEN = 32_000
MAX_EVENTS = 50
MAX_ACTIONS = 50
MAX_METADATA_CHARS = 2_000


def _bounded_metadata(value: Any) -> Any:
    if value is None:
        return None
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except TypeError as exc:
        raise ValueError("event metadata is not serializable") from exc
    if len(text) > MAX_METADATA_CHARS:
        raise ValueError(
            f"event metadata exceeds {MAX_METADATA_CHARS} characters"
        )
    return value


class SituationInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(max_length=MAX_ID_LEN)
    title: str = Field(max_length=MAX_TITLE_LEN)
    description: str | None = Field(default=None, max_length=MAX_DESCRIPTION_LEN)
    status: str = Field(max_length=MAX_STATUS_LEN)


class EventItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(max_length=MAX_ID_LEN)
    type: str = Field(max_length=MAX_STATUS_LEN)
    title: str = Field(max_length=MAX_TITLE_LEN)
    description: str | None = Field(default=None, max_length=MAX_DESCRIPTION_LEN)
    occurredAt: str | None = Field(default=None, max_length=64)
    metadata: Any | None = None

    @field_validator("metadata")
    @classmethod
    def _cap_metadata(cls, value: Any) -> Any:
        return _bounded_metadata(value)


class ActionItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(max_length=MAX_ID_LEN)
    title: str = Field(max_length=MAX_TITLE_LEN)
    description: str | None = Field(default=None, max_length=MAX_DESCRIPTION_LEN)
    status: str = Field(max_length=MAX_STATUS_LEN)
    dueDate: str | None = Field(default=None, max_length=64)


class ProjectContext(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(max_length=MAX_ID_LEN)
    name: str = Field(max_length=MAX_TITLE_LEN)
    description: str | None = Field(default=None, max_length=MAX_DESCRIPTION_LEN)
    pointA: str | None = Field(default=None, max_length=MAX_DESCRIPTION_LEN)
    pointB: str | None = Field(default=None, max_length=MAX_DESCRIPTION_LEN)
    status: str = Field(max_length=MAX_STATUS_LEN)


class AnalysisContext(BaseModel):
    model_config = ConfigDict(extra="ignore")

    events: list[EventItem] = Field(default_factory=list, max_length=MAX_EVENTS)
    actions: list[ActionItem] = Field(default_factory=list, max_length=MAX_ACTIONS)
    project: ProjectContext | None = None


class ProtocolContext(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(max_length=MAX_ID_LEN)
    version: int
    content: str = Field(max_length=MAX_PROTOCOL_CONTENT_LEN)


class AnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    requestId: str = Field(max_length=MAX_ID_LEN)
    situation: SituationInput
    context: AnalysisContext = Field(default_factory=AnalysisContext)
    protocol: ProtocolContext | None = None


class Recommendation(BaseModel):
    title: str
    description: str


class AnalysisResponse(BaseModel):
    requestId: str
    status: Literal["COMPLETED", "FAILED"]
    summary: str | None = None
    reasoning: str | None = None
    protocolVersionId: str | None = None
    recommendations: list[Recommendation] = Field(default_factory=list)
    externalJobId: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
