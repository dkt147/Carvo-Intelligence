"""Request/response contract shared with the CARVO backend.

Field names are camelCase on purpose: they must match the JSON the backend
sends (`src/modules/ai/ai.types.ts`) and expects back (`AiAnalysisResponse`).
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class SituationInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    title: str
    description: str | None = None
    status: str


class EventItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    type: str
    title: str
    description: str | None = None
    occurredAt: str | None = None
    metadata: Any | None = None


class ActionItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    title: str
    description: str | None = None
    status: str
    dueDate: str | None = None


class ProjectContext(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    name: str
    description: str | None = None
    pointA: str | None = None
    pointB: str | None = None
    status: str


class AnalysisContext(BaseModel):
    model_config = ConfigDict(extra="ignore")

    events: list[EventItem] = Field(default_factory=list)
    actions: list[ActionItem] = Field(default_factory=list)
    project: ProjectContext | None = None


class ProtocolContext(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    version: int
    content: str


class AnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    requestId: str
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
