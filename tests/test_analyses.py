from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.analysis.analyzer import Analyzer
from app.api.analyses import get_analyzer
from app.main import app

VALID_PAYLOAD = {
    "requestId": "req-123",
    "situation": {
        "id": "sit-1",
        "title": "Minister missed a committed milestone",
        "description": "The milestone due last week was not delivered.",
        "status": "OPEN",
    },
    "context": {
        "events": [
            {
                "id": "evt-1",
                "type": "PROMISE",
                "title": "Committed to deliver by Friday",
                "occurredAt": "2026-09-01T10:00:00.000Z",
            }
        ],
        "actions": [
            {"id": "act-1", "title": "Prepare report", "status": "IN_PROGRESS"}
        ],
        "project": None,
    },
    "protocol": None,
}


class FakeProvider:
    model = "fake-model"

    def __init__(self, payload: dict[str, Any] | None = None, exc: Exception | None = None):
        self._payload = payload
        self._exc = exc

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        if self._exc is not None:
            raise self._exc
        return self._payload or {}


def client_with(provider: FakeProvider) -> TestClient:
    app.dependency_overrides[get_analyzer] = lambda: Analyzer(provider)
    return TestClient(app)


@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    app.dependency_overrides.clear()


def test_health():
    assert TestClient(app).get("/health").json() == {"status": "ok"}


def test_completed_analysis_echoes_request_id_and_passes_recommendations():
    provider = FakeProvider(
        {
            "summary": "The commitment was not met.",
            "reasoning": "Event evt-1 records a promise; no completion event exists.",
            "recommendations": [
                {"title": "Follow up", "description": "Ask for a revised delivery date."}
            ],
            "protocolVersionId": None,
            "metadata": {"uncertainties": ["No protocol supplied"]},
        }
    )
    body = client_with(provider).post("/api/v1/analyses", json=VALID_PAYLOAD).json()

    assert body["requestId"] == "req-123"
    assert body["status"] == "COMPLETED"
    assert body["recommendations"] == [
        {"title": "Follow up", "description": "Ask for a revised delivery date."}
    ]
    assert body["metadata"]["analysisVersion"] == "phase-2"
    assert body["metadata"]["model"] == "fake-model"
    assert body["metadata"]["retrievedCount"] == 0
    assert body["metadata"]["uncertainties"] == ["No protocol supplied"]


def test_malformed_recommendations_are_dropped():
    provider = FakeProvider(
        {
            "summary": "x",
            "recommendations": [
                {"title": "kept", "description": "ok"},
                {"title": "missing description"},
                "not-an-object",
            ],
        }
    )
    body = client_with(provider).post("/api/v1/analyses", json=VALID_PAYLOAD).json()

    assert body["recommendations"] == [{"title": "kept", "description": "ok"}]


def test_provider_failure_returns_failed_status_on_contract():
    provider = FakeProvider(exc=RuntimeError("model unavailable"))
    response = client_with(provider).post("/api/v1/analyses", json=VALID_PAYLOAD)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "FAILED"
    assert body["requestId"] == "req-123"
    assert body["error"] == "model unavailable"


def test_invalid_request_returns_error_shape():
    provider = FakeProvider({})
    response = client_with(provider).post(
        "/api/v1/analyses", json={"requestId": "only-this"}
    )

    assert response.status_code == 422
    assert response.json()["error"] == "Invalid analysis request"


def test_api_key_is_enforced_when_configured(monkeypatch):
    monkeypatch.setattr("app.api.analyses.settings.ai_service_api_key", "secret")
    c = client_with(FakeProvider({"summary": "ok"}))

    assert c.post("/api/v1/analyses", json=VALID_PAYLOAD).status_code == 401
    assert (
        c.post(
            "/api/v1/analyses",
            json=VALID_PAYLOAD,
            headers={"Authorization": "Bearer wrong"},
        ).status_code
        == 401
    )
    assert (
        c.post(
            "/api/v1/analyses",
            json=VALID_PAYLOAD,
            headers={"Authorization": "Bearer secret"},
        ).status_code
        == 200
    )
