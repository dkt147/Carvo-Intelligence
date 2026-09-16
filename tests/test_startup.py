import logging

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import app
from app.runtime import initialize_resources, load_retriever, readiness_payload
from tests.test_analyses import VALID_PAYLOAD


def test_missing_index_does_not_load_embedder(tmp_path, monkeypatch):
    def _boom():
        raise AssertionError("embedder must not load when the index is absent")

    monkeypatch.setattr("app.runtime.build_embedder", _boom)
    retriever, size, embedder_ready = load_retriever(
        Settings(index_dir=str(tmp_path), llm_api_key="k")
    )
    assert retriever is None
    assert size == 0
    assert embedder_ready is False


def test_missing_manifest_fails_startup(tmp_path, monkeypatch):
    (tmp_path / "index.faiss").write_bytes(b"not-a-faiss-index")
    (tmp_path / "chunks.json").write_text("[]", encoding="utf-8")
    monkeypatch.setattr("app.runtime.build_embedder", lambda: object())

    with pytest.raises(RuntimeError, match="manifest.json"):
        load_retriever(Settings(index_dir=str(tmp_path), llm_api_key="k"))


def test_health_stays_ok_when_not_ready():
    with TestClient(app) as client:
        health = client.get("/health")
        ready = client.get("/ready")

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert ready.status_code == 503
    body = ready.json()
    assert body["status"] == "not_ready"
    assert body["llm"] is True
    assert body["retriever"] is False
    assert body["embedder"] is False
    assert body["indexSize"] == 0


def test_ready_ok_when_resources_present():
    class _Provider:
        model = "ready-model"

    class _Index:
        ntotal = 4

    class _Retriever:
        _index = _Index()

    app_state = app.state
    previous = {
        "llm_provider": getattr(app_state, "llm_provider", None),
        "retriever": getattr(app_state, "retriever", None),
        "index_size": getattr(app_state, "index_size", 0),
        "embedder_ready": getattr(app_state, "embedder_ready", False),
    }
    try:
        app_state.llm_provider = _Provider()
        app_state.retriever = _Retriever()
        app_state.index_size = 4
        app_state.embedder_ready = True
        payload = readiness_payload(app)
        assert payload["status"] == "ok"
        assert payload["indexSize"] == 4
        assert payload["retriever"] is True
        assert payload["embedder"] is True
    finally:
        for key, value in previous.items():
            setattr(app_state, key, value)


def test_analyses_use_provider_from_app_state():
    class FakeProvider:
        model = "from-state"

        def complete_json(self, system: str, user: str) -> dict:
            return {"summary": "from-state", "recommendations": []}

    with TestClient(app) as client:
        client.app.state.llm_provider = FakeProvider()
        client.app.state.retriever = None
        body = client.post("/api/v1/analyses", json=VALID_PAYLOAD).json()

    assert body["status"] == "COMPLETED"
    assert body["metadata"]["model"] == "from-state"
    assert body["metadata"]["retrievedCount"] == 0


def test_access_log_has_request_id_path_status_duration_without_secrets(caplog):
    caplog.set_level(logging.INFO, logger="carvo.access")
    with TestClient(app) as client:
        response = client.get(
            "/health",
            headers={"Authorization": "Bearer super-secret-token", "X-Request-Id": "req-log-1"},
        )

    assert response.status_code == 200
    assert response.headers["x-request-id"] == "req-log-1"
    text = caplog.text
    assert "request_id=req-log-1" in text
    assert "path=/health" in text
    assert "status=200" in text
    assert "duration_ms=" in text
    assert "super-secret-token" not in text
    assert "Minister missed" not in text


def test_access_log_does_not_record_analysis_body(caplog):
    class FakeProvider:
        model = "fake-model"

        def complete_json(self, system: str, user: str) -> dict:
            return {"summary": "ok"}

    caplog.set_level(logging.INFO, logger="carvo.access")
    with TestClient(app) as client:
        client.app.state.llm_provider = FakeProvider()
        client.post("/api/v1/analyses", json=VALID_PAYLOAD)

    text = caplog.text
    assert "path=/api/v1/analyses" in text
    assert "Minister missed a committed milestone" not in text
    assert "LLM_API_KEY" not in text


def test_initialize_resources_fails_on_missing_manifest(tmp_path, monkeypatch):
    (tmp_path / "index.faiss").write_bytes(b"not-a-faiss-index")
    (tmp_path / "chunks.json").write_text("[]", encoding="utf-8")
    monkeypatch.setattr("app.runtime.build_embedder", lambda: object())
    monkeypatch.setattr("app.runtime.build_llm_provider", lambda: object())

    class _App:
        state = type("S", (), {})()

    with pytest.raises(RuntimeError, match="manifest.json"):
        initialize_resources(
            _App(),  # type: ignore[arg-type]
            Settings(index_dir=str(tmp_path), llm_api_key="k", llm_provider="groq"),
        )
