# CARVO Intelligence Service

Standalone Python (FastAPI) service that implements the CARVO backend's external
analysis contract. Phase 1: contract-compatible analysis backed by OpenAI.

The backend calls `POST {AI_SERVICE_URL}/api/v1/analyses` (default
`http://localhost:8000`) and expects the final `AiAnalysisResponse` synchronously.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | liveness check (process is up) |
| `GET` | `/ready` | readiness: LLM, FAISS index, embedder. HTTP 503 if analysis cannot run |
| `POST` | `/api/v1/analyses` | run an analysis for one situation |

Request body matches `src/modules/ai/ai.types.ts` (`requestId`, `situation`,
`context.events`, `context.actions`, optional `context.project`, optional
`protocol`). Response matches `AiAnalysisResponse` (`requestId`, `status`,
`summary`, `reasoning`, `protocolVersionId`, `recommendations[]`, `metadata`).
`requestId` is always echoed back. Internal failures return HTTP 200 with
`status: "FAILED"` and `error`; bad input returns HTTP 422 with `{ "error": ... }`.

## Scope (Phase 1)

- Parse request, interpret situation + events + actions (+ project/protocol when supplied).
- Produce structured summary, reasoning, recommendations.
- Conservative evidence policy: no invented protocol rules, facts, or scores;
  uncertainty and missing-info questions go under `metadata`.
- Knowledge/protocol retrieval and document ingestion are **not** in this phase —
  the service works on whatever context the backend sends.

## Run

```bash
cd D:\carvo-intelligence
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env   # then set LLM_API_KEY for the chosen LLM_PROVIDER
uvicorn app.main:app --port 8000 --reload
```

The API process exits at startup if `LLM_PROVIDER` is not `groq` or `openai`,
or if `LLM_API_KEY` / the resolved model is empty. Analyses are not served
with a missing key.

### Providers

- **LLM**: `LLM_PROVIDER=groq` (default, free tier) or `openai`. Set
  `LLM_API_KEY` to that provider's key only — there is no Groq/OpenAI fallback.
  OpenAI-compatible endpoints (e.g. Ollama) use `LLM_PROVIDER=openai` with
  `LLM_BASE_URL` / `LLM_MODEL`.
- **Embeddings**: `EMBEDDINGS_PROVIDER=local` uses `fastembed`
  (`intfloat/multilingual-e5-small`, ONNX, no API key, ~130 MB one-time download,
  handles Hebrew). `openai` uses `text-embedding-3-small` and `OPENAI_API_KEY`
  (embeddings only, never the chat LLM).

### Build the retrieval index

```bash
.venv\Scripts\python -m app.ingestion.build_index          # all protocol docs
.venv\Scripts\python -m app.ingestion.build_index --limit 3   # quick smoke test
```

Writes `data/index/{index.faiss,chunks.json}`; the service auto-loads it on boot.
Without an index the service still runs (retrieval simply returns nothing).

## Test

```bash
pytest
```

Tests stub the LLM provider. A dummy `LLM_API_KEY` is set in `tests/conftest.py`
so startup validation can run; no real provider key is required.

## Config (`.env`)

| Var | Default | Notes |
| --- | --- | --- |
| `PORT` | `8000` | must match backend `AI_SERVICE_URL` |
| `AI_SERVICE_API_KEY` | *(empty)* | if set, `Authorization: Bearer <key>` is required |
| `LLM_PROVIDER` | `groq` | `groq` \| `openai` (required; unknown values fail startup) |
| `LLM_API_KEY` | *(empty)* | **required to start**. Key for `LLM_PROVIDER` only |
| `LLM_MODEL` / `LLM_BASE_URL` | *(provider default)* | override only if needed |
| `EMBEDDINGS_PROVIDER` | `local` | `local` (fastembed, free) \| `openai` |
| `OPENAI_API_KEY` | *(empty)* | embeddings only, when `EMBEDDINGS_PROVIDER=openai` |
| `LOCAL_EMBEDDING_MODEL` | `intfloat/multilingual-e5-small` | fastembed model id |
| `PROTOCOLS_DIR` | `D:/protocolscarvo` | source `.docx` protocols |
| `INDEX_DIR` | `data/index` | where the FAISS index is written |
| `RETRIEVAL_TOP_K` | `6` | chunks injected per analysis |
| `LOG_LEVEL` | `INFO` | |
