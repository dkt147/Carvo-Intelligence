import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.analyses import router as analyses_router
from app.config import settings
from app.runtime import initialize_resources, readiness_payload

logging.basicConfig(level=settings.log_level.upper())
logger = logging.getLogger(__name__)
access_logger = logging.getLogger("carvo.access")


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_resources(app, settings)
    yield


app = FastAPI(
    title="CARVO Intelligence Service",
    version="0.1.0",
    lifespan=lifespan,
)
app.include_router(analyses_router)


@app.middleware("http")
async def access_log(request: Request, call_next):
    started = time.perf_counter()
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    try:
        response = await call_next(request)
    except Exception:
        duration_ms = (time.perf_counter() - started) * 1000
        access_logger.info(
            "request_id=%s path=%s status=%s duration_ms=%.1f",
            request_id,
            request.url.path,
            500,
            duration_ms,
        )
        raise

    duration_ms = (time.perf_counter() - started) * 1000
    access_logger.info(
        "request_id=%s path=%s status=%s duration_ms=%.1f",
        request_id,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    response.headers["X-Request-Id"] = request_id
    return response


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready(request: Request) -> JSONResponse:
    payload = readiness_payload(request.app)
    status_code = 200 if payload["status"] == "ok" else 503
    return JSONResponse(status_code=status_code, content=payload)


@app.exception_handler(RequestValidationError)
async def _validation_handler(
    _request: Request, exc: RequestValidationError
) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content=jsonable_encoder(
            {"error": "Invalid analysis request", "details": exc.errors()}
        ),
    )
