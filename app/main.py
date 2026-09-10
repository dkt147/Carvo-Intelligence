import logging

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.analyses import router as analyses_router
from app.config import settings

logging.basicConfig(level=settings.log_level.upper())

app = FastAPI(title="CARVO Intelligence Service", version="0.1.0")
app.include_router(analyses_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


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
