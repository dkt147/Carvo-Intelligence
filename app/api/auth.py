"""Bearer auth and body-size guards for the analysis API (Q19)."""

from __future__ import annotations

import hashlib
import hmac

from fastapi import HTTPException, Request

from app.config import settings

# Match carvo-backend express.json({ limit: "2mb" }).
MAX_ANALYSIS_BODY_BYTES = 2 * 1024 * 1024
ANALYSES_PATH = "/api/v1/analyses"
AUTH_ERROR = "Invalid or missing API key"
BODY_TOO_LARGE_ERROR = "Request body too large"


def extract_bearer_token(authorization: str | None) -> str:
    if not authorization:
        return ""
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer":
        return ""
    return token.strip()


def bearer_matches(authorization: str | None, expected: str) -> bool:
    """Constant-time compare of the Bearer token against the configured secret."""
    provided = extract_bearer_token(authorization)
    return hmac.compare_digest(
        hashlib.sha256(provided.encode("utf-8")).digest(),
        hashlib.sha256(expected.encode("utf-8")).digest(),
    )


def require_api_key(request: Request) -> None:
    expected = (settings.ai_service_api_key or "").strip()
    if not expected:
        return
    if not bearer_matches(request.headers.get("authorization"), expected):
        raise HTTPException(status_code=401, detail=AUTH_ERROR)


def analyses_auth_and_size_response(request: Request):
    """Return a Response if this analyses request should be rejected before parsing.

    Checks Authorization (when configured) then Content-Length. Does not read
    the body or log the token.
    """
    from fastapi.responses import JSONResponse

    if request.method != "POST" or request.url.path != ANALYSES_PATH:
        return None

    expected = (settings.ai_service_api_key or "").strip()
    if expected and not bearer_matches(request.headers.get("authorization"), expected):
        return JSONResponse(status_code=401, content={"error": AUTH_ERROR})

    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            declared = int(content_length)
        except ValueError:
            return JSONResponse(
                status_code=413, content={"error": BODY_TOO_LARGE_ERROR}
            )
        if declared > MAX_ANALYSIS_BODY_BYTES:
            return JSONResponse(
                status_code=413, content={"error": BODY_TOO_LARGE_ERROR}
            )
    return None
