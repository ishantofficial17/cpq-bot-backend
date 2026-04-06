"""
middleware.py — FastAPI middleware for request tracing and access logging.

Assigns a unique request_id to every incoming request and logs:
  - method, path, status_code, duration_ms on completion
  - Attaches request_id as a response header (X-Request-ID) so clients can
    reference it when reporting issues.
"""

import time
import uuid
import logging
import contextvars

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

log = logging.getLogger("access")

# Context variable so any code in the request path can access the request_id
current_request_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "current_request_id", default=""
)


class RequestTracingMiddleware(BaseHTTPMiddleware):
    """
    Assigns a unique request_id and logs access info for every request.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = str(uuid.uuid4())[:8]  # short 8-char ID for readability
        current_request_id.set(request_id)

        # Attach to request state so endpoint handlers can access it
        request.state.request_id = request_id

        start = time.perf_counter()
        response: Response | None = None

        try:
            response = await call_next(request)
            return response
        finally:
            duration_ms = round((time.perf_counter() - start) * 1000, 1)
            status_code = response.status_code if response else 500

            # Add request_id header to response
            if response is not None:
                response.headers["X-Request-ID"] = request_id

            log.info(
                "%s %s → %d (%.1fms)",
                request.method,
                request.url.path,
                status_code,
                duration_ms,
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": status_code,
                    "duration_ms": duration_ms,
                },
            )
