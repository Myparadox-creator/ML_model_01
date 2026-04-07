"""
Logging Middleware
====================
Structured request/response logging with timing, request IDs, and error tracking.
Supports both JSON and text output formats.
"""

import time
import uuid
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("api.access")


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware that logs every request with:
    - Request ID (unique per request, added to response headers)
    - Method + path
    - Response status code
    - Duration in milliseconds
    - Client IP
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        # Generate unique request ID
        request_id = str(uuid.uuid4())[:8]
        start_time = time.time()

        # Add request ID to state for downstream use
        request.state.request_id = request_id

        # Process request
        try:
            response = await call_next(request)
        except Exception as e:
            # Log unhandled exceptions
            duration_ms = (time.time() - start_time) * 1000
            logger.error(
                f"[{request_id}] {request.method} {request.url.path} → 500 "
                f"({duration_ms:.1f}ms) ERROR: {str(e)}"
            )
            raise

        duration_ms = (time.time() - start_time) * 1000

        # Add headers to response
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{duration_ms:.1f}ms"

        # Log the request
        status_code = response.status_code
        client_ip = request.client.host if request.client else "unknown"

        # Color-code by status
        if status_code >= 500:
            log_level = logging.ERROR
        elif status_code >= 400:
            log_level = logging.WARNING
        else:
            log_level = logging.INFO

        logger.log(
            log_level,
            f"[{request_id}] {request.method} {request.url.path} → {status_code} "
            f"({duration_ms:.1f}ms) [{client_ip}]",
        )

        return response


def setup_logging(log_level: str = "INFO", log_format: str = "text"):
    """
    Configure application logging.
    
    Args:
        log_level: DEBUG, INFO, WARNING, ERROR
        log_format: "json" or "text"
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    if log_format == "json":
        import json

        class JsonFormatter(logging.Formatter):
            def format(self, record):
                log_data = {
                    "timestamp": self.formatTime(record),
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                }
                if record.exc_info:
                    log_data["exception"] = self.formatException(record.exc_info)
                return json.dumps(log_data)

        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s │ %(levelname)-8s │ %(name)-20s │ %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root_logger.addHandler(console)

    # Suppress noisy loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
