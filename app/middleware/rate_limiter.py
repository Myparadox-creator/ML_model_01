"""
Rate Limiter Middleware
========================
Token bucket rate limiting per client IP.
Protects the API from abuse while allowing burst traffic.
"""

import time
import logging
from collections import defaultdict
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class RateLimiter(BaseHTTPMiddleware):
    """
    Token bucket rate limiter.
    
    - Each client IP gets a bucket of tokens
    - Tokens are replenished at a fixed rate
    - Each request consumes one token
    - If no tokens remain, the request is rejected with 429
    """

    def __init__(self, app, requests_per_minute: int = None):
        super().__init__(app)
        self.rpm = requests_per_minute or settings.RATE_LIMIT_PER_MINUTE
        self.buckets: dict = defaultdict(lambda: {"tokens": self.rpm, "last_refill": time.time()})

    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for docs and health check
        if request.url.path in ("/docs", "/redoc", "/openapi.json", "/health"):
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        bucket = self.buckets[client_ip]

        # Refill tokens based on elapsed time
        now = time.time()
        elapsed = now - bucket["last_refill"]
        tokens_to_add = elapsed * (self.rpm / 60.0)  # tokens per second
        bucket["tokens"] = min(self.rpm, bucket["tokens"] + tokens_to_add)
        bucket["last_refill"] = now

        # Check if request is allowed
        if bucket["tokens"] < 1:
            logger.warning(f"🚫 Rate limit exceeded for {client_ip}")
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Rate limit exceeded. Please wait before making more requests.",
                    "retry_after_seconds": int(60 / self.rpm),
                },
                headers={
                    "Retry-After": str(int(60 / self.rpm)),
                    "X-RateLimit-Limit": str(self.rpm),
                    "X-RateLimit-Remaining": "0",
                },
            )

        # Consume a token
        bucket["tokens"] -= 1

        response = await call_next(request)

        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(self.rpm)
        response.headers["X-RateLimit-Remaining"] = str(int(bucket["tokens"]))

        return response
