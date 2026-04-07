"""
Cache Service
===============
Simple caching layer with Redis support and in-memory fallback.
If Redis is not available or not enabled, uses a basic dict cache with TTL.

Usage:
    cache = get_cache()
    cache.set("key", value, ttl=300)
    result = cache.get("key")
"""

import time
import json
import logging
from typing import Any, Optional

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class InMemoryCache:
    """
    Simple in-memory cache with TTL support.
    Not suitable for multi-process deployments — use Redis for that.
    """

    def __init__(self):
        self._store: dict = {}
        self._expiry: dict = {}

    def get(self, key: str) -> Optional[Any]:
        """Get a cached value. Returns None if expired or not found."""
        if key in self._store:
            if key in self._expiry and time.time() > self._expiry[key]:
                # Expired — clean up
                del self._store[key]
                del self._expiry[key]
                return None
            return self._store[key]
        return None

    def set(self, key: str, value: Any, ttl: int = 300):
        """Set a cache value with TTL in seconds (default: 5 min)."""
        self._store[key] = value
        self._expiry[key] = time.time() + ttl

    def delete(self, key: str):
        """Remove a cached value."""
        self._store.pop(key, None)
        self._expiry.pop(key, None)

    def clear(self):
        """Clear all cached values."""
        self._store.clear()
        self._expiry.clear()

    def stats(self) -> dict:
        """Get cache statistics."""
        # Clean expired entries
        now = time.time()
        expired = [k for k, exp in self._expiry.items() if now > exp]
        for k in expired:
            self.delete(k)

        return {
            "backend": "in-memory",
            "entries": len(self._store),
            "status": "active",
        }


class RedisCache:
    """
    Redis-backed cache for production deployments.
    Falls back to InMemoryCache if Redis connection fails.
    """

    def __init__(self, redis_url: str):
        self._fallback = InMemoryCache()
        self._redis = None

        try:
            import redis
            self._redis = redis.from_url(redis_url, decode_responses=True)
            self._redis.ping()
            logger.info("✅ Redis cache connected")
        except ImportError:
            logger.warning("⚠️ redis package not installed — using in-memory cache")
            self._redis = None
        except Exception as e:
            logger.warning(f"⚠️ Redis connection failed: {e} — using in-memory cache")
            self._redis = None

    def get(self, key: str) -> Optional[Any]:
        if self._redis:
            try:
                value = self._redis.get(key)
                if value:
                    return json.loads(value)
                return None
            except Exception:
                return self._fallback.get(key)
        return self._fallback.get(key)

    def set(self, key: str, value: Any, ttl: int = 300):
        if self._redis:
            try:
                self._redis.setex(key, ttl, json.dumps(value, default=str))
                return
            except Exception:
                pass
        self._fallback.set(key, value, ttl)

    def delete(self, key: str):
        if self._redis:
            try:
                self._redis.delete(key)
                return
            except Exception:
                pass
        self._fallback.delete(key)

    def clear(self):
        if self._redis:
            try:
                self._redis.flushdb()
                return
            except Exception:
                pass
        self._fallback.clear()

    def stats(self) -> dict:
        if self._redis:
            try:
                info = self._redis.info("memory")
                return {
                    "backend": "redis",
                    "used_memory": info.get("used_memory_human", "unknown"),
                    "keys": self._redis.dbsize(),
                    "status": "connected",
                }
            except Exception:
                pass
        return self._fallback.stats()


# ── Singleton ────────────────────────────────────────────────────────────────
_cache = None


def get_cache():
    """Get or create the global cache instance."""
    global _cache
    if _cache is None:
        if settings.REDIS_ENABLED:
            _cache = RedisCache(settings.REDIS_URL)
        else:
            _cache = InMemoryCache()
            logger.info("📦 Using in-memory cache (Redis disabled)")
    return _cache
