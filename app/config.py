"""
Application Configuration
==========================
Loads settings from environment variables / .env file.
Uses Pydantic BaseSettings for validation and type coercion.
"""

import os
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # ── App ──────────────────────────────────────────────────
    APP_NAME: str = "Shipment Delay Early Warning System"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = False

    # ── Database ─────────────────────────────────────────────
    DATABASE_URL: str = "sqlite:///./shipment_delay.db"

    # ── JWT ──────────────────────────────────────────────────
    JWT_SECRET_KEY: str = "change-me-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRY_HOURS: int = 24

    # ── Redis ────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_ENABLED: bool = False

    # ── External APIs ────────────────────────────────────────
    OPENWEATHER_API_KEY: str = ""
    OPENWEATHER_ENABLED: bool = False

    # ── Rate Limiting ────────────────────────────────────────
    RATE_LIMIT_PER_MINUTE: int = 100

    # ── Logging ──────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"  # "json" or "text"

    # ── ML Models ────────────────────────────────────────────
    DEFAULT_MODEL: str = "xgboost"
    MODELS_DIR: str = "models"

    @property
    def BASE_DIR(self) -> str:
        """Project root directory."""
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    @property
    def MODELS_PATH(self) -> str:
        """Absolute path to models directory."""
        return os.path.join(self.BASE_DIR, self.MODELS_DIR)

    @property
    def DATA_DIR(self) -> str:
        """Absolute path to data directory."""
        return os.path.join(self.BASE_DIR, "data")

    @property
    def OUTPUTS_DIR(self) -> str:
        """Absolute path to outputs directory."""
        return os.path.join(self.BASE_DIR, "outputs")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Cached settings singleton. Call this everywhere you need config."""
    return Settings()
