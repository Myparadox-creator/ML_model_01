"""
🚛 Shipment Delay Early Warning System — FastAPI Application
=============================================================
Production-ready API with JWT auth, ML predictions, SHAP explanations,
and actionable recommendations for logistics delay prevention.

Usage:
    uvicorn app.main:app --reload --port 8000
    Then visit: http://localhost:8000/docs
"""

import os
import json
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

from app.config import get_settings
from app.database import create_all_tables
from app.middleware.logging_middleware import LoggingMiddleware, setup_logging
from app.middleware.rate_limiter import RateLimiter
from app.services.ml_service import get_ml_service
from app.services.recommendation import generate_recommendations

# ── Routers ──────────────────────────────────────────────────────────────────
from app.auth.router import router as auth_router
from app.routers.predictions import router as predictions_router
from app.routers.shipments import router as shipments_router
from app.routers.carriers import router as carriers_router
from app.routers.alerts import router as alerts_router
from app.routers.analytics import router as analytics_router

settings = get_settings()
logger = logging.getLogger(__name__)


# ── Lifespan (startup/shutdown) ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifecycle management.
    - Startup: configure logging, create DB tables, load ML models
    - Shutdown: cleanup
    """
    # ── STARTUP ──────────────────────────────────────────────
    setup_logging(log_level=settings.LOG_LEVEL, log_format=settings.LOG_FORMAT)

    logger.info("=" * 60)
    logger.info(f"🚛 {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info("=" * 60)

    # Create database tables
    logger.info("📦 Creating database tables...")
    create_all_tables()
    logger.info("✅ Database ready")

    # Load ML models
    logger.info("🤖 Loading ML models...")
    ml_service = get_ml_service()
    try:
        ml_service.load_models()
    except Exception as e:
        logger.error(f"⚠️ ML models failed to load: {e}")
        logger.error("   Prediction endpoints will return 503 until models are available.")
        logger.error("   Run 'python main.py' to train models first.")

    logger.info("🚀 Application ready!")
    logger.info(f"   📖 Docs: http://localhost:8000/docs")
    logger.info(f"   🔑 Register: POST /auth/register")
    logger.info(f"   🔮 Predict:  POST /api/v1/predict")

    yield  # ← Application runs here

    # ── SHUTDOWN ─────────────────────────────────────────────
    logger.info("👋 Shutting down gracefully...")


# ── FastAPI App ──────────────────────────────────────────────────────────────
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="""
## 🚛 AI-Powered Early Warning System for Shipment Delays

Predicts shipment delays **48–72 hours in advance** with:
- **ML Predictions** — XGBoost, Random Forest, Logistic Regression
- **SHAP Explanations** — "Storm risk contributed 40% to delay"
- **Actionable Recommendations** — reroute, switch carrier, adjust ETA
- **Real-time Alerts** — automatic flagging of high-risk shipments

### Getting Started
1. **Register** → `POST /auth/register`
2. **Login** → `POST /auth/login` (get JWT token)
3. **Predict** → `POST /api/v1/predict` (with Bearer token)

### Architecture
- **Backend**: FastAPI + SQLAlchemy + JWT
- **ML**: XGBoost + SHAP + scikit-learn
- **Database**: SQLite (dev) / PostgreSQL (prod)
- **Caching**: In-memory (dev) / Redis (prod)
    """,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── Middleware (order matters: last added = first executed) ───────────────────
app.add_middleware(RateLimiter)
app.add_middleware(LoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register Routers ────────────────────────────────────────────────────────
app.include_router(auth_router)
app.include_router(predictions_router)
app.include_router(shipments_router)
app.include_router(carriers_router)
app.include_router(alerts_router)
app.include_router(analytics_router)


# ── Root Endpoints ──────────────────────────────────────────────────────────
@app.get("/", tags=["System"])
def root():
    """API root — returns system info and links."""
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": "/docs",
        "endpoints": {
            "auth": "/auth/register, /auth/login",
            "predict": "/api/v1/predict",
            "shipments": "/api/v1/shipments",
            "carriers": "/api/v1/carriers",
            "alerts": "/api/v1/alerts",
            "analytics": "/api/v1/analytics/dashboard",
        },
    }


@app.get("/health", tags=["System"])
def health_check():
    """Health check endpoint for monitoring and load balancers."""
    ml_service = get_ml_service()
    return {
        "status": "healthy",
        "ml_ready": ml_service.is_ready,
        "models_loaded": ml_service.available_models,
        "version": settings.APP_VERSION,
    }


# ══════════════════════════════════════════════════════════════════════════════
# LEGACY ENDPOINTS — backward compatibility for the existing frontend
# These are PUBLIC (no JWT) so the dashboard works without login.
# The frontend calls /shipments, /model-info, /analytics, /predict directly.
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/shipments", tags=["Legacy"])
def legacy_shipments(limit: int = 15):
    """Legacy: list recent shipments from CSV (for frontend dashboard)."""
    try:
        csv_path = os.path.join(settings.DATA_DIR, "shipments.csv")
        df = pd.read_csv(csv_path)
        return df.tail(limit).iloc[::-1].to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model-info", tags=["Legacy"])
def legacy_model_info():
    """Legacy: model performance metrics (for frontend analytics)."""
    try:
        metrics_path = os.path.join(settings.OUTPUTS_DIR, "model_metrics.json")
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        if isinstance(metrics, dict) and "metrics" in metrics:
            metrics_list = metrics["metrics"]
        else:
            metrics_list = metrics
        best_model = max(metrics_list, key=lambda x: x.get("ROC-AUC", 0))
        return {"metrics": metrics_list, "best_model": best_model["Model"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analytics", tags=["Legacy"])
def legacy_analytics():
    """Legacy: risk distribution + route stats (for frontend dashboard)."""
    try:
        csv_path = os.path.join(settings.DATA_DIR, "shipments.csv")
        df = pd.read_csv(csv_path)
        total = len(df)
        if total == 0:
            return {"risk_distribution": [], "route_stats": []}

        high = len(df[df["delay_probability"] >= 0.7])
        medium = len(df[(df["delay_probability"] >= 0.4) & (df["delay_probability"] < 0.7)])
        low = total - high - medium

        risk_dist = [
            {"name": "Low Risk", "value": round((low / total) * 100), "color": "#22c55e"},
            {"name": "Medium Risk", "value": round((medium / total) * 100), "color": "#eab308"},
            {"name": "High Risk", "value": round((high / total) * 100), "color": "#ef4444"},
        ]

        route_stats = []
        for rtype in ["highway", "local", "mixed"]:
            subset = df[df["route_type"] == rtype]
            rate = round((len(subset[subset["delayed"] == 1]) / len(subset)) * 100) if len(subset) > 0 else 0
            route_stats.append({"route": rtype.capitalize(), "delay_rate": rate})

        return {"risk_distribution": risk_dist, "route_stats": route_stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


from pydantic import BaseModel


class LegacyPredictionRequest(BaseModel):
    origin: str
    destination: str
    distance_km: float
    route_type: str
    departure_hour: int
    day_of_week: int
    is_weekend: int
    carrier_reliability_score: float
    weather_severity: float
    traffic_congestion: float
    has_news_disruption: int
    model_name: str = "xgboost"


@app.post("/predict", tags=["Legacy"])
def legacy_predict(req: LegacyPredictionRequest):
    """Legacy: predict delay (for frontend predict page — no auth required)."""
    ml_service = get_ml_service()
    if not ml_service.is_ready:
        raise HTTPException(status_code=503, detail="ML models not loaded")

    shipment_data = req.model_dump(exclude={"model_name"})

    try:
        result = ml_service.predict(shipment_data, model_name=req.model_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Generate recommendations
    recommendations = generate_recommendations(result, shipment_data)

    return {
        "delay_probability": result["delay_probability"],
        "risk_level": result["risk_level"],
        "predicted_delayed": result["predicted_delayed"],
        "model_used": result["model_used"],
        "recommended_actions": recommendations,
        "reasons": result.get("reasons", []),
    }
