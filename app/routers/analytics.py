"""
Analytics Router
==================
Dashboard analytics and model performance endpoints.

Endpoints:
    GET /api/v1/analytics/dashboard    — risk distribution + route stats
    GET /api/v1/analytics/model-info   — model performance metrics
    GET /api/v1/analytics/trends       — delay trends over time
    GET /api/v1/analytics/weather      — current weather for cities
    GET /api/v1/analytics/traffic      — traffic for a route
"""

import os
import json
import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.database import get_db
from app.auth.dependencies import get_current_user
from app.models.user import User
from app.models.shipment import Shipment
from app.models.prediction import Prediction
from app.models.alert import Alert
from app.config import get_settings
from app.services.weather_service import get_weather, get_route_weather
from app.services.traffic_service import get_traffic
from app.services.cache_service import get_cache

logger = logging.getLogger(__name__)
settings = get_settings()
router = APIRouter(prefix="/api/v1/analytics", tags=["Analytics"])


@router.get("/dashboard", summary="Dashboard overview")
def get_dashboard(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Comprehensive dashboard data including:
    - Total shipments, predictions, and alerts
    - Risk distribution (LOW/MEDIUM/HIGH percentages)
    - Route-level delay rates
    """
    cache = get_cache()
    cached = cache.get("analytics:dashboard")
    if cached:
        return cached

    # Counts
    total_shipments = db.query(Shipment).count()
    total_predictions = db.query(Prediction).count()
    active_alerts = db.query(Alert).filter(Alert.is_resolved == False).count()

    # Risk distribution from predictions
    high = db.query(Prediction).filter(Prediction.risk_level == "HIGH").count()
    medium = db.query(Prediction).filter(Prediction.risk_level == "MEDIUM").count()
    low = db.query(Prediction).filter(Prediction.risk_level == "LOW").count()
    total_pred = high + medium + low

    risk_distribution = []
    if total_pred > 0:
        risk_distribution = [
            {"name": "Low Risk", "value": round(low / total_pred * 100), "color": "#22c55e"},
            {"name": "Medium Risk", "value": round(medium / total_pred * 100), "color": "#eab308"},
            {"name": "High Risk", "value": round(high / total_pred * 100), "color": "#ef4444"},
        ]

    # Route stats
    route_stats = []
    for route_type in ["highway", "local", "mixed"]:
        total_rt = db.query(Shipment).filter(Shipment.route_type == route_type).count()
        delayed_rt = db.query(Shipment).filter(
            Shipment.route_type == route_type,
            Shipment.status == "delayed",
        ).count()
        rate = round((delayed_rt / total_rt * 100) if total_rt > 0 else 0)
        route_stats.append({"route": route_type.capitalize(), "delay_rate": rate, "total": total_rt})

    result = {
        "summary": {
            "total_shipments": total_shipments,
            "total_predictions": total_predictions,
            "active_alerts": active_alerts,
        },
        "risk_distribution": risk_distribution,
        "route_stats": route_stats,
    }

    cache.set("analytics:dashboard", result, ttl=60)
    return result


@router.get("/model-info", summary="Model performance metrics")
def get_model_info(
    current_user: User = Depends(get_current_user),
):
    """Get performance metrics (accuracy, precision, recall, F1, ROC-AUC) for all trained models."""
    try:
        metrics_path = os.path.join(settings.OUTPUTS_DIR, "model_metrics.json")
        if not os.path.exists(metrics_path):
            raise HTTPException(status_code=404, detail="Model metrics not found. Run training pipeline first.")

        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        # Handle both formats (list or dict with "metrics" key)
        if isinstance(metrics, dict) and "metrics" in metrics:
            metrics_list = metrics["metrics"]
        else:
            metrics_list = metrics

        best_model = max(metrics_list, key=lambda x: x.get("ROC-AUC", 0))

        return {
            "metrics": metrics_list,
            "best_model": best_model["Model"],
            "best_roc_auc": best_model["ROC-AUC"],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trends", summary="Delay trends")
def get_trends(
    days: int = 7,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get delay prediction trends over time."""
    from datetime import datetime, timedelta, timezone

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    predictions = (
        db.query(Prediction)
        .filter(Prediction.created_at >= cutoff)
        .order_by(Prediction.created_at)
        .all()
    )

    # Group by date
    daily_stats = {}
    for p in predictions:
        date_str = p.created_at.strftime("%Y-%m-%d") if p.created_at else "unknown"
        if date_str not in daily_stats:
            daily_stats[date_str] = {"date": date_str, "total": 0, "high_risk": 0, "avg_probability": 0, "sum_probability": 0}
        daily_stats[date_str]["total"] += 1
        daily_stats[date_str]["sum_probability"] += p.delay_probability
        if p.risk_level == "HIGH":
            daily_stats[date_str]["high_risk"] += 1

    # Calculate averages
    for stats in daily_stats.values():
        if stats["total"] > 0:
            stats["avg_probability"] = round(stats["sum_probability"] / stats["total"], 4)
        del stats["sum_probability"]

    return {
        "period_days": days,
        "total_predictions": len(predictions),
        "daily_trends": list(daily_stats.values()),
    }


@router.get("/weather/{city}", summary="Get weather for a city")
def get_city_weather(
    city: str,
    current_user: User = Depends(get_current_user),
):
    """Get current weather conditions (mock or real) for a specific city."""
    return get_weather(city)


@router.get("/weather", summary="Get route weather")
def get_weather_for_route(
    origin: str,
    destination: str,
    current_user: User = Depends(get_current_user),
):
    """Get weather conditions along a route (origin + destination + combined severity)."""
    return get_route_weather(origin, destination)


@router.get("/traffic", summary="Get traffic for a route")
def get_route_traffic(
    origin: str,
    destination: str,
    departure_hour: Optional[int] = None,
    current_user: User = Depends(get_current_user),
):
    """Get traffic congestion data for a route."""
    return get_traffic(origin, destination, departure_hour)
