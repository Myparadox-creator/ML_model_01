"""
Predictions Router
====================
The core prediction API — predicts shipment delays with SHAP explanations
and actionable recommendations.

Endpoints:
    POST /api/v1/predict           — run prediction on shipment data
    GET  /api/v1/predictions       — list stored predictions (paginated)
    GET  /api/v1/predictions/{id}  — get prediction details
"""

import json
import time
import logging
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session

from app.database import get_db
from app.auth.dependencies import get_current_user
from app.models.user import User
from app.models.shipment import Shipment
from app.models.prediction import Prediction as PredictionModel
from app.models.alert import Alert
from app.schemas.prediction import (
    PredictionRequest,
    PredictionResponse,
    DelayReason,
    PredictionHistoryResponse,
    PredictionListResponse,
)
from app.services.ml_service import get_ml_service
from app.services.recommendation import generate_recommendations
from app.services.cache_service import get_cache

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["Predictions"])


def _create_alert_if_high_risk(
    db: Session,
    shipment_id: str,
    delay_prob: float,
    risk_level: str,
    reasons: list,
    user_id: int = None,
):
    """Background task: create an alert for high-risk predictions."""
    if risk_level == "HIGH":
        top_reason = reasons[0]["factor"] if reasons else "High delay probability"
        alert = Alert(
            shipment_id=shipment_id,
            user_id=user_id,
            alert_type="HIGH_RISK",
            severity="HIGH" if delay_prob >= 0.8 else "MEDIUM",
            title=f"⚠️ High delay risk for {shipment_id}",
            message=f"Delay probability: {delay_prob:.0%}. Top factor: {top_reason}",
            delay_probability=delay_prob,
        )
        db.add(alert)
        db.commit()
        logger.info(f"🚨 Alert created for shipment {shipment_id}")


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict shipment delay",
    description="Predicts the probability of a shipment delay with SHAP-based explanations and corrective action recommendations.",
)
def predict_delay(
    req: PredictionRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    🔮 **Core Prediction Endpoint**
    
    Accepts shipment features and returns:
    - **delay_probability**: 0.0 to 1.0
    - **risk_level**: LOW / MEDIUM / HIGH
    - **reasons**: SHAP-based explanations of top contributing factors
    - **recommendations**: Actionable steps to mitigate delay
    """
    ml_service = get_ml_service()
    if not ml_service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML models not loaded. Please wait for system initialization.",
        )

    shipment_data = req.model_dump(exclude={"model_name"})

    # Check cache
    cache = get_cache()
    cache_key = f"prediction:{hash(json.dumps(shipment_data, sort_keys=True))}"
    cached = cache.get(cache_key)
    if cached:
        logger.info("📦 Prediction served from cache")
        return PredictionResponse(**cached)

    # Run prediction
    try:
        result = ml_service.predict(shipment_data, model_name=req.model_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    # Generate recommendations
    recommendations = generate_recommendations(result, shipment_data)
    result["recommendations"] = recommendations

    # Format reasons
    reasons = [DelayReason(**r) for r in result.get("reasons", [])]
    result["reasons"] = reasons

    # Store prediction in DB (create shipment if needed)
    try:
        import uuid
        shipment_id = f"SHP-{uuid.uuid4().hex[:8].upper()}"

        # Create shipment record
        shipment = Shipment(
            shipment_id=shipment_id,
            **{k: v for k, v in shipment_data.items() if k not in ("model_name",)},
        )
        db.add(shipment)

        # Create prediction record
        prediction_record = PredictionModel(
            shipment_id=shipment_id,
            delay_probability=result["delay_probability"],
            risk_level=result["risk_level"],
            predicted_delayed=1 if result["predicted_delayed"] else 0,
            model_used=result["model_used"],
            reasons=json.dumps([r.model_dump() for r in reasons]),
            recommendations=json.dumps(recommendations),
            prediction_time_ms=result.get("prediction_time_ms"),
        )
        db.add(prediction_record)
        db.commit()

        # Background: create alert if high risk
        background_tasks.add_task(
            _create_alert_if_high_risk,
            db, shipment_id,
            result["delay_probability"],
            result["risk_level"],
            [r.model_dump() for r in reasons],
            current_user.id,
        )
    except Exception as e:
        logger.warning(f"Failed to store prediction: {e}")
        db.rollback()

    # Cache result for 5 minutes
    response = PredictionResponse(**result)
    cache.set(cache_key, response.model_dump(), ttl=300)

    return response


@router.get(
    "/predictions",
    response_model=PredictionListResponse,
    summary="List prediction history",
)
def list_predictions(
    page: int = 1,
    per_page: int = 20,
    risk_level: str = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get paginated list of stored predictions, optionally filtered by risk level."""
    query = db.query(PredictionModel)

    if risk_level:
        query = query.filter(PredictionModel.risk_level == risk_level.upper())

    total = query.count()
    predictions = (
        query.order_by(PredictionModel.created_at.desc())
        .offset((page - 1) * per_page)
        .limit(per_page)
        .all()
    )

    return PredictionListResponse(
        predictions=[PredictionHistoryResponse.model_validate(p) for p in predictions],
        total=total,
        page=page,
        per_page=per_page,
    )


@router.get(
    "/predictions/{prediction_id}",
    response_model=PredictionHistoryResponse,
    summary="Get prediction details",
)
def get_prediction(
    prediction_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get details of a specific stored prediction."""
    prediction = db.query(PredictionModel).filter(PredictionModel.id == prediction_id).first()
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return prediction
