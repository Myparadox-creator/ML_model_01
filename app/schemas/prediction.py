"""
Prediction Schemas
====================
Request/response schemas for the prediction API.
"""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """
    Schema for the POST /predict endpoint.
    Contains all features needed for delay prediction.
    """
    origin: str = Field(..., examples=["Mumbai"])
    destination: str = Field(..., examples=["Delhi"])
    distance_km: float = Field(..., gt=0, examples=[1400.0])
    route_type: str = Field(..., pattern="^(highway|local|mixed)$", examples=["highway"])
    departure_hour: int = Field(..., ge=0, le=23, examples=[14])
    day_of_week: int = Field(..., ge=0, le=6, examples=[2])
    is_weekend: int = Field(0, ge=0, le=1)
    carrier_reliability_score: float = Field(..., ge=0, le=1, examples=[0.85])
    weather_severity: float = Field(0.0, ge=0, le=10, examples=[3.5])
    traffic_congestion: float = Field(0.0, ge=0, le=10, examples=[4.2])
    has_news_disruption: int = Field(0, ge=0, le=1)
    model_name: str = Field("xgboost", examples=["xgboost"])


class DelayReason(BaseModel):
    """A single contributing factor to a delay prediction."""
    factor: str = Field(..., examples=["High weather severity (8.5/10)"])
    contribution: str = Field(..., examples=["35%"])
    direction: str = Field(..., examples=["increases delay risk"])


class PredictionResponse(BaseModel):
    """
    Full prediction response with probability, reasons, and recommendations.
    This is the main output of the Early Warning System.
    """
    delay_probability: float = Field(..., ge=0, le=1, examples=[0.82])
    risk_level: str = Field(..., examples=["HIGH"])
    predicted_delayed: bool = Field(..., examples=[True])
    model_used: str = Field(..., examples=["xgboost"])
    reasons: List[DelayReason] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    prediction_time_ms: Optional[float] = Field(None, examples=[45.2])

    class Config:
        from_attributes = True


class PredictionHistoryResponse(BaseModel):
    """Schema for stored prediction records."""
    id: int
    shipment_id: str
    delay_probability: float
    risk_level: str
    predicted_delayed: int
    model_used: str
    reasons: Optional[str]
    recommendations: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


class PredictionListResponse(BaseModel):
    """Paginated list of predictions."""
    predictions: List[PredictionHistoryResponse]
    total: int
    page: int
    per_page: int
