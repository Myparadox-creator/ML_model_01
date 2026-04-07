"""
Shipment Schemas
==================
Request/response schemas for shipment-related endpoints.
"""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field


class ShipmentCreate(BaseModel):
    """Schema for creating a new shipment."""
    origin: str = Field(..., examples=["Mumbai"])
    destination: str = Field(..., examples=["Delhi"])
    distance_km: float = Field(..., gt=0, examples=[1400.0])
    route_type: str = Field(..., pattern="^(highway|local|mixed)$", examples=["highway"])
    departure_hour: int = Field(..., ge=0, le=23, examples=[14])
    day_of_week: int = Field(..., ge=0, le=6, examples=[2])
    is_weekend: int = Field(0, ge=0, le=1)
    carrier_id: Optional[str] = Field(None, examples=["CARRIER_005"])
    carrier_reliability_score: float = Field(..., ge=0, le=1, examples=[0.85])
    weather_severity: float = Field(0.0, ge=0, le=10, examples=[3.5])
    traffic_congestion: float = Field(0.0, ge=0, le=10, examples=[4.2])
    has_news_disruption: int = Field(0, ge=0, le=1)
    holiday_indicator: int = Field(0, ge=0, le=1)


class ShipmentResponse(BaseModel):
    """Schema for shipment data in responses."""
    id: int
    shipment_id: str
    origin: str
    destination: str
    distance_km: float
    route_type: str
    departure_hour: int
    day_of_week: int
    is_weekend: int
    carrier_id: Optional[str]
    carrier_reliability_score: float
    weather_severity: float
    traffic_congestion: float
    has_news_disruption: int
    holiday_indicator: int
    status: str
    created_at: datetime

    class Config:
        from_attributes = True


class ShipmentListResponse(BaseModel):
    """Paginated list of shipments."""
    shipments: List[ShipmentResponse]
    total: int
    page: int
    per_page: int


class ShipmentStatusUpdate(BaseModel):
    """Schema for updating shipment status."""
    status: str = Field(..., pattern="^(pending|in_transit|delivered|delayed)$")
    actual_delay_hours: Optional[float] = None
