"""
Alert Schemas
===============
Request/response schemas for alert-related endpoints.
"""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field


class AlertResponse(BaseModel):
    """Schema for alert data in responses."""
    id: int
    shipment_id: str
    alert_type: str
    severity: str
    title: str
    message: str
    delay_probability: Optional[float]
    is_resolved: bool
    resolved_at: Optional[datetime]
    resolved_by: Optional[str]
    resolution_notes: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


class AlertResolve(BaseModel):
    """Schema for resolving an alert."""
    resolved_by: str = Field(..., examples=["john_doe"])
    resolution_notes: Optional[str] = Field(None, examples=["Carrier switched to CARRIER_003"])


class AlertListResponse(BaseModel):
    """Paginated list of alerts."""
    alerts: List[AlertResponse]
    total: int
    unresolved: int
