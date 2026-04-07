"""
Shipments Router
==================
CRUD operations for shipment management.

Endpoints:
    POST /api/v1/shipments           — create new shipment
    GET  /api/v1/shipments           — list shipments (paginated)
    GET  /api/v1/shipments/{id}      — get shipment details
    PUT  /api/v1/shipments/{id}      — update shipment status
"""

import uuid
import logging
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.auth.dependencies import get_current_user
from app.models.user import User
from app.models.shipment import Shipment
from app.schemas.shipment import (
    ShipmentCreate,
    ShipmentResponse,
    ShipmentListResponse,
    ShipmentStatusUpdate,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/shipments", tags=["Shipments"])


@router.post(
    "",
    response_model=ShipmentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new shipment",
)
def create_shipment(
    data: ShipmentCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Register a new shipment in the system for tracking and prediction."""
    shipment_id = f"SHP-{uuid.uuid4().hex[:8].upper()}"

    shipment = Shipment(
        shipment_id=shipment_id,
        origin=data.origin,
        destination=data.destination,
        distance_km=data.distance_km,
        route_type=data.route_type,
        departure_hour=data.departure_hour,
        day_of_week=data.day_of_week,
        is_weekend=data.is_weekend,
        carrier_id=data.carrier_id,
        carrier_reliability_score=data.carrier_reliability_score,
        weather_severity=data.weather_severity,
        traffic_congestion=data.traffic_congestion,
        has_news_disruption=data.has_news_disruption,
        holiday_indicator=data.holiday_indicator,
        status="pending",
    )

    db.add(shipment)
    db.commit()
    db.refresh(shipment)

    logger.info(f"📦 Shipment created: {shipment_id} ({data.origin} → {data.destination})")
    return shipment


@router.get(
    "",
    response_model=ShipmentListResponse,
    summary="List all shipments",
)
def list_shipments(
    page: int = 1,
    per_page: int = 20,
    status_filter: str = None,
    origin: str = None,
    destination: str = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get paginated list of shipments with optional filters."""
    query = db.query(Shipment)

    if status_filter:
        query = query.filter(Shipment.status == status_filter)
    if origin:
        query = query.filter(Shipment.origin == origin)
    if destination:
        query = query.filter(Shipment.destination == destination)

    total = query.count()
    shipments = (
        query.order_by(Shipment.created_at.desc())
        .offset((page - 1) * per_page)
        .limit(per_page)
        .all()
    )

    return ShipmentListResponse(
        shipments=[ShipmentResponse.model_validate(s) for s in shipments],
        total=total,
        page=page,
        per_page=per_page,
    )


@router.get(
    "/{shipment_id}",
    response_model=ShipmentResponse,
    summary="Get shipment details",
)
def get_shipment(
    shipment_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get details of a specific shipment by its ID."""
    shipment = db.query(Shipment).filter(Shipment.shipment_id == shipment_id).first()
    if not shipment:
        raise HTTPException(status_code=404, detail=f"Shipment {shipment_id} not found")
    return shipment


@router.put(
    "/{shipment_id}",
    response_model=ShipmentResponse,
    summary="Update shipment status",
)
def update_shipment(
    shipment_id: str,
    update: ShipmentStatusUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Update the status of a shipment (e.g., mark as delivered or delayed)."""
    shipment = db.query(Shipment).filter(Shipment.shipment_id == shipment_id).first()
    if not shipment:
        raise HTTPException(status_code=404, detail=f"Shipment {shipment_id} not found")

    shipment.status = update.status
    if update.actual_delay_hours is not None:
        shipment.actual_delay_hours = update.actual_delay_hours

    db.commit()
    db.refresh(shipment)

    logger.info(f"📦 Shipment {shipment_id} updated → status={update.status}")
    return shipment
