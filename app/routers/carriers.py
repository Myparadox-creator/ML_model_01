"""
Carriers Router
=================
Endpoints for carrier management and reliability tracking.

Endpoints:
    GET  /api/v1/carriers          — list all carriers
    GET  /api/v1/carriers/{id}     — carrier detail + stats
"""

import logging
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from app.database import get_db
from app.auth.dependencies import get_current_user
from app.models.user import User
from app.models.carrier import Carrier
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/carriers", tags=["Carriers"])


class CarrierResponse(BaseModel):
    id: int
    carrier_id: str
    name: str
    reliability_score: float
    total_shipments: int
    delayed_shipments: int
    avg_delay_hours: float
    is_active: int

    class Config:
        from_attributes = True


class CarrierDetailResponse(CarrierResponse):
    delay_rate: float


@router.get("", response_model=List[CarrierResponse], summary="List all carriers")
def list_carriers(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get all registered carriers with their reliability scores."""
    carriers = db.query(Carrier).order_by(Carrier.reliability_score.desc()).all()

    # If no carriers in DB, seed them from training data
    if not carriers:
        carriers = _seed_carriers(db)

    return carriers


@router.get("/{carrier_id}", response_model=CarrierDetailResponse, summary="Get carrier details")
def get_carrier(
    carrier_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get detailed information about a specific carrier."""
    carrier = db.query(Carrier).filter(Carrier.carrier_id == carrier_id).first()
    if not carrier:
        raise HTTPException(status_code=404, detail=f"Carrier {carrier_id} not found")

    return CarrierDetailResponse(
        id=carrier.id,
        carrier_id=carrier.carrier_id,
        name=carrier.name,
        reliability_score=carrier.reliability_score,
        total_shipments=carrier.total_shipments,
        delayed_shipments=carrier.delayed_shipments,
        avg_delay_hours=carrier.avg_delay_hours,
        is_active=carrier.is_active,
        delay_rate=carrier.delay_rate,
    )


def _seed_carriers(db: Session) -> list:
    """Seed carriers into DB from training data configuration."""
    import random
    random.seed(42)

    carrier_names = [
        "SpeedFreight Logistics", "BlueArrow Transport", "SafeShip Express",
        "Delta Cargo India", "NovaTrans Pvt Ltd", "RapidHaul Services",
        "PrimeFleet Movers", "TrustLine Freight", "GreenPath Logistics",
        "StarRoute Carriers", "SwiftMove India", "OceanBridge Cargo",
        "HillTrack Transport", "MetroLink Freight", "CrossRoad Logistics",
        "PeakTime Movers", "SilverWing Transport", "GoldStar Freight",
        "RedLine Express", "Evergreen Carriers",
    ]

    carriers = []
    for i in range(1, 21):
        carrier_id = f"CARRIER_{i:03d}"
        reliability = round(random.uniform(0.55, 0.97), 4)
        total = random.randint(50, 500)
        delayed = int(total * (1 - reliability) * random.uniform(0.8, 1.2))

        carrier = Carrier(
            carrier_id=carrier_id,
            name=carrier_names[i - 1],
            reliability_score=reliability,
            total_shipments=total,
            delayed_shipments=delayed,
            avg_delay_hours=round(random.uniform(1.5, 8.0), 2),
            is_active=1,
        )
        db.add(carrier)
        carriers.append(carrier)

    db.commit()
    logger.info(f"🚛 Seeded {len(carriers)} carriers into database")
    return carriers
