"""
Shipment ORM Model
====================
Represents a shipment record with all features used for delay prediction.
"""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from app.database import Base


class Shipment(Base):
    __tablename__ = "shipments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    shipment_id = Column(String(50), unique=True, nullable=False, index=True)

    # Route info
    origin = Column(String(100), nullable=False, index=True)
    destination = Column(String(100), nullable=False, index=True)
    distance_km = Column(Float, nullable=False)
    route_type = Column(String(20), nullable=False)  # highway, local, mixed

    # Timing
    departure_hour = Column(Integer, nullable=False)
    day_of_week = Column(Integer, nullable=False)
    is_weekend = Column(Integer, default=0)
    dispatch_time = Column(DateTime, nullable=True)
    estimated_arrival = Column(DateTime, nullable=True)

    # Carrier
    carrier_id = Column(String(50), ForeignKey("carriers.carrier_id"), nullable=True)
    carrier_reliability_score = Column(Float, nullable=False)

    # External factors
    weather_severity = Column(Float, default=0.0)
    traffic_congestion = Column(Float, default=0.0)
    has_news_disruption = Column(Integer, default=0)
    holiday_indicator = Column(Integer, default=0)

    # Status
    status = Column(String(30), default="in_transit")  # pending, in_transit, delivered, delayed
    actual_delay_hours = Column(Float, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    carrier = relationship("Carrier", back_populates="shipments")
    predictions = relationship("Prediction", back_populates="shipment", cascade="all, delete-orphan")
    alerts = relationship("Alert", back_populates="shipment", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Shipment(id={self.id}, shipment_id='{self.shipment_id}', {self.origin}→{self.destination})>"
