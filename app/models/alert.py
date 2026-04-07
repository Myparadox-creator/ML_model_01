"""
Alert ORM Model
=================
High-risk shipment alerts that require human attention.
"""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from app.database import Base


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    shipment_id = Column(
        String(50),
        ForeignKey("shipments.shipment_id"),
        nullable=False,
        index=True,
    )
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)

    # Alert details
    alert_type = Column(String(50), nullable=False)  # HIGH_RISK, WEATHER, CARRIER, TRAFFIC
    severity = Column(String(10), nullable=False)  # LOW, MEDIUM, HIGH, CRITICAL
    title = Column(String(200), nullable=False)
    message = Column(Text, nullable=False)
    delay_probability = Column(Float, nullable=True)

    # Resolution
    is_resolved = Column(Boolean, default=False, index=True)
    resolved_at = Column(DateTime, nullable=True)
    resolved_by = Column(String(100), nullable=True)
    resolution_notes = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    # Relationships
    shipment = relationship("Shipment", back_populates="alerts")
    user = relationship("User", back_populates="alerts")

    def __repr__(self):
        return (
            f"<Alert(shipment='{self.shipment_id}', "
            f"type='{self.alert_type}', resolved={self.is_resolved})>"
        )
