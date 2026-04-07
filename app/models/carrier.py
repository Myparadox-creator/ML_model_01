"""
Carrier ORM Model
===================
Represents logistics carriers with their reliability metrics.
"""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.orm import relationship
from app.database import Base


class Carrier(Base):
    __tablename__ = "carriers"

    id = Column(Integer, primary_key=True, autoincrement=True)
    carrier_id = Column(String(50), unique=True, nullable=False, index=True)
    name = Column(String(100), nullable=False)
    reliability_score = Column(Float, default=0.8)
    total_shipments = Column(Integer, default=0)
    delayed_shipments = Column(Integer, default=0)
    avg_delay_hours = Column(Float, default=0.0)
    is_active = Column(Integer, default=1)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    shipments = relationship("Shipment", back_populates="carrier")

    @property
    def delay_rate(self) -> float:
        """Percentage of shipments that were delayed."""
        if self.total_shipments == 0:
            return 0.0
        return round((self.delayed_shipments / self.total_shipments) * 100, 2)

    def __repr__(self):
        return f"<Carrier(carrier_id='{self.carrier_id}', reliability={self.reliability_score:.2f})>"
