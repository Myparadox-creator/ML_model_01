"""
Prediction ORM Model
======================
Stores every delay prediction with SHAP-based explanations and recommendations.
"""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey
from sqlalchemy.orm import relationship
from app.database import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    shipment_id = Column(
        String(50),
        ForeignKey("shipments.shipment_id"),
        nullable=False,
        index=True,
    )

    # Prediction results
    delay_probability = Column(Float, nullable=False)
    risk_level = Column(String(10), nullable=False, index=True)  # LOW, MEDIUM, HIGH
    predicted_delayed = Column(Integer, nullable=False)  # 0 or 1
    model_used = Column(String(50), nullable=False)

    # SHAP explanations (JSON string)
    reasons = Column(Text, nullable=True)  # JSON array of {factor, contribution, direction}
    recommendations = Column(Text, nullable=True)  # JSON array of strings

    # Metadata
    prediction_time_ms = Column(Float, nullable=True)  # latency tracking
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    # Relationships
    shipment = relationship("Shipment", back_populates="predictions")

    def __repr__(self):
        return (
            f"<Prediction(shipment='{self.shipment_id}', "
            f"prob={self.delay_probability:.2f}, risk='{self.risk_level}')>"
        )
