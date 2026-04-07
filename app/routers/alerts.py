"""
Alerts Router
===============
Endpoints for managing high-risk shipment alerts.

Endpoints:
    GET  /api/v1/alerts              — list alerts (filterable)
    POST /api/v1/alerts/{id}/resolve — mark alert as resolved
"""

import logging
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.auth.dependencies import get_current_user
from app.models.user import User
from app.models.alert import Alert
from app.schemas.alert import AlertResponse, AlertResolve, AlertListResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/alerts", tags=["Alerts"])


@router.get("", response_model=AlertListResponse, summary="List alerts")
def list_alerts(
    resolved: bool = None,
    severity: str = None,
    limit: int = 50,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Get alert list, optionally filtered by resolution status or severity.
    
    - **resolved**: true/false to filter by resolution status
    - **severity**: LOW, MEDIUM, HIGH, CRITICAL
    """
    query = db.query(Alert)

    if resolved is not None:
        query = query.filter(Alert.is_resolved == resolved)
    if severity:
        query = query.filter(Alert.severity == severity.upper())

    total = query.count()
    unresolved = db.query(Alert).filter(Alert.is_resolved == False).count()

    alerts = query.order_by(Alert.created_at.desc()).limit(limit).all()

    return AlertListResponse(
        alerts=[AlertResponse.model_validate(a) for a in alerts],
        total=total,
        unresolved=unresolved,
    )


@router.post(
    "/{alert_id}/resolve",
    response_model=AlertResponse,
    summary="Resolve an alert",
)
def resolve_alert(
    alert_id: int,
    resolve_data: AlertResolve,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Mark an alert as resolved with notes about the action taken."""
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")

    if alert.is_resolved:
        raise HTTPException(status_code=400, detail="Alert is already resolved")

    alert.is_resolved = True
    alert.resolved_at = datetime.now(timezone.utc)
    alert.resolved_by = resolve_data.resolved_by
    alert.resolution_notes = resolve_data.resolution_notes

    db.commit()
    db.refresh(alert)

    logger.info(f"✅ Alert {alert_id} resolved by {resolve_data.resolved_by}")
    return alert
