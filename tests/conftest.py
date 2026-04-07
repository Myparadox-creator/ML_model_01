"""
Test Fixtures (conftest.py)
==============================
Shared fixtures for all test modules.
Provides a test database, test client, and auth helpers.
"""

import os
import sys
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Override database BEFORE importing app
os.environ["DATABASE_URL"] = "sqlite:///./test_shipment_delay.db"
os.environ["JWT_SECRET_KEY"] = "test-secret-key"
os.environ["REDIS_ENABLED"] = "false"
os.environ["LOG_LEVEL"] = "WARNING"

from app.database import Base, get_db, engine
from app.main import app


# ── Test Database ────────────────────────────────────────────────────────────
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    """Override the database dependency to use test database."""
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(scope="session", autouse=True)
def setup_database():
    """Create all tables once before tests, drop after."""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)
    # Clean up test database file (may fail on Windows due to file locks)
    db_path = "test_shipment_delay.db"
    try:
        if os.path.exists(db_path):
            os.remove(db_path)
    except PermissionError:
        pass  # Windows file lock — harmless


@pytest.fixture
def db():
    """Provide a clean database session for each test."""
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def client():
    """Provide a TestClient for making API requests."""
    return TestClient(app)


# ── Auth Helpers ─────────────────────────────────────────────────────────────
@pytest.fixture
def test_user_data():
    """Default test user data."""
    return {
        "username": "testuser",
        "email": "test@example.com",
        "password": "testpass123",
        "full_name": "Test User",
    }


@pytest.fixture
def auth_headers(client, test_user_data):
    """
    Register a test user and return auth headers with JWT token.
    Use this fixture for endpoints that require authentication.
    """
    # Register
    client.post("/auth/register", json=test_user_data)

    # Login
    response = client.post(
        "/auth/login",
        data={"username": test_user_data["username"], "password": test_user_data["password"]},
    )

    if response.status_code == 200:
        token = response.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}

    # If user already exists, just login
    response = client.post(
        "/auth/login",
        data={"username": test_user_data["username"], "password": test_user_data["password"]},
    )
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def sample_shipment_data():
    """Sample shipment data for prediction tests."""
    return {
        "origin": "Mumbai",
        "destination": "Delhi",
        "distance_km": 1400.0,
        "route_type": "highway",
        "departure_hour": 14,
        "day_of_week": 2,
        "is_weekend": 0,
        "carrier_reliability_score": 0.85,
        "weather_severity": 3.5,
        "traffic_congestion": 4.2,
        "has_news_disruption": 0,
        "model_name": "xgboost",
    }


@pytest.fixture
def high_risk_shipment_data():
    """High-risk shipment data (should predict high delay probability)."""
    return {
        "origin": "Mumbai",
        "destination": "Kolkata",
        "distance_km": 2050.0,
        "route_type": "local",
        "departure_hour": 8,
        "day_of_week": 5,
        "is_weekend": 1,
        "carrier_reliability_score": 0.52,
        "weather_severity": 9.0,
        "traffic_congestion": 8.5,
        "has_news_disruption": 1,
        "model_name": "xgboost",
    }
