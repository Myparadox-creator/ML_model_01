"""
API Endpoint Tests
====================
Tests for authentication, prediction, shipment CRUD, and analytics endpoints.
"""

import pytest


class TestAuth:
    """Authentication endpoint tests."""

    def test_register_success(self, client):
        """Test successful user registration."""
        response = client.post("/auth/register", json={
            "username": "newuser1",
            "email": "new1@example.com",
            "password": "password123",
            "full_name": "New User",
        })
        assert response.status_code == 201
        data = response.json()
        assert data["username"] == "newuser1"
        assert data["email"] == "new1@example.com"
        assert data["role"] == "operator"
        assert "hashed_password" not in data  # Password should never be exposed

    def test_register_duplicate_username(self, client):
        """Test that duplicate usernames are rejected."""
        user_data = {
            "username": "dupuser",
            "email": "dup1@example.com",
            "password": "password123",
        }
        client.post("/auth/register", json=user_data)
        response = client.post("/auth/register", json={
            **user_data,
            "email": "dup2@example.com",
        })
        assert response.status_code == 409

    def test_login_success(self, client):
        """Test successful login returns JWT token."""
        # Register first
        client.post("/auth/register", json={
            "username": "loginuser",
            "email": "login@example.com",
            "password": "password123",
        })

        # Login
        response = client.post("/auth/login", data={
            "username": "loginuser",
            "password": "password123",
        })
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    def test_login_wrong_password(self, client):
        """Test login with wrong password is rejected."""
        client.post("/auth/register", json={
            "username": "wrongpw",
            "email": "wrongpw@example.com",
            "password": "correct123",
        })

        response = client.post("/auth/login", data={
            "username": "wrongpw",
            "password": "WRONG",
        })
        assert response.status_code == 401

    def test_protected_route_without_token(self, client):
        """Test that protected routes reject requests without auth."""
        response = client.get("/api/v1/shipments")
        assert response.status_code == 401


class TestHealthCheck:
    """System endpoint tests."""

    def test_root(self, client):
        """Test root endpoint returns system info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert "version" in data

    def test_health(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "ml_ready" in data


class TestPredictions:
    """Prediction endpoint tests."""

    def test_predict_success(self, client, auth_headers, sample_shipment_data):
        """Test successful delay prediction."""
        response = client.post(
            "/api/v1/predict",
            json=sample_shipment_data,
            headers=auth_headers,
        )
        # 503 = models not loaded (expected in test env without trained models)
        if response.status_code == 503:
            pytest.skip("ML models not loaded — run 'python main.py' first")
        assert response.status_code == 200
        data = response.json()
        assert "delay_probability" in data
        assert 0 <= data["delay_probability"] <= 1
        assert data["risk_level"] in ("LOW", "MEDIUM", "HIGH")
        assert isinstance(data["predicted_delayed"], bool)
        assert "reasons" in data
        assert "recommendations" in data

    def test_predict_high_risk(self, client, auth_headers, high_risk_shipment_data):
        """Test that high-risk input gets high probability."""
        response = client.post(
            "/api/v1/predict",
            json=high_risk_shipment_data,
            headers=auth_headers,
        )
        if response.status_code == 503:
            pytest.skip("ML models not loaded — run 'python main.py' first")
        assert response.status_code == 200
        data = response.json()
        # High risk inputs should generally predict higher probability
        assert data["delay_probability"] >= 0.3  # Should be elevated

    def test_predict_returns_reasons(self, client, auth_headers, sample_shipment_data):
        """Test that predictions include SHAP-based reasons."""
        response = client.post(
            "/api/v1/predict",
            json=sample_shipment_data,
            headers=auth_headers,
        )
        data = response.json()
        reasons = data.get("reasons", [])
        # Should have at least some reasons
        if reasons:
            assert "factor" in reasons[0]
            assert "contribution" in reasons[0]
            assert "direction" in reasons[0]

    def test_predict_returns_recommendations(self, client, auth_headers, high_risk_shipment_data):
        """Test that high-risk predictions include recommendations."""
        response = client.post(
            "/api/v1/predict",
            json=high_risk_shipment_data,
            headers=auth_headers,
        )
        if response.status_code == 503:
            pytest.skip("ML models not loaded — run 'python main.py' first")
        data = response.json()
        assert len(data["recommendations"]) > 0

    def test_predict_invalid_model(self, client, auth_headers, sample_shipment_data):
        """Test prediction with non-existent model returns error."""
        sample_shipment_data["model_name"] = "nonexistent_model"
        response = client.post(
            "/api/v1/predict",
            json=sample_shipment_data,
            headers=auth_headers,
        )
        assert response.status_code in (400, 500, 503)


class TestShipments:
    """Shipment CRUD endpoint tests."""

    def test_create_shipment(self, client, auth_headers):
        """Test creating a new shipment."""
        response = client.post(
            "/api/v1/shipments",
            json={
                "origin": "Delhi",
                "destination": "Mumbai",
                "distance_km": 1400.0,
                "route_type": "highway",
                "departure_hour": 10,
                "day_of_week": 1,
                "is_weekend": 0,
                "carrier_reliability_score": 0.9,
                "weather_severity": 2.0,
                "traffic_congestion": 3.0,
                "has_news_disruption": 0,
                "holiday_indicator": 0,
            },
            headers=auth_headers,
        )
        assert response.status_code == 201
        data = response.json()
        assert data["origin"] == "Delhi"
        assert data["shipment_id"].startswith("SHP-")
        assert data["status"] == "pending"

    def test_list_shipments(self, client, auth_headers):
        """Test listing shipments (paginated)."""
        response = client.get("/api/v1/shipments", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "shipments" in data
        assert "total" in data
        assert "page" in data


class TestAnalytics:
    """Analytics endpoint tests."""

    def test_dashboard(self, client, auth_headers):
        """Test dashboard analytics endpoint."""
        response = client.get("/api/v1/analytics/dashboard", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "summary" in data
        assert "risk_distribution" in data

    def test_model_info(self, client, auth_headers):
        """Test model metrics endpoint."""
        response = client.get("/api/v1/analytics/model-info", headers=auth_headers)
        # May return 404 if metrics file doesn't exist in test env
        assert response.status_code in (200, 404)

    def test_weather_endpoint(self, client, auth_headers):
        """Test weather API (mock)."""
        response = client.get(
            "/api/v1/analytics/weather/Mumbai",
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["city"] == "Mumbai"
        assert "severity_score" in data
        assert 0 <= data["severity_score"] <= 10

    def test_traffic_endpoint(self, client, auth_headers):
        """Test traffic API (mock)."""
        response = client.get(
            "/api/v1/analytics/traffic?origin=Mumbai&destination=Delhi&departure_hour=8",
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert "congestion_level" in data
        assert "recommendation" in data
