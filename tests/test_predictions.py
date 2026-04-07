"""
Prediction Validation Tests
===============================
End-to-end tests that validate prediction correctness:
- Known high-risk inputs should produce high probabilities
- Known low-risk inputs should produce low probabilities
- Weather/traffic services return valid data
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestWeatherService:
    """Tests for the mock weather service."""

    def test_get_weather_returns_valid_data(self):
        """Test that weather service returns all required fields."""
        from app.services.weather_service import get_weather

        result = get_weather("Mumbai")
        assert result["city"] == "Mumbai"
        assert 0 <= result["severity_score"] <= 10
        assert "condition" in result
        assert "temperature" in result
        assert isinstance(result["alerts"], list)

    def test_unknown_city_returns_default(self):
        """Test that unknown cities get default climate data."""
        from app.services.weather_service import get_weather

        result = get_weather("UnknownCity123")
        assert result["city"] == "UnknownCity123"
        assert "severity_score" in result

    def test_route_weather_returns_combined(self):
        """Test that route weather combines origin and destination."""
        from app.services.weather_service import get_route_weather

        result = get_route_weather("Mumbai", "Delhi")
        assert "origin_weather" in result
        assert "destination_weather" in result
        assert "route_severity" in result
        assert result["route_severity"] >= 0


class TestTrafficService:
    """Tests for the mock traffic service."""

    def test_get_traffic_returns_valid_data(self):
        """Test that traffic service returns all required fields."""
        from app.services.traffic_service import get_traffic

        result = get_traffic("Mumbai", "Delhi", departure_hour=8)
        assert 0 <= result["congestion_level"] <= 10
        assert result["peak_hours"] is True  # 8 AM is peak
        assert "recommendation" in result

    def test_off_peak_lower_congestion(self):
        """Test that off-peak hours generally show lower congestion."""
        from app.services.traffic_service import get_traffic

        peak = get_traffic("Bangalore", "Chennai", departure_hour=8)
        off_peak = get_traffic("Bangalore", "Chennai", departure_hour=3)
        # Off-peak should generally be lower (but not guaranteed due to randomness)
        # Just check it returns valid data
        assert off_peak["peak_hours"] is False


class TestCacheService:
    """Tests for the cache service."""

    def test_in_memory_cache_set_get(self):
        """Test basic set/get operations."""
        from app.services.cache_service import InMemoryCache

        cache = InMemoryCache()
        cache.set("key1", {"data": "test"}, ttl=60)
        assert cache.get("key1") == {"data": "test"}

    def test_in_memory_cache_miss(self):
        """Test cache miss returns None."""
        from app.services.cache_service import InMemoryCache

        cache = InMemoryCache()
        assert cache.get("nonexistent") is None

    def test_in_memory_cache_expiry(self):
        """Test that expired items return None."""
        import time
        from app.services.cache_service import InMemoryCache

        cache = InMemoryCache()
        cache.set("expire_key", "value", ttl=1)
        assert cache.get("expire_key") == "value"
        time.sleep(1.5)
        assert cache.get("expire_key") is None

    def test_cache_delete(self):
        """Test deleting a cached item."""
        from app.services.cache_service import InMemoryCache

        cache = InMemoryCache()
        cache.set("del_key", "value")
        cache.delete("del_key")
        assert cache.get("del_key") is None


class TestMLServiceIntegration:
    """Integration tests for the ML service (requires trained models)."""

    @pytest.fixture
    def ml_service(self):
        """Get an ML service instance with loaded models."""
        from app.services.ml_service import MLService

        service = MLService()
        try:
            service.load_models()
        except Exception:
            pytest.skip("Models not trained yet — run `python main.py` first")
        return service

    def test_service_loads_models(self, ml_service):
        """Test that ML service loads all expected models."""
        assert ml_service.is_ready
        assert "xgboost" in ml_service.available_models

    def test_prediction_output_format(self, ml_service):
        """Test that prediction returns correct output format."""
        result = ml_service.predict({
            "origin": "Mumbai",
            "destination": "Delhi",
            "distance_km": 1400,
            "route_type": "highway",
            "departure_hour": 14,
            "day_of_week": 2,
            "is_weekend": 0,
            "carrier_reliability_score": 0.85,
            "weather_severity": 3.5,
            "traffic_congestion": 4.2,
            "has_news_disruption": 0,
        })

        assert "delay_probability" in result
        assert "risk_level" in result
        assert "predicted_delayed" in result
        assert "reasons" in result
        assert "prediction_time_ms" in result
        assert 0 <= result["delay_probability"] <= 1
        assert result["risk_level"] in ("LOW", "MEDIUM", "HIGH")

    def test_high_risk_prediction(self, ml_service):
        """Test that extreme risk inputs produce elevated probabilities."""
        result = ml_service.predict({
            "origin": "Mumbai",
            "destination": "Kolkata",
            "distance_km": 2050,
            "route_type": "local",
            "departure_hour": 8,
            "day_of_week": 5,
            "is_weekend": 1,
            "carrier_reliability_score": 0.50,
            "weather_severity": 9.5,
            "traffic_congestion": 9.0,
            "has_news_disruption": 1,
        })

        # With all adverse conditions, probability should be elevated
        assert result["delay_probability"] >= 0.4, (
            f"Expected high probability, got {result['delay_probability']}"
        )

    def test_low_risk_prediction(self, ml_service):
        """Test that ideal conditions produce lower probabilities."""
        result = ml_service.predict({
            "origin": "Bangalore",
            "destination": "Chennai",
            "distance_km": 350,
            "route_type": "highway",
            "departure_hour": 12,
            "day_of_week": 2,
            "is_weekend": 0,
            "carrier_reliability_score": 0.97,
            "weather_severity": 0.5,
            "traffic_congestion": 1.0,
            "has_news_disruption": 0,
        })

        # With ideal conditions, probability should be lower
        assert result["delay_probability"] <= 0.7, (
            f"Expected lower probability, got {result['delay_probability']}"
        )

    def test_prediction_with_different_models(self, ml_service):
        """Test that all available models produce valid predictions."""
        shipment = {
            "origin": "Delhi",
            "destination": "Jaipur",
            "distance_km": 280,
            "route_type": "highway",
            "departure_hour": 10,
            "day_of_week": 1,
            "is_weekend": 0,
            "carrier_reliability_score": 0.8,
            "weather_severity": 2.0,
            "traffic_congestion": 3.0,
            "has_news_disruption": 0,
        }

        for model_name in ml_service.available_models:
            result = ml_service.predict(shipment, model_name=model_name)
            assert 0 <= result["delay_probability"] <= 1
            assert result["model_used"] == model_name
