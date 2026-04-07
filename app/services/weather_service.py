"""
Weather Service
=================
Mock weather API that simulates OpenWeatherMap responses.
Replace with real API by setting OPENWEATHER_ENABLED=true and providing API key.

The mock generates realistic weather patterns:
- Seasonal variation (monsoon = more rain)
- City-specific climate patterns
- Random weather events (storms)
"""

import random
import logging
from datetime import datetime
from typing import Optional

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# ── City Climate Profiles ────────────────────────────────────────────────────
# Base weather characteristics per city (used by mock)
CITY_CLIMATE = {
    "Mumbai": {"base_rain": 0.6, "base_temp": 30, "storm_prob": 0.15},
    "Delhi": {"base_rain": 0.3, "base_temp": 28, "storm_prob": 0.08},
    "Bangalore": {"base_rain": 0.4, "base_temp": 25, "storm_prob": 0.05},
    "Chennai": {"base_rain": 0.5, "base_temp": 32, "storm_prob": 0.12},
    "Hyderabad": {"base_rain": 0.3, "base_temp": 29, "storm_prob": 0.06},
    "Pune": {"base_rain": 0.4, "base_temp": 27, "storm_prob": 0.07},
    "Ahmedabad": {"base_rain": 0.2, "base_temp": 33, "storm_prob": 0.05},
    "Jaipur": {"base_rain": 0.15, "base_temp": 30, "storm_prob": 0.04},
    "Kolkata": {"base_rain": 0.5, "base_temp": 29, "storm_prob": 0.10},
    "Lucknow": {"base_rain": 0.25, "base_temp": 28, "storm_prob": 0.05},
    "Guwahati": {"base_rain": 0.6, "base_temp": 26, "storm_prob": 0.12},
    "Patna": {"base_rain": 0.35, "base_temp": 28, "storm_prob": 0.07},
    "Indore": {"base_rain": 0.3, "base_temp": 29, "storm_prob": 0.05},
    "Bhopal": {"base_rain": 0.3, "base_temp": 28, "storm_prob": 0.05},
}

DEFAULT_CLIMATE = {"base_rain": 0.3, "base_temp": 28, "storm_prob": 0.05}


def get_weather(city: str) -> dict:
    """
    Get current weather conditions for a city.
    
    Returns:
        {
            "city": "Mumbai",
            "temperature": 32.5,
            "humidity": 78,
            "wind_speed": 15.2,
            "rain_probability": 0.65,
            "condition": "Thunderstorm",
            "severity_score": 7.8,   # 0-10 scale for ML input
            "alerts": ["Heavy rain warning"]
        }
    """
    if settings.OPENWEATHER_ENABLED and settings.OPENWEATHER_API_KEY:
        return _fetch_real_weather(city)
    return _mock_weather(city)


def _mock_weather(city: str) -> dict:
    """Generate realistic mock weather data."""
    climate = CITY_CLIMATE.get(city, DEFAULT_CLIMATE)
    now = datetime.now()
    month = now.month

    # Seasonal adjustment (June-Sep = monsoon → more rain)
    seasonal_factor = 1.0
    if 6 <= month <= 9:
        seasonal_factor = 1.8  # Monsoon multiplier
    elif month in (10, 11):
        seasonal_factor = 1.3  # Post-monsoon

    # Generate weather values
    rain_prob = min(climate["base_rain"] * seasonal_factor + random.uniform(-0.15, 0.15), 1.0)
    temperature = climate["base_temp"] + random.uniform(-5, 5)
    humidity = 40 + rain_prob * 50 + random.uniform(-10, 10)
    wind_speed = random.uniform(5, 15) + (20 if random.random() < climate["storm_prob"] else 0)

    # Determine condition
    is_storm = random.random() < climate["storm_prob"] * seasonal_factor
    if is_storm:
        condition = "Thunderstorm"
    elif rain_prob > 0.6:
        condition = "Heavy Rain"
    elif rain_prob > 0.3:
        condition = "Light Rain"
    elif temperature > 38:
        condition = "Extreme Heat"
    else:
        condition = "Clear"

    # Calculate severity score (0-10 for ML consumption)
    severity = _calculate_severity(rain_prob, wind_speed, is_storm, temperature)

    # Alerts
    alerts = []
    if is_storm:
        alerts.append("⛈️ Thunderstorm warning — expect delays")
    if rain_prob > 0.7:
        alerts.append("🌧️ Heavy rain advisory")
    if wind_speed > 30:
        alerts.append("💨 High wind advisory — unsafe for open transport")
    if temperature > 42:
        alerts.append("🌡️ Extreme heat warning")

    return {
        "city": city,
        "temperature": round(temperature, 1),
        "humidity": round(min(max(humidity, 20), 100), 1),
        "wind_speed": round(wind_speed, 1),
        "rain_probability": round(max(rain_prob, 0), 2),
        "condition": condition,
        "severity_score": round(severity, 2),
        "alerts": alerts,
    }


def _calculate_severity(rain_prob: float, wind_speed: float, is_storm: bool, temp: float) -> float:
    """
    Compute a 0-10 weather severity score.
    This directly maps to the `weather_severity` ML feature.
    """
    score = 0.0
    score += rain_prob * 3.0          # Rain: up to 3 points
    score += (wind_speed / 50) * 2.5  # Wind: up to 2.5 points
    if is_storm:
        score += 3.0                  # Storm: +3 points
    if temp > 40:
        score += 1.0                  # Extreme heat: +1
    if temp > 45:
        score += 0.5                  # Dangerous heat: +0.5

    return min(score, 10.0)


def get_route_weather(origin: str, destination: str) -> dict:
    """
    Get weather for both origin and destination, plus a combined severity.
    Used by the prediction pipeline to enrich shipment data.
    """
    origin_weather = get_weather(origin)
    dest_weather = get_weather(destination)

    # Combined severity = max of the two (worst case scenario for route)
    combined_severity = max(origin_weather["severity_score"], dest_weather["severity_score"])

    return {
        "origin_weather": origin_weather,
        "destination_weather": dest_weather,
        "route_severity": round(combined_severity, 2),
        "worst_condition": (
            origin_weather["condition"]
            if origin_weather["severity_score"] >= dest_weather["severity_score"]
            else dest_weather["condition"]
        ),
    }


def _fetch_real_weather(city: str) -> dict:
    """
    Fetch real weather from OpenWeatherMap API.
    TODO: Implement when user provides API key.
    """
    import httpx

    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            "q": f"{city},IN",
            "appid": settings.OPENWEATHER_API_KEY,
            "units": "metric",
        }
        response = httpx.get(url, params=params, timeout=5.0)
        response.raise_for_status()
        data = response.json()

        rain_prob = data.get("rain", {}).get("1h", 0) / 10  # Normalize
        wind_speed = data.get("wind", {}).get("speed", 0)
        temp = data.get("main", {}).get("temp", 28)
        condition = data.get("weather", [{}])[0].get("main", "Clear")
        is_storm = condition.lower() in ("thunderstorm", "squall", "tornado")

        return {
            "city": city,
            "temperature": temp,
            "humidity": data.get("main", {}).get("humidity", 50),
            "wind_speed": wind_speed,
            "rain_probability": round(min(rain_prob, 1.0), 2),
            "condition": condition,
            "severity_score": round(_calculate_severity(rain_prob, wind_speed, is_storm, temp), 2),
            "alerts": [],
        }
    except Exception as e:
        logger.warning(f"Real weather API failed for {city}: {e} — falling back to mock")
        return _mock_weather(city)
