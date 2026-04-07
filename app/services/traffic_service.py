"""
Traffic Service
=================
Mock traffic congestion API that simulates realistic traffic patterns.
Generates congestion levels based on time-of-day, route type, and city.

Replace with real Google Maps / TomTom API by implementing _fetch_real_traffic().
"""

import random
import math
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


# ── City Congestion Profiles ─────────────────────────────────────────────────
CITY_CONGESTION = {
    "Mumbai": {"base": 7.0, "peak_multiplier": 1.4},
    "Delhi": {"base": 7.5, "peak_multiplier": 1.5},
    "Bangalore": {"base": 8.0, "peak_multiplier": 1.6},  # Infamous traffic
    "Chennai": {"base": 6.0, "peak_multiplier": 1.3},
    "Hyderabad": {"base": 5.5, "peak_multiplier": 1.3},
    "Pune": {"base": 5.0, "peak_multiplier": 1.2},
    "Ahmedabad": {"base": 4.5, "peak_multiplier": 1.2},
    "Jaipur": {"base": 4.0, "peak_multiplier": 1.1},
    "Kolkata": {"base": 6.5, "peak_multiplier": 1.4},
    "Lucknow": {"base": 4.5, "peak_multiplier": 1.2},
    "Guwahati": {"base": 3.5, "peak_multiplier": 1.1},
    "Patna": {"base": 4.0, "peak_multiplier": 1.2},
    "Indore": {"base": 3.5, "peak_multiplier": 1.1},
    "Bhopal": {"base": 3.0, "peak_multiplier": 1.1},
}

DEFAULT_CONGESTION = {"base": 4.0, "peak_multiplier": 1.2}


def get_traffic(origin: str, destination: str, departure_hour: int = None) -> dict:
    """
    Get traffic congestion data for a route.
    
    Args:
        origin: Origin city
        destination: Destination city
        departure_hour: Hour of departure (0-23), defaults to current hour
        
    Returns:
        {
            "origin": "Mumbai",
            "destination": "Delhi",
            "congestion_level": 7.2,     # 0-10 scale for ML input
            "estimated_delay_minutes": 45,
            "peak_hours": True,
            "recommendation": "Delay dispatch by 2 hours to avoid peak traffic"
        }
    """
    if departure_hour is None:
        departure_hour = datetime.now().hour

    # Get congestion for both cities and average (route passes through outskirts)
    origin_profile = CITY_CONGESTION.get(origin, DEFAULT_CONGESTION)
    dest_profile = CITY_CONGESTION.get(destination, DEFAULT_CONGESTION)

    # Time-of-day multiplier (peaks at 8-10 AM and 5-8 PM)
    time_factor = _time_of_day_factor(departure_hour)

    # Route type factor (highway = less congestion near cities)
    origin_congestion = origin_profile["base"] * time_factor * origin_profile["peak_multiplier"]
    dest_congestion = dest_profile["base"] * time_factor * dest_profile["peak_multiplier"]

    # Route congestion = weighted average (origin matters more for departure delay)
    congestion = 0.6 * origin_congestion + 0.4 * dest_congestion
    congestion = min(congestion + random.uniform(-0.5, 0.5), 10.0)
    congestion = max(congestion, 0.0)

    # Estimated delay in minutes based on congestion
    estimated_delay_min = int(congestion * 8 + random.uniform(-5, 10))
    estimated_delay_min = max(estimated_delay_min, 0)

    # Is this peak hours?
    is_peak = departure_hour in range(7, 11) or departure_hour in range(17, 21)

    # Generate recommendation
    recommendation = _generate_recommendation(congestion, departure_hour, is_peak)

    return {
        "origin": origin,
        "destination": destination,
        "congestion_level": round(congestion, 2),
        "estimated_delay_minutes": estimated_delay_min,
        "peak_hours": is_peak,
        "recommendation": recommendation,
    }


def _time_of_day_factor(hour: int) -> float:
    """
    Compute traffic multiplier based on time of day.
    Uses a double-peak sine wave (morning + evening rush).
    
    Returns a value between 0.3 (night) and 1.0 (rush hour).
    """
    # Morning peak centered at 8:30
    morning_peak = math.exp(-((hour - 8.5) ** 2) / 4)
    # Evening peak centered at 18:00
    evening_peak = math.exp(-((hour - 18.0) ** 2) / 5)
    # Lunch bump centered at 13:00
    lunch_bump = 0.3 * math.exp(-((hour - 13.0) ** 2) / 3)

    factor = 0.3 + 0.7 * max(morning_peak, evening_peak, lunch_bump)
    return round(factor, 3)


def _generate_recommendation(congestion: float, hour: int, is_peak: bool) -> str:
    """Generate actionable traffic recommendation."""
    if congestion >= 8:
        return "⚠️ Severe congestion — strongly recommend delaying dispatch or using alternate route"
    elif congestion >= 6:
        if is_peak:
            # Find next non-peak hour
            if hour < 11:
                return f"🕐 Delay dispatch to {11}:00 to avoid morning rush"
            elif hour < 17:
                return "✅ Current window is reasonable — proceed with caution"
            else:
                return f"🕐 Delay dispatch to {21}:00 to avoid evening rush"
        return "🔶 Moderate congestion — expect 30-60 min additional travel time"
    elif congestion >= 4:
        return "🟡 Light congestion — minimal impact expected"
    else:
        return "🟢 Roads are clear — optimal dispatch window"
