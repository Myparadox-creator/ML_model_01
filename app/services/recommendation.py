"""
Recommendation Engine
=======================
Hybrid rule-based + ML-driven recommendation system.
Takes prediction results + context and generates actionable corrective actions.

Architecture:
    1. Rule-based triggers (deterministic, transparent)
    2. Context-aware adjustments (weather, traffic, carrier history)
    3. Priority ranking (critical → informational)
"""

import logging
from typing import List
from app.services.weather_service import get_route_weather
from app.services.traffic_service import get_traffic

logger = logging.getLogger(__name__)


# ── Recommendation Templates ────────────────────────────────────────────────
RECOMMENDATIONS = {
    # Weather-related
    "reroute_weather": {
        "action": "🌧️ Reroute shipment to avoid weather-affected region",
        "priority": 1,
        "category": "WEATHER",
    },
    "delay_weather": {
        "action": "⏳ Delay dispatch by 6-12 hours until weather clears",
        "priority": 2,
        "category": "WEATHER",
    },
    "weather_protection": {
        "action": "📦 Ensure weather-proof packaging for cargo",
        "priority": 3,
        "category": "WEATHER",
    },

    # Carrier-related
    "switch_carrier": {
        "action": "🔄 Switch to a higher-reliability carrier for this route",
        "priority": 1,
        "category": "CARRIER",
    },
    "carrier_backup": {
        "action": "📋 Prepare backup carrier in case of primary carrier failure",
        "priority": 2,
        "category": "CARRIER",
    },
    "monitor_carrier": {
        "action": "👁️ Enable real-time tracking for this carrier",
        "priority": 3,
        "category": "CARRIER",
    },

    # Traffic-related
    "delay_traffic": {
        "action": "🕐 Delay dispatch by 2-4 hours to avoid peak congestion",
        "priority": 2,
        "category": "TRAFFIC",
    },
    "alternate_route": {
        "action": "🛣️ Use alternate highway route to bypass congestion",
        "priority": 2,
        "category": "TRAFFIC",
    },
    "night_dispatch": {
        "action": "🌙 Consider night dispatch (10PM-4AM) for minimal traffic",
        "priority": 3,
        "category": "TRAFFIC",
    },

    # General risk
    "escalate": {
        "action": "🚨 Escalate to operations manager — high delay risk detected",
        "priority": 1,
        "category": "ESCALATION",
    },
    "notify_customer": {
        "action": "📱 Proactively notify customer of potential delay and updated ETA",
        "priority": 2,
        "category": "COMMUNICATION",
    },
    "adjust_eta": {
        "action": "⏰ Adjust estimated arrival time to account for delays",
        "priority": 2,
        "category": "PLANNING",
    },
    "split_shipment": {
        "action": "📦 Consider splitting into relay legs to reduce per-leg risk",
        "priority": 3,
        "category": "PLANNING",
    },
    "standard_tracking": {
        "action": "✅ Standard monitoring — no immediate action needed",
        "priority": 4,
        "category": "MONITORING",
    },
}


def generate_recommendations(
    prediction_result: dict,
    shipment_data: dict,
) -> List[str]:
    """
    Generate ranked recommendations based on prediction + context.
    
    Args:
        prediction_result: Output from MLService.predict()
        shipment_data: Raw shipment features dict
        
    Returns:
        Sorted list of recommendation strings (most critical first)
    """
    triggered = []
    prob = prediction_result.get("delay_probability", 0)
    risk = prediction_result.get("risk_level", "LOW")

    weather_sev = shipment_data.get("weather_severity", 0)
    traffic_cong = shipment_data.get("traffic_congestion", 0)
    carrier_rel = shipment_data.get("carrier_reliability_score", 1.0)
    distance = shipment_data.get("distance_km", 0)
    departure_hour = shipment_data.get("departure_hour", 12)
    is_weekend = shipment_data.get("is_weekend", 0)
    has_disruption = shipment_data.get("has_news_disruption", 0)

    # ── Rule 1: Critical delay risk ──────────────────────────────────────
    if prob >= 0.8:
        triggered.append(RECOMMENDATIONS["escalate"])
        triggered.append(RECOMMENDATIONS["notify_customer"])

    # ── Rule 2: Weather-based ────────────────────────────────────────────
    if weather_sev >= 8:
        triggered.append(RECOMMENDATIONS["reroute_weather"])
        triggered.append(RECOMMENDATIONS["weather_protection"])
    elif weather_sev >= 6:
        triggered.append(RECOMMENDATIONS["delay_weather"])
        triggered.append(RECOMMENDATIONS["weather_protection"])
    elif weather_sev >= 4:
        triggered.append(RECOMMENDATIONS["weather_protection"])

    # ── Rule 3: Carrier reliability ──────────────────────────────────────
    if carrier_rel < 0.55:
        triggered.append(RECOMMENDATIONS["switch_carrier"])
    elif carrier_rel < 0.70:
        triggered.append(RECOMMENDATIONS["carrier_backup"])
        triggered.append(RECOMMENDATIONS["monitor_carrier"])
    elif carrier_rel < 0.80:
        triggered.append(RECOMMENDATIONS["monitor_carrier"])

    # ── Rule 4: Traffic congestion ───────────────────────────────────────
    if traffic_cong >= 8:
        triggered.append(RECOMMENDATIONS["alternate_route"])
        triggered.append(RECOMMENDATIONS["delay_traffic"])
    elif traffic_cong >= 6:
        is_peak = departure_hour in range(7, 11) or departure_hour in range(17, 21)
        if is_peak:
            triggered.append(RECOMMENDATIONS["delay_traffic"])
        else:
            triggered.append(RECOMMENDATIONS["alternate_route"])
    elif traffic_cong >= 4 and departure_hour in range(7, 10):
        triggered.append(RECOMMENDATIONS["night_dispatch"])

    # ── Rule 5: Long distance + adverse conditions ───────────────────────
    if distance > 1500 and weather_sev > 5:
        triggered.append(RECOMMENDATIONS["split_shipment"])

    # ── Rule 6: Weekend + reliability concern ────────────────────────────
    if is_weekend and carrier_rel < 0.75:
        triggered.append(RECOMMENDATIONS["notify_customer"])

    # ── Rule 7: Disruption events ────────────────────────────────────────
    if has_disruption:
        triggered.append(RECOMMENDATIONS["reroute_weather"])
        triggered.append(RECOMMENDATIONS["notify_customer"])

    # ── Rule 8: Medium risk — adjust ETA ─────────────────────────────────
    if risk in ("MEDIUM", "HIGH"):
        triggered.append(RECOMMENDATIONS["adjust_eta"])

    # ── Default: low risk ────────────────────────────────────────────────
    if not triggered:
        triggered.append(RECOMMENDATIONS["standard_tracking"])

    # Deduplicate and sort by priority
    seen_actions = set()
    unique = []
    for r in triggered:
        if r["action"] not in seen_actions:
            seen_actions.add(r["action"])
            unique.append(r)

    unique.sort(key=lambda x: x["priority"])

    # Return just the action strings (top 5)
    return [r["action"] for r in unique[:5]]
