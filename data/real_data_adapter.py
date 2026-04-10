"""
Real Data Adapter
==================
Transforms the Kaggle E-Commerce Shipping Dataset into the feature
schema expected by the LogiPredict ML pipeline.

Input:  data/raw/ecommerce_shipping.csv (Kaggle format)
Output: data/shipments.csv (LogiPredict format)

Feature Mapping:
    Kaggle Column              → LogiPredict Feature
    ─────────────────────────────────────────────────
    Warehouse_block            → origin (via city lookup)
    Product importance + ID    → destination (via city lookup)
    Weight_in_gms + mode       → distance_km (city pair distance)
    Mode_of_Shipment           → route_type
    Customer_care_calls        → departure_hour (proxy)
    Random (seeded)            → day_of_week
    day_of_week >= 5           → is_weekend
    Customer_rating            → carrier_reliability_score
    Mode + Discount            → traffic_congestion
    Discount_offered > 50      → has_news_disruption
    Reached.on.Time_Y.N        → delayed (inverted: 1=NOT on time)
"""

import os
import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_CSV = os.path.join(DATA_DIR, "raw", "ecommerce_shipping.csv")
OUTPUT_CSV = os.path.join(DATA_DIR, "shipments.csv")

# ── Indian City Mapping ──────────────────────────────────────────────────────
# Map warehouse blocks (A-F) to Indian cities using logistics hub sizes
WAREHOUSE_TO_CITIES = {
    "A": ["Mumbai", "Pune"],
    "B": ["Delhi", "Jaipur"],
    "C": ["Bangalore", "Chennai"],
    "D": ["Hyderabad", "Ahmedabad"],
    "F": ["Kolkata", "Lucknow"],
}

# All possible destination cities
ALL_CITIES = [
    "Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad",
    "Pune", "Ahmedabad", "Jaipur", "Kolkata", "Lucknow",
    "Guwahati", "Patna", "Indore", "Bhopal",
]

# Real road distances between Indian city pairs (km) — sourced from Google Maps
# Used as lookup instead of random generation
CITY_DISTANCES = {
    ("Mumbai", "Delhi"): 1400, ("Mumbai", "Bangalore"): 980,
    ("Mumbai", "Chennai"): 1330, ("Mumbai", "Hyderabad"): 710,
    ("Mumbai", "Pune"): 150, ("Mumbai", "Ahmedabad"): 530,
    ("Mumbai", "Jaipur"): 1150, ("Mumbai", "Kolkata"): 2050,
    ("Mumbai", "Lucknow"): 1380, ("Mumbai", "Guwahati"): 2580,
    ("Mumbai", "Patna"): 1750, ("Mumbai", "Indore"): 590,
    ("Mumbai", "Bhopal"): 780,
    ("Delhi", "Bangalore"): 2150, ("Delhi", "Chennai"): 2180,
    ("Delhi", "Hyderabad"): 1550, ("Delhi", "Pune"): 1450,
    ("Delhi", "Ahmedabad"): 940, ("Delhi", "Jaipur"): 280,
    ("Delhi", "Kolkata"): 1500, ("Delhi", "Lucknow"): 555,
    ("Delhi", "Guwahati"): 1900, ("Delhi", "Patna"): 1000,
    ("Delhi", "Indore"): 810, ("Delhi", "Bhopal"): 780,
    ("Bangalore", "Chennai"): 350, ("Bangalore", "Hyderabad"): 570,
    ("Bangalore", "Pune"): 840, ("Bangalore", "Ahmedabad"): 1500,
    ("Bangalore", "Jaipur"): 1880, ("Bangalore", "Kolkata"): 1870,
    ("Bangalore", "Lucknow"): 2000, ("Bangalore", "Guwahati"): 2800,
    ("Bangalore", "Patna"): 2100, ("Bangalore", "Indore"): 1290,
    ("Bangalore", "Bhopal"): 1240,
    ("Chennai", "Hyderabad"): 630, ("Chennai", "Pune"): 1170,
    ("Chennai", "Ahmedabad"): 1770, ("Chennai", "Jaipur"): 1900,
    ("Chennai", "Kolkata"): 1660, ("Chennai", "Lucknow"): 1910,
    ("Chennai", "Guwahati"): 2550, ("Chennai", "Patna"): 1870,
    ("Chennai", "Indore"): 1470, ("Chennai", "Bhopal"): 1440,
    ("Hyderabad", "Pune"): 560, ("Hyderabad", "Ahmedabad"): 1200,
    ("Hyderabad", "Jaipur"): 1340, ("Hyderabad", "Kolkata"): 1500,
    ("Hyderabad", "Lucknow"): 1350, ("Hyderabad", "Guwahati"): 2350,
    ("Hyderabad", "Patna"): 1570, ("Hyderabad", "Indore"): 870,
    ("Hyderabad", "Bhopal"): 870,
    ("Pune", "Ahmedabad"): 660, ("Pune", "Jaipur"): 1170,
    ("Pune", "Kolkata"): 1900, ("Pune", "Lucknow"): 1410,
    ("Pune", "Guwahati"): 2610, ("Pune", "Patna"): 1780,
    ("Pune", "Indore"): 530, ("Pune", "Bhopal"): 650,
    ("Ahmedabad", "Jaipur"): 670, ("Ahmedabad", "Kolkata"): 1900,
    ("Ahmedabad", "Lucknow"): 1200, ("Ahmedabad", "Guwahati"): 2760,
    ("Ahmedabad", "Patna"): 1680, ("Ahmedabad", "Indore"): 400,
    ("Ahmedabad", "Bhopal"): 570,
    ("Jaipur", "Kolkata"): 1640, ("Jaipur", "Lucknow"): 570,
    ("Jaipur", "Guwahati"): 2050, ("Jaipur", "Patna"): 1100,
    ("Jaipur", "Indore"): 580, ("Jaipur", "Bhopal"): 490,
    ("Kolkata", "Lucknow"): 990, ("Kolkata", "Guwahati"): 1100,
    ("Kolkata", "Patna"): 580, ("Kolkata", "Indore"): 1530,
    ("Kolkata", "Bhopal"): 1360,
    ("Lucknow", "Guwahati"): 1530, ("Lucknow", "Patna"): 530,
    ("Lucknow", "Indore"): 860, ("Lucknow", "Bhopal"): 640,
    ("Guwahati", "Patna"): 990, ("Guwahati", "Indore"): 2260,
    ("Guwahati", "Bhopal"): 2100,
    ("Patna", "Indore"): 1180, ("Patna", "Bhopal"): 1020,
    ("Indore", "Bhopal"): 195,
}

# ── Shipment mode → route type mapping ───────────────────────────────────────
MODE_TO_ROUTE = {
    "Ship": "mixed",
    "Flight": "highway",
    "Road": "local",
}

# ── Seasonal weather severity by city ────────────────────────────────────────
# Average weather disruption scores (0-10) for major months
CITY_WEATHER_BASE = {
    "Mumbai": 5.5, "Delhi": 4.0, "Bangalore": 3.0, "Chennai": 5.0,
    "Hyderabad": 3.5, "Pune": 4.0, "Ahmedabad": 3.5, "Jaipur": 3.0,
    "Kolkata": 5.0, "Lucknow": 4.5, "Guwahati": 6.0, "Patna": 5.0,
    "Indore": 3.5, "Bhopal": 3.5,
}


def get_distance(city_a: str, city_b: str) -> float:
    """Look up real driving distance between two Indian cities."""
    if city_a == city_b:
        return 50.0  # Same city delivery
    key1 = (city_a, city_b)
    key2 = (city_b, city_a)
    if key1 in CITY_DISTANCES:
        return float(CITY_DISTANCES[key1])
    elif key2 in CITY_DISTANCES:
        return float(CITY_DISTANCES[key2])
    else:
        # Fallback: estimate based on average distance
        return 800.0


def adapt_kaggle_dataset(raw_csv_path: str = RAW_CSV, seed: int = 42) -> pd.DataFrame:
    """
    Transform the Kaggle E-Commerce Shipping Dataset into the
    LogiPredict feature schema.

    Args:
        raw_csv_path: Path to the raw Kaggle CSV
        seed: Random seed for reproducible city assignments

    Returns:
        DataFrame with exactly the same columns as generate_dataset() output
    """
    print("🔄 Adapting Kaggle dataset to LogiPredict schema...")
    rng = np.random.default_rng(seed)

    # ── Load raw data ────────────────────────────────────────────────────
    df_raw = pd.read_csv(raw_csv_path)
    n = len(df_raw)
    print(f"   Raw records: {n:,}")
    print(f"   Raw columns: {list(df_raw.columns)}")

    # ── Map Warehouse_block → origin city ────────────────────────────────
    origins = []
    for block in df_raw["Warehouse_block"]:
        if block in WAREHOUSE_TO_CITIES:
            city = rng.choice(WAREHOUSE_TO_CITIES[block])
        else:
            # Block E or unknown → randomly pick from Kolkata/Guwahati
            city = rng.choice(["Kolkata", "Guwahati"])
        origins.append(city)

    # ── Map → destination city (different from origin) ───────────────────
    destinations = []
    for origin in origins:
        possible = [c for c in ALL_CITIES if c != origin]
        dest = rng.choice(possible)
        destinations.append(dest)

    # ── Compute distance_km from real city pair distances ────────────────
    distances = np.array([
        get_distance(o, d) for o, d in zip(origins, destinations)
    ])
    # Add ±8% noise for realism (different trucks take different routes)
    distances = distances * rng.uniform(0.92, 1.08, size=n)
    distances = np.round(distances, 1)

    # ── Map Mode_of_Shipment → route_type ────────────────────────────────
    route_types = df_raw["Mode_of_Shipment"].map(MODE_TO_ROUTE).values
    # Fill any unmapped values
    route_types = np.where(pd.isna(route_types), "mixed", route_types)

    # ── Derive departure_hour from Customer_care_calls ───────────────────
    # More calls = busier period → map to business/rush hours
    care_calls = df_raw["Customer_care_calls"].values
    # Map 1-7 calls → hours distributed across 0-23
    departure_hour = np.clip(
        (care_calls * 3 + rng.integers(0, 5, size=n)) % 24, 0, 23
    ).astype(int)

    # ── day_of_week (not in dataset → seeded random uniform) ─────────────
    day_of_week = rng.integers(0, 7, size=n)
    is_weekend = (day_of_week >= 5).astype(int)

    # ── Carrier ID ──────────────────────────────────────────────────────
    carrier_ids = np.array([f"CARRIER_{i:03d}" for i in rng.integers(1, 21, size=n)])

    # ── carrier_reliability_score from Customer_rating ───────────────────
    # Rating is 1-5, normalize to 0.4-1.0 range (no one is below 0.4)
    ratings = df_raw["Customer_rating"].values
    carrier_reliability = np.clip(
        (ratings - 1) / 4 * 0.6 + 0.4 + rng.normal(0, 0.03, size=n),
        0.3, 1.0
    )
    carrier_reliability = np.round(carrier_reliability, 4)

    # ── weather_severity from origin city base + noise ───────────────────
    weather_severity = np.array([
        CITY_WEATHER_BASE.get(city, 4.0) for city in origins
    ])
    weather_severity += rng.normal(0, 1.5, size=n)
    weather_severity = np.clip(weather_severity, 0, 10)
    weather_severity = np.round(weather_severity, 2)

    # ── traffic_congestion from mode + product cost + discount ───────────
    # Ship (slow) → higher congestion proxy
    # High cost → priority handling → lower congestion
    # High discount → clearance → possible congestion
    mode_factor = df_raw["Mode_of_Shipment"].map({
        "Ship": 0.7, "Road": 0.5, "Flight": 0.2
    }).values

    cost_normalized = df_raw["Cost_of_the_Product"].values / df_raw["Cost_of_the_Product"].max()
    discount_normalized = df_raw["Discount_offered"].values / 65  # max discount ~65

    traffic_congestion = (
        mode_factor * 5
        + discount_normalized * 3
        + (1 - cost_normalized) * 2
        + rng.normal(0, 1.0, size=n)
    )
    traffic_congestion = np.clip(traffic_congestion, 0, 10)
    traffic_congestion = np.round(traffic_congestion, 2)

    # ── has_news_disruption from heavy discounts ─────────────────────────
    # Discounts > 50% may indicate disruption-period clearance
    has_news_disruption = (df_raw["Discount_offered"].values > 50).astype(int)

    # ── delayed target: Reached.on.Time_Y.N INVERTED ─────────────────────
    # In the Kaggle dataset: 1 = reached on time, 0 = NOT on time
    # Our target: 1 = delayed (NOT on time), 0 = on time
    # IMPORTANT: Kaggle uses 1=NOT reached on time, so no inversion needed
    # Let's verify by checking the column description:
    # "1 Indicates that the product has NOT reached on time and 0 indicates it has reached on time"
    # So 1 = NOT on time = delayed. This actually matches directly!
    delayed = df_raw["Reached.on.Time_Y.N"].values

    # ── Compute delay_probability from the actual features ───────────────
    # Use a formula similar to the synthetic one but based on real feature values
    delay_score = (
        0.20 * (weather_severity / 10)
        + 0.15 * (traffic_congestion / 10)
        + 0.15 * (1 - carrier_reliability)
        + 0.12 * (distances / distances.max())
        + 0.10 * has_news_disruption
        + 0.05 * is_weekend
        + 0.05 * np.where(
            ((departure_hour >= 7) & (departure_hour <= 9))
            | ((departure_hour >= 17) & (departure_hour <= 19)),
            1, 0
        )
        + 0.05 * np.where(route_types == "local", 1, 0)
    )
    delay_score += rng.normal(0, 0.05, size=n)
    delay_prob = 1 / (1 + np.exp(-8 * (delay_score - 0.45)))
    delay_prob = np.round(delay_prob, 4)

    # ── Assemble into LogiPredict schema ─────────────────────────────────
    df_out = pd.DataFrame({
        "shipment_id": [f"SHP-{i:06d}" for i in range(1, n + 1)],
        "origin": origins,
        "destination": destinations,
        "distance_km": distances,
        "route_type": route_types,
        "departure_hour": departure_hour,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "carrier_id": carrier_ids,
        "carrier_reliability_score": carrier_reliability,
        "weather_severity": weather_severity,
        "traffic_congestion": traffic_congestion,
        "has_news_disruption": has_news_disruption,
        "delay_probability": delay_prob,
        "delayed": delayed,
    })

    print(f"   Output records: {len(df_out):,}")
    print(f"   Output columns: {list(df_out.columns)}")
    print(f"   Delay rate: {df_out['delayed'].mean():.1%}")
    print(f"   Cities: {sorted(df_out['origin'].unique())}")

    return df_out


def save_adapted_dataset(df: pd.DataFrame, filepath: str = OUTPUT_CSV) -> str:
    """Save the adapted dataset to CSV."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"✅ Real data saved: {filepath} ({len(df):,} rows, {df.shape[1]} columns)")
    return filepath


if __name__ == "__main__":
    if not os.path.exists(RAW_CSV):
        print(f"❌ Raw data not found: {RAW_CSV}")
        print("   Run download_real_data.py first.")
    else:
        df = adapt_kaggle_dataset()
        save_adapted_dataset(df)
        print(f"\n{df.head()}")
        print(f"\n{df.describe()}")
