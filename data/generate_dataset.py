"""
Shipment Dataset Generator
============================
Generates shipment data for the AI-powered logistics delay prediction system.

Modes:
  - real:      Downloads and adapts the Kaggle E-Commerce Shipping Dataset
               (10,999 real records with actual on-time/delayed labels)
  - synthetic: Generates ~10,000 synthetic records using numpy.random (fallback)

Usage:
    python data/generate_dataset.py                  # defaults to real mode
    python data/generate_dataset.py --mode real       # force real data
    python data/generate_dataset.py --mode synthetic  # force synthetic data
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

# ── Configuration ────────────────────────────────────────────────────────────
SEED = 42
NUM_SAMPLES = 10_000
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "shipments.csv")

# City pairs with approximate distances (km)
ROUTES = [
    ("Mumbai", "Delhi", 1400), ("Delhi", "Kolkata", 1500),
    ("Bangalore", "Chennai", 350), ("Hyderabad", "Pune", 560),
    ("Mumbai", "Ahmedabad", 530), ("Delhi", "Jaipur", 280),
    ("Kolkata", "Guwahati", 1100), ("Chennai", "Hyderabad", 630),
    ("Pune", "Bangalore", 840), ("Jaipur", "Lucknow", 570),
    ("Lucknow", "Patna", 530), ("Ahmedabad", "Indore", 400),
    ("Indore", "Bhopal", 195), ("Guwahati", "Patna", 990),
    ("Mumbai", "Kolkata", 2050), ("Delhi", "Chennai", 2180),
    ("Bangalore", "Delhi", 2150), ("Hyderabad", "Kolkata", 1500),
    ("Pune", "Delhi", 1450), ("Chennai", "Kolkata", 1660),
]

CARRIERS = [f"CARRIER_{i:03d}" for i in range(1, 21)]  # 20 carriers
ROUTE_TYPES = ["highway", "local", "mixed"]


def generate_dataset(n_samples: int = NUM_SAMPLES, seed: int = SEED) -> pd.DataFrame:
    """Generate a synthetic shipment dataset with realistic delay patterns."""
    rng = np.random.default_rng(seed)

    # ── Select random routes ─────────────────────────────────────────────
    route_indices = rng.integers(0, len(ROUTES), size=n_samples)
    origins = [ROUTES[i][0] for i in route_indices]
    destinations = [ROUTES[i][1] for i in route_indices]
    base_distances = np.array([ROUTES[i][2] for i in route_indices], dtype=float)

    # Add noise to distances (±10%)
    distances = base_distances * rng.uniform(0.9, 1.1, size=n_samples)

    # ── Route type ───────────────────────────────────────────────────────
    route_types = rng.choice(ROUTE_TYPES, size=n_samples, p=[0.5, 0.2, 0.3])

    # ── Timing features ──────────────────────────────────────────────────
    departure_hour = rng.integers(0, 24, size=n_samples)
    day_of_week = rng.integers(0, 7, size=n_samples)  # 0=Mon, 6=Sun
    is_weekend = (day_of_week >= 5).astype(int)

    # ── Carrier features ─────────────────────────────────────────────────
    carrier_ids = rng.choice(CARRIERS, size=n_samples)
    # Each carrier has a base reliability score
    carrier_base_reliability = {c: rng.uniform(0.5, 0.98) for c in CARRIERS}
    carrier_reliability = np.array([
        np.clip(carrier_base_reliability[c] + rng.normal(0, 0.05), 0.1, 1.0)
        for c in carrier_ids
    ])

    # ── External factors ─────────────────────────────────────────────────
    weather_severity = rng.uniform(0, 10, size=n_samples)     # 0=clear, 10=extreme
    traffic_congestion = rng.uniform(0, 10, size=n_samples)   # 0=free, 10=gridlock
    has_news_disruption = rng.choice([0, 1], size=n_samples, p=[0.85, 0.15])

    # ── Compute delay probability (realistic formula) ────────────────────
    delay_score = (
        0.20 * (weather_severity / 10)           # weather contributes ~20%
        + 0.15 * (traffic_congestion / 10)        # traffic contributes ~15%
        + 0.15 * (1 - carrier_reliability)         # low reliability → higher risk
        + 0.12 * (distances / distances.max())     # longer distance → higher risk
        + 0.10 * has_news_disruption               # disruption events
        + 0.05 * is_weekend                        # weekend shipments slightly riskier
        + 0.05 * np.where(                         # rush hour departures (7-9, 17-19)
            ((departure_hour >= 7) & (departure_hour <= 9))
            | ((departure_hour >= 17) & (departure_hour <= 19)),
            1, 0
        )
        + 0.05 * np.where(route_types == "local", 1, 0)  # local routes riskier
    )

    # Add noise and apply sigmoid-like mapping
    delay_score += rng.normal(0, 0.08, size=n_samples)
    delay_prob = 1 / (1 + np.exp(-8 * (delay_score - 0.45)))  # sigmoid centered at 0.45

    # Binary target: delayed or not
    delayed = (rng.random(size=n_samples) < delay_prob).astype(int)

    # ── Assemble DataFrame ───────────────────────────────────────────────
    df = pd.DataFrame({
        "shipment_id": [f"SHP-{i:06d}" for i in range(1, n_samples + 1)],
        "origin": origins,
        "destination": destinations,
        "distance_km": np.round(distances, 1),
        "route_type": route_types,
        "departure_hour": departure_hour,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "carrier_id": carrier_ids,
        "carrier_reliability_score": np.round(carrier_reliability, 4),
        "weather_severity": np.round(weather_severity, 2),
        "traffic_congestion": np.round(traffic_congestion, 2),
        "has_news_disruption": has_news_disruption,
        "delay_probability": np.round(delay_prob, 4),
        "delayed": delayed,
    })

    return df


def save_dataset(df: pd.DataFrame, filepath: str = OUTPUT_FILE) -> str:
    """Save dataset to CSV."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"✅ Dataset saved: {filepath} ({len(df):,} rows, {df.shape[1]} columns)")
    print(f"   Delay rate: {df['delayed'].mean():.1%}")
    return filepath


def generate_real_dataset() -> pd.DataFrame:
    """Generate dataset from real Kaggle data (download if needed)."""
    from data.download_real_data import download_dataset, RAW_CSV
    from data.real_data_adapter import adapt_kaggle_dataset

    # Step 1: Download if needed
    raw_path = download_dataset()
    if raw_path is None:
        print("⚠️  Real data download failed. Falling back to synthetic data.")
        return None

    # Step 2: Adapt to our schema
    df = adapt_kaggle_dataset(raw_path)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate shipment dataset for delay prediction"
    )
    parser.add_argument(
        "--mode",
        choices=["real", "synthetic"],
        default="real",
        help="Data mode: 'real' (Kaggle dataset) or 'synthetic' (numpy random). Default: real",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=NUM_SAMPLES,
        help=f"Number of samples for synthetic mode. Default: {NUM_SAMPLES}",
    )
    args = parser.parse_args()

    if args.mode == "real":
        df = generate_real_dataset()
        if df is None:
            print("⚠️  Falling back to synthetic mode...")
            df = generate_dataset(n_samples=args.samples)
    else:
        df = generate_dataset(n_samples=args.samples)

    save_dataset(df)
    print(df.head())
    print(df.describe())

