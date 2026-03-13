"""
🚛 AI-Powered Shipment Delay Prediction — Main Pipeline
==========================================================
Runs the complete ML pipeline:
  1. Generate synthetic shipment dataset
  2. Preprocess and split data
  3. Train all models (Logistic Regression, Random Forest, XGBoost)
  4. Evaluate and compare models
  
Usage:
    python main.py
"""

import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))


def main():
    total_start = time.time()

    print("=" * 70)
    print("🚛  AI-POWERED SHIPMENT DELAY PREDICTION SYSTEM")
    print("   Early Warning System for Logistics SLA Management")
    print("=" * 70)

    # ── Step 1: Generate Dataset ─────────────────────────────────────────
    print("\n\n📦 STEP 1: Generating Synthetic Shipment Dataset")
    print("-" * 50)
    from data.generate_dataset import generate_dataset, save_dataset

    df = generate_dataset(n_samples=10_000)
    save_dataset(df)
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")

    # ── Step 2: Preprocess Data ──────────────────────────────────────────
    print("\n\n🔧 STEP 2: Preprocessing & Feature Engineering")
    print("-" * 50)
    from src.preprocessing import prepare_data

    X_train, X_test, y_train, y_test, preprocessor, feature_names = prepare_data(df)

    # ── Step 3: Train Models ─────────────────────────────────────────────
    print("\n\n🤖 STEP 3: Training ML Models")
    print("-" * 50)
    from src.train_models import train_all_models

    models = train_all_models(X_train, y_train)

    # ── Step 4: Evaluate Models ──────────────────────────────────────────
    print("\n\n📊 STEP 4: Evaluating & Comparing Models")
    print("-" * 50)
    from src.evaluate import evaluate_all_models

    comparison_df = evaluate_all_models(models, X_test, y_test, feature_names)

    # ── Summary ──────────────────────────────────────────────────────────
    total_time = time.time() - total_start
    print("\n\n" + "=" * 70)
    print("🎉 PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Dataset:    data/shipments.csv (10,000 shipments)")
    print(f"   Models:     models/ (3 trained models + preprocessor)")
    print(f"   Outputs:    outputs/ (metrics, plots, comparisons)")
    print(f"\n   🚀 To start the prediction API:")
    print(f"      uvicorn api.app:app --reload --port 8000")
    print(f"      Then visit: http://localhost:8000/docs")
    print("=" * 70)


if __name__ == "__main__":
    main()
