"""
🚛 AI-Powered Shipment Delay Prediction — Main Pipeline
==========================================================
Runs the complete ML pipeline:
  1. Load data (real Kaggle data or synthetic fallback)
  2. Preprocess and split data
  3. Train all models (Logistic Regression, Random Forest, XGBoost)
  4. Evaluate and compare models

Usage:
    python main.py                # real data (default)
    python main.py --mode real    # force real Kaggle data
    python main.py --mode synthetic  # force synthetic data
"""

import sys
import os
import time
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))


def main(mode: str = "real"):
    total_start = time.time()

    print("=" * 70)
    print("🚛  AI-POWERED SHIPMENT DELAY PREDICTION SYSTEM")
    print("   Early Warning System for Logistics SLA Management")
    print(f"   Data mode: {mode.upper()}")
    print("=" * 70)

    data_source = mode

    # ── Step 1: Load / Generate Dataset ──────────────────────────────────
    if mode == "real":
        print("\n\n📦 STEP 1: Loading Real Shipment Dataset (Kaggle)")
        print("-" * 50)

        try:
            from data.download_real_data import download_dataset
            from data.real_data_adapter import adapt_kaggle_dataset, save_adapted_dataset

            raw_path = download_dataset()
            if raw_path is None:
                raise RuntimeError("Download failed")

            df = adapt_kaggle_dataset(raw_path)
            save_adapted_dataset(df)
            data_source = "real"
            print(f"   ✅ Real dataset loaded: {df.shape}")

        except Exception as e:
            print(f"\n   ⚠️  Real data failed: {e}")
            print("   Falling back to synthetic data...")
            from data.generate_dataset import generate_dataset, save_dataset
            df = generate_dataset(n_samples=10_000)
            save_dataset(df)
            data_source = "synthetic"
    else:
        print("\n\n📦 STEP 1: Generating Synthetic Shipment Dataset")
        print("-" * 50)
        from data.generate_dataset import generate_dataset, save_dataset
        df = generate_dataset(n_samples=10_000)
        save_dataset(df)
        data_source = "synthetic"

    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Data source: {data_source}")

    # ── Step 2: Preprocess Data ──────────────────────────────────────────
    print("\n\n🔧 STEP 2: Preprocessing & Feature Engineering")
    print("-" * 50)
    from src.preprocessing import prepare_data

    X_train, X_test, y_train, y_test, preprocessor, feature_names = prepare_data(df)

    # ── Step 3: Train Models ─────────────────────────────────────────────
    print("\n\n🤖 STEP 3: Training ML Models")
    print("-" * 50)
    from src.train_models import train_all_models

    # Use GridSearchCV for real data (finds optimal params), skip for synthetic
    use_gridsearch = (data_source == "real")
    models = train_all_models(X_train, y_train, use_gridsearch=use_gridsearch)

    # ── Step 4: Evaluate Models ──────────────────────────────────────────
    print("\n\n📊 STEP 4: Evaluating & Comparing Models")
    print("-" * 50)
    from src.evaluate import evaluate_all_models

    comparison_df = evaluate_all_models(
        models, X_test, y_test, feature_names, data_source=data_source
    )

    # ── Summary ──────────────────────────────────────────────────────────
    total_time = time.time() - total_start
    n_rows = len(df)
    print("\n\n" + "=" * 70)
    print("🎉 PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Data source: {data_source}")
    print(f"   Dataset:    data/shipments.csv ({n_rows:,} shipments)")
    print(f"   Models:     models/ (3 trained models + preprocessor)")
    print(f"   Outputs:    outputs/ (metrics, plots, comparisons)")
    print(f"\n   🚀 To start the prediction API:")
    print(f"      uvicorn api.app:app --reload --port 8000")
    print(f"      Then visit: http://localhost:8000/docs")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the full ML pipeline for shipment delay prediction"
    )
    parser.add_argument(
        "--mode",
        choices=["real", "synthetic"],
        default="real",
        help="Data mode: 'real' (Kaggle dataset) or 'synthetic' (numpy random). Default: real",
    )
    args = parser.parse_args()
    main(mode=args.mode)
