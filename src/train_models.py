"""
Model Training Module
======================
Trains three ML models for shipment delay prediction:
  1. Logistic Regression  — baseline probability model
  2. Random Forest         — captures non-linear patterns
  3. XGBoost               — industry standard for tabular data
"""

import os
import time
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def train_logistic_regression(X_train, y_train) -> LogisticRegression:
    """Train a Logistic Regression model (baseline)."""
    print("\n🔹 Training Logistic Regression...")
    start = time.time()

    model = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver="lbfgs",
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train, y_train)

    elapsed = time.time() - start
    print(f"   ✅ Done in {elapsed:.2f}s | Train accuracy: {model.score(X_train, y_train):.4f}")
    return model


def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    """Train a Random Forest model (non-linear patterns)."""
    print("\n🔹 Training Random Forest...")
    start = time.time()

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    elapsed = time.time() - start
    print(f"   ✅ Done in {elapsed:.2f}s | Train accuracy: {model.score(X_train, y_train):.4f}")
    return model


def train_xgboost(X_train, y_train) -> XGBClassifier:
    """Train an XGBoost model (industry standard for tabular data)."""
    print("\n🔹 Training XGBoost...")
    start = time.time()

    # Calculate scale_pos_weight for imbalanced classes
    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    model = XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_train, y_train)

    elapsed = time.time() - start
    print(f"   ✅ Done in {elapsed:.2f}s | Train accuracy: {model.score(X_train, y_train):.4f}")
    return model


def save_model(model, name: str) -> str:
    """Save a trained model to disk."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    filepath = os.path.join(MODELS_DIR, f"{name}.joblib")
    joblib.dump(model, filepath)
    print(f"   💾 Saved: {filepath}")
    return filepath


def train_all_models(X_train, y_train) -> dict:
    """
    Train all three models and save them.

    Returns:
        Dictionary mapping model name → trained model object
    """
    print("=" * 60)
    print("🚀 MODEL TRAINING PIPELINE")
    print("=" * 60)

    models = {}

    # 1. Logistic Regression
    lr_model = train_logistic_regression(X_train, y_train)
    save_model(lr_model, "logistic_regression")
    models["Logistic Regression"] = lr_model

    # 2. Random Forest
    rf_model = train_random_forest(X_train, y_train)
    save_model(rf_model, "random_forest")
    models["Random Forest"] = rf_model

    # 3. XGBoost
    xgb_model = train_xgboost(X_train, y_train)
    save_model(xgb_model, "xgboost")
    models["XGBoost"] = xgb_model

    print("\n" + "=" * 60)
    print(f"✅ All {len(models)} models trained and saved!")
    print("=" * 60)

    return models


if __name__ == "__main__":
    # Quick standalone test
    from preprocessing import load_data, prepare_data

    df = load_data()
    X_train, X_test, y_train, y_test, preprocessor, feature_names = prepare_data(df)
    models = train_all_models(X_train, y_train)
