"""
Model Training Module
======================
Trains three ML models for shipment delay prediction:
  1. Logistic Regression  — baseline probability model
  2. Random Forest         — captures non-linear patterns
  3. XGBoost               — industry standard for tabular data (with GridSearchCV)

For real data, XGBoost uses GridSearchCV to find optimal hyperparameters
since the true data patterns are unknown (unlike hand-crafted synthetic data).
"""

import os
import time
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from xgboost import XGBClassifier

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def _print_cv_scores(model, X_train, y_train, model_name: str):
    """Run 5-fold cross-validation and print scores."""
    print(f"   📊 Running 5-fold cross-validation...")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc", n_jobs=-1)
    print(f"   CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"   CV folds:   {[f'{s:.4f}' for s in cv_scores]}")
    return cv_scores


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

    _print_cv_scores(model, X_train, y_train, "Logistic Regression")
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

    _print_cv_scores(model, X_train, y_train, "Random Forest")
    return model


def train_xgboost(X_train, y_train, use_gridsearch: bool = True) -> XGBClassifier:
    """
    Train an XGBoost model with optional GridSearchCV hyperparameter tuning.

    Args:
        X_train: Training features
        y_train: Training labels
        use_gridsearch: If True, run GridSearchCV to find best params.
                        Takes ~2-3 min but finds optimal hyperparameters.
    """
    print("\n🔹 Training XGBoost...")
    start = time.time()

    # Calculate scale_pos_weight for imbalanced classes
    n_pos = int(np.sum(y_train == 1))
    n_neg = int(np.sum(y_train == 0))
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    print(f"   Class balance: {n_neg} negative / {n_pos} positive (ratio: {scale_pos_weight:.2f})")

    if use_gridsearch:
        print("   🔍 Running GridSearchCV (this may take 2-3 minutes)...")

        # Focused parameter grid — good coverage without excessive combinations
        param_grid = {
            "n_estimators": [200, 300, 500],
            "max_depth": [4, 6, 8],
            "learning_rate": [0.05, 0.1, 0.2],
            "subsample": [0.8, 0.9],
            "colsample_bytree": [0.7, 0.8],
        }

        base_model = XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )

        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=3,
            scoring="roc_auc",
            n_jobs=-1,
            verbose=0,
            refit=True,
        )

        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_

        elapsed_grid = time.time() - start
        print(f"   🏆 Best GridSearchCV params (in {elapsed_grid:.1f}s):")
        for param, value in grid_search.best_params_.items():
            print(f"      {param}: {value}")
        print(f"   Best CV ROC-AUC: {grid_search.best_score_:.4f}")

    else:
        # Use default params (faster, for synthetic data or testing)
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

    # Final cross-validation on best model
    _print_cv_scores(model, X_train, y_train, "XGBoost")
    return model


def save_model(model, name: str) -> str:
    """Save a trained model to disk."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    filepath = os.path.join(MODELS_DIR, f"{name}.joblib")
    joblib.dump(model, filepath)
    print(f"   💾 Saved: {filepath}")
    return filepath


def train_all_models(X_train, y_train, use_gridsearch: bool = True) -> dict:
    """
    Train all three models and save them.

    Args:
        X_train: Training features
        y_train: Training labels
        use_gridsearch: If True, run GridSearchCV for XGBoost.

    Returns:
        Dictionary mapping model name → trained model object
    """
    print("=" * 60)
    print("🚀 MODEL TRAINING PIPELINE")
    if use_gridsearch:
        print("   (GridSearchCV enabled for XGBoost)")
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

    # 3. XGBoost (with optional GridSearchCV)
    xgb_model = train_xgboost(X_train, y_train, use_gridsearch=use_gridsearch)
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
    models = train_all_models(X_train, y_train, use_gridsearch=True)
