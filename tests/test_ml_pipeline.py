"""
ML Pipeline Tests
===================
Tests for data generation, preprocessing, model training, and SHAP explanations.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestDataGeneration:
    """Tests for the synthetic data generator."""

    def test_dataset_shape(self):
        """Test that generated dataset has correct shape."""
        from data.generate_dataset import generate_dataset
        df = generate_dataset(n_samples=100, seed=42)
        assert len(df) == 100
        assert df.shape[1] == 15  # 15 columns (including delay_probability)

    def test_required_columns(self):
        """Test that all required columns exist."""
        from data.generate_dataset import generate_dataset
        df = generate_dataset(n_samples=50)
        required = [
            "shipment_id", "origin", "destination", "distance_km",
            "route_type", "departure_hour", "day_of_week", "is_weekend",
            "carrier_id", "carrier_reliability_score",
            "weather_severity", "traffic_congestion",
            "has_news_disruption", "delayed",
        ]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_value_ranges(self):
        """Test that generated values are within expected ranges."""
        from data.generate_dataset import generate_dataset
        df = generate_dataset(n_samples=500)

        assert df["distance_km"].min() > 0, "Distance should be positive"
        assert df["departure_hour"].between(0, 23).all(), "Hour out of range"
        assert df["day_of_week"].between(0, 6).all(), "Day out of range"
        assert df["is_weekend"].isin([0, 1]).all(), "is_weekend should be binary"
        assert df["carrier_reliability_score"].between(0, 1.1).all(), "Reliability out of range"
        assert df["weather_severity"].between(0, 10.5).all(), "Weather out of range"
        assert df["traffic_congestion"].between(0, 10.5).all(), "Traffic out of range"
        assert df["delayed"].isin([0, 1]).all(), "Target should be binary"

    def test_delay_rate_reasonable(self):
        """Test that the delay rate is realistic (not all 0 or all 1)."""
        from data.generate_dataset import generate_dataset
        df = generate_dataset(n_samples=1000)
        rate = df["delayed"].mean()
        assert 0.1 < rate < 0.9, f"Delay rate {rate:.1%} seems unrealistic"


class TestPreprocessing:
    """Tests for the preprocessing pipeline."""

    def test_prepare_data_returns_correct_types(self):
        """Test that prepare_data returns correct tuple types."""
        from data.generate_dataset import generate_dataset
        from src.preprocessing import prepare_data

        df = generate_dataset(n_samples=200)
        X_train, X_test, y_train, y_test, preprocessor, feature_names = prepare_data(df, save_models=False)

        assert isinstance(X_train, np.ndarray), "X_train should be numpy array"
        assert isinstance(X_test, np.ndarray), "X_test should be numpy array"
        assert len(y_train) + len(y_test) == 200, "Train+test should equal total"
        assert len(feature_names) > 0, "Feature names should not be empty"

    def test_preprocessor_handles_unknown_categories(self):
        """Test that preprocessor handles unseen categories gracefully."""
        from data.generate_dataset import generate_dataset
        from src.preprocessing import prepare_data

        df = generate_dataset(n_samples=200)
        _, _, _, _, preprocessor, _ = prepare_data(df, save_models=False)

        # Create input with unknown city
        unknown_input = pd.DataFrame([{
            "origin": "UnknownCity",
            "destination": "Delhi",
            "distance_km": 500,
            "route_type": "highway",
            "departure_hour": 10,
            "day_of_week": 3,
            "is_weekend": 0,
            "carrier_reliability_score": 0.8,
            "weather_severity": 3.0,
            "traffic_congestion": 4.0,
            "has_news_disruption": 0,
        }])

        # Should not raise (handle_unknown="ignore")
        result = preprocessor.transform(unknown_input)
        assert result is not None

    def test_stratified_split_preserves_ratio(self):
        """Test that train/test split preserves class balance."""
        from data.generate_dataset import generate_dataset
        from src.preprocessing import prepare_data

        df = generate_dataset(n_samples=1000)
        _, _, y_train, y_test, _, _ = prepare_data(df, save_models=False)

        train_rate = y_train.mean()
        test_rate = y_test.mean()
        # Stratified split should keep rates within 3%
        assert abs(train_rate - test_rate) < 0.03, "Stratification failed"


class TestModelTraining:
    """Tests for model training (quick smoke tests with tiny data)."""

    def test_logistic_regression_trains(self):
        """Test that logistic regression trains without error."""
        from src.train_models import train_logistic_regression
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)
        model = train_logistic_regression(X, y)
        assert hasattr(model, "predict_proba")
        probs = model.predict_proba(X)
        assert probs.shape == (100, 2)

    def test_random_forest_trains(self):
        """Test that random forest trains without error."""
        from src.train_models import train_random_forest
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)
        model = train_random_forest(X, y)
        assert hasattr(model, "feature_importances_")

    def test_xgboost_trains(self):
        """Test that XGBoost trains without error."""
        from src.train_models import train_xgboost
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)
        model = train_xgboost(X, y)
        assert hasattr(model, "predict_proba")


class TestRecommendationEngine:
    """Tests for the recommendation engine."""

    def test_high_risk_generates_recommendations(self):
        """Test that high-risk scenarios produce actionable recommendations."""
        from app.services.recommendation import generate_recommendations

        prediction = {"delay_probability": 0.9, "risk_level": "HIGH"}
        shipment = {
            "weather_severity": 9.0,
            "traffic_congestion": 8.0,
            "carrier_reliability_score": 0.5,
            "distance_km": 2000,
            "departure_hour": 8,
            "is_weekend": 1,
            "has_news_disruption": 1,
        }

        recs = generate_recommendations(prediction, shipment)
        assert len(recs) > 0, "Should generate recommendations for high risk"
        assert any("escalate" in r.lower() or "🚨" in r for r in recs), "Should include escalation"

    def test_low_risk_gets_standard_monitoring(self):
        """Test that low-risk scenarios get standard monitoring."""
        from app.services.recommendation import generate_recommendations

        prediction = {"delay_probability": 0.15, "risk_level": "LOW"}
        shipment = {
            "weather_severity": 1.0,
            "traffic_congestion": 2.0,
            "carrier_reliability_score": 0.95,
            "distance_km": 300,
            "departure_hour": 12,
            "is_weekend": 0,
            "has_news_disruption": 0,
        }

        recs = generate_recommendations(prediction, shipment)
        assert len(recs) > 0, "Should have at least one recommendation"
        assert any("standard" in r.lower() or "✅" in r for r in recs), "Should include standard monitoring"
