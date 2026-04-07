"""
ML Service
=============
Handles loading ML models, running predictions, and generating
SHAP-based explanations. This is the brain of the Early Warning System.

Architecture:
    - Models and preprocessor are loaded once at startup (singleton)
    - SHAP TreeExplainer is pre-computed for fast per-prediction explanations
    - Thread-safe prediction with proper error handling
"""

import os
import time
import logging
import numpy as np
import pandas as pd
import joblib

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# ── Feature names must match the training pipeline ───────────────────────────
PREDICTION_FEATURES = [
    "origin", "destination", "distance_km", "route_type",
    "departure_hour", "day_of_week", "is_weekend",
    "carrier_reliability_score", "weather_severity",
    "traffic_congestion", "has_news_disruption",
]

# Human-readable labels for SHAP explanation output
FEATURE_LABELS = {
    "distance_km": "Shipping distance",
    "departure_hour": "Departure time",
    "day_of_week": "Day of week",
    "is_weekend": "Weekend shipment",
    "carrier_reliability_score": "Carrier reliability",
    "weather_severity": "Weather severity",
    "traffic_congestion": "Traffic congestion",
    "has_news_disruption": "News/disruption event",
    "origin": "Origin city",
    "destination": "Destination city",
    "route_type": "Route type",
}


class MLService:
    """
    Singleton service for ML model operations.
    
    Usage:
        ml = MLService()
        ml.load_models()  # Call once at startup
        result = ml.predict(shipment_data)
    """

    def __init__(self):
        self.models = {}
        self.preprocessor = None
        self.shap_explainer = None
        self.feature_names = None
        self._loaded = False

    def load_models(self):
        """Load all trained models and preprocessor from disk."""
        models_path = settings.MODELS_PATH

        try:
            # Load preprocessor
            preprocessor_path = os.path.join(models_path, "preprocessor.joblib")
            if os.path.exists(preprocessor_path):
                self.preprocessor = joblib.load(preprocessor_path)
                logger.info("✅ Preprocessor loaded")
            else:
                logger.warning(f"⚠️ Preprocessor not found at {preprocessor_path}")
                return

            # Load models
            model_files = {
                "logistic_regression": "logistic_regression.joblib",
                "random_forest": "random_forest.joblib",
                "xgboost": "xgboost.joblib",
            }

            for name, filename in model_files.items():
                filepath = os.path.join(models_path, filename)
                if os.path.exists(filepath):
                    self.models[name] = joblib.load(filepath)
                    logger.info(f"✅ Model loaded: {name}")
                else:
                    logger.warning(f"⚠️ Model not found: {filepath}")

            # Extract feature names from preprocessor
            self._extract_feature_names()

            # Initialize SHAP explainer for XGBoost (fastest for tree models)
            self._init_shap_explainer()

            self._loaded = True
            logger.info(f"🚀 ML Service ready — {len(self.models)} models loaded")

        except Exception as e:
            logger.error(f"❌ Failed to load models: {e}")
            raise

    def _extract_feature_names(self):
        """Get feature names from the fitted preprocessor."""
        if self.preprocessor is None:
            return

        feature_names = []
        # Numeric features (first transformer)
        num_features = self.preprocessor.transformers_[0][2]  # column names
        feature_names.extend(num_features)

        # Categorical features (one-hot encoded)
        try:
            cat_encoder = self.preprocessor.named_transformers_["cat"].named_steps["encoder"]
            cat_input_features = self.preprocessor.transformers_[1][2]
            cat_feature_names = cat_encoder.get_feature_names_out(cat_input_features)
            feature_names.extend(cat_feature_names.tolist())
        except Exception:
            pass

        self.feature_names = feature_names
        logger.info(f"📋 {len(feature_names)} features extracted from preprocessor")

    def _init_shap_explainer(self):
        """Initialize SHAP TreeExplainer for the XGBoost model."""
        if "xgboost" not in self.models:
            logger.warning("⚠️ XGBoost model not available — SHAP disabled")
            return

        try:
            import shap
            self.shap_explainer = shap.TreeExplainer(self.models["xgboost"])
            logger.info("✅ SHAP TreeExplainer initialized for XGBoost")
        except ImportError:
            logger.warning("⚠️ SHAP library not installed — explanations disabled")
        except Exception as e:
            logger.warning(f"⚠️ SHAP initialization failed: {e}")

    @property
    def is_ready(self) -> bool:
        """Check if models are loaded and ready for predictions."""
        return self._loaded and self.preprocessor is not None and len(self.models) > 0

    @property
    def available_models(self) -> list:
        """List of loaded model names."""
        return list(self.models.keys())

    def predict(self, shipment_data: dict, model_name: str = None) -> dict:
        """
        Run a delay prediction with SHAP explanations.
        
        Args:
            shipment_data: Dict with shipment features
            model_name: Which model to use (default: from config)
            
        Returns:
            {
                "delay_probability": 0.82,
                "risk_level": "HIGH",
                "predicted_delayed": True,
                "model_used": "xgboost",
                "reasons": [...],
                "recommendations": [...],
                "prediction_time_ms": 45.2
            }
        """
        if not self.is_ready:
            raise RuntimeError("ML models not loaded. Run load_models() first.")

        start_time = time.time()

        # Determine model
        model_name = model_name or settings.DEFAULT_MODEL
        if model_name not in self.models:
            available = ", ".join(self.available_models)
            raise ValueError(f"Model '{model_name}' not found. Available: {available}")

        model = self.models[model_name]

        # Prepare input DataFrame
        df = pd.DataFrame([{k: shipment_data.get(k) for k in PREDICTION_FEATURES}])

        # Preprocess
        try:
            X = self.preprocessor.transform(df)
        except Exception as e:
            raise ValueError(f"Preprocessing failed: {e}")

        # Predict
        prob = float(model.predict_proba(X)[0][1])
        pred = int(model.predict(X)[0])

        # Risk level
        if prob >= 0.7:
            risk_level = "HIGH"
        elif prob >= 0.4:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        # SHAP explanations
        reasons = self._explain_prediction(X, shipment_data)

        elapsed_ms = (time.time() - start_time) * 1000

        return {
            "delay_probability": round(prob, 4),
            "risk_level": risk_level,
            "predicted_delayed": pred == 1,
            "model_used": model_name,
            "reasons": reasons,
            "prediction_time_ms": round(elapsed_ms, 2),
        }

    def _explain_prediction(self, X_processed, raw_data: dict) -> list:
        """
        Generate human-readable explanations using SHAP values.
        
        Returns a list of top contributing factors sorted by importance:
        [
            {"factor": "Storm risk (weather_severity=8.5)", "contribution": "40%", "direction": "increases delay"},
            ...
        ]
        """
        if self.shap_explainer is None or self.feature_names is None:
            # Fallback: rule-based explanations
            return self._fallback_explanations(raw_data)

        try:
            shap_values = self.shap_explainer.shap_values(X_processed)

            # For binary classification, shap_values may be a list [class_0, class_1]
            # or a single array. We want class_1 (delay) contributions.
            if isinstance(shap_values, list):
                sv = shap_values[1][0]  # First (only) sample, class 1
            else:
                sv = shap_values[0]  # First sample

            # Pair features with their SHAP values
            feature_contributions = []
            total_abs_shap = np.sum(np.abs(sv))

            if total_abs_shap == 0:
                return self._fallback_explanations(raw_data)

            for i, (fname, shap_val) in enumerate(zip(self.feature_names, sv)):
                # Calculate percentage contribution
                pct = abs(shap_val) / total_abs_shap * 100

                if pct < 3:  # Skip negligible contributions
                    continue

                # Get human-readable label
                label = self._format_feature_label(fname, raw_data)
                direction = "increases delay risk" if shap_val > 0 else "decreases delay risk"

                feature_contributions.append({
                    "factor": label,
                    "contribution": f"{pct:.0f}%",
                    "direction": direction,
                    "_abs_shap": abs(shap_val),  # For sorting (removed from output later)
                })

            # Sort by absolute contribution and take top 6
            feature_contributions.sort(key=lambda x: x["_abs_shap"], reverse=True)
            top_reasons = feature_contributions[:6]

            # Remove internal sort key
            for r in top_reasons:
                del r["_abs_shap"]

            return top_reasons

        except Exception as e:
            logger.warning(f"SHAP explanation failed: {e}")
            return self._fallback_explanations(raw_data)

    def _format_feature_label(self, feature_name: str, raw_data: dict) -> str:
        """Convert a feature name into a human-readable description with value."""
        # Handle one-hot encoded features (e.g., "origin_Mumbai")
        for cat in ["origin", "destination", "route_type"]:
            if feature_name.startswith(f"{cat}_"):
                value = feature_name.split("_", 1)[1]
                base_label = FEATURE_LABELS.get(cat, cat)
                return f"{base_label}: {value}"

        # Handle numeric features
        base_label = FEATURE_LABELS.get(feature_name, feature_name)
        value = raw_data.get(feature_name)

        if value is not None:
            if feature_name == "carrier_reliability_score":
                return f"{base_label} ({value:.0%})"
            elif feature_name in ("weather_severity", "traffic_congestion"):
                return f"{base_label} ({value:.1f}/10)"
            elif feature_name == "distance_km":
                return f"{base_label} ({value:.0f} km)"
            elif feature_name == "departure_hour":
                return f"{base_label} ({value}:00)"
            elif feature_name == "is_weekend":
                return "Weekend shipment" if value else "Weekday shipment"
            elif feature_name == "has_news_disruption":
                return "Active disruption event" if value else "No disruption"
            else:
                return f"{base_label} ({value})"

        return base_label

    def _fallback_explanations(self, raw_data: dict) -> list:
        """Rule-based explanations when SHAP is unavailable."""
        reasons = []

        ws = raw_data.get("weather_severity", 0)
        if ws >= 7:
            reasons.append({"factor": f"Severe weather ({ws:.1f}/10)", "contribution": "High", "direction": "increases delay risk"})
        elif ws >= 4:
            reasons.append({"factor": f"Moderate weather ({ws:.1f}/10)", "contribution": "Medium", "direction": "increases delay risk"})

        tc = raw_data.get("traffic_congestion", 0)
        if tc >= 7:
            reasons.append({"factor": f"Heavy traffic ({tc:.1f}/10)", "contribution": "High", "direction": "increases delay risk"})
        elif tc >= 4:
            reasons.append({"factor": f"Moderate traffic ({tc:.1f}/10)", "contribution": "Medium", "direction": "increases delay risk"})

        cr = raw_data.get("carrier_reliability_score", 1)
        if cr < 0.6:
            reasons.append({"factor": f"Low carrier reliability ({cr:.0%})", "contribution": "High", "direction": "increases delay risk"})

        dist = raw_data.get("distance_km", 0)
        if dist > 1500:
            reasons.append({"factor": f"Long distance ({dist:.0f} km)", "contribution": "Medium", "direction": "increases delay risk"})

        if raw_data.get("has_news_disruption"):
            reasons.append({"factor": "Active disruption event", "contribution": "High", "direction": "increases delay risk"})

        if raw_data.get("is_weekend"):
            reasons.append({"factor": "Weekend shipment", "contribution": "Low", "direction": "increases delay risk"})

        return reasons[:6]


# ── Singleton Instance ───────────────────────────────────────────────────────
_ml_service = None


def get_ml_service() -> MLService:
    """Get or create the global MLService singleton."""
    global _ml_service
    if _ml_service is None:
        _ml_service = MLService()
    return _ml_service
