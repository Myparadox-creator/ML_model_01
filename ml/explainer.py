"""
SHAP Model Explainability
===========================
Standalone module for generating SHAP-based explanations.
Can be used both in the API (real-time) and offline (batch analysis).

Usage:
    from ml.explainer import ShipmentExplainer
    
    explainer = ShipmentExplainer(model, preprocessor, feature_names)
    explanation = explainer.explain(shipment_data)
"""

import os
import numpy as np
import pandas as pd
import joblib
import logging

logger = logging.getLogger(__name__)


class ShipmentExplainer:
    """
    Generates human-readable explanations for shipment delay predictions
    using SHAP (SHapley Additive exPlanations).
    
    SHAP tells us: "How much did each feature push the prediction
    from the base rate towards the final prediction?"
    """

    def __init__(self, model, preprocessor, feature_names: list):
        self.model = model
        self.preprocessor = preprocessor
        self.feature_names = feature_names
        self.shap_explainer = None

        self._init_explainer()

    def _init_explainer(self):
        """Initialize SHAP explainer based on model type."""
        try:
            import shap

            model_type = type(self.model).__name__

            if model_type in ("XGBClassifier", "RandomForestClassifier"):
                self.shap_explainer = shap.TreeExplainer(self.model)
                logger.info(f"✅ SHAP TreeExplainer initialized for {model_type}")
            else:
                # Logistic Regression — use LinearExplainer or KernelExplainer
                # LinearExplainer is fast for linear models
                self.shap_explainer = shap.LinearExplainer(
                    self.model,
                    masker=np.zeros((1, len(self.feature_names))),
                )
                logger.info(f"✅ SHAP LinearExplainer initialized for {model_type}")

        except ImportError:
            logger.warning("SHAP not installed — `pip install shap`")
        except Exception as e:
            logger.warning(f"SHAP init failed: {e}")

    def explain(self, raw_features: dict, top_n: int = 6) -> list:
        """
        Generate top-N explanations for a single prediction.
        
        Args:
            raw_features: Dict of feature name → value (before preprocessing)
            top_n: Number of top contributing factors to return
            
        Returns:
            List of dicts:
            [
                {
                    "factor": "High weather severity (8.5/10)",
                    "contribution_pct": 35.2,
                    "shap_value": 0.42,
                    "direction": "increases delay risk"
                },
                ...
            ]
        """
        if self.shap_explainer is None:
            return []

        try:
            # Prepare input
            from src.preprocessing import NUMERIC_FEATURES, CATEGORICAL_FEATURES

            all_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
            df = pd.DataFrame([{k: raw_features.get(k) for k in all_features}])
            X = self.preprocessor.transform(df)

            # Compute SHAP values
            shap_values = self.shap_explainer.shap_values(X)

            # Handle binary classification output format
            if isinstance(shap_values, list):
                sv = shap_values[1][0]  # Class 1 (delayed), first sample
            else:
                sv = shap_values[0]  # First sample

            # Build explanation
            total_abs = np.sum(np.abs(sv))
            if total_abs == 0:
                return []

            explanations = []
            for i, (fname, shap_val) in enumerate(zip(self.feature_names, sv)):
                pct = abs(shap_val) / total_abs * 100
                if pct < 2:  # Skip negligible
                    continue

                explanations.append({
                    "factor": self._format_factor(fname, raw_features),
                    "contribution_pct": round(pct, 1),
                    "shap_value": round(float(shap_val), 4),
                    "direction": "increases delay risk" if shap_val > 0 else "decreases delay risk",
                })

            # Sort by absolute contribution
            explanations.sort(key=lambda x: x["contribution_pct"], reverse=True)
            return explanations[:top_n]

        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")
            return []

    def _format_factor(self, feature_name: str, raw_data: dict) -> str:
        """Format a feature name + value into a human-readable string."""
        labels = {
            "weather_severity": ("Weather severity", "/10"),
            "traffic_congestion": ("Traffic congestion", "/10"),
            "carrier_reliability_score": ("Carrier reliability", ""),
            "distance_km": ("Shipping distance", " km"),
            "departure_hour": ("Departure time", ":00"),
            "day_of_week": ("Day of week", ""),
            "is_weekend": ("Weekend", ""),
            "has_news_disruption": ("Disruption event", ""),
        }

        # Handle one-hot encoded features
        for cat in ["origin", "destination", "route_type"]:
            if feature_name.startswith(f"{cat}_"):
                value = feature_name.split("_", 1)[1]
                return f"{cat.replace('_', ' ').title()}: {value}"

        # Handle numeric features
        if feature_name in labels:
            label, suffix = labels[feature_name]
            value = raw_data.get(feature_name)
            if value is not None:
                if feature_name == "carrier_reliability_score":
                    return f"{label} ({value:.0%})"
                elif feature_name == "is_weekend":
                    return "Weekend shipment" if value else "Weekday shipment"
                elif feature_name == "has_news_disruption":
                    return "Active disruption" if value else "No disruption"
                return f"{label} ({value}{suffix})"
            return label

        return feature_name

    def batch_explain(self, features_list: list, top_n: int = 6) -> list:
        """Generate explanations for a batch of predictions."""
        return [self.explain(f, top_n) for f in features_list]


def create_explainer_from_disk(
    models_dir: str = None,
    model_name: str = "xgboost",
) -> ShipmentExplainer:
    """
    Factory function to create an explainer from saved model files.
    
    Usage:
        explainer = create_explainer_from_disk()
        reasons = explainer.explain({"weather_severity": 8.5, ...})
    """
    if models_dir is None:
        models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

    model = joblib.load(os.path.join(models_dir, f"{model_name}.joblib"))
    preprocessor = joblib.load(os.path.join(models_dir, "preprocessor.joblib"))

    # Get feature names
    from src.preprocessing import get_feature_names
    feature_names = get_feature_names(preprocessor)

    return ShipmentExplainer(model, preprocessor, feature_names)
