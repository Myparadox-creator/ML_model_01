"""
FastAPI Prediction API
=======================
Serves the trained shipment delay prediction models via REST endpoints.

Endpoints:
  GET  /health      — Server health check
  GET  /model-info  — Model metadata and performance metrics
  POST /predict     — Predict shipment delay probability
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

# ── App Setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="🚛 Shipment Delay Prediction API",
    description="AI-powered logistics early warning system for predicting shipment delays.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load Models on Startup ───────────────────────────────────────────────────
models = {}
preprocessor = None
metrics = None


@app.on_event("startup")
def load_models():
    """Load saved models and preprocessor on server startup."""
    global models, preprocessor, metrics

    try:
        preprocessor = joblib.load(os.path.join(MODELS_DIR, "preprocessor.joblib"))
        models = {
            "logistic_regression": joblib.load(os.path.join(MODELS_DIR, "logistic_regression.joblib")),
            "random_forest": joblib.load(os.path.join(MODELS_DIR, "random_forest.joblib")),
            "xgboost": joblib.load(os.path.join(MODELS_DIR, "xgboost.joblib")),
        }
        print(f"✅ Loaded {len(models)} models and preprocessor")
    except FileNotFoundError as e:
        print(f"⚠️  Model loading failed: {e}")
        print("   Run `python main.py` first to train models.")

    # Load metrics if available
    metrics_path = os.path.join(OUTPUTS_DIR, "model_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)


# ── Request / Response Schemas ───────────────────────────────────────────────
class ShipmentInput(BaseModel):
    """Input schema for shipment delay prediction."""
    origin: str = Field(..., example="Mumbai", description="Origin city")
    destination: str = Field(..., example="Delhi", description="Destination city")
    distance_km: float = Field(..., example=1400.0, ge=1, description="Distance in kilometers")
    route_type: str = Field(..., example="highway", description="Route type: highway, local, mixed")
    departure_hour: int = Field(..., example=14, ge=0, le=23, description="Hour of departure (0-23)")
    day_of_week: int = Field(..., example=2, ge=0, le=6, description="Day of week (0=Mon, 6=Sun)")
    is_weekend: int = Field(..., example=0, ge=0, le=1, description="1 if weekend, 0 otherwise")
    carrier_reliability_score: float = Field(..., example=0.85, ge=0, le=1, description="Carrier reliability (0-1)")
    weather_severity: float = Field(..., example=6.5, ge=0, le=10, description="Weather severity (0-10)")
    traffic_congestion: float = Field(..., example=7.2, ge=0, le=10, description="Traffic congestion (0-10)")
    has_news_disruption: int = Field(..., example=1, ge=0, le=1, description="1 if news disruption, 0 otherwise")
    model_name: Optional[str] = Field(
        default="xgboost",
        description="Model to use: logistic_regression, random_forest, or xgboost",
    )


class PredictionResponse(BaseModel):
    """Output schema for delay prediction."""
    delay_probability: float
    risk_level: str
    risk_color: str
    predicted_delayed: bool
    model_used: str
    recommended_actions: list[str]


# ── Helper Functions ─────────────────────────────────────────────────────────
def get_risk_level(probability: float) -> tuple[str, str]:
    """Map delay probability to risk level and color."""
    if probability >= 0.7:
        return "HIGH", "red"
    elif probability >= 0.4:
        return "MEDIUM", "yellow"
    else:
        return "LOW", "green"


def get_recommended_actions(probability: float, weather: float, traffic: float) -> list[str]:
    """Generate AI-recommended actions based on risk factors."""
    actions = []

    if probability >= 0.7:
        actions.append("🚨 Alert operations team immediately")
        actions.append("📞 Notify client of potential delay")

    if weather >= 7:
        actions.append("🌧️ Consider rerouting to avoid weather-affected zones")
    if traffic >= 7:
        actions.append("🚗 Reroute via alternate highway to bypass congestion")
    if probability >= 0.5:
        actions.append("🚚 Assign faster carrier if available")
    if probability >= 0.4:
        actions.append("📊 Monitor shipment with increased frequency")

    if not actions:
        actions.append("✅ Shipment on track — no action needed")

    return actions


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health")
def health_check():
    """Server health check."""
    return {
        "status": "healthy",
        "models_loaded": len(models),
        "models_available": list(models.keys()),
        "preprocessor_loaded": preprocessor is not None,
    }


@app.get("/model-info")
def model_info():
    """Return model metadata and performance metrics."""
    if metrics is None:
        return {"message": "No metrics available. Run evaluation first."}

    return {
        "models": list(models.keys()),
        "metrics": metrics,
        "best_model": max(metrics, key=lambda x: x["ROC-AUC"])["Model"] if metrics else None,
    }


@app.get("/shipments")
def get_shipments(limit: int = 100):
    """Return recent shipments from the dataset for the dashboard."""
    data_path = os.path.join(BASE_DIR, "data", "shipments.csv")
    if not os.path.exists(data_path):
        raise HTTPException(status_code=404, detail="Shipments dataset not found. Run main.py first.")
    
    try:
        df = pd.read_csv(data_path).head(limit)
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analytics")
def get_analytics():
    """Return aggregated analytics for the dashboard charts."""
    data_path = os.path.join(BASE_DIR, "data", "shipments.csv")
    if not os.path.exists(data_path):
        raise HTTPException(status_code=404, detail="Dataset not found")
        
    df = pd.read_csv(data_path)
    
    # 1. Risk distribution (using delay_probability)
    high = len(df[df["delay_probability"] >= 0.7])
    medium = len(df[(df["delay_probability"] >= 0.4) & (df["delay_probability"] < 0.7)])
    low = len(df[df["delay_probability"] < 0.4])
    total_risk = high + medium + low
    
    # 2. Delay rate by route type
    route_stats = []
    for route in df["route_type"].unique():
        subset = df[df["route_type"] == route]
        if len(subset) > 0:
            rate = (subset["delayed"].sum() / len(subset)) * 100
            route_stats.append({
                "route": route.capitalize(),
                "delay_rate": round(rate, 1)
            })
            
    return {
        "risk_distribution": [
            {"name": "Low Risk", "value": round((low/total_risk)*100, 1), "color": "#22c55e"},
            {"name": "Medium Risk", "value": round((medium/total_risk)*100, 1), "color": "#eab308"},
            {"name": "High Risk", "value": round((high/total_risk)*100, 1), "color": "#ef4444"}
        ],
        "route_stats": route_stats
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_delay(shipment: ShipmentInput):
    """Predict shipment delay probability."""
    if not models:
        raise HTTPException(status_code=503, detail="Models not loaded. Run training pipeline first.")

    model_name = shipment.model_name or "xgboost"
    if model_name not in models:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_name}' not found. Available: {list(models.keys())}",
        )

    # Prepare input as DataFrame (matching preprocessing pipeline format)
    input_data = pd.DataFrame([{
        "origin": shipment.origin,
        "destination": shipment.destination,
        "distance_km": shipment.distance_km,
        "route_type": shipment.route_type,
        "departure_hour": shipment.departure_hour,
        "day_of_week": shipment.day_of_week,
        "is_weekend": shipment.is_weekend,
        "carrier_reliability_score": shipment.carrier_reliability_score,
        "weather_severity": shipment.weather_severity,
        "traffic_congestion": shipment.traffic_congestion,
        "has_news_disruption": shipment.has_news_disruption,
    }])

    # Preprocess
    try:
        X = preprocessor.transform(input_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Preprocessing error: {str(e)}")

    # Predict
    model = models[model_name]
    probability = float(model.predict_proba(X)[0][1])
    predicted = bool(model.predict(X)[0])
    risk_level, risk_color = get_risk_level(probability)
    actions = get_recommended_actions(probability, shipment.weather_severity, shipment.traffic_congestion)

    return PredictionResponse(
        delay_probability=round(probability, 4),
        risk_level=risk_level,
        risk_color=risk_color,
        predicted_delayed=predicted,
        model_used=model_name,
        recommended_actions=actions,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
