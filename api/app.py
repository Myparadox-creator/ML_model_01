import os
import json
import joblib
import pandas as pd
import asyncio
import random
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Shipment Delay Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

models = {}
preprocessor = None

@app.on_event("startup")
async def startup_event():
    global models, preprocessor
    try:
        preprocessor = joblib.load(os.path.join(MODELS_DIR, "preprocessor.joblib"))
        for model_name in ["logistic_regression", "random_forest", "xgboost"]:
            model_path = os.path.join(MODELS_DIR, f"{model_name}.joblib")
            if os.path.exists(model_path):
                models[model_name] = joblib.load(model_path)
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")
        
    # Start live generation
    asyncio.create_task(generate_live_shipments())

CITIES = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Hyderabad', 'Pune', 'Ahmedabad', 'Jaipur', 'Kolkata', 'Lucknow', 'Guwahati', 'Patna', 'Indore', 'Bhopal']

async def generate_live_shipments():
    while True:
        try:
            if "xgboost" in models and preprocessor is not None:
                origin = random.choice(CITIES)
                destination = random.choice([c for c in CITIES if c != origin])
                
                new_shipment = {
                    "origin": origin,
                    "destination": destination,
                    "distance_km": random.uniform(200, 2500),
                    "route_type": random.choice(["highway", "local", "mixed"]),
                    "departure_hour": random.randint(0, 23),
                    "day_of_week": random.randint(0, 6),
                }
                new_shipment["is_weekend"] = 1 if new_shipment["day_of_week"] >= 5 else 0
                new_shipment["carrier_reliability_score"] = random.uniform(0.5, 0.99)
                new_shipment["weather_severity"] = random.uniform(0, 10)
                new_shipment["traffic_congestion"] = random.uniform(0, 10)
                new_shipment["has_news_disruption"] = 1 if random.random() > 0.9 else 0
                
                df_new = pd.DataFrame([new_shipment])
                X = preprocessor.transform(df_new)
                prob = float(models["xgboost"].predict_proba(X)[0][1])
                pred = int(models["xgboost"].predict(X)[0])
                
                # ID and labels (must place in correct order if appending, but pandas matches columns)
                new_shipment["shipment_id"] = f"SHP-LIVE-{random.randint(1000, 99999)}"
                new_shipment["carrier_id"] = f"CARRIER_{random.randint(1, 20):03d}"
                new_shipment["delay_probability"] = prob
                new_shipment["delayed"] = pred
                
                csv_path = os.path.join(DATA_DIR, "shipments.csv")
                if os.path.exists(csv_path):
                    df_to_append = pd.DataFrame([new_shipment])
                    # Reorder to match existing
                    existing_columns = pd.read_csv(csv_path, nrows=0).columns
                    for col in existing_columns:
                        if col not in df_to_append:
                            df_to_append[col] = 0
                    df_to_append = df_to_append[existing_columns]
                    df_to_append.to_csv(csv_path, mode='a', header=False, index=False)
        except Exception as e:
            print(f"Live generator error: {e}")
            
        await asyncio.sleep(5)

class PredictionRequest(BaseModel):
    origin: str
    destination: str
    distance_km: float
    route_type: str
    departure_hour: int
    day_of_week: int
    is_weekend: int
    carrier_reliability_score: float
    weather_severity: float
    traffic_congestion: float
    has_news_disruption: int
    model_name: str = "xgboost"

@app.get("/health")
def health_check():
    return {"status": "ok", "models_loaded": list(models.keys())}

@app.get("/model-info")
def get_model_info():
    try:
        with open(os.path.join(OUTPUTS_DIR, "model_metrics.json"), "r") as f:
            metrics = json.load(f)
            # Find best model based on ROC-AUC
            if isinstance(metrics, dict) and "metrics" in metrics:
                best_model = max(metrics["metrics"], key=lambda x: x.get("ROC-AUC", 0))["Model"]
                return metrics
            else:
                best_model = max(metrics, key=lambda x: x.get("ROC-AUC", 0))["Model"]
                return {"metrics": metrics, "best_model": best_model}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/shipments")
def get_shipments(limit: int = 15):
    try:
        df = pd.read_csv(os.path.join(DATA_DIR, "shipments.csv"))
        # Take the most recent shipments (tail) and reverse to show newest first
        return df.tail(limit).iloc[::-1].to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics")
def get_analytics():
    try:
        df = pd.read_csv(os.path.join(DATA_DIR, "shipments.csv"))
        # Calculate live risk distribution
        total = len(df)
        if total == 0:
            return {"risk_distribution": [], "route_stats": []}
            
        high = len(df[df["delay_probability"] >= 0.7])
        medium = len(df[(df["delay_probability"] >= 0.4) & (df["delay_probability"] < 0.7)])
        low = total - high - medium
        
        risk_dist = [
            {"name": "Low Risk", "value": round((low/total)*100), "color": "#22c55e"},
            {"name": "Medium Risk", "value": round((medium/total)*100), "color": "#eab308"},
            {"name": "High Risk", "value": round((high/total)*100), "color": "#ef4444"}
        ]
        
        # Calculate route stats
        route_stats = []
        for rtype in ["highway", "local", "mixed"]:
            subset = df[df["route_type"] == rtype]
            if len(subset) > 0:
                rate = round((len(subset[subset["delayed"] == 1]) / len(subset)) * 100)
            else:
                rate = 0
            route_stats.append({"route": rtype.capitalize(), "delay_rate": rate})
            
        return {"risk_distribution": risk_dist, "route_stats": route_stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
def predict(req: PredictionRequest):
    if req.model_name not in models:
        raise HTTPException(status_code=400, detail=f"Model {req.model_name} not found")
    
    # Create DataFrame
    df = pd.DataFrame([req.dict(exclude={"model_name"})])
    
    # Preprocess
    try:
        X = preprocessor.transform(df)
        prob = float(models[req.model_name].predict_proba(X)[0][1])
        pred = int(models[req.model_name].predict(X)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
    
    if prob >= 0.7:
        risk_level = "HIGH"
        actions = ["Escalate to manager", "Consider alternate carrier"]
    elif prob >= 0.4:
        risk_level = "MEDIUM"
        actions = ["Monitor closely", "Notify receiver"]
    else:
        risk_level = "LOW"
        actions = ["Standard tracking"]
        
    return {
        "delay_probability": prob,
        "risk_level": risk_level,
        "predicted_delayed": pred == 1,
        "model_used": req.model_name,
        "recommended_actions": actions
    }
