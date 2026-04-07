# 🚛 AI Early Warning System for Shipment Delays

> Production-ready ML system that predicts shipment delays **48–72 hours in advance**, explains *why* delays happen using SHAP, and suggests corrective actions.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green?logo=fastapi)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange)
![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)

---

## ✨ Features

- **🔮 Delay Prediction** — Binary classification with probability output (not just yes/no)
- **🧠 SHAP Explanations** — "Storm risk contributed 35%, low carrier reliability 20%"
- **💡 Smart Recommendations** — Reroute, change carrier, adjust ETA, notify customer
- **🔒 JWT Authentication** — Secure API with role-based access
- **🌧️ Weather Integration** — Mock/real OpenWeatherMap for route weather
- **🚗 Traffic Analysis** — Time-of-day congestion patterns per city
- **🗄️ Database** — SQLite (dev) / PostgreSQL (prod) with full ORM
- **📦 Caching** — In-memory (dev) / Redis (prod) with graceful fallback
- **🚨 Auto Alerts** — High-risk predictions auto-create alerts for operators
- **⏱️ Rate Limiting** — Token bucket per-IP protection
- **📊 Analytics Dashboard API** — Risk distribution, model metrics, delay trends
- **🐳 Docker Ready** — Multi-stage Dockerfile + docker-compose

---

## 🏗️ Architecture

```
ML_model_01/
├── app/                      # Backend (FastAPI)
│   ├── main.py               # App factory + lifespan
│   ├── config.py             # Settings from .env
│   ├── database.py           # SQLAlchemy setup
│   ├── auth/                 # JWT authentication
│   ├── models/               # ORM models (5 tables)
│   ├── schemas/              # Pydantic request/response
│   ├── routers/              # API endpoints (5 routers)
│   ├── services/             # Business logic
│   │   ├── ml_service.py     # Model loading + SHAP
│   │   ├── weather_service.py
│   │   ├── traffic_service.py
│   │   ├── recommendation.py
│   │   └── cache_service.py
│   └── middleware/           # Logging + rate limiting
├── ml/                       # ML explainability
│   └── explainer.py          # SHAP TreeExplainer
├── src/                      # ML pipeline
│   ├── preprocessing.py
│   ├── train_models.py
│   └── evaluate.py
├── data/                     # Dataset (10K shipments)
├── models/                   # Trained .joblib files
├── outputs/                  # Evaluation plots + metrics
├── tests/                    # 44 unit/integration tests
├── frontend/                 # Vite+React dashboard
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Models (first time only)
```bash
python main.py
```

### 3. Start the API
```bash
uvicorn app.main:app --reload --port 8000
```

### 4. Open Docs
Visit **http://localhost:8000/docs**

### 5. Register → Login → Predict
```bash
# Register
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","email":"admin@test.com","password":"admin123"}'

# Login
curl -X POST http://localhost:8000/auth/login \
  -d "username=admin&password=admin123"

# Predict (use token from login response)
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "origin": "Mumbai",
    "destination": "Kolkata",
    "distance_km": 2050,
    "route_type": "local",
    "departure_hour": 8,
    "day_of_week": 5,
    "is_weekend": 1,
    "carrier_reliability_score": 0.52,
    "weather_severity": 9.0,
    "traffic_congestion": 8.5,
    "has_news_disruption": 1
  }'
```

**Response:**
```json
{
  "delay_probability": 0.87,
  "risk_level": "HIGH",
  "predicted_delayed": true,
  "model_used": "xgboost",
  "reasons": [
    {"factor": "Weather severity (9.0/10)", "contribution": "35%", "direction": "increases delay risk"},
    {"factor": "Traffic congestion (8.5/10)", "contribution": "22%", "direction": "increases delay risk"},
    {"factor": "Carrier reliability (52%)", "contribution": "18%", "direction": "increases delay risk"}
  ],
  "recommendations": [
    "🚨 Escalate to operations manager — high delay risk detected",
    "📱 Proactively notify customer of potential delay and updated ETA",
    "🌧️ Reroute shipment to avoid weather-affected region"
  ]
}
```

---

## 🐳 Docker

```bash
docker-compose up -d
# App:   http://localhost:8000
# Redis: localhost:6379
```

---

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# With coverage
python -m pytest tests/ --cov=app --cov=ml --cov-report=term-missing
```

---

## 📊 ML Models

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | ~0.74 | ~0.72 | ~0.75 | ~0.73 | ~0.82 |
| Random Forest | ~0.88 | ~0.87 | ~0.89 | ~0.88 | ~0.95 |
| **XGBoost** ⭐ | ~0.90 | ~0.89 | ~0.91 | ~0.90 | ~0.96 |

---

## 🔑 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/auth/register` | Register user |
| POST | `/auth/login` | Get JWT token |
| POST | `/api/v1/predict` | **Predict delay** |
| GET | `/api/v1/predictions` | Prediction history |
| POST | `/api/v1/shipments` | Create shipment |
| GET | `/api/v1/shipments` | List shipments |
| GET | `/api/v1/carriers` | List carriers |
| GET | `/api/v1/alerts` | View alerts |
| POST | `/api/v1/alerts/{id}/resolve` | Resolve alert |
| GET | `/api/v1/analytics/dashboard` | Dashboard data |
| GET | `/api/v1/analytics/weather/{city}` | Weather |
| GET | `/api/v1/analytics/traffic` | Traffic |
| GET | `/health` | Health check |

---

## ⚙️ Configuration

Copy `.env.example` to `.env` and configure:

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `sqlite:///./shipment_delay.db` | Database connection |
| `JWT_SECRET_KEY` | `change-me` | JWT signing key |
| `REDIS_ENABLED` | `false` | Enable Redis caching |
| `OPENWEATHER_API_KEY` | empty | Real weather API key |
| `RATE_LIMIT_PER_MINUTE` | `100` | API rate limit |

---

## 📝 License

MIT
