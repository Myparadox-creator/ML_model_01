import { useState } from 'react';
import { Send, Zap } from 'lucide-react';
import RiskBadge from '../components/RiskBadge';

const API_BASE = 'http://localhost:8000';

const CITIES = [
  'Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Hyderabad',
  'Pune', 'Ahmedabad', 'Jaipur', 'Kolkata', 'Lucknow',
  'Guwahati', 'Patna', 'Indore', 'Bhopal',
];

const defaultForm = {
  origin: 'Mumbai',
  destination: 'Delhi',
  distance_km: 1400,
  route_type: 'highway',
  departure_hour: 14,
  day_of_week: 2,
  is_weekend: 0,
  carrier_reliability_score: 0.85,
  weather_severity: 5,
  traffic_congestion: 5,
  has_news_disruption: 0,
  model_name: 'xgboost',
};

export default function Predict() {
  const [form, setForm] = useState(defaultForm);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const update = (field, value) => {
    setForm((prev) => ({ ...prev, [field]: value }));
  };

  const handlePredict = async () => {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(form),
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || 'Prediction failed');
      }

      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err.message || 'Could not connect to API. Is the server running?');
    } finally {
      setLoading(false);
    }
  };

  const gaugeColor = result
    ? result.risk_level === 'HIGH' ? 'var(--accent-red)'
    : result.risk_level === 'MEDIUM' ? 'var(--accent-yellow)'
    : 'var(--accent-green)'
    : 'var(--accent-blue)';

  return (
    <div>
      <div className="page-header">
        <h1>Predict Shipment Delay</h1>
        <p>Enter shipment details to get AI-powered delay probability</p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1.4fr 1fr', gap: 24, alignItems: 'start' }}>
        {/* Input Form */}
        <div className="glass-card animate-in" style={{ padding: 28 }}>
          <h3 style={{ marginBottom: 24, fontSize: '1.1rem', fontWeight: 700 }}>
            📋 Shipment Details
          </h3>

          <div className="form-grid">
            {/* Origin */}
            <div className="form-group">
              <label>Origin City</label>
              <select value={form.origin} onChange={(e) => update('origin', e.target.value)}>
                {CITIES.map((c) => <option key={c} value={c}>{c}</option>)}
              </select>
            </div>

            {/* Destination */}
            <div className="form-group">
              <label>Destination City</label>
              <select value={form.destination} onChange={(e) => update('destination', e.target.value)}>
                {CITIES.map((c) => <option key={c} value={c}>{c}</option>)}
              </select>
            </div>

            {/* Distance */}
            <div className="form-group">
              <label>Distance (km)</label>
              <input
                type="number"
                value={form.distance_km}
                onChange={(e) => update('distance_km', Number(e.target.value))}
                min="1"
              />
            </div>

            {/* Route Type */}
            <div className="form-group">
              <label>Route Type</label>
              <select value={form.route_type} onChange={(e) => update('route_type', e.target.value)}>
                <option value="highway">Highway</option>
                <option value="local">Local</option>
                <option value="mixed">Mixed</option>
              </select>
            </div>

            {/* Departure Hour */}
            <div className="form-group">
              <label>Departure Hour</label>
              <input
                type="number"
                value={form.departure_hour}
                onChange={(e) => update('departure_hour', Number(e.target.value))}
                min="0" max="23"
              />
            </div>

            {/* Day of Week */}
            <div className="form-group">
              <label>Day of Week</label>
              <select value={form.day_of_week} onChange={(e) => {
                const day = Number(e.target.value);
                update('day_of_week', day);
                update('is_weekend', day >= 5 ? 1 : 0);
              }}>
                {['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                  .map((d, i) => <option key={d} value={i}>{d}</option>)}
              </select>
            </div>
          </div>

          {/* Sliders */}
          <div style={{ marginTop: 24, display: 'flex', flexDirection: 'column', gap: 20 }}>
            <div className="form-group">
              <label>Carrier Reliability Score</label>
              <div className="slider-container">
                <input
                  type="range" min="0" max="100" step="1"
                  value={Math.round(form.carrier_reliability_score * 100)}
                  onChange={(e) => update('carrier_reliability_score', Number(e.target.value) / 100)}
                />
                <span className="slider-value">{(form.carrier_reliability_score * 100).toFixed(0)}%</span>
              </div>
            </div>

            <div className="form-group">
              <label>Weather Severity</label>
              <div className="slider-container">
                <input
                  type="range" min="0" max="10" step="0.5"
                  value={form.weather_severity}
                  onChange={(e) => update('weather_severity', Number(e.target.value))}
                />
                <span className="slider-value">{form.weather_severity}</span>
              </div>
            </div>

            <div className="form-group">
              <label>Traffic Congestion</label>
              <div className="slider-container">
                <input
                  type="range" min="0" max="10" step="0.5"
                  value={form.traffic_congestion}
                  onChange={(e) => update('traffic_congestion', Number(e.target.value))}
                />
                <span className="slider-value">{form.traffic_congestion}</span>
              </div>
            </div>

            {/* News Disruption Toggle */}
            <div className="form-group">
              <label>News Disruption Event</label>
              <div
                className="toggle-switch"
                onClick={() => update('has_news_disruption', form.has_news_disruption ? 0 : 1)}
              >
                <div className={`toggle-track ${form.has_news_disruption ? 'active' : ''}`}>
                  <div className="toggle-thumb" />
                </div>
                <span style={{ color: form.has_news_disruption ? 'var(--accent-orange)' : 'var(--text-muted)' }}>
                  {form.has_news_disruption ? 'Active disruption detected' : 'No disruption'}
                </span>
              </div>
            </div>
          </div>

          {/* Model Selector */}
          <div style={{ marginTop: 24 }}>
            <label style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.5px', display: 'block', marginBottom: 10 }}>
              AI Model
            </label>
            <div className="model-selector">
              {[
                { id: 'logistic_regression', label: 'Logistic Regression' },
                { id: 'random_forest', label: 'Random Forest' },
                { id: 'xgboost', label: 'XGBoost' },
              ].map((m) => (
                <button
                  key={m.id}
                  className={`model-pill ${form.model_name === m.id ? 'active' : ''}`}
                  onClick={() => update('model_name', m.id)}
                >
                  {m.label}
                </button>
              ))}
            </div>
          </div>

          {/* Submit */}
          <button
            className="btn btn-primary btn-lg"
            style={{ width: '100%', marginTop: 28 }}
            onClick={handlePredict}
            disabled={loading}
          >
            {loading ? (
              <><div className="spinner" style={{ width: 18, height: 18, borderWidth: 2 }} /> Analyzing...</>
            ) : (
              <><Zap size={18} /> Predict Delay Probability</>
            )}
          </button>
        </div>

        {/* Result Panel */}
        <div className="glass-card animate-in" style={{ position: 'sticky', top: 32 }}>
          {!result && !error && (
            <div style={{ padding: 48, textAlign: 'center' }}>
              <div style={{ fontSize: 48, marginBottom: 16, opacity: 0.3 }}>🎯</div>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
                Fill in shipment details and click<br />"Predict" to see results
              </p>
            </div>
          )}

          {error && (
            <div style={{ padding: 32, textAlign: 'center' }}>
              <div style={{ fontSize: 40, marginBottom: 12 }}>⚠️</div>
              <p style={{ color: 'var(--accent-red)', fontSize: '0.9rem', marginBottom: 8, fontWeight: 600 }}>
                Prediction Failed
              </p>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>{error}</p>
            </div>
          )}

          {result && (
            <div className="prediction-result">
              <div className="probability-gauge">
                <div
                  className="probability-circle"
                  style={{
                    '--gauge-color': gaugeColor,
                    '--gauge-percent': `${(result.delay_probability * 100).toFixed(0)}%`,
                  }}
                >
                  <div className="probability-value" style={{ color: gaugeColor }}>
                    {(result.delay_probability * 100).toFixed(1)}%
                  </div>
                  <div className="probability-label">Delay Probability</div>
                </div>
              </div>

              <RiskBadge level={result.risk_level} />

              <p style={{ margin: '16px 0 4px', color: 'var(--text-muted)', fontSize: '0.8rem' }}>
                Model: <strong style={{ color: 'var(--text-primary)' }}>{result.model_used}</strong>
              </p>

              <p style={{ color: 'var(--text-muted)', fontSize: '0.8rem', marginBottom: 20 }}>
                Predicted: <strong style={{ color: result.predicted_delayed ? 'var(--accent-red)' : 'var(--accent-green)' }}>
                  {result.predicted_delayed ? '⚠️ DELAYED' : '✅ ON-TIME'}
                </strong>
              </p>

              <h4 style={{ textAlign: 'left', fontSize: '0.85rem', fontWeight: 700, marginBottom: 12, color: 'var(--text-secondary)' }}>
                Recommended Actions
              </h4>
              <ul className="actions-list">
                {result.recommended_actions.map((action, i) => (
                  <li key={i}>{action}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
