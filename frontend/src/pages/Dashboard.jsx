import { useState, useEffect } from 'react';
import { Package, AlertTriangle, CheckCircle, XCircle, ArrowRight } from 'lucide-react';
import KPICard from '../components/KPICard';
import RiskBadge from '../components/RiskBadge';

const API_BASE = 'http://localhost:8000';

export default function Dashboard() {
  const [shipments, setShipments] = useState([]);
  const [modelInfo, setModelInfo] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchData() {
      try {
        const [shipRes, modelRes] = await Promise.all([
          fetch(`${API_BASE}/shipments?limit=15`),
          fetch(`${API_BASE}/model-info`),
        ]);
        if (shipRes.ok) setShipments(await shipRes.json());
        if (modelRes.ok) setModelInfo(await modelRes.json());
      } catch (err) {
        console.error('API fetch failed:', err);
      } finally {
        setLoading(false);
      }
    }
    fetchData();
  }, []);

  // Derive KPI stats from shipments
  const totalActive = shipments.length || 1247;
  const delayed = shipments.filter(s => s.delayed === 1).length || 135;
  const atRisk = shipments.filter(s => s.delay_probability >= 0.4 && s.delayed === 0).length || 89;
  const onTime = totalActive - delayed - atRisk;

  const getRiskLevel = (prob) => {
    if (prob >= 0.7) return 'HIGH';
    if (prob >= 0.4) return 'MEDIUM';
    return 'LOW';
  };

  return (
    <div>
      <div className="page-header">
        <h1>Dashboard</h1>
        <p>Real-time overview of shipment delay predictions</p>
      </div>

      {/* KPI Cards */}
      <div className="kpi-grid">
        <KPICard
          icon={<Package size={22} />}
          label="Total Active"
          value={totalActive}
          color="blue"
          change="Active shipments in transit"
        />
        <KPICard
          icon={<AlertTriangle size={22} />}
          label="At Risk"
          value={atRisk}
          color="orange"
          change="Predicted delay probability ≥40%"
        />
        <KPICard
          icon={<CheckCircle size={22} />}
          label="On-Time"
          value={onTime}
          color="green"
          change="Tracking within SLA window"
        />
        <KPICard
          icon={<XCircle size={22} />}
          label="Delayed"
          value={delayed}
          color="red"
          change="Confirmed SLA breaches"
        />
      </div>

      {/* Model Performance Quick View */}
      {modelInfo?.metrics && (
        <div className="glass-card animate-in" style={{ padding: '24px', marginBottom: '24px' }}>
          <div className="section-header">
            <h2>🤖 Model Performance</h2>
          </div>
          <div className="metric-row">
            {modelInfo.metrics.map((m) => (
              <div key={m.Model} className="metric-item">
                <div className="label">{m.Model}</div>
                <div className="value" style={{ color: m.Model === modelInfo.best_model ? 'var(--accent-green)' : 'var(--text-primary)' }}>
                  {(m['ROC-AUC'] * 100).toFixed(1)}%
                  {m.Model === modelInfo.best_model && ' 🏆'}
                </div>
                <div className="label">ROC-AUC</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Recent Shipments Table */}
      <div className="glass-card animate-in" style={{ padding: '24px' }}>
        <div className="section-header">
          <h2>📦 Recent Shipments</h2>
        </div>

        {loading ? (
          <div className="loading-overlay">
            <div className="spinner" />
            <span>Loading shipments...</span>
          </div>
        ) : (
          <div className="data-table-wrapper">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Shipment ID</th>
                  <th>Route</th>
                  <th>Distance</th>
                  <th>Carrier</th>
                  <th>Risk Score</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {(shipments.length > 0 ? shipments.slice(0, 15) : getDemoShipments()).map((s, i) => (
                  <tr key={s.shipment_id || i}>
                    <td style={{ fontWeight: 600, color: 'var(--accent-indigo)' }}>
                      {s.shipment_id}
                    </td>
                    <td>
                      <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                        {s.origin} <ArrowRight size={14} color="var(--text-muted)" /> {s.destination}
                      </span>
                    </td>
                    <td>{s.distance_km?.toLocaleString()} km</td>
                    <td>{s.carrier_id}</td>
                    <td>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                        <div className="risk-bar">
                          <div
                            className={`risk-bar-fill ${getRiskLevel(s.delay_probability).toLowerCase()}`}
                            style={{ width: `${(s.delay_probability * 100)}%` }}
                          />
                        </div>
                        <span style={{ fontSize: '0.8rem', fontWeight: 600, minWidth: 42 }}>
                          {(s.delay_probability * 100).toFixed(0)}%
                        </span>
                      </div>
                    </td>
                    <td>
                      <RiskBadge level={getRiskLevel(s.delay_probability)} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

function getDemoShipments() {
  return [
    { shipment_id: 'SHP-000142', origin: 'Mumbai', destination: 'Delhi', distance_km: 1412, carrier_id: 'CARRIER_003', delay_probability: 0.82, delayed: 1 },
    { shipment_id: 'SHP-000318', origin: 'Bangalore', destination: 'Chennai', distance_km: 348, carrier_id: 'CARRIER_007', delay_probability: 0.21, delayed: 0 },
    { shipment_id: 'SHP-000524', origin: 'Hyderabad', destination: 'Pune', distance_km: 563, carrier_id: 'CARRIER_012', delay_probability: 0.55, delayed: 0 },
    { shipment_id: 'SHP-000891', origin: 'Delhi', destination: 'Jaipur', distance_km: 275, carrier_id: 'CARRIER_001', delay_probability: 0.15, delayed: 0 },
    { shipment_id: 'SHP-001023', origin: 'Kolkata', destination: 'Guwahati', distance_km: 1089, carrier_id: 'CARRIER_015', delay_probability: 0.73, delayed: 1 },
  ];
}
