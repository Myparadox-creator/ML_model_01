import { useState, useEffect } from 'react';
import { Package, AlertTriangle, CheckCircle, XCircle, ArrowRight } from 'lucide-react';
import KPICard from '../components/KPICard';
import RiskBadge from '../components/RiskBadge';

const API_BASE = 'http://localhost:8000';

export default function Dashboard() {
  const [shipments, setShipments] = useState([]);
  const [modelInfo, setModelInfo] = useState(null);
  const [loading, setLoading] = useState(true);
  const [isOffline, setIsOffline] = useState(false);

  useEffect(() => {
    async function fetchData() {
      try {
        const [shipRes, modelRes] = await Promise.all([
          fetch(`${API_BASE}/shipments?limit=15`),
          fetch(`${API_BASE}/model-info`),
        ]);
        if (shipRes.ok) {
          setShipments(await shipRes.json());
          setIsOffline(false);
        } else {
          setIsOffline(true);
        }
        if (modelRes.ok) setModelInfo(await modelRes.json());
      } catch (err) {
        console.error('API fetch failed:', err);
        setIsOffline(true);
      } finally {
        setLoading(false);
      }
    }
    
    fetchData();
    const interval = setInterval(fetchData, 5000);
    return () => clearInterval(interval);
  }, []);

  // Derive KPI stats from shipments
  const totalActive = shipments.length;
  const delayed = shipments.filter(s => s.delayed === 1).length;
  const atRisk = shipments.filter(s => s.delay_probability >= 0.4 && s.delayed === 0).length;
  const onTime = totalActive - delayed - atRisk;

  const getRiskLevel = (prob) => {
    if (prob >= 0.7) return 'HIGH';
    if (prob >= 0.4) return 'MEDIUM';
    return 'LOW';
  };

  return (
    <div>
      <div className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h1>Dashboard</h1>
          <p>Real-time overview of shipment delay predictions</p>
        </div>
        {isOffline && (
          <div style={{ padding: '8px 16px', borderRadius: 20, background: 'rgba(239, 68, 68, 0.15)', color: '#ef4444', fontWeight: 600, display: 'flex', alignItems: 'center', gap: 8 }}>
            <div style={{ width: 8, height: 8, borderRadius: '50%', background: '#ef4444', boxShadow: '0 0 8px #ef4444' }} />
            API OFFLINE
          </div>
        )}
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
                {shipments.map((s, i) => (
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
                {!isOffline && shipments.length === 0 && !loading && (
                   <tr><td colSpan="6" style={{ textAlign: 'center', padding: 24, color: 'var(--text-muted)' }}>No recent shipments found.</td></tr>
                )}
                {isOffline && (
                   <tr><td colSpan="6" style={{ textAlign: 'center', padding: 24, color: '#ef4444' }}>Backend API is offline. Cannot load shipments.</td></tr>
                )}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
