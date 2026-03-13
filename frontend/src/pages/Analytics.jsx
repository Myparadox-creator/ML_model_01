import { useState, useEffect } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  PieChart, Pie, Cell, ResponsiveContainer, RadarChart, PolarGrid,
  PolarAngleAxis, PolarRadiusAxis, Radar,
} from 'recharts';

const API_BASE = 'http://localhost:8000';

const CHART_COLORS = ['#3b82f6', '#22c55e', '#f97316', '#ef4444', '#a855f7', '#06b6d4'];

export default function Analytics() {
  const [metrics, setMetrics] = useState(null);
  const [analytics, setAnalytics] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchData() {
      try {
        const [modelRes, analyticsRes] = await Promise.all([
          fetch(`${API_BASE}/model-info`),
          fetch(`${API_BASE}/analytics`),
        ]);
        if (modelRes.ok) setMetrics(await modelRes.json());
        if (analyticsRes.ok) setAnalytics(await analyticsRes.json());
      } catch (err) {
        console.error('Analytics fetch failed:', err);
      } finally {
        setLoading(false);
      }
    }
    fetchData();
  }, []);

  // Prepare chart data from model metrics
  const modelComparison = metrics?.metrics?.map((m) => ({
    name: m.Model.replace('Logistic Regression', 'Log. Reg.'),
    Accuracy: m.Accuracy,
    Precision: m.Precision,
    Recall: m.Recall,
    F1: m['F1-Score'],
    'ROC-AUC': m['ROC-AUC'],
  })) || getDefaultModelData();

  const radarData = metrics?.metrics?.length
    ? ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'].map((metric) => {
        const point = { metric };
        metrics.metrics.forEach((m) => {
          const key = m.Model.replace('Logistic Regression', 'Log. Reg.');
          point[key] = m[metric] || 0;
        });
        return point;
      })
    : getDefaultRadarData();

  // Risk distribution
  const riskDistribution = analytics?.risk_distribution || [
    { name: 'Low Risk', value: 65, color: '#22c55e' },
    { name: 'Medium Risk', value: 22, color: '#eab308' },
    { name: 'High Risk', value: 13, color: '#ef4444' },
  ];

  // Route delay rates
  const routeData = analytics?.route_stats || [
    { route: 'Highway', delay_rate: 28 },
    { route: 'Local', delay_rate: 38 },
    { route: 'Mixed', delay_rate: 32 },
  ];

  const customTooltipStyle = {
    backgroundColor: '#1e293b',
    border: '1px solid rgba(148, 163, 184, 0.12)',
    borderRadius: '12px',
    padding: '12px 16px',
    color: '#f1f5f9',
    fontSize: '0.8rem',
  };

  if (loading) {
    return (
      <div>
        <div className="page-header">
          <h1>Analytics</h1>
          <p>Model performance and shipment insights</p>
        </div>
        <div className="loading-overlay">
          <div className="spinner" />
          <span>Loading analytics...</span>
        </div>
      </div>
    );
  }

  return (
    <div>
      <div className="page-header">
        <h1>Analytics</h1>
        <p>Model performance comparison and shipment delay insights</p>
      </div>

      {/* Model Metrics Cards */}
      {metrics?.metrics && (
        <div className="glass-card animate-in" style={{ padding: 24, marginBottom: 24 }}>
          <h3 style={{ marginBottom: 16, fontSize: '1.1rem', fontWeight: 700 }}>
            🏆 Model Performance Summary
          </h3>
          <div className="metric-row">
            {metrics.metrics.map((m) => (
              <div key={m.Model} className="metric-item" style={{
                borderLeft: `3px solid ${m.Model === metrics.best_model ? 'var(--accent-green)' : 'var(--border-subtle)'}`,
              }}>
                <div className="label">{m.Model} {m.Model === metrics.best_model ? '🏆' : ''}</div>
                <div className="value" style={{
                  color: m.Model === metrics.best_model ? 'var(--accent-green)' : 'var(--text-primary)',
                  fontSize: '1.3rem',
                }}>
                  {(m['ROC-AUC'] * 100).toFixed(1)}%
                </div>
                <div className="label" style={{ marginTop: 4 }}>
                  F1: {m['F1-Score'].toFixed(3)} | Acc: {m.Accuracy.toFixed(3)}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Charts Grid */}
      <div className="charts-grid">
        {/* Model Comparison Bar Chart */}
        <div className="glass-card chart-card animate-in">
          <h3>📊 Model Comparison</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={modelComparison} barCategoryGap="20%">
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.1)" />
              <XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 12 }} />
              <YAxis tick={{ fill: '#94a3b8', fontSize: 12 }} domain={[0, 1]} />
              <Tooltip contentStyle={customTooltipStyle} />
              <Legend wrapperStyle={{ fontSize: 12, color: '#94a3b8' }} />
              <Bar dataKey="Accuracy" fill="#3b82f6" radius={[4, 4, 0, 0]} />
              <Bar dataKey="F1" fill="#22c55e" radius={[4, 4, 0, 0]} />
              <Bar dataKey="ROC-AUC" fill="#f97316" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Radar Chart */}
        <div className="glass-card chart-card animate-in">
          <h3>🎯 Multi-metric Radar</h3>
          <ResponsiveContainer width="100%" height={300}>
            <RadarChart data={radarData}>
              <PolarGrid stroke="rgba(148,163,184,0.15)" />
              <PolarAngleAxis dataKey="metric" tick={{ fill: '#94a3b8', fontSize: 11 }} />
              <PolarRadiusAxis tick={{ fill: '#64748b', fontSize: 10 }} domain={[0, 1]} />
              <Radar name="Log. Reg." dataKey="Log. Reg." stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.15} strokeWidth={2} />
              <Radar name="Random Forest" dataKey="Random Forest" stroke="#22c55e" fill="#22c55e" fillOpacity={0.1} strokeWidth={2} />
              <Radar name="XGBoost" dataKey="XGBoost" stroke="#f97316" fill="#f97316" fillOpacity={0.1} strokeWidth={2} />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              <Tooltip contentStyle={customTooltipStyle} />
            </RadarChart>
          </ResponsiveContainer>
        </div>

        {/* Risk Distribution Pie */}
        <div className="glass-card chart-card animate-in">
          <h3>🛡️ Risk Distribution</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={riskDistribution}
                cx="50%" cy="50%"
                innerRadius={70}
                outerRadius={110}
                paddingAngle={4}
                dataKey="value"
                label={({ name, value }) => `${name} (${value}%)`}
                labelLine={{ stroke: '#64748b' }}
              >
                {riskDistribution.map((entry, index) => (
                  <Cell key={index} fill={entry.color || CHART_COLORS[index]} stroke="none" />
                ))}
              </Pie>
              <Tooltip contentStyle={customTooltipStyle} />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Route Delay Rates */}
        <div className="glass-card chart-card animate-in">
          <h3>🛣️ Delay Rate by Route Type</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={routeData} barCategoryGap="30%">
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.1)" />
              <XAxis dataKey="route" tick={{ fill: '#94a3b8', fontSize: 12 }} />
              <YAxis tick={{ fill: '#94a3b8', fontSize: 12 }} unit="%" />
              <Tooltip contentStyle={customTooltipStyle} />
              <Bar dataKey="delay_rate" radius={[6, 6, 0, 0]}>
                {routeData.map((_, index) => (
                  <Cell key={index} fill={CHART_COLORS[index]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

function getDefaultModelData() {
  return [
    { name: 'Log. Reg.', Accuracy: 0.627, Precision: 0.415, Recall: 0.591, F1: 0.487, 'ROC-AUC': 0.663 },
    { name: 'Random Forest', Accuracy: 0.678, Precision: 0.448, Recall: 0.308, F1: 0.365, 'ROC-AUC': 0.642 },
    { name: 'XGBoost', Accuracy: 0.659, Precision: 0.419, Recall: 0.348, F1: 0.380, 'ROC-AUC': 0.614 },
  ];
}

function getDefaultRadarData() {
  return [
    { metric: 'Accuracy', 'Log. Reg.': 0.627, 'Random Forest': 0.678, XGBoost: 0.659 },
    { metric: 'Precision', 'Log. Reg.': 0.415, 'Random Forest': 0.448, XGBoost: 0.419 },
    { metric: 'Recall', 'Log. Reg.': 0.591, 'Random Forest': 0.308, XGBoost: 0.348 },
    { metric: 'F1-Score', 'Log. Reg.': 0.487, 'Random Forest': 0.365, XGBoost: 0.380 },
    { metric: 'ROC-AUC', 'Log. Reg.': 0.663, 'Random Forest': 0.642, XGBoost: 0.614 },
  ];
}
