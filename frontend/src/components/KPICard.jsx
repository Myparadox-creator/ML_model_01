export default function KPICard({ icon, label, value, color, change }) {
  return (
    <div className={`glass-card kpi-card ${color} animate-in`}>
      <div className="kpi-header">
        <div className={`kpi-icon ${color}`}>
          {icon}
        </div>
        <span className="kpi-label">{label}</span>
      </div>
      <div className={`kpi-value ${color}`}>
        {typeof value === 'number' ? value.toLocaleString() : value}
      </div>
      {change && <div className="kpi-change">{change}</div>}
    </div>
  );
}
