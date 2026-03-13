export default function RiskBadge({ level }) {
  const riskClass = level?.toLowerCase() || 'low';
  return (
    <span className={`risk-badge ${riskClass}`}>
      <span className="dot" />
      {level || 'LOW'}
    </span>
  );
}
