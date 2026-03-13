import { LayoutDashboard, Crosshair, Truck, BarChart3, Activity } from 'lucide-react';

export default function Sidebar({ activePage, onNavigate }) {
  const navItems = [
    { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { id: 'predict', label: 'Predict Delay', icon: Crosshair },
    { id: 'shipments', label: 'Shipments', icon: Truck },
    { id: 'analytics', label: 'Analytics', icon: BarChart3 },
  ];

  return (
    <aside className="sidebar">
      <div className="sidebar-logo">
        <div className="logo-icon">
          <div className="logo-symbol">🚛</div>
          <div>
            <h2>LogiPredict</h2>
            <span>AI Early Warning System</span>
          </div>
        </div>
      </div>

      <nav className="sidebar-nav">
        {navItems.map((item) => {
          const Icon = item.icon;
          return (
            <button
              key={item.id}
              className={`nav-link ${activePage === item.id ? 'active' : ''}`}
              onClick={() => onNavigate(item.id)}
            >
              <Icon className="nav-icon" />
              <span>{item.label}</span>
            </button>
          );
        })}
      </nav>

      <div className="sidebar-footer">
        <div className="status">
          <div className="status-dot" />
          <span>API Connected</span>
        </div>
      </div>
    </aside>
  );
}
