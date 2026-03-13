import { useState } from 'react';
import Sidebar from './components/Sidebar';
import Dashboard from './pages/Dashboard';
import Predict from './pages/Predict';
import Shipments from './pages/Shipments';
import Analytics from './pages/Analytics';

function App() {
  const [activePage, setActivePage] = useState('dashboard');

  const renderPage = () => {
    switch (activePage) {
      case 'dashboard':
        return <Dashboard />;
      case 'predict':
        return <Predict />;
      case 'shipments':
        return <Shipments />;
      case 'analytics':
        return <Analytics />;
      default:
        return <Dashboard />;
    }
  };

  return (
    <div className="app-layout">
      <Sidebar activePage={activePage} onNavigate={setActivePage} />
      <main className="main-content">
        {renderPage()}
      </main>
    </div>
  );
}

export default App;
