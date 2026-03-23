import { useState, useEffect } from 'react';
import { Search, ArrowRight, ChevronLeft, ChevronRight } from 'lucide-react';
import RiskBadge from '../components/RiskBadge';

const API_BASE = 'http://localhost:8000';
const PAGE_SIZE = 20;

export default function Shipments() {
  const [shipments, setShipments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [isOffline, setIsOffline] = useState(false);
  const [search, setSearch] = useState('');
  const [filterStatus, setFilterStatus] = useState('all');
  const [page, setPage] = useState(1);

  useEffect(() => {
    async function fetchShipments() {
      try {
        const res = await fetch(`${API_BASE}/shipments?limit=200`);
        if (res.ok) {
          setShipments(await res.json());
          setIsOffline(false);
        } else {
          setIsOffline(true);
        }
      } catch (err) {
        console.error('Failed to fetch shipments:', err);
        setIsOffline(true);
      } finally {
        setLoading(false);
      }
    }
    fetchShipments();
    const interval = setInterval(fetchShipments, 5000);
    return () => clearInterval(interval);
  }, []);

  const getRisk = (prob) => {
    if (prob >= 0.7) return 'HIGH';
    if (prob >= 0.4) return 'MEDIUM';
    return 'LOW';
  };

  // Filter & search
  const filtered = shipments.filter((s) => {
    const matchesSearch =
      !search ||
      s.shipment_id?.toLowerCase().includes(search.toLowerCase()) ||
      s.origin?.toLowerCase().includes(search.toLowerCase()) ||
      s.destination?.toLowerCase().includes(search.toLowerCase()) ||
      s.carrier_id?.toLowerCase().includes(search.toLowerCase());

    const risk = getRisk(s.delay_probability);
    const matchesFilter =
      filterStatus === 'all' ||
      (filterStatus === 'high' && risk === 'HIGH') ||
      (filterStatus === 'medium' && risk === 'MEDIUM') ||
      (filterStatus === 'low' && risk === 'LOW') ||
      (filterStatus === 'delayed' && s.delayed === 1);

    return matchesSearch && matchesFilter;
  });

  const totalPages = Math.ceil(filtered.length / PAGE_SIZE);
  const paginated = filtered.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE);

  return (
    <div>
      <div className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h1>Shipments</h1>
          <p>Browse and search all shipment records ({filtered.length.toLocaleString()} results)</p>
        </div>
        {isOffline && (
          <div style={{ padding: '8px 16px', borderRadius: 20, background: 'rgba(239, 68, 68, 0.15)', color: '#ef4444', fontWeight: 600, display: 'flex', alignItems: 'center', gap: 8 }}>
            <div style={{ width: 8, height: 8, borderRadius: '50%', background: '#ef4444', boxShadow: '0 0 8px #ef4444' }} />
            API OFFLINE
          </div>
        )}
      </div>

      {/* Search & Filter */}
      <div className="search-bar">
        <div className="search-input-wrapper">
          <Search className="search-icon" />
          <input
            type="text"
            placeholder="Search by ID, city, or carrier..."
            value={search}
            onChange={(e) => { setSearch(e.target.value); setPage(1); }}
          />
        </div>
        <select
          className="filter-select"
          value={filterStatus}
          onChange={(e) => { setFilterStatus(e.target.value); setPage(1); }}
        >
          <option value="all">All Statuses</option>
          <option value="high">🔴 High Risk</option>
          <option value="medium">🟡 Medium Risk</option>
          <option value="low">🟢 Low Risk</option>
          <option value="delayed">❌ Delayed</option>
        </select>
      </div>

      {/* Table */}
      <div className="glass-card animate-in" style={{ padding: 24 }}>
        {loading ? (
          <div className="loading-overlay">
            <div className="spinner" />
            <span>Loading shipments...</span>
          </div>
        ) : (
          <>
            <div className="data-table-wrapper">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Shipment ID</th>
                    <th>Route</th>
                    <th>Distance</th>
                    <th>Route Type</th>
                    <th>Carrier</th>
                    <th>Reliability</th>
                    <th>Weather</th>
                    <th>Traffic</th>
                    <th>Delay Prob</th>
                    <th>Risk</th>
                  </tr>
                </thead>
                <tbody>
                  {paginated.map((s, i) => (
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
                      <td>
                        <span style={{
                          padding: '2px 10px',
                          borderRadius: 100,
                          fontSize: '0.75rem',
                          background: 'rgba(99, 102, 241, 0.1)',
                          color: 'var(--accent-indigo)',
                        }}>
                          {s.route_type}
                        </span>
                      </td>
                      <td>{s.carrier_id}</td>
                      <td>{(s.carrier_reliability_score * 100).toFixed(0)}%</td>
                      <td>
                        <span style={{ color: s.weather_severity >= 7 ? 'var(--accent-red)' : 'var(--text-secondary)' }}>
                          {s.weather_severity?.toFixed(1)}
                        </span>
                      </td>
                      <td>
                        <span style={{ color: s.traffic_congestion >= 7 ? 'var(--accent-red)' : 'var(--text-secondary)' }}>
                          {s.traffic_congestion?.toFixed(1)}
                        </span>
                      </td>
                      <td>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                          <div className="risk-bar">
                            <div
                              className={`risk-bar-fill ${getRisk(s.delay_probability).toLowerCase()}`}
                              style={{ width: `${s.delay_probability * 100}%` }}
                            />
                          </div>
                          <span style={{ fontSize: '0.8rem', fontWeight: 600, minWidth: 38 }}>
                            {(s.delay_probability * 100).toFixed(0)}%
                          </span>
                        </div>
                      </td>
                      <td>
                        <RiskBadge level={getRisk(s.delay_probability)} />
                      </td>
                    </tr>
                  ))}
                  {!isOffline && paginated.length === 0 && !loading && (
                    <tr><td colSpan="10" style={{ textAlign: 'center', padding: 24, color: 'var(--text-muted)' }}>No shipments found.</td></tr>
                  )}
                  {isOffline && (
                    <tr><td colSpan="10" style={{ textAlign: 'center', padding: 24, color: '#ef4444' }}>Backend API is offline. Cannot load shipments.</td></tr>
                  )}
                </tbody>
              </table>
            </div>

            {/* Pagination */}
            {totalPages > 1 && (
              <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', gap: 16, marginTop: 24 }}>
                <button
                  className="btn btn-outline"
                  onClick={() => setPage((p) => Math.max(1, p - 1))}
                  disabled={page === 1}
                  style={{ padding: '8px 14px' }}
                >
                  <ChevronLeft size={16} />
                </button>
                <span style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>
                  Page {page} of {totalPages}
                </span>
                <button
                  className="btn btn-outline"
                  onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                  disabled={page === totalPages}
                  style={{ padding: '8px 14px' }}
                >
                  <ChevronRight size={16} />
                </button>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
