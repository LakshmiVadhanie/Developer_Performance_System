import { useState, useRef, useEffect, type ReactElement } from 'react';
import './App.css';

import Login    from './components/Login';
import Home     from './components/Home';
import Dashboard from './components/Dashboard';
import Chatbot  from './components/Chatbot';

type Page = 'login' | 'home' | 'dashboard';

// ── Nav items for dashboard sidebar ─────────────────────────
const navItems = [
  { icon: 'dashboard',   label: 'Dashboard'  },
  { icon: 'monitoring',  label: 'Monitoring Metrics' },
];

const notifications = [
  { icon: 'commit',      color: 'var(--accent)',    title: 'k.morris pushed feat/lstm-fix',      time: '12m ago' },
  { icon: 'merge',       color: 'var(--secondary)', title: 'PR #842 merged into main',           time: '48m ago' },
  { icon: 'bug_report',  color: 'var(--error)',     title: 'Build failed on staging-v4',         time: '2h ago'  },
  { icon: 'trending_up', color: '#2dd4bf',          title: 'LSTM R² improved to 0.58',           time: '3h ago'  },
  { icon: 'group',       color: 'var(--primary)',   title: '87 active developers this sprint',   time: '1d ago'  },
];

// PlaceholderPage removed unused

const pages: Record<string, ReactElement> = {
  Dashboard: <></>,
  'Monitoring Metrics': (
    <div style={{ display: 'flex', flex: 1, justifyContent: 'center', alignItems: 'center', width: '100%', height: '100%' }}>
      <div style={{ width: '100%', maxWidth: '1600px', height: '100%', minHeight: '700px', borderRadius: '16px', overflow: 'hidden', border: '1px solid rgba(139, 92, 246, 0.3)', boxShadow: '0 20px 40px rgba(0,0,0,0.6)', background: '#0b0a14' }}>
        <iframe src="http://localhost:3000/d/devinsight_ml_home?kiosk" width="100%" height="100%" frameBorder="0" style={{ display: 'block', width: '100%', height: '100%' }} title="Grafana Metrics" />
      </div>
    </div>
  ),
};

const searchIndex = [
  { label: 'k.morris → feat/lstm-fix',              type: 'Commit',  id:'git' },
  { label: 's.chen merged PR #842',                  type: 'Merge',   id:'git' },
  { label: 'r.patel patched fix/auth-leak',          type: 'Commit',  id:'git' },
  { label: 'Build failed on staging-v4',             type: 'Alert',   id:'deploy' },
  { label: 'Predicted commits: 184 (+12.5%)',        type: 'Metric',  id:'stats' },
  { label: 'Active PRs: 42 (-5.2%)',                 type: 'Metric',  id:'stats' },
  { label: 'LSTM R² Score: 0.58',                    type: 'Model',   id:'stats' },
  { label: 'Active Developers: 87 (+2.4%)',          type: 'Metric',  id:'stats' },
  { label: 'Reviews Given: 95 (+18.1%)',             type: 'Metric',  id:'stats' },
  { label: 'Build Frequency Heatmap — peak Thursday', type: 'Widget', id:'charts' },
  { label: 'Flow State Efficiency: 72% Focus',       type: 'Widget',  id:'flow' },
  { label: 'PR Cycle Distribution: avg 18.4 hrs',    type: 'Widget',  id:'flow' },
  { label: 'Coding Velocity: +24% week-over-week',   type: 'Widget',  id:'charts' },
  { label: 'Deployment: Production v2.4.1-rc ✅',   type: 'Deploy',  id:'deploy' },
  { label: 'Deployment: Staging fix/auth-leak ✅',  type: 'Deploy',  id:'deploy' },
  { label: 'Git Activity: 10 events today',          type: 'Widget',  id:'git' },
  { label: 'LSTM Forecast — 184 commits tomorrow',   type: 'Model',   id:'stats' },
];

// ID → section selector mapping for scroll
const sectionMap: Record<string, string> = {
  stats:  '.stats-row',
  charts: '.charts-row',
  flow:   '.flow-row',
  git:    '.activity-card',
  deploy: '.deploy-card',
};

function App() {
  const [page,       setPage]       = useState<Page>('login');
  const [username,   setUsername]   = useState('');
  const [activePage, setActivePage] = useState('Dashboard');
  const [search,     setSearch]     = useState('');
  const [showNotifs, setShowNotifs] = useState(false);
  const notifRef = useRef<HTMLDivElement>(null);

  // Close notif panel on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (notifRef.current && !notifRef.current.contains(e.target as Node)) {
        setShowNotifs(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  const handleLogin = (user: string) => {
    setUsername(user);
    setPage('home');
  };

  const handleStart = () => {
    setPage('dashboard');
    setActivePage('Dashboard');
  };

  const handleSearchSelect = (id: string) => {
    setSearch('');
    setActivePage('Dashboard');
    // Small delay to let sidebar switch render, then scroll
    setTimeout(() => {
      const el = document.querySelector(sectionMap[id] || '.dash');
      if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
  };

  // ── Pages: Login ───────────────────────────────────────────
  if (page === 'login') {
    return (
      <>
        <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700;800&family=Inter:wght@400;500;600&display=swap" rel="stylesheet" />
        <link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap" rel="stylesheet" />
        <Login onLogin={(u) => handleLogin(u || 'User')} />
      </>
    );
  }

  // ── Pages: Home ────────────────────────────────────────────
  if (page === 'home') {
    return (
      <>
        <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700;800&family=Inter:wght@400;500;600&display=swap" rel="stylesheet" />
        <link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap" rel="stylesheet" />
        <Home onStart={handleStart} username={username} />
      </>
    );
  }

  // ── Pages: Dashboard ───────────────────────────────────────
  const isDashboard = activePage === 'Dashboard';

  return (
    <div className="app-root">
      <div className="bg-orbs">
        <div className="orb orb-1" /><div className="orb orb-2" /><div className="orb orb-3" />
      </div>
      <link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap" rel="stylesheet" />

      {/* ── Sidebar ── */}
      <aside className="sidebar">
        <div className="sidebar-logo" style={{ cursor: 'pointer' }} onClick={() => setPage('home')}>
          <div className="logo-icon">⌘</div>
          <div className="logo-text">
            <div className="logo-title">DevInsight</div>
            <div className="logo-sub">AI Workspace</div>
          </div>
        </div>
        <nav className="sidebar-nav">
          {navItems.map(item => (
            <div key={item.label} className={`nav-item ${activePage === item.label ? 'active' : ''}`}
              onClick={() => setActivePage(item.label)} title={item.label}>
              <span className="material-symbols-outlined nav-icon">{item.icon}</span>
              <span className="nav-label">{item.label}</span>
            </div>
          ))}
        </nav>
        <div className="sidebar-footer">
          <div className="avatar">{username.slice(0, 2).toUpperCase() || 'LV'}</div>
          <div className="user-info">
            <div className="user-name">{username || 'Lakshmi'}</div>
            <div className="user-role">Senior Architect</div>
          </div>
        </div>
      </aside>

      {/* ── Top bar ── */}
      <header className="topbar">
        <div className="topbar-search">
          <span className="material-symbols-outlined search-icon">search</span>
          <input
            type="text"
            placeholder="Search commits, developers, metrics..."
            value={search}
            onChange={e => setSearch(e.target.value)}
            onKeyDown={e => { if (e.key === 'Escape') setSearch(''); }}
          />
          {search && (
            <button onClick={() => setSearch('')}
              style={{ position:'absolute', right:'0.6rem', background:'none', border:'none', cursor:'pointer', color:'var(--text-muted)', fontSize:'1rem' }}>✕</button>
          )}
        </div>

        {/* Search results */}
        {search && (() => {
          const filtered = searchIndex.filter(r => r.label.toLowerCase().includes(search.toLowerCase()));
          return (
            <div className="search-results">
              {filtered.length === 0 ? (
                <div className="search-result-item">
                  <span className="sr-label" style={{ color: 'var(--text-muted)' }}>No results for "{search}"</span>
                </div>
              ) : filtered.map((r, i) => (
                <div key={i} className="search-result-item" onClick={() => handleSearchSelect(r.id)}>
                  <span className="sr-label">{r.label}</span>
                  <span className="sr-type">{r.type}</span>
                </div>
              ))}
            </div>
          );
        })()}

        <div className="topbar-right">
          {/* Notifications */}
          <div style={{ position: 'relative' }} ref={notifRef}>
            <button className="topbar-btn" title="Notifications" onClick={() => setShowNotifs(v => !v)}>
              <span className="material-symbols-outlined" style={{ fontSize: '1.25rem' }}>notifications</span>
              <span className="notif-dot" />
            </button>
            {showNotifs && (
              <div className="notif-panel">
                <div className="notif-header">
                  <span className="notif-title">Notifications</span>
                  <span className="notif-badge">5 new</span>
                </div>
                {notifications.map((n, i) => (
                  <div key={i} className="notif-item">
                    <span className="material-symbols-outlined notif-icon" style={{ color: n.color }}>{n.icon}</span>
                    <div className="notif-body">
                      <p className="notif-text">{n.title}</p>
                      <span className="notif-time">{n.time}</span>
                    </div>
                  </div>
                ))}
                <div className="notif-footer" onClick={() => setShowNotifs(false)}>Mark all as read</div>
              </div>
            )}
          </div>

          <button className="topbar-btn" title="Terminal">
            <span className="material-symbols-outlined" style={{ fontSize: '1.25rem' }}>terminal</span>
          </button>
          <div className="topbar-divider" />
          <span className="workspace-badge">Workspace: Alpha-7</span>
        </div>
      </header>

      {/* ── Main canvas ── */}
      <div className="main-canvas" style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
        <main style={{ flex: 1, overflowY: 'hidden', position: 'relative', display: 'flex', flexDirection: 'column' }}>
          {isDashboard ? (
            <div style={{ display: 'flex', gap: '2rem', height: '100%', padding: '2rem' }}>
              <div className="dashboard-section" style={{ flex: 1.5, overflowY: 'auto', paddingRight: '1rem' }}>
                <Dashboard />
              </div>
              <div className="chatbot-section" style={{ flex: 1, height: '100%', minWidth: '400px', borderRadius: '16px', overflow: 'hidden', border: '1px solid rgba(139, 92, 246, 0.2)', boxShadow: '0 12px 40px rgba(0,0,0,0.6)' }}>
                <Chatbot />
              </div>
            </div>
          ) : (
            <div style={{ flex: 1, overflowY: 'auto', padding: '2rem' }}>
              {pages[activePage]}
            </div>
          )}
        </main>
      </div>
    </div>
  );
}

export default App;
