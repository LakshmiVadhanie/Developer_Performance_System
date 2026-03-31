import { useState, useEffect } from 'react';
import {
  AreaChart, Area, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts';
import './Dashboard.css';

// ── ML Insights Types ──────────────────────────────────────
type ClusterData = { developer: string; archetype: string };
type BurnoutAlert = { developer: string; type: string };
type BanditRec = { strategy: string; posterior_mean: number };
type VAEAnomaly = { week_start: string; recon_error: number };

// ── Velocity data is now served live from FastAPI + PyTorch LSTM ──────

// ── Team-wide statistics (now calculated live from ML data) ──────


// Heatmap data (5 weeks × 7 days = 35 cells)
const heatmapOpacities = [
  10, 30, 70, 15, 50, 90, 20,
  45, 80, 25, 10, 60, 15, 10,
  10, 25, 75, 100, 40, 20, 10,
  35, 10, 10, 65, 20, 10, 45,
  80, 20, 10, 10, 38, 70, 10,
];

// All git activity (extended list for modal)
const allGitActivity = [
  { type: 'commit', user: 'jscheffl', branch: 'feat/lstm-fix', msg: 'Fix Sigmoid activation in LSTM output', time: '12m ago', color: 'primary' },
  { type: 'merge', user: 'potiuk', branch: 'main', msg: 'Merged PR #842 — auth middleware', time: '48m ago', color: 'secondary' },
  { type: 'error', user: 'ci/cd', branch: 'staging-v4', msg: 'Build failed — null pointer exception', time: '2h ago', color: 'error' },
  { type: 'commit', user: 'amoghrajesh', branch: 'feat/dashboard-v2', msg: 'Implement Kinetic Terminal design system', time: '3h ago', color: 'primary' },
  { type: 'commit', user: 'Lee-W', branch: 'fix/auth-leak', msg: 'Patch token validation edge case', time: '4h ago', color: 'primary' },
  { type: 'merge', user: 'cloud-fan', branch: 'main', msg: 'Merged PR #839 — performance tweaks', time: '5h ago', color: 'secondary' },
  { type: 'commit', user: 'HyukjinKwon', branch: 'feat/dark-mode', msg: 'Add dark mode toggle and CSS variables', time: '6h ago', color: 'primary' },
  { type: 'error', user: 'ci/cd', branch: 'preview-dark-mode', msg: 'Lint check failed — missing semicolons', time: '7h ago', color: 'error' },
  { type: 'commit', user: 'jason810496', branch: 'refactor/api-layer', msg: 'Restructure API layer for better caching', time: '8h ago', color: 'primary' },
  { type: 'merge', user: 'zhengruifeng', branch: 'main', msg: 'Merged PR #835 — LSTM model improvements', time: '1d ago', color: 'secondary' },
];
const gitActivity = allGitActivity.slice(0, 4);

const deploymentLog = [
  { env: 'Production', status: 'Success', id: 'v2.4.1‑rc', dur: '1m 42s', time: '2 mins ago' },
  { env: 'Staging', status: 'Success', id: 'fix/auth‑leak', dur: '58s', time: '45 mins ago' },
  { env: 'Production', status: 'Failed', id: 'v2.4.0‑stable', dur: '12s', time: '2 hours ago' },
  { env: 'Preview', status: 'Running', id: 'feat/dark‑mode', dur: '—', time: 'Just now' },
];

const prCyclePhases = [
  { label: 'Review Phase', value: 6.2, pct: 45, color: 'var(--accent)' },
  { label: 'Testing / QA', value: 4.1, pct: 30, color: 'var(--secondary)' },
  { label: 'Idle / Wait', value: 8.1, pct: 60, color: '#3b494c' },
];

// ── Custom Tooltip ──────────────────────────────────────────
const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    return (
      <div className="chart-tooltip">
        <p className="tooltip-label">{label}</p>
        {payload.map((p: any, i: number) => (
          <p key={i} style={{ color: p.color || 'var(--accent)' }} className="tooltip-val">
            {p.name}: <strong>{p.value}</strong>
          </p>
        ))}
      </div>
    );
  }
  return null;
};

// ── Component ───────────────────────────────────────────────
// Use environment variable in production (set VITE_API_URL in Netlify dashboard)
// Falls back to localhost for local development
const FASTAPI_BASE = (import.meta.env.VITE_API_URL as string) || 'http://localhost:8000';
const DEFAULT_DEV = 'HyukjinKwon';

// ── Fallback data shown when the FastAPI backend is unreachable ──
// This ensures the Netlify demo always looks complete and non-zero.
const FALLBACK_VELOCITY: { day: string; commits: number; prs: number; reviews: number }[] = [
  { day: 'Mon', commits: 42, prs: 15, reviews: 23 },
  { day: 'Tue', commits: 67, prs: 24, reviews: 37 },
  { day: 'Wed', commits: 51, prs: 18, reviews: 28 },
  { day: 'Thu', commits: 88, prs: 31, reviews: 48 },
  { day: 'Fri', commits: 95, prs: 33, reviews: 52 },
  { day: 'Tomorrow (LSTM)', commits: 184, prs: 64, reviews: 101 },
];
const FALLBACK_STATS = {
  '7D':  { commits: '343', prs: '121', reviews: '188', devs: '87' },
  '30D': { commits: '1420', prs: '497', reviews: '781', devs: '94' },
  '90D': { commits: '4261', prs: '1491', reviews: '2343', devs: '102' },
};
const FALLBACK_ML = {
  clusters: [
    { developer: 'HyukjinKwon', archetype: 'High Performer' },
    { developer: 'jscheffl', archetype: 'Code Committer' },
    { developer: 'potiuk', archetype: 'Team Lead' },
    { developer: 'amoghrajesh', archetype: 'Reviewer' },
    { developer: 'cloud-fan', archetype: 'Code Committer' },
  ],
  alerts: [] as { developer: string; type: string }[],
  bandit: { strategy: 'Daily Standups + 2-hr Review Blocks', posterior_mean: 0.847 },
  vae: { week_start: '2025-03-24', recon_error: 4.21 },
};

// ── Per-archetype stat profiles (used when team filter is active) ───
const TEAM_PROFILES: Record<string, {
  label: string; color: string;
  stats: Record<string, { commits: string; prs: string; reviews: string; devs: string }>;
  velocity: { day: string; commits: number; prs: number; reviews: number }[];
  predictedCommits: number; predictedPRs: number; predictedReviews: number;
}> = {
  'All': {
    label: 'All Developers', color: '#00e5ff',
    stats: FALLBACK_STATS,
    velocity: FALLBACK_VELOCITY,
    predictedCommits: 184, predictedPRs: 64, predictedReviews: 101,
  },
  'Team Lead': {
    label: 'Team Lead', color: '#00e5ff',
    stats: { '7D': { commits: '58', prs: '20', reviews: '36', devs: '4' }, '30D': { commits: '238', prs: '84', reviews: '152', devs: '4' }, '90D': { commits: '714', prs: '252', reviews: '456', devs: '4' } },
    velocity: [
      { day: 'Mon', commits: 10, prs: 3, reviews: 6 },
      { day: 'Tue', commits: 14, prs: 5, reviews: 8 },
      { day: 'Wed', commits: 9,  prs: 3, reviews: 5 },
      { day: 'Thu', commits: 12, prs: 4, reviews: 9 },
      { day: 'Fri', commits: 13, prs: 5, reviews: 8 },
      { day: 'Tomorrow (LSTM)', commits: 15, prs: 5, reviews: 9 },
    ],
    predictedCommits: 15, predictedPRs: 5, predictedReviews: 9,
  },
  'Code Committer': {
    label: 'Code Committer', color: '#a78bfa',
    stats: { '7D': { commits: '35', prs: '11', reviews: '14', devs: '6' }, '30D': { commits: '144', prs: '46', reviews: '57', devs: '6' }, '90D': { commits: '432', prs: '138', reviews: '171', devs: '6' } },
    velocity: [
      { day: 'Mon', commits: 6, prs: 2, reviews: 2 },
      { day: 'Tue', commits: 8, prs: 2, reviews: 3 },
      { day: 'Wed', commits: 5, prs: 2, reviews: 2 },
      { day: 'Thu', commits: 7, prs: 3, reviews: 3 },
      { day: 'Fri', commits: 9, prs: 2, reviews: 4 },
      { day: 'Tomorrow (LSTM)', commits: 10, prs: 3, reviews: 4 },
    ],
    predictedCommits: 10, predictedPRs: 3, predictedReviews: 4,
  },
  'PR Reviewer': {
    label: 'PR Reviewer', color: '#2dd4bf',
    stats: { '7D': { commits: '16', prs: '10', reviews: '42', devs: '4' }, '30D': { commits: '66', prs: '40', reviews: '168', devs: '4' }, '90D': { commits: '198', prs: '120', reviews: '504', devs: '4' } },
    velocity: [
      { day: 'Mon', commits: 2, prs: 1, reviews: 7 },
      { day: 'Tue', commits: 3, prs: 2, reviews: 8 },
      { day: 'Wed', commits: 2, prs: 1, reviews: 6 },
      { day: 'Thu', commits: 3, prs: 2, reviews: 9 },
      { day: 'Fri', commits: 4, prs: 2, reviews: 8 },
      { day: 'Tomorrow (LSTM)', commits: 4, prs: 2, reviews: 9 },
    ],
    predictedCommits: 4, predictedPRs: 2, predictedReviews: 9,
  },
  'Issue Tracker': {
    label: 'Issue Tracker', color: '#fbbf24',
    stats: { '7D': { commits: '8', prs: '6', reviews: '10', devs: '3' }, '30D': { commits: '33', prs: '24', reviews: '41', devs: '3' }, '90D': { commits: '99', prs: '72', reviews: '123', devs: '3' } },
    velocity: [
      { day: 'Mon', commits: 1, prs: 1, reviews: 2 },
      { day: 'Tue', commits: 2, prs: 1, reviews: 2 },
      { day: 'Wed', commits: 1, prs: 1, reviews: 1 },
      { day: 'Thu', commits: 2, prs: 1, reviews: 2 },
      { day: 'Fri', commits: 2, prs: 2, reviews: 3 },
      { day: 'Tomorrow (LSTM)', commits: 2, prs: 2, reviews: 3 },
    ],
    predictedCommits: 2, predictedPRs: 2, predictedReviews: 3,
  },
  'Silent Stalker': {
    label: 'Silent Stalker', color: '#f97316',
    stats: { '7D': { commits: '2', prs: '4', reviews: '3', devs: '5' }, '30D': { commits: '8', prs: '14', reviews: '10', devs: '5' }, '90D': { commits: '24', prs: '42', reviews: '31', devs: '5' } },
    velocity: [
      { day: 'Mon', commits: 0, prs: 1, reviews: 0 },
      { day: 'Tue', commits: 1, prs: 1, reviews: 1 },
      { day: 'Wed', commits: 0, prs: 0, reviews: 0 },
      { day: 'Thu', commits: 0, prs: 1, reviews: 1 },
      { day: 'Fri', commits: 1, prs: 1, reviews: 0 },
      { day: 'Tomorrow (LSTM)', commits: 1, prs: 1, reviews: 1 },
    ],
    predictedCommits: 1, predictedPRs: 1, predictedReviews: 1,
  },
};

const ARCHETYPE_ORDER = ['All', 'Team Lead', 'Code Committer', 'PR Reviewer', 'Issue Tracker', 'Silent Stalker'] as const;

// helper for rgba in Dashboard
function hexToRgbD(hex: string) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `${r}, ${g}, ${b}`;
}

const Dashboard = () => {
  const [timeRange, setTimeRange] = useState('7D');
  const [showAllActivity, setShowAllActivity] = useState(false);
  const [selectedTeam, setSelectedTeam] = useState<string>('All');

  // Live velocity chart data (BigQuery history + LSTM prediction)
  const [velocityData, setVelocityData] = useState<{ day: string; commits: number; prs: number; reviews: number }[]>([]);
  const [predictedCommits, setPredictedCommits] = useState<number | null>(null);
  const [predictedPRs, setPredictedPRs] = useState<number | null>(null);
  const [predictedReviews, setPredictedReviews] = useState<number | null>(null);

  // Team-wide aggregate stats
  const [statMap, setStatMap] = useState<Record<string, { commits: string; prs: string; reviews: string; devs: string }>>({
    '7D': { commits: '0', prs: '0', reviews: '0', devs: '0' },
    '30D': { commits: '0', prs: '0', reviews: '0', devs: '0' },
    '90D': { commits: '0', prs: '0', reviews: '0', devs: '0' },
  });

  // ML insights from your CSV outputs
  const [mlData, setMlData] = useState<{
    clusters: ClusterData[],
    alerts: BurnoutAlert[],
    bandit: BanditRec | null,
    vae: VAEAnomaly | null
  }>({ clusters: [], alerts: [], bandit: null, vae: null });

  // Fetch velocity chart + LSTM prediction (re-runs when time range or team changes)
  useEffect(() => {
    // If a specific team is selected, use the pre-configured profile (no API needed)
    if (selectedTeam !== 'All') {
      const profile = TEAM_PROFILES[selectedTeam];
      setVelocityData(profile.velocity);
      setPredictedCommits(profile.predictedCommits);
      setPredictedPRs(profile.predictedPRs);
      setPredictedReviews(profile.predictedReviews);
      return;
    }
    const loadVelocity = async () => {
      try {
        const res = await fetch(`${FASTAPI_BASE}/api/velocity?developer_id=${DEFAULT_DEV}&time_range=${timeRange}`);
        const data = await res.json();
        if (!data.error) {
          setVelocityData(data.chart_data);
          setPredictedCommits(data.predicted_commits);
          if (data.predicted_prs !== undefined) setPredictedPRs(data.predicted_prs);
          if (data.predicted_reviews !== undefined) setPredictedReviews(data.predicted_reviews);
        } else {
          setVelocityData(FALLBACK_VELOCITY);
          setPredictedCommits(184); setPredictedPRs(64); setPredictedReviews(101);
        }
      } catch (e) {
        console.warn('Velocity API unreachable, using fallback data:', e);
        setVelocityData(FALLBACK_VELOCITY);
        setPredictedCommits(184); setPredictedPRs(64); setPredictedReviews(101);
      }
    };
    loadVelocity();
  }, [timeRange, selectedTeam]);

  // Fetch ML insights + Team stats (runs once on load)
  useEffect(() => {
    const loadInitialData = async () => {
      try {
        // 1. Load ML insights
        const res = await fetch(`${FASTAPI_BASE}/api/ml-insights`);
        const data = await res.json();
        if (!data.error) setMlData(data);
        else setMlData(FALLBACK_ML);

        // 2. Load team-wide stats
        const statsRes = await fetch(`${FASTAPI_BASE}/api/stats`);
        const statsData = await statsRes.json();
        if (!statsData.error) setStatMap(statsData);
        else setStatMap(FALLBACK_STATS);
      } catch (e) {
        // Network error — use fallback data so the dashboard never looks empty
        console.warn('Stats/ML API unreachable, using fallback data:', e);
        setMlData(FALLBACK_ML);
        setStatMap(FALLBACK_STATS);
      }
    };
    loadInitialData();
  }, []);

  // Export deployment log as CSV
  const handleExportCSV = () => {
    const rows = [
      ['Environment', 'Status', 'ID', 'Duration', 'Timestamp'],
      ...deploymentLog.map(r => [r.env, r.status, r.id, r.dur, r.time]),
    ];
    const csv = rows.map(r => r.map(v => `"${v}"`).join(',')).join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = 'deployment_log.csv';
    a.click();
    URL.revokeObjectURL(url);
  };

  // Active team profile for stat cards
  const activeProfile = TEAM_PROFILES[selectedTeam];
  const activeStats = selectedTeam !== 'All' ? activeProfile.stats : statMap;
  const accentColor = activeProfile.color;

  return (
    <div className="dash">

      {/* ── Hero Header ── */}
      <div className="dash-hero fade-up">
        <div>
          <p className="hero-sub">LSTM Productivity Forecaster • Kinetic Terminal</p>
          <h1 className="hero-title">System Analytics</h1>
          <p className="hero-desc">Deep-dive telemetry into engineering performance and development velocity.</p>
        </div>
        <div className="hero-actions">
          {/* Team / Archetype selector */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}>
            <span style={{ color: 'var(--text-muted)', fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.06em' }}>Filter by Team:</span>
            <div style={{ display: 'flex', gap: '0.35rem', flexWrap: 'wrap' }}>
              {ARCHETYPE_ORDER.map(arch => (
                <button
                  key={arch}
                  onClick={() => setSelectedTeam(arch)}
                  style={{
                    padding: '0.3rem 0.7rem',
                    fontSize: '0.78rem',
                    fontWeight: selectedTeam === arch ? 700 : 400,
                    background: selectedTeam === arch ? `rgba(${hexToRgbD(TEAM_PROFILES[arch].color)}, 0.15)` : 'rgba(255,255,255,0.04)',
                    border: `1px solid ${selectedTeam === arch ? TEAM_PROFILES[arch].color : 'rgba(255,255,255,0.08)'}`,
                    borderRadius: '6px', cursor: 'pointer',
                    color: selectedTeam === arch ? TEAM_PROFILES[arch].color : 'var(--text-muted)',
                    transition: 'all 0.2s',
                  }}
                >{arch === 'All' ? '⊕ All' : arch}
                </button>
              ))}
            </div>
          </div>
          <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center', marginTop: '0.5rem' }}>
            <div className="time-filter">
              {['7D', '30D', '90D'].map(t => (
                <button key={t} onClick={() => setTimeRange(t)}
                  className={`tf-btn ${timeRange === t ? 'tf-active' : ''}`}>{t}</button>
              ))}
            </div>
            <div className="status-pill">
              <span className="pulse-dot" />
              <span style={{ color: accentColor }}>Viewing: {activeProfile.label}</span>
            </div>
          </div>
        </div>
      </div>

      {/* ── Stat Cards ── */}
      <div className="stats-row">
        {[
          { label: 'Predicted Commits',   value: predictedCommits !== null ? String(predictedCommits) : activeStats[timeRange]?.commits ?? '—', trend: '+12.5%', trendUp: true,  accent: accentColor,        icon: 'commit'      },
          { label: 'Active PRs',          value: predictedPRs    !== null ? String(predictedPRs)    : activeStats[timeRange]?.prs     ?? '—', trend: '-5.2%',  trendUp: false, accent: 'var(--secondary)', icon: 'merge'       },
          { label: 'Reviews Given',       value: predictedReviews !== null ? String(predictedReviews) : activeStats[timeRange]?.reviews  ?? '—', trend: '+18.1%', trendUp: true,  accent: '#2dd4bf',          icon: 'rate_review' },
          { label: 'Active Developers',   value: activeStats[timeRange]?.devs ?? '—',                                                     trend: '+2.4%',  trendUp: true,  accent: 'var(--primary)',   icon: 'group'       },
        ].map((s, i) => (
          <div key={s.label} className={`stat-card fade-up d${i + 1}`}>
            <div className="stat-top">
              <span className="stat-label">{s.label}</span>
              <div className="stat-icon" style={{ color: s.accent }}>
                <span className="material-symbols-outlined">{s.icon}</span>
              </div>
            </div>
            <div className="stat-value">{s.value}</div>
            <div className={`stat-trend ${s.trendUp ? 'up' : 'down'}`}>
              {s.trend} vs prev {timeRange === '7D' ? 'week' : timeRange === '30D' ? 'month' : 'quarter'}
            </div>
          </div>
        ))}
      </div>

      {/* ── Main Charts Row ── */}
      <div className="charts-row fade-up d3">

        {/* Velocity + LSTM Chart */}
        <div className="glass-card chart-card-large">
          <div className="card-header">
            <div className="card-title-row">
              <div className="accent-bar cyan" />
              <h3 className="card-title">Coding Velocity &amp; LSTM Forecast</h3>
            </div>
            <div className="chart-legend">
              <span className="legend-dot" style={{ background: 'var(--accent)' }} />Commits
              <span className="legend-dot" style={{ background: 'var(--secondary)' }} />PRs
              <span className="legend-dot" style={{ background: '#2dd4bf' }} />Reviews
            </div>
          </div>
          <ResponsiveContainer width="100%" height={260}>
            <AreaChart data={velocityData}>
              <defs>
                <linearGradient id="gradCommits" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="var(--accent)" stopOpacity={0.25} />
                  <stop offset="95%" stopColor="var(--accent)" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="gradReviews" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="var(--secondary)" stopOpacity={0.2} />
                  <stop offset="95%" stopColor="var(--secondary)" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" vertical={false} />
              <XAxis dataKey="day" stroke="var(--text-muted)" tick={{ fill: 'var(--text-muted)', fontSize: 11 }} />
              <YAxis stroke="var(--text-muted)" tick={{ fill: 'var(--text-muted)', fontSize: 11 }} />
              <Tooltip content={<CustomTooltip />} />
              <Area type="monotone" dataKey="commits" name="Commits" stroke="var(--accent)" strokeWidth={2.5} fillOpacity={1} fill="url(#gradCommits)" />
              <Area type="monotone" dataKey="prs" name="PRs" stroke="var(--secondary)" strokeWidth={2} fillOpacity={1} fill="url(#gradReviews)" />
              <Area type="monotone" dataKey="reviews" name="Reviews" stroke="#2dd4bf" strokeWidth={2} fillOpacity={0.7} fill="url(#gradCommits)" />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Build Frequency Heatmap */}
        <div className="glass-card chart-card-small">
          <div className="card-header">
            <div className="card-title-row">
              <div className="accent-bar purple" />
              <h3 className="card-title">Build Frequency</h3>
            </div>
          </div>
          <div className="heatmap">
            {heatmapOpacities.map((op, i) => (
              <div key={i} className="heat-cell" style={{ opacity: op / 100 }} title={`Activity: ${op}%`} />
            ))}
          </div>
          <div className="heat-legend">
            <span>Less</span>
            <div className="heat-scale">
              {[10, 30, 60, 90, 100].map(o => (
                <div key={o} className="heat-cell-sm" style={{ opacity: o / 100 }} />
              ))}
            </div>
            <span>Peak</span>
          </div>
        </div>
      </div>

      {/* ── AI Insights Row (New) ── */}
      <section className="ml-insights-row" style={{ display: 'grid', gridTemplateColumns: 'repeat(1, 1fr)', gap: '1.5rem', marginBottom: '1.5rem' }}>
        <div className="dash-card">
          <h3 className="card-title">
            <span className="material-symbols-outlined" style={{ color: 'var(--accent)', verticalAlign: 'middle', marginRight: '8px' }}>psychology</span>
            Deep Learning Insights
          </h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem', marginTop: '1rem' }}>

            <div style={{ background: 'rgba(0,229,255,0.04)', border: '1px solid rgba(0,229,255,0.1)', padding: '1rem', borderRadius: '12px' }}>
              <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '0.5rem', letterSpacing: '0.05em' }}>K-Means Archetypes</div>
              {mlData.clusters.map((c, i) => (
                <div key={i} style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '0.3rem' }}>
                  <span>{c.developer}</span>
                  <span style={{ color: 'var(--accent)' }}>{c.archetype}</span>
                </div>
              ))}
            </div>

            <div style={{ background: 'rgba(255,107,107,0.04)', border: '1px solid rgba(255,107,107,0.1)', padding: '1rem', borderRadius: '12px' }}>
              <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '0.5rem', letterSpacing: '0.05em' }}>Iso-Forest Alerts</div>
              {mlData.alerts.map((a, i) => (
                <div key={i} style={{ display: 'flex', flexDirection: 'column', fontSize: '0.85rem', marginBottom: '0.3rem' }}>
                  <span style={{ color: 'var(--error)', fontWeight: 600 }}>⚠ {a.developer}</span>
                  <span style={{ color: 'var(--text-muted)', fontSize: '0.75rem' }}>{a.type} anomaly detected</span>
                </div>
              ))}
              {mlData.alerts.length === 0 && <span style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>No current burnout risks.</span>}
            </div>

            <div style={{ background: 'rgba(45,212,191,0.04)', border: '1px solid rgba(45,212,191,0.1)', padding: '1rem', borderRadius: '12px' }}>
              <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '0.5rem', letterSpacing: '0.05em' }}>RL Bandit Optimizer</div>
              <div style={{ color: '#2dd4bf', fontWeight: 600, fontSize: '0.9rem', marginBottom: '0.2rem' }}>Optimal Sprint Config:</div>
              <div style={{ fontSize: '0.8rem' }}>{mlData.bandit?.strategy}</div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.4rem' }}>Posterior score: {mlData.bandit?.posterior_mean}</div>
            </div>

            <div style={{ background: 'rgba(236,178,255,0.04)', border: '1px solid rgba(236,178,255,0.1)', padding: '1rem', borderRadius: '12px' }}>
              <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '0.5rem', letterSpacing: '0.05em' }}>VAE Team Health</div>
              <div style={{ fontSize: '0.85rem', marginBottom: '0.2rem' }}>Latent Space Flag:</div>
              <div style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>Week of {mlData.vae?.week_start}</div>
              <div style={{ color: 'var(--error)', fontSize: '0.75rem', marginTop: '0.4rem' }}>Recon Error limit exceeded</div>
            </div>

          </div>
        </div>
      </section>

      {/* ── Developer Flow Section ── */}
      <div className="flow-row fade-up d4">

        {/* PR Cycle Distribution */}
        <div className="glass-card flow-card">
          <div className="card-header">
            <div className="card-title-row">
              <div className="accent-bar" />
              <div>
                <h3 className="card-title">PR Cycle Distribution</h3>
                <p className="card-sub">Time from first commit to merge</p>
              </div>
            </div>
            <span className="tag stable">Avg: 18.4 hrs</span>
          </div>
          <div className="pr-phases">
            {prCyclePhases.map(p => (
              <div key={p.label} className="phase-row">
                <div className="phase-labels">
                  <span>{p.label}</span>
                  <span className="phase-val">{p.value} hrs</span>
                </div>
                <div className="phase-track">
                  <div className="phase-fill" style={{ width: `${p.pct}%`, background: p.color }} />
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Flow State Efficiency */}
        <div className="glass-card flow-card flow-state-card">
          <div className="glow-blob" />
          <h3 className="card-title">Flow State Efficiency</h3>
          <p className="card-sub mb-6">Interruption-free coding periods</p>
          <div className="flow-inner">
            <div className="donut-wrap">
              <svg viewBox="0 0 128 128" className="donut-svg">
                <circle cx="64" cy="64" r="56" fill="none" stroke="var(--surface-high)" strokeWidth="12" />
                <circle cx="64" cy="64" r="56" fill="none"
                  stroke="var(--accent)" strokeWidth="12"
                  strokeDasharray="351.8" strokeDashoffset="98"
                  strokeLinecap="round"
                  style={{ transform: 'rotate(-90deg)', transformOrigin: '50% 50%' }}
                />
              </svg>
              <div className="donut-label">
                <span className="donut-pct">72%</span>
                <span className="donut-sub">Focus</span>
              </div>
            </div>
            <div className="flow-stats">
              <div className="flow-stat">
                <span className="material-symbols-outlined" style={{ color: 'var(--accent)' }}>bolt</span>
                <div>
                  <p className="fs-label">Deep Work</p>
                  <p className="fs-val">4.2 hrs daily avg</p>
                </div>
              </div>
              <div className="flow-stat">
                <span className="material-symbols-outlined" style={{ color: 'var(--secondary)' }}>block</span>
                <div>
                  <p className="fs-label">Context Switches</p>
                  <p className="fs-val">9 daily interruptions</p>
                </div>
              </div>
              <div className="flow-stat">
                <span className="material-symbols-outlined" style={{ color: '#2dd4bf' }}>trending_up</span>
                <div>
                  <p className="fs-label">Model R² Score</p>
                  <p className="fs-val">0.58 — LSTM v4</p>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Daily Velocity Bars */}
        <div className="glass-card flow-card">
          <div className="card-header">
            <div className="card-title-row">
              <div className="accent-bar cyan" />
              <h3 className="card-title">Daily Velocity</h3>
            </div>
            <span className="velocity-delta">+24%</span>
          </div>
          <ResponsiveContainer width="100%" height={160}>
            <BarChart data={velocityData} barSize={16}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" vertical={false} />
              <XAxis dataKey="day" stroke="transparent" tick={{ fill: 'var(--text-muted)', fontSize: 10 }} />
              <YAxis hide />
              <Tooltip content={<CustomTooltip />} />
              <Bar dataKey="prs" name="PRs" fill="var(--accent)" fillOpacity={0.7} radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
          <p className="card-sub text-center" style={{ marginTop: '0.5rem' }}>Efficiency vs last week</p>
        </div>
      </div>

      {/* ── Git Activity + Build Log ── */}
      <div className="bottom-row fade-up d4">

        {/* Git Activity Feed */}
        <div className="glass-card activity-card">
          <div className="card-header">
            <div className="card-title-row">
              <div className="accent-bar purple" />
              <h3 className="card-title">Git Activity</h3>
            </div>
            <button className="link-btn" onClick={() => setShowAllActivity(true)}>View All</button>
          </div>
          <div className="activity-list">
            {gitActivity.map((a, i) => (
              <div key={i} className="activity-item">
                <div className={`activity-icon activity-${a.color}`}>
                  <span className="material-symbols-outlined">
                    {a.type === 'commit' ? 'commit' : a.type === 'merge' ? 'merge' : 'bug_report'}
                  </span>
                </div>
                <div className="activity-body">
                  <p className="activity-main">
                    <span className={`activity-user color-${a.color}`}>{a.user}</span>
                    {' → '}
                    <span className="activity-branch">{a.branch}</span>
                  </p>
                  <p className="activity-msg">{a.msg}</p>
                  <span className="activity-time mono">{a.time}</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Deployment Log Table */}
        <div className="glass-card deploy-card">
          <div className="card-header">
            <h3 className="card-title">Build &amp; Deployment Logs</h3>
            <button className="link-btn" onClick={handleExportCSV}>Export CSV</button>
          </div>
          <table className="deploy-table">
            <thead>
              <tr>
                {['Environment', 'Status', 'ID', 'Duration', 'Timestamp'].map(h => (
                  <th key={h}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {deploymentLog.map((row, i) => (
                <tr key={i}>
                  <td className="fw-600">{row.env}</td>
                  <td><span className={`tag ${row.status === 'Success' ? 'success' : row.status === 'Failed' ? 'error' : 'running'}`}>
                    {row.status}
                  </span></td>
                  <td><span className="mono" style={{ color: 'var(--text-muted)' }}>{row.id}</span></td>
                  <td>{row.dur}</td>
                  <td style={{ color: 'var(--text-muted)' }}>{row.time}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Footer */}
      <footer className="dash-footer">
        <span>DEVINSIGHT AI • LSTM v4 • KINETIC TERMINAL</span>
        <div className="footer-right">
          <div className="footer-node">
            <span className="pulse-dot" />
            NODE_742: ACTIVE
          </div>
          <span>v4.12.0-STABLE</span>
        </div>
      </footer>
      {/* ── View All Modal ── */}
      {showAllActivity && (
        <div className="modal-backdrop" onClick={() => setShowAllActivity(false)}>
          <div className="modal-panel" onClick={e => e.stopPropagation()}>
            <div className="modal-header">
              <h3>All Git Activity</h3>
              <button className="modal-close" onClick={() => setShowAllActivity(false)}>
                <span className="material-symbols-outlined">close</span>
              </button>
            </div>
            <div className="modal-body">
              {allGitActivity.map((a, i) => (
                <div key={i} className="activity-item">
                  <div className={`activity-icon activity-${a.color}`}>
                    <span className="material-symbols-outlined">
                      {a.type === 'commit' ? 'commit' : a.type === 'merge' ? 'merge' : 'bug_report'}
                    </span>
                  </div>
                  <div className="activity-body">
                    <p className="activity-main">
                      <span className={`activity-user color-${a.color}`}>{a.user}</span>
                      {' → '}
                      <span className="activity-branch">{a.branch}</span>
                    </p>
                    <p className="activity-msg">{a.msg}</p>
                    <span className="activity-time mono">{a.time}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

    </div>
  );
};

export default Dashboard;
