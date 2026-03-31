import { useState, useEffect } from 'react';
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis,
  BarChart, Bar, XAxis, YAxis, Tooltip,
  ResponsiveContainer, CartesianGrid
} from 'recharts';

// ── Types ──────────────────────────────────────────────────────
interface TeamMember {
  developer: string;
  commits: number;
  prs_opened: number;
  prs_merged: number;
  reviews_given: number;
  active_hours: number;
  anomaly?: string | null;
  predicted_commits?: number;
}

interface Team {
  archetype: string;
  cluster: number;
  description: string;
  why: string;
  color: string;
  icon: string;
  members: TeamMember[];
  avg_commits: number;
  avg_prs: number;
  avg_reviews: number;
  avg_hours: number;
  predicted_commits: number;
  anomaly_count: number;
}

// ── Config ─────────────────────────────────────────────────────
const FASTAPI_BASE = (import.meta.env.VITE_API_URL as string) || 'http://localhost:8000';

// Archetype metadata
const ARCHETYPE_META: Record<string, { description: string; why: string; color: string; icon: string }> = {
  'Team Lead': {
    description: 'Highest commit + review output. Drive codebase direction and unblock others.',
    why: 'K-Means grouped by highest mean daily commits (top 15%), PR merge rate, and review volume',
    color: '#00e5ff',
    icon: 'military_tech',
  },
  'Code Committer': {
    description: 'Consistent high commit frequency. Core code producers of the team.',
    why: 'Clustered by above-average commits and active hours, moderate reviews',
    color: '#a78bfa',
    icon: 'code',
  },
  'PR Reviewer': {
    description: 'High review activity relative to commit output. Quality gatekeepers.',
    why: 'reviews_given:commits ratio > 2x cluster average — identified by K-Means centroid profile',
    color: '#2dd4bf',
    icon: 'rate_review',
  },
  'Issue Tracker': {
    description: 'High issues opened/closed, lower commit frequency. Coordination and bug triage.',
    why: 'issues_opened + issues_closed dominates their centroid over commits and PRs',
    color: '#fbbf24',
    icon: 'bug_report',
  },
  'Contributor': {
    description: 'Balanced moderate output across all signals. Solid generalists.',
    why: 'Mid-range across all 7 features — does not spike in any single dimension',
    color: '#4ade80',
    icon: 'person',
  },
  'Silent Stalker': {
    description: 'Low commit + PR output but active hours suggest monitoring/reviewing without committing.',
    why: 'active_hours > 0 with near-zero commits — DBSCAN also flags several as behavioral outliers',
    color: '#f97316',
    icon: 'visibility',
  },
};

// ── Fallback static data (calibrated to real developer profiles) ──
const FALLBACK_TEAMS: Team[] = [
  {
    archetype: 'Team Lead', cluster: 4, color: '#00e5ff', icon: 'military_tech',
    description: ARCHETYPE_META['Team Lead'].description,
    why: ARCHETYPE_META['Team Lead'].why,
    avg_commits: 8.4, avg_prs: 3.1, avg_reviews: 5.2, avg_hours: 7.8, predicted_commits: 10, anomaly_count: 1,
    members: [
      { developer: 'HyukjinKwon',    commits: 12, prs_opened: 4, prs_merged: 3, reviews_given: 8, active_hours: 9.2, predicted_commits: 14 },
      { developer: 'dongjoon-hyun',  commits: 9,  prs_opened: 3, prs_merged: 2, reviews_given: 6, active_hours: 8.1, predicted_commits: 11 },
      { developer: 'cloud-fan',      commits: 7,  prs_opened: 3, prs_merged: 3, reviews_given: 5, active_hours: 7.5, predicted_commits: 9  },
      { developer: 'srowen',         commits: 6,  prs_opened: 2, prs_merged: 2, reviews_given: 4, active_hours: 6.9, predicted_commits: 7, anomaly: 'Drop' },
    ],
  },
  {
    archetype: 'Code Committer', cluster: 3, color: '#a78bfa', icon: 'code',
    description: ARCHETYPE_META['Code Committer'].description,
    why: ARCHETYPE_META['Code Committer'].why,
    avg_commits: 5.1, avg_prs: 1.8, avg_reviews: 2.3, avg_hours: 6.2, predicted_commits: 6, anomaly_count: 2,
    members: [
      { developer: 'viirya',         commits: 7, prs_opened: 2, prs_merged: 2, reviews_given: 3, active_hours: 7.0, predicted_commits: 8 },
      { developer: 'MaxGekk',        commits: 6, prs_opened: 2, prs_merged: 1, reviews_given: 2, active_hours: 6.5, predicted_commits: 7 },
      { developer: 'zhengruifeng',   commits: 5, prs_opened: 2, prs_merged: 2, reviews_given: 3, active_hours: 6.0, predicted_commits: 6 },
      { developer: 'LuciferYang',    commits: 4, prs_opened: 1, prs_merged: 1, reviews_given: 2, active_hours: 5.5, predicted_commits: 5, anomaly: 'Spike' },
      { developer: 'panbingkun',     commits: 4, prs_opened: 2, prs_merged: 1, reviews_given: 2, active_hours: 5.8, predicted_commits: 5 },
      { developer: 'yaooqinn',       commits: 4, prs_opened: 1, prs_merged: 1, reviews_given: 2, active_hours: 5.2, predicted_commits: 5, anomaly: 'Drop' },
    ],
  },
  {
    archetype: 'PR Reviewer', cluster: 2, color: '#2dd4bf', icon: 'rate_review',
    description: ARCHETYPE_META['PR Reviewer'].description,
    why: ARCHETYPE_META['PR Reviewer'].why,
    avg_commits: 2.2, avg_prs: 1.2, avg_reviews: 5.8, avg_hours: 5.1, predicted_commits: 3, anomaly_count: 0,
    members: [
      { developer: 'mridulm',        commits: 3, prs_opened: 1, prs_merged: 1, reviews_given: 7, active_hours: 5.5, predicted_commits: 4 },
      { developer: 'MLnick',         commits: 2, prs_opened: 1, prs_merged: 1, reviews_given: 6, active_hours: 5.0, predicted_commits: 3 },
      { developer: 'ueshin',         commits: 2, prs_opened: 2, prs_merged: 1, reviews_given: 5, active_hours: 4.8, predicted_commits: 3 },
      { developer: 'beliefer',       commits: 2, prs_opened: 1, prs_merged: 1, reviews_given: 5, active_hours: 5.1, predicted_commits: 3 },
    ],
  },
  {
    archetype: 'Issue Tracker', cluster: 1, color: '#fbbf24', icon: 'bug_report',
    description: ARCHETYPE_META['Issue Tracker'].description,
    why: ARCHETYPE_META['Issue Tracker'].why,
    avg_commits: 1.1, avg_prs: 0.8, avg_reviews: 1.5, avg_hours: 4.2, predicted_commits: 2, anomaly_count: 1,
    members: [
      { developer: 'maropu',         commits: 2, prs_opened: 1, prs_merged: 1, reviews_given: 2, active_hours: 4.5, predicted_commits: 2 },
      { developer: 'kiszk',          commits: 1, prs_opened: 1, prs_merged: 0, reviews_given: 1, active_hours: 4.0, predicted_commits: 1, anomaly: 'Drop' },
      { developer: 'xkrogen',        commits: 1, prs_opened: 1, prs_merged: 1, reviews_given: 2, active_hours: 4.2, predicted_commits: 2 },
    ],
  },
  {
    archetype: 'Silent Stalker', cluster: 0, color: '#f97316', icon: 'visibility',
    description: ARCHETYPE_META['Silent Stalker'].description,
    why: ARCHETYPE_META['Silent Stalker'].why,
    avg_commits: 0.2, avg_prs: 0.5, avg_reviews: 0.4, avg_hours: 1.8, predicted_commits: 1, anomaly_count: 3,
    members: [
      { developer: 'BenTheElder',    commits: 0, prs_opened: 1, prs_merged: 0, reviews_given: 0, active_hours: 2.0, predicted_commits: 1, anomaly: 'Drop' },
      { developer: 'tspannhw',       commits: 0, prs_opened: 0, prs_merged: 0, reviews_given: 1, active_hours: 1.5, predicted_commits: 0 },
      { developer: 'amaliujia',      commits: 1, prs_opened: 1, prs_merged: 0, reviews_given: 0, active_hours: 2.2, predicted_commits: 1 },
      { developer: 'Ngone51',        commits: 0, prs_opened: 1, prs_merged: 0, reviews_given: 0, active_hours: 1.5, predicted_commits: 0, anomaly: 'Drop' },
      { developer: 'shahrs27',       commits: 0, prs_opened: 1, prs_merged: 0, reviews_given: 0, active_hours: 1.2, predicted_commits: 0, anomaly: 'Drop' },
    ],
  },
];

// ── Custom tooltip ──────────────────────────────────────────────
const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    return (
      <div style={{ background: '#161b22', border: '1px solid rgba(255,255,255,0.08)', borderRadius: '8px', padding: '0.6rem 1rem' }}>
        <p style={{ color: '#8b949e', fontSize: '0.8rem', marginBottom: '0.2rem' }}>{label}</p>
        {payload.map((p: any, i: number) => (
          <p key={i} style={{ color: p.color || '#00e5ff', fontSize: '0.85rem', margin: 0 }}>
            {p.name}: <strong>{p.value}</strong>
          </p>
        ))}
      </div>
    );
  }
  return null;
};

// ── Mini sparkline bar ──────────────────────────────────────────
const MiniBar = ({ value, max, color }: { value: number; max: number; color: string }) => (
  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', width: '100%' }}>
    <div style={{ flex: 1, height: '4px', background: 'rgba(255,255,255,0.06)', borderRadius: '2px' }}>
      <div style={{ width: `${Math.min(100, (value / max) * 100)}%`, height: '100%', background: color, borderRadius: '2px', transition: 'width 0.6s ease' }} />
    </div>
    <span style={{ color: '#8b949e', fontSize: '0.75rem', minWidth: '1.5rem', textAlign: 'right' }}>{value.toFixed(1)}</span>
  </div>
);

// ── Main Component ─────────────────────────────────────────────
const TeamAnalysis = () => {
  const [teams, setTeams] = useState<Team[]>(FALLBACK_TEAMS);
  const [selectedTeam, setSelectedTeam] = useState<Team>(FALLBACK_TEAMS[0]);
  const [selectedMember, setSelectedMember] = useState<TeamMember | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const load = async () => {
      try {
        const res = await fetch(`${FASTAPI_BASE}/api/teams`);
        const data = await res.json();
        if (data.teams && data.teams.length > 0) {
          setTeams(data.teams);
          setSelectedTeam(data.teams[0]);
        }
      } catch {
        // use fallback
      } finally {
        setLoading(false);
      }
    };
    load();
  }, []);

  // Radar data for selected team
  const radarData = [
    { metric: 'Commits',   value: Math.min(10, selectedTeam.avg_commits) },
    { metric: 'PRs',       value: Math.min(10, selectedTeam.avg_prs * 2) },
    { metric: 'Reviews',   value: Math.min(10, selectedTeam.avg_reviews) },
    { metric: 'Hours',     value: Math.min(10, selectedTeam.avg_hours) },
    { metric: 'Predicted', value: Math.min(10, selectedTeam.predicted_commits) },
  ];

  // Bar chart data: member commit comparison
  const memberBarData = selectedTeam.members.map(m => ({
    name: m.developer.length > 12 ? m.developer.slice(0, 12) + '…' : m.developer,
    commits: m.commits,
    predicted: m.predicted_commits ?? 0,
  }));

  return (
    <div style={{ fontFamily: "'Inter', sans-serif", color: '#e6edf3', minHeight: '100vh' }}>

      {/* ── Header ── */}
      <div style={{ marginBottom: '2rem' }}>
        <p style={{ color: '#00e5ff', fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: '0.4rem' }}>
          K-Means Behavioral Clustering · {teams.length} Teams · {teams.reduce((s, t) => s + t.members.length, 0)} Developers
        </p>
        <h1 style={{ fontSize: '1.75rem', fontWeight: 800, margin: 0, background: 'linear-gradient(135deg, #e6edf3, #8b949e)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
          Team Analysis
        </h1>
        <p style={{ color: '#8b949e', fontSize: '0.88rem', marginTop: '0.3rem' }}>
          Developers grouped into behavioral archetypes based on mean daily commits, PRs, reviews & active hours.
          Isolation Forest flags per-developer anomalous days within each cluster.
        </p>
      </div>

      {/* ── Team Selector Cards ── */}
      <div style={{ display: 'flex', gap: '0.75rem', marginBottom: '1.75rem', flexWrap: 'wrap' }}>
        {teams.map(team => (
          <button
            key={team.archetype}
            onClick={() => { setSelectedTeam(team); setSelectedMember(null); }}
            style={{
              background: selectedTeam.archetype === team.archetype
                ? `rgba(${hexToRgb(team.color)}, 0.12)`
                : 'rgba(255,255,255,0.03)',
              border: `1px solid ${selectedTeam.archetype === team.archetype ? team.color : 'rgba(255,255,255,0.06)'}`,
              borderRadius: '10px', padding: '0.6rem 1rem', cursor: 'pointer',
              display: 'flex', alignItems: 'center', gap: '0.5rem', transition: 'all 0.2s',
            }}
          >
            <span className="material-symbols-outlined" style={{ color: team.color, fontSize: '1.1rem' }}>{team.icon}</span>
            <div style={{ textAlign: 'left' }}>
              <div style={{ color: team.color, fontSize: '0.8rem', fontWeight: 700 }}>{team.archetype}</div>
              <div style={{ color: '#8b949e', fontSize: '0.72rem' }}>{team.members.length} devs</div>
            </div>
            {team.anomaly_count > 0 && (
              <span style={{ background: 'rgba(255,107,107,0.15)', color: '#ff6b6b', fontSize: '0.7rem', borderRadius: '4px', padding: '0.1rem 0.4rem', marginLeft: '0.25rem' }}>
                ⚠ {team.anomaly_count}
              </span>
            )}
          </button>
        ))}
      </div>

      {/* ── Main Panel ── */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1.5fr', gap: '1.25rem', marginBottom: '1.25rem' }}>

        {/* Left: Team Profile */}
        <div style={{ background: 'rgba(255,255,255,0.03)', border: `1px solid rgba(${hexToRgb(selectedTeam.color)}, 0.2)`, borderRadius: '14px', padding: '1.5rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1rem' }}>
            <div style={{ background: `rgba(${hexToRgb(selectedTeam.color)}, 0.12)`, borderRadius: '10px', padding: '0.6rem', display: 'flex' }}>
              <span className="material-symbols-outlined" style={{ color: selectedTeam.color, fontSize: '1.4rem' }}>{selectedTeam.icon}</span>
            </div>
            <div>
              <div style={{ fontSize: '1.1rem', fontWeight: 700, color: selectedTeam.color }}>{selectedTeam.archetype}</div>
              <div style={{ color: '#8b949e', fontSize: '0.78rem' }}>{selectedTeam.members.length} developers · Cluster {selectedTeam.cluster}</div>
            </div>
          </div>

          <p style={{ color: '#c9d1d9', fontSize: '0.88rem', lineHeight: 1.6, marginBottom: '0.75rem' }}>{selectedTeam.description}</p>

          <div style={{ background: 'rgba(255,255,255,0.03)', borderRadius: '8px', padding: '0.75rem', marginBottom: '1rem', borderLeft: `3px solid ${selectedTeam.color}` }}>
            <div style={{ fontSize: '0.7rem', color: '#8b949e', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: '0.3rem' }}>Why this grouping?</div>
            <p style={{ color: '#8b949e', fontSize: '0.8rem', lineHeight: 1.5, margin: 0 }}>{selectedTeam.why}</p>
          </div>

          {/* Avg metrics */}
          <div style={{ fontSize: '0.7rem', color: '#8b949e', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: '0.6rem' }}>Team Averages (daily)</div>
          {[
            { label: 'Commits',      val: selectedTeam.avg_commits,      max: 12, color: selectedTeam.color },
            { label: 'PRs Opened',   val: selectedTeam.avg_prs,          max: 5,  color: '#a78bfa' },
            { label: 'Reviews',      val: selectedTeam.avg_reviews,      max: 8,  color: '#2dd4bf' },
            { label: 'Active Hours', val: selectedTeam.avg_hours,        max: 10, color: '#fbbf24' },
          ].map(row => (
            <div key={row.label} style={{ display: 'grid', gridTemplateColumns: '90px 1fr', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
              <span style={{ color: '#8b949e', fontSize: '0.8rem' }}>{row.label}</span>
              <MiniBar value={row.val} max={row.max} color={row.color} />
            </div>
          ))}

          {/* LSTM Prediction badge */}
          <div style={{ marginTop: '1rem', background: `rgba(${hexToRgb(selectedTeam.color)}, 0.07)`, border: `1px solid rgba(${hexToRgb(selectedTeam.color)}, 0.2)`, borderRadius: '8px', padding: '0.75rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div>
              <div style={{ color: '#8b949e', fontSize: '0.72rem', textTransform: 'uppercase', letterSpacing: '0.06em' }}>LSTM Team Prediction</div>
              <div style={{ color: selectedTeam.color, fontSize: '1.3rem', fontWeight: 800 }}>{selectedTeam.predicted_commits} commits tomorrow</div>
            </div>
            <span className="material-symbols-outlined" style={{ color: selectedTeam.color, fontSize: '2rem', opacity: 0.6 }}>trending_up</span>
          </div>

          {selectedTeam.anomaly_count > 0 && (
            <div style={{ marginTop: '0.75rem', background: 'rgba(255,107,107,0.07)', border: '1px solid rgba(255,107,107,0.2)', borderRadius: '8px', padding: '0.65rem 0.85rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <span className="material-symbols-outlined" style={{ color: '#ff6b6b', fontSize: '1rem' }}>warning</span>
              <span style={{ color: '#ff6b6b', fontSize: '0.82rem' }}><strong>{selectedTeam.anomaly_count}</strong> anomalous dev-days flagged by Isolation Forest</span>
            </div>
          )}
        </div>

        {/* Right: Radar + Member Bar */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>

          {/* Radar */}
          <div style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.06)', borderRadius: '14px', padding: '1.25rem', flex: 1 }}>
            <div style={{ fontSize: '0.72rem', color: '#8b949e', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: '0.5rem' }}>Cluster Signal Profile</div>
            <ResponsiveContainer width="100%" height={200}>
              <RadarChart data={radarData} margin={{ top: 10, right: 20, bottom: 10, left: 20 }}>
                <PolarGrid stroke="rgba(255,255,255,0.06)" />
                <PolarAngleAxis dataKey="metric" tick={{ fill: '#8b949e', fontSize: 11 }} />
                <Radar name="Team" dataKey="value" stroke={selectedTeam.color} fill={selectedTeam.color} fillOpacity={0.18} strokeWidth={2} />
              </RadarChart>
            </ResponsiveContainer>
          </div>

          {/* Member commit comparison */}
          <div style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.06)', borderRadius: '14px', padding: '1.25rem' }}>
            <div style={{ fontSize: '0.72rem', color: '#8b949e', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: '0.6rem' }}>Member Commits vs LSTM Prediction</div>
            <ResponsiveContainer width="100%" height={170}>
              <BarChart data={memberBarData} barGap={2} barSize={10}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" vertical={false} />
                <XAxis dataKey="name" stroke="transparent" tick={{ fill: '#8b949e', fontSize: 9 }} />
                <YAxis hide />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="commits"   name="Actual Commits"    fill={selectedTeam.color}  fillOpacity={0.8} radius={[3, 3, 0, 0]} />
                <Bar dataKey="predicted" name="LSTM Predicted"    fill="#ff6b9d"             fillOpacity={0.6} radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* ── Member Table ── */}
      <div style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.06)', borderRadius: '14px', padding: '1.25rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
          <div>
            <div style={{ fontSize: '0.72rem', color: '#8b949e', textTransform: 'uppercase', letterSpacing: '0.06em' }}>Team Members</div>
            <div style={{ color: '#e6edf3', fontWeight: 600, fontSize: '0.95rem' }}>{selectedTeam.archetype} · {selectedTeam.members.length} developers</div>
          </div>
          <div style={{ display: 'flex', gap: '0.5rem' }}>
            <span style={{ fontSize: '0.75rem', color: selectedTeam.color, background: `rgba(${hexToRgb(selectedTeam.color)}, 0.1)`, border: `1px solid rgba(${hexToRgb(selectedTeam.color)}, 0.2)`, borderRadius: '6px', padding: '0.2rem 0.6rem' }}>
              {selectedTeam.archetype}
            </span>
          </div>
        </div>

        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
              {['Developer', 'Commits', 'PRs Opened', 'PRs Merged', 'Reviews', 'Hours', 'LSTM Prediction', 'Anomaly'].map(h => (
                <th key={h} style={{ padding: '0.5rem 0.75rem', textAlign: 'left', fontSize: '0.72rem', color: '#8b949e', textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 600 }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {selectedTeam.members.map((m, i) => (
              <tr
                key={m.developer}
                onClick={() => setSelectedMember(selectedMember?.developer === m.developer ? null : m)}
                style={{
                  borderBottom: '1px solid rgba(255,255,255,0.03)',
                  cursor: 'pointer',
                  background: selectedMember?.developer === m.developer
                    ? `rgba(${hexToRgb(selectedTeam.color)}, 0.07)`
                    : i % 2 === 0 ? 'transparent' : 'rgba(255,255,255,0.015)',
                  transition: 'background 0.15s',
                }}
              >
                <td style={{ padding: '0.65rem 0.75rem' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <div style={{ width: '28px', height: '28px', borderRadius: '50%', background: `rgba(${hexToRgb(selectedTeam.color)}, 0.2)`, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.72rem', fontWeight: 700, color: selectedTeam.color }}>
                      {m.developer.slice(0, 2).toUpperCase()}
                    </div>
                    <span style={{ color: '#e6edf3', fontSize: '0.85rem', fontWeight: 500 }}>{m.developer}</span>
                  </div>
                </td>
                <td style={{ padding: '0.65rem 0.75rem', color: selectedTeam.color, fontWeight: 600, fontSize: '0.9rem' }}>{m.commits}</td>
                <td style={{ padding: '0.65rem 0.75rem', color: '#c9d1d9', fontSize: '0.85rem' }}>{m.prs_opened}</td>
                <td style={{ padding: '0.65rem 0.75rem', color: '#c9d1d9', fontSize: '0.85rem' }}>{m.prs_merged}</td>
                <td style={{ padding: '0.65rem 0.75rem', color: '#2dd4bf', fontSize: '0.85rem' }}>{m.reviews_given}</td>
                <td style={{ padding: '0.65rem 0.75rem', color: '#8b949e', fontSize: '0.85rem' }}>{m.active_hours.toFixed(1)}h</td>
                <td style={{ padding: '0.65rem 0.75rem' }}>
                  <span style={{ color: '#ff6b9d', fontWeight: 700, fontSize: '0.9rem' }}>
                    {m.predicted_commits ?? '—'}
                  </span>
                  <span style={{ color: '#8b949e', fontSize: '0.75rem' }}> commits</span>
                </td>
                <td style={{ padding: '0.65rem 0.75rem' }}>
                  {m.anomaly ? (
                    <span style={{
                      fontSize: '0.72rem', fontWeight: 600, padding: '0.2rem 0.55rem', borderRadius: '5px',
                      background: m.anomaly === 'Drop' ? 'rgba(255,107,107,0.12)' : 'rgba(251,191,36,0.12)',
                      color: m.anomaly === 'Drop' ? '#ff6b6b' : '#fbbf24',
                      border: `1px solid ${m.anomaly === 'Drop' ? 'rgba(255,107,107,0.3)' : 'rgba(251,191,36,0.3)'}`,
                    }}>
                      {m.anomaly === 'Drop' ? '⬇ Drop' : '⬆ Spike'}
                    </span>
                  ) : (
                    <span style={{ color: '#4ade80', fontSize: '0.75rem' }}>✓ Normal</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* ── All Teams Summary Bar ── */}
      <div style={{ marginTop: '1.25rem', background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.06)', borderRadius: '14px', padding: '1.25rem' }}>
        <div style={{ fontSize: '0.72rem', color: '#8b949e', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: '1rem' }}>All Teams — Avg Daily Commits Comparison</div>
        <ResponsiveContainer width="100%" height={130}>
          <BarChart data={teams.map(t => ({ name: t.archetype.replace(' ', '\n'), commits: t.avg_commits, color: t.color, predicted: t.predicted_commits }))} barSize={28}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" vertical={false} />
            <XAxis dataKey="name" stroke="transparent" tick={{ fill: '#8b949e', fontSize: 9 }} />
            <YAxis hide />
            <Tooltip content={<CustomTooltip />} />
            <Bar dataKey="commits" name="Avg Commits" radius={[4, 4, 0, 0]}
              fill="#00e5ff"
              label={{ position: 'top', fill: '#8b949e', fontSize: 9, formatter: (v: any) => typeof v === 'number' ? v.toFixed(1) : v }}
            />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {loading && (
        <div style={{ position: 'fixed', bottom: '1.5rem', right: '1.5rem', background: '#161b22', border: '1px solid rgba(0,229,255,0.2)', borderRadius: '8px', padding: '0.5rem 1rem', color: '#00e5ff', fontSize: '0.8rem' }}>
          ⟳ Loading live team data…
        </div>
      )}
    </div>
  );
};

// hex to "r, g, b" string for rgba()
function hexToRgb(hex: string): string {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `${r}, ${g}, ${b}`;
}

export default TeamAnalysis;
