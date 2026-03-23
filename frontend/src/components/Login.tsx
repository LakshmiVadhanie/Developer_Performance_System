import { useState } from 'react';
import './Login.css';

type Props = { onLogin: (username: string) => void };

export default function Login({ onLogin }: Props) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError]       = useState('');
  const [loading, setLoading]   = useState(false);
  // Particle background removed

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!username || !password) {
      setError('Please enter both username and password.');
      return;
    }

    const validUsers = ['lakshmi', 'pranav', 'vaishnavi'];
    if (validUsers.includes(username.toLowerCase()) && password === '12345') {
      setError('');
      setLoading(true);
      setTimeout(() => onLogin(username), 800);
    } else {
      setError('Invalid username or password.');
    }
  };

  return (
    <div className="login-root">
      {/* Canvas and Orbs removed */}

      <div className="login-panel">
        {/* Brand */}
        <div className="login-brand">
          <div className="login-logo">⌘</div>
          <p className="login-brand-name">DevInsight</p>
          <p className="login-brand-sub">AI Workspace</p>
        </div>

        <h1 className="login-title">Welcome back</h1>
        <p className="login-desc">Sign in to access your team's performance intelligence.</p>

        <form className="login-form" onSubmit={handleSubmit}>
          <div className="login-field">
            <label>Username</label>
            <input
              type="text"
              placeholder="e.g. l.vadhanie"
              value={username}
              onChange={e => setUsername(e.target.value)}
              autoComplete="username"
            />
          </div>
          <div className="login-field">
            <label>Password</label>
            <input
              type="password"
              placeholder="••••••••"
              value={password}
              onChange={e => setPassword(e.target.value)}
              autoComplete="current-password"
            />
          </div>

          {error && <p className="login-error">{error}</p>}

          <button type="submit" className={`login-btn ${loading ? 'loading' : ''}`} disabled={loading}>
            {loading ? <span className="login-spinner" /> : 'Sign In'}
          </button>
        </form>

        <p className="login-hint">Demo: any username + password works ✦</p>
      </div>
    </div>
  );
}
