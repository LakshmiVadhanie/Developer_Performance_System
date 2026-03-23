import { useEffect, useState } from 'react';
import './Home.css';

type Props = { onStart: () => void; username?: string };

// Removed QUOTES and TECH_STACK

export default function Home({ onStart, username }: Props) {
// Removed quoteIdx and canvasRef
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });
  const [visible, setVisible] = useState(false);

  // Reveal on mount
  useEffect(() => {
    const t = setTimeout(() => setVisible(true), 80);
    return () => clearTimeout(t);
  }, []);

  // Rotate quotes removed

  // Mouse parallax
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      setMousePos({ x: e.clientX / window.innerWidth, y: e.clientY / window.innerHeight });
    };
    window.addEventListener('mousemove', handler);
    return () => window.removeEventListener('mousemove', handler);
  }, []);

  // Particle grid canvas removed

  const px = (mousePos.x - 0.5) * 30;
  const py = (mousePos.y - 0.5) * 20;

  return (
    <div className={`home-root ${visible ? 'home-visible' : ''}`}>
      {/* Canvas and orbs removed */}

      {/* ── Top bar ── */}
      <header className="home-header">
        <div className="home-logo">
          <div className="home-logo-icon">⌘</div>
          <span>DevInsight</span>
        </div>
        {username && (
          <div className="home-user-pill">
            <span className="home-user-dot" />
            {username}
          </div>
        )}
      </header>

      {/* ── Hero ── */}
      <main className="home-hero">
        <div
          className="home-hero-inner"
          style={{ transform: `translate(${px * 0.25}px, ${py * 0.2}px)` }}
        >
        </div>
      </main>

      {/* Floating UI overlay at the bottom center */}
      <div style={{ position: 'absolute', bottom: '2%', left: '0', right: '0', display: 'flex', flexDirection: 'column', alignItems: 'center', zIndex: 40 }}>
        
        {/* Quote */}
        <div style={{ fontSize: '1.25rem', fontStyle: 'italic', marginBottom: '1.2rem', color: 'rgba(255,255,255,0.95)', textAlign: 'center', maxWidth: '700px', fontWeight: 300, letterSpacing: '0.04em', textShadow: '0 4px 12px rgba(0,0,0,0.8)' }}>
          "You can't optimize what you don't measure.<br/>DevInsight isolates the signal from the noise."
        </div>
        
        {/* Tech Stack */}
        <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap', justifyContent: 'center', maxWidth: '800px', marginBottom: '1.8rem' }}>
            {['Transformers', 'BigQuery', 'PyTorch', 'React', 'Grafana', 'Prometheus'].map(tech => (
              <span key={tech} style={{ background: 'rgba(0,0,0,0.6)', border: '1px solid rgba(255,255,255,0.2)', padding: '0.5rem 1.2rem', borderRadius: '25px', fontSize: '0.8rem', letterSpacing: '0.05em', color: 'rgba(255,255,255,0.85)', backdropFilter: 'blur(10px)' }}>{tech}</span>
            ))}
        </div>

        {/* Start Analysis Button */}
        <button 
          className="home-cta" 
          onClick={onStart} 
          style={{ 
            boxShadow: '0 0 30px rgba(0, 229, 255, 0.4)', 
            padding: '1rem 3.5rem',
            borderRadius: '50px',
            transform: 'scale(1.05)',
            border: '1px solid rgba(0, 229, 255, 0.5)'
          }}
        >
          <span className="home-cta-text" style={{ fontSize: '1.15rem', fontWeight: 600 }}>Start Analysis</span>
          <span className="home-cta-arrow" style={{ fontSize: '1.3rem', marginLeft: '0.6rem' }}>→</span>
        </button>

      </div>
    </div>
  );
}
