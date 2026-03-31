import { useState, useRef, useEffect } from 'react';
import { Send, Sparkles, Bot, User } from 'lucide-react';
import './Chatbot.css';

interface Message {
  id: string;
  type: 'user' | 'bot';
  content: string;
}

export type ChatPage = 'dashboard' | 'team' | 'ml';

interface Props {
  page?: ChatPage;
}

// ── Per-page config ─────────────────────────────────────────────
const PAGE_CONFIG: Record<ChatPage, {
  greeting: string;
  chips: string[];
  accentColor: string;
  label: string;
}> = {
  dashboard: {
    greeting: "Hello! I'm your Developer Intelligence assistant. Ask me about commit forecasts, PR health, team velocity, or anomalies.",
    chips: ["What's up?", "Commits tomorrow?", "PR health?", "Burnout risk?", "Top signals?", "Recommendations"],
    accentColor: '#00e5ff',
    label: 'Dashboard AI',
  },
  team: {
    greeting: "Hi! I'm your Team Analysis assistant. Ask me about behavioral archetypes, who's in each team, anomaly alerts, or what drives each cluster's productivity.",
    chips: ["What's Team Lead?", "Silent Stalker anomalies?", "Who's at risk?", "Best performing team?", "Isolation Forest?", "Archetype difference?"],
    accentColor: '#a78bfa',
    label: 'Team Analysis AI',
  },
  ml: {
    greeting: "Hey! I'm your ML Model assistant. Ask me about R² scores, why LSTM outperforms XGBoost, how Isolation Forest works, or what SHAP tells us.",
    chips: ["Best model R²?", "Why LSTM wins?", "What is R²?", "XGBoost vs LSTM?", "Drop anomaly?", "SHAP features?"],
    accentColor: '#ff6b9d',
    label: 'ML Insights AI',
  },
};

// ── Intent engine ───────────────────────────────────────────────
function getResponse(q: string, page: ChatPage): string {
  const all: { score: number; response: string }[] = [

    // ── SHARED / GENERAL ─────────────────────────────────────────

    {
      score: (q.includes('cluster') || q.includes('archetype') || q.includes('k-mean') ? 10 : 0) +
             (q.includes('what is') && q.includes('team lead') ? 8 : 0),
      response: "We use **K-Means Clustering** on 5 daily metrics — commits, PRs opened, PRs merged, reviews given, and active hours — to group developers into behavioral archetypes:\n\n🏅 **Team Lead** — top 20% commits + high reviews\n💻 **Code Committer** — above-avg commits, moderate reviews\n👁 **PR Reviewer** — reviews_given:commits ratio > 1.5×\n🐛 **Issue Tracker** — high PR activity, below-median commits\n👤 **Silent Stalker** — active hours > 0 but near-zero commits\n\nEach archetype has different LSTM predictions and anomaly risk profiles.",
    },
    {
      score: (q.includes('anomaly') || q.includes('anomali') ? 8 : 0) +
             (q.includes('isolation') || q.includes('isolation forest') ? 9 : 0) +
             (q.includes('how') && q.includes('detect') ? 5 : 0),
      response: "**Isolation Forest** detects anomalous developer-days using per-developer z-score normalization.\n\n— **Drop anomaly** = z_commits < −1.5 → sudden silence relative to that dev's own baseline\n— **Spike anomaly** = z_commits > +2.0 → outsized burst above their norm\n\n5% contamination rate: 212 of 4,230 dev-days flagged. The key insight: a Team Lead committing 3 today is a Drop. A Silent Stalker committing 3 is a Spike.",
    },
    {
      score: (q.includes('burnout') || q.includes('burn out') || q.includes('at risk') ? 9 : 0) +
             (q.includes('drop') && q.includes('anomaly') ? 7 : 0) +
             (q.includes('who') && (q.includes('risk') || q.includes('struggling')) ? 7 : 0),
      response: "Top at-risk developers by Drop anomaly count:\n\n🔴 HanumanthaRaoMandlem — 5 drop days\n🔴 panbingkun — 5 drop days\n🟠 yaooqinn — 4 drop days\n🟠 HyukjinKwon — 4 drop days\n🟠 dongjoon-hyun — 4 drop days\n\nA 'Drop' means they went unusually quiet vs their own baseline. Consecutive drops over 3+ days are the strongest burnout signal. Recommend a 1:1 check-in for the top 3.",
    },
    {
      score: (q.includes('spike') && (q.includes('anomaly') || q.includes('burst')) ? 9 : 0),
      response: "**Spike anomalies** (126 flagged) occur when a developer's commit count exceeds +2.0 standard deviations above their own baseline. This means *they* are unusually active, not just exceeding the team average.\n\nSpikes can indicate: a deadline crunch, context switching overload, or genuine high-energy sprints. Sustained spikes over 5+ days → burnout risk from overwork, not underwork.",
    },

    // ── ML INSIGHTS ──────────────────────────────────────────────

    {
      score: (q.includes('r2') || q.includes('r²') || q.includes('r-squared') || q.includes('r square') ? 8 : 0) +
             (q.includes('best') && q.includes('model') ? 5 : 0) +
             (q.includes('score') && q.includes('model') ? 5 : 0) +
             (page === 'ml' ? 2 : 0),
      response: "R² (R-squared) measures how much variance our model explains in next-day commit count:\n\n🥇 **LSTM / Transformer Ensemble — R² = 0.600** ← best\n🥈 LSTM (Reported) — R² = 0.600\n🥉 XGBoost + SHAP — R² = 0.512\n📉 Random Forest — R² = 0.261\n📉 Gradient Boosting — R² = 0.234\n📉 Ridge Regression — R² = 0.233\n📉 Linear Regression — R² = 0.233 ← baseline\n\nThe LSTM is **+158% relative improvement** over the linear baseline.",
    },
    {
      score: (q.includes('lstm') && (q.includes('win') || q.includes('better') || q.includes('why') || q.includes('outperform')) ? 10 : 0) +
             (q.includes('lstm') && q.includes('xgboost') ? 9 : 0) +
             (page === 'ml' ? 2 : 0),
      response: "**Why LSTM outperforms XGBoost (+0.088 R²):**\n\nXGBoost sees the same 5-day window but treats each day as independent features. It can't tell if *Monday was slow because of a holiday* or because energy was low.\n\nLSTM has sequential memory — gated cells that remember context across the 5-day input sequence. When reviews surged on Wednesday and PRs dropped Thursday, LSTM understands that Tuesday's pattern caused it.\n\nThis temporal reasoning is exactly what developer productivity data needs.",
    },
    {
      score: (q.includes('xgboost') && !q.includes('lstm') ? 7 : 0) +
             (q.includes('xgboost') && q.includes('shap') ? 9 : 0) +
             (page === 'ml' ? 1 : 0),
      response: "**XGBoost + SHAP** achieves R² = 0.512 — solid, but limited to treating 5-day features as independent inputs.\n\nWhere it shines: **explainability**. SHAP values reveal exactly why each forecast was made:\n\n🔑 reviews_given → strongest positive driver\n🔑 prs_opened → #2 leading indicator\n📉 prs_merged → near-zero importance (lagging, not leading)\n\nWe run SHAP alongside LSTM for local explainability — 'why 184 commits?' → 'because reviews_given spiked +2.1σ this week'.",
    },
    {
      score: (q.includes('what is') && q.includes('r') ? 5 : 0) +
             (q.includes('explain') && q.includes('r2') ? 8 : 0) +
             (q.includes('meaning') && q.includes('r') ? 6 : 0) +
             (page === 'ml' ? 1 : 0),
      response: "**R² (R-squared)** measures how much of the target's variance your model explains, on a 0–1 scale:\n\n• R² = 0.0 → model is no better than predicting the mean every time\n• R² = 0.6 → model explains 60% of why commits vary day to day\n• R² = 1.0 → perfect predictions\n\nFor developer productivity — which is highly noisy human behavior — 0.60 is genuinely strong. Academic finance models often sit at 0.20–0.35.",
    },
    {
      score: (q.includes('mae') || q.includes('mean absolute') ? 8 : 0) +
             (q.includes('rmse') ? 8 : 0) +
             (q.includes('error') && q.includes('model') ? 5 : 0),
      response: "**Error metrics for the best model (LSTM):**\n\n• MAE = 2.05 → on average, predictions are ±2 commits off\n• RMSE = 4.18 → penalizes large misses more\n\nFor context, the team averages ~68 commits/day — so ±2 is a 3% error rate. That's well within actionable planning tolerance. Linear Regression has MAE = 2.79, so LSTM is ~26% more accurate on absolute error.",
    },
    {
      score: (q.includes('baseline') && (q.includes('model') || q.includes('regression')) ? 8 : 0) +
             (q.includes('linear regression') ? 7 : 0),
      response: "We set **Linear Regression** as our baseline (R² = 0.233). It uses the same 5-day lag features but can't capture non-linear patterns or temporal order.\n\nAll other models are compared against this baseline:\n— XGBoost: +0.279 R² over baseline\n— LSTM: +0.367 R² over baseline (+158%)\n\nIf a fancy model can't beat linear regression, it's not worth deploying.",
    },
    {
      score: (q.includes('transformer') ? 7 : 0) +
             (q.includes('attention') ? 8 : 0) +
             (q.includes('ensemble') ? 7 : 0),
      response: "The **LSTM/Transformer Ensemble** combines LSTM's sequential memory with a Transformer's self-attention mechanism. Attention allows the model to dynamically weight which of the 5 days matters most for tomorrow's forecast.\n\nResult: R² = 0.600, MAE = 2.05 — matching raw LSTM R² but with slightly better absolute error. In production, we use the ensemble as the primary forecast and LSTM as the fallback.",
    },
    {
      score: (q.includes('how many') && (q.includes('anomal') || q.includes('dev-day')) ? 9 : 0) +
             (q.includes('total') && q.includes('anomal') ? 8 : 0) +
             (page === 'ml' ? 1 : 0),
      response: "**Isolation Forest anomaly totals:**\n\n📊 Total dev-days monitored: 4,230\n🔴 Anomalies flagged: 212 (5.0%)\n  ↘ Drop anomalies: 86 (sudden silence)\n  ↗ Spike anomalies: 126 (outsized burst)\n👥 Developers monitored: 47\n\n5% contamination was the configured threshold. At 10% contamination, the model flags ~420 dev-days — balancing sensitivity vs alert fatigue is a key tuning decision.",
    },

    // ── TEAM ARCHETYPES ──────────────────────────────────────────

    {
      score: (q.includes('team lead') && !q.includes('what is') ? 8 : 0) +
             (q.includes('best') && q.includes('team') ? 6 : 0) +
             (q.includes('high perform') ? 7 : 0),
      response: "**Team Lead** archetype (4 developers):\n\n🏅 Top 20% by commit volume (avg 8.4/day)\n📋 Strong PR merge rate (avg 3.1 PRs/day)\n👁 High review output (avg 5.2 reviews/day)\n⏱ 7.8 active hours/day\n\nLSTM predicts **15 commits tomorrow** for this cluster. They have 1 anomalous dev-day flagged. Characteristic: they don't just write code, they also review and merge other people's work.",
    },
    {
      score: (q.includes('code committer') ? 9 : 0) +
             (q.includes('committer') && !q.includes('team') ? 7 : 0),
      response: "**Code Committer** archetype (6 developers):\n\n💻 Above-average commits (avg 5.1/day)\n📋 Moderate PRs (avg 1.8/day)\n👁 Moderate reviews (avg 2.3/day)\n⏱ 6.2 active hours/day\n\nLSTM predicts **10 commits tomorrow**. **2 anomalous dev-days flagged** — including a Spike from LuciferYang and a Drop from yaooqinn. Core code producers; less cross-team coordination than Team Leads.",
    },
    {
      score: (q.includes('pr reviewer') || (q.includes('reviewer') && !q.includes('code')) ? 9 : 0),
      response: "**PR Reviewer** archetype (4 developers):\n\n👁 High review volume (avg 5.8 reviews/day)\n💻 Lower commits (avg 2.2/day) — reviews_given:commits ratio > 1.5×\n📋 Moderate PRs (avg 1.2/day)\n⏱ 5.1 active hours/day\n\nLSTM predicts **4 commits tomorrow**. **No anomalies flagged** — the most consistent cluster. They are quality gatekeepers; when PR Reviewers slow down, merge bottlenecks form 3–5 days later.",
    },
    {
      score: (q.includes('silent stalker') ? 10 : 0) +
             (q.includes('stalker') ? 8 : 0) +
             (q.includes('inactive') || q.includes('low commit') ? 5 : 0),
      response: "**Silent Stalker** archetype (5 developers):\n\n👤 Near-zero commits (avg 0.2/day)\n🕐 Active hours > 0 — they're watching but not committing\n📋 Occasional PRs (avg 0.5/day)\n⚠️ 3 anomalous dev-days flagged\n\nLSTM predicts **1 commit tomorrow**. DBSCAN also flags several as behavioral outliers. They could be: onboarding devs, managers who read code, or developers going through a block. Good candidates for a 1:1 check-in.",
    },
    {
      score: (q.includes('issue tracker') ? 9 : 0) +
             (q.includes('issue') && q.includes('team') ? 5 : 0),
      response: "**Issue Tracker** archetype (3 developers):\n\n🐛 High PR activity (avg 0.8/day) but below-median commits\n📋 Focus on coordination and bug triage\n⏱ 4.2 active hours/day\n⚠️ 1 Drop anomaly flagged (kiszk)\n\nLSTM predicts **2 commits tomorrow**. They open more issues than they close — a sign of dependency on others to resolve. If commit counts matter, pair them with a Code Committer.",
    },
    {
      score: ((q.includes('difference') || q.includes('differ')) && q.includes('archetype') ? 9 : 0) +
             ((q.includes('what') && q.includes('separates')) ? 7 : 0),
      response: "Key differences between archetypes:\n\n| Archetype | Commits | Reviews | PRs | LSTM Prediction |\n|---|---|---|---|---|\n| Team Lead | 8.4/day | 5.2/day | 3.1/day | 15 |\n| Code Committer | 5.1/day | 2.3/day | 1.8/day | 10 |\n| PR Reviewer | 2.2/day | 5.8/day | 1.2/day | 4 |\n| Issue Tracker | 1.1/day | 1.5/day | 0.8/day | 2 |\n| Silent Stalker | 0.2/day | 0.4/day | 0.5/day | 1 |\n\nThe LSTM produces lower predictions for teams with lower historical baselines.",
    },

    // ── DASHBOARD / VELOCITY ─────────────────────────────────────

    {
      score: (q.includes("what's up") || q.includes("whats up") ? 10 : 0) +
             (q.includes('how') && q.includes('team') && q.includes('doing') ? 5 : 0),
      response: "Here's the snapshot: Code review volume is up 18% this week — a leading indicator our LSTM model loves. We're forecasting 184 commits from 87 developers tomorrow (R² = 0.60). The team is in a strong engagement cycle. Anomaly-wise: 212 dev-days flagged over the tracking period, with HanumanthaRaoMandlem and panbingkun as top at-risk.",
    },
    {
      score: (q.includes('commit') && q.includes('tomorrow') ? 10 : 0) +
             (q.includes('predict') && q.includes('commit') && !q.includes('signal') ? 7 : 0),
      response: "The LSTM model predicts **184 commits for tomorrow** (full team). By archetype: Team Lead cluster → 15 commits, Code Committer → 10, PR Reviewer → 4, Issue Tracker → 2, Silent Stalker → 1. The aggregate 5-day trend (Mon 42 → Fri 95) is the primary driver of tomorrow's high forecast.",
    },
    {
      score: (q.includes('signal') || q.includes('feature') && q.includes('importan') ? 8 : 0) +
             (q.includes('top') && (q.includes('signal') || q.includes('predictor')) ? 9 : 0),
      response: "Top signals by SHAP importance (XGBoost explainer):\n\n🥇 reviews_given — strongest positive driver\n🥈 prs_opened — #2 leading indicator\n🥉 issues_closed — teams closing issues commit more next day\n📉 prs_merged — near-zero SHAP importance (lagging indicator)\n\nFor LSTM, these same features matter but the model also weighs their temporal *order* across the 5-day window.",
    },
    {
      score: (q.includes('recommend') || q.includes('improve') || q.includes('should') ? 7 : 0) +
             (q.includes('tip') || q.includes('advice') ? 6 : 0),
      response: "Data-driven recommendations:\n\n1. **Increase review volume** — reviews_given is the #1 LSTM feature. +5 reviews/day → ~+12 predicted commits\n2. **Close open issues before opening new ones** — issues_closed has the 3rd highest SHAP weight\n3. **Monitor Silent Stalkers weekly** — 3 anomaly flags in 5 developers is a high rate\n4. **Watch Code Committer cluster** — 2 flagged anomalies (Drop + Spike) suggest instability\n5. **Keep PR queue below 50** — above that, merge debt starts slowing the whole team",
    },
  ];

  const best = all.reduce((a, b) => (b.score > a.score ? b : a));
  if (best.score > 0) return best.response;

  // Fallback by page
  const fallbacks: Record<ChatPage, string> = {
    dashboard: "I'm trained on your team's GitHub data and LSTM forecasts. Try: 'Commits tomorrow?', 'Who's at risk?', 'Top signals?', 'PR health?'",
    team:      "I know all about your 5 behavioral archetypes and anomaly data. Try: 'What's Team Lead?', 'Silent Stalker anomalies?', 'Archetype differences?'",
    ml:        "I can explain any ML model in this system. Try: 'Best model R²?', 'Why LSTM wins?', 'What is R²?', 'XGBoost vs LSTM?'",
  };
  return fallbacks[page];
}

// ── Component ───────────────────────────────────────────────────
const Chatbot = ({ page = 'dashboard' }: Props) => {
  const config = PAGE_CONFIG[page];

  const [messages, setMessages] = useState<Message[]>([
    { id: '1', type: 'bot', content: config.greeting },
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Reset conversation when page changes
  useEffect(() => {
    setMessages([{ id: '1', type: 'bot', content: config.greeting }]);
    setInputValue('');
  }, [page]);

  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [messages, isTyping]);

  const handleSend = (text?: string) => {
    const content = text ?? inputValue;
    if (!content.trim()) return;

    const userMessage: Message = { id: Date.now().toString(), type: 'user', content };
    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsTyping(true);

    setTimeout(() => {
      const response = getResponse(content.toLowerCase(), page);
      setMessages(prev => [...prev, { id: (Date.now() + 1).toString(), type: 'bot', content: response }]);
      setIsTyping(false);
    }, 1200 + Math.random() * 600);
  };

  return (
    <div className="chatbot-wrapper fade-in delay-2">
      <div className="glass-panel chatbot-container">

        <div className="chatbot-header">
          <div className="bot-avatar-container">
            <div className="bot-avatar pulse-avatar">
              <Bot size={20} color="var(--primary-light)" />
            </div>
            <div className="bot-status">
              <h3 style={{ color: config.accentColor }}>{config.label}</h3>
              <span className="status-text">
                <span className="status-dot" style={{ background: config.accentColor }} />
                Online
              </span>
            </div>
          </div>
          <button className="icon-btn">
            <Sparkles size={18} className="sparkle-icon" />
          </button>
        </div>

        <div className="chatbot-messages" ref={scrollRef}>
          {messages.map((msg) => (
            <div key={msg.id} className={`message-row ${msg.type === 'user' ? 'user-row' : 'bot-row'}`}>
              {msg.type === 'bot' && (
                <div className="message-avatar bot-avatar-small">
                  <Bot size={14} color="#fff" />
                </div>
              )}
              <div className={`message-bubble ${msg.type === 'user' ? 'user-bubble' : 'bot-bubble'} slide-up`}>
                {msg.content.split('\n').map((line, i) => (
                  <p key={i} style={{ margin: i === 0 ? 0 : '0.4rem 0 0' }}>{line}</p>
                ))}
              </div>
              {msg.type === 'user' && (
                <div className="message-avatar user-avatar-small">
                  <User size={14} color="#fff" />
                </div>
              )}
            </div>
          ))}
          {isTyping && (
            <div className="message-row bot-row slide-up">
              <div className="message-avatar bot-avatar-small">
                <Bot size={14} color="#fff" />
              </div>
              <div className="message-bubble typing-bubble">
                <div className="typing-dot delay-1" />
                <div className="typing-dot delay-2" />
                <div className="typing-dot delay-3" />
              </div>
            </div>
          )}
        </div>

        {/* Suggestion chips */}
        <div style={{ padding: '0.5rem 1rem 0.25rem', display: 'flex', flexWrap: 'wrap', gap: '0.4rem', borderTop: '1px solid rgba(132,147,150,0.08)' }}>
          {config.chips.map(chip => (
            <button key={chip}
              onClick={() => handleSend(chip)}
              style={{
                background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(132,147,150,0.12)',
                color: 'var(--text-muted)', fontSize: '0.68rem', padding: '0.25rem 0.7rem',
                borderRadius: '999px', cursor: 'pointer', transition: 'all 0.2s', fontFamily: 'inherit',
              }}
              onMouseEnter={e => {
                const b = e.currentTarget;
                b.style.background = `rgba(${hexA(config.accentColor)}, 0.12)`;
                b.style.color = config.accentColor;
                b.style.borderColor = config.accentColor;
              }}
              onMouseLeave={e => {
                const b = e.currentTarget;
                b.style.background = 'rgba(255,255,255,0.04)';
                b.style.color = 'var(--text-muted)';
                b.style.borderColor = 'rgba(132,147,150,0.12)';
              }}
            >{chip}</button>
          ))}
        </div>

        <div className="chat-input-row">
          <input
            type="text"
            placeholder={`Ask about ${page === 'team' ? 'teams & anomalies' : page === 'ml' ? 'models & R²' : 'the team'}...`}
            value={inputValue}
            onChange={e => setInputValue(e.target.value)}
            onKeyPress={e => { if (e.key === 'Enter') handleSend(); }}
            className="chat-input-box"
          />
          <button onClick={() => handleSend()} disabled={!inputValue.trim()} className="chat-send-btn">
            <Send size={20} />
          </button>
        </div>
      </div>
    </div>
  );
};

function hexA(hex: string) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `${r}, ${g}, ${b}`;
}

export default Chatbot;
