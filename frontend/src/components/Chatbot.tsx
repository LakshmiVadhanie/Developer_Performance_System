import React, { useState, useRef, useEffect } from 'react';
import { Send, Sparkles, Bot, User } from 'lucide-react';
import './Chatbot.css';

interface Message {
  id: string;
  type: 'user' | 'bot';
  content: string;
}

const Chatbot = () => {
  const [messages, setMessages] = useState<Message[]>([
    { id: '1', type: 'bot', content: 'Hello! I am your Developer Intelligence assistant. How can I help you analyze the team\'s performance today?' }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, isTyping]);

  const handleSend = () => {
    if (!inputValue.trim()) return;

    const userMessage: Message = { id: Date.now().toString(), type: 'user', content: inputValue };
    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsTyping(true);

    // Smart keyword-matching response engine
    setTimeout(() => {
      const q = userMessage.content.toLowerCase();

      // Every intent has weighted keywords. Highest total score wins.
      const intents: { score: number; response: string }[] = [
        {
          score: (
            (q.includes("what's up") || q.includes("whats up") ? 10 : 0) +
            (q.includes('snapshot') ? 5 : 0) +
            (q.includes('how') && q.includes('team') && q.includes('doing') ? 5 : 0) +
            ((q === 'what is up' || q === "what's up" || q === "whats up") ? 8 : 0)
          ),
          response: "Here's the snapshot: Code review volume is up 18% this week — a leading indicator our LSTM model loves. We're forecasting 184 commits from 87 developers tomorrow, with an R² confidence of 0.58. The team is in a strong engagement cycle right now."
        },
        {
          // signals / features / what drives predictions
          score: (
            (q.includes('signal') ? 6 : 0) +
            (q.includes('driv') ? 6 : 0) +
            (q.includes('feature') && !q.includes('feature dev') ? 5 : 0) +
            (q.includes('indicator') ? 5 : 0) +
            (q.includes('importan') ? 5 : 0) +
            (q.includes('what') && q.includes('predict') && !q.includes('how many') ? 3 : 0)
          ),
          response: "The top signals driving commit predictions are: (1) prs_opened — the strongest positive indicator, (2) issues_closed — teams closing issues tend to commit more the next day, and (3) reviews_given — more reviews correlates with higher output. Interestingly, prs_merged has near-zero importance — merging isn't a leading indicator."
        },
        {
          // HISTORICAL commits — last N days / past week / recent
          score: (
            (q.includes('last') && q.includes('commit') ? 10 : 0) +
            (q.includes('past') && q.includes('commit') ? 10 : 0) +
            ((q.includes('5 day') || q.includes('five day') || q.includes('5day')) ? 8 : 0) +
            ((q.includes('3 day') || q.includes('three day') || q.includes('7 day') || q.includes('week')) && q.includes('commit') ? 6 : 0) +
            (q.includes('how many') && q.includes('commit') && (q.includes('last') || q.includes('past') || q.includes('day') || q.includes('week')) ? 8 : 0)
          ),
          response: "Here's the commit breakdown for the last 5 days: Mon 42 · Tue 67 · Wed 51 · Thu 88 · Fri 95. Total = 343 commits over 5 days, averaging 68.6/day. Thursday–Friday is the team's peak shipping window. The LSTM model uses this 5-day rolling window as part of its forecast — the upward trend is why it's predicting 184 tomorrow."
        },
        {
          // FORECAST commits — tomorrow / predicted / how many tomorrow
          score: (
            (q.includes('commit') && q.includes('tomorrow') ? 10 : 0) +
            (q.includes('commit') && q.includes('forecast') && !q.includes('signal') ? 8 : 0) +
            (q.includes('predict') && q.includes('commit') && !q.includes('signal') && !q.includes('driv') && !q.includes('last') && !q.includes('past') ? 7 : 0) +
            (q.includes('how many') && q.includes('commit') && !q.includes('last') && !q.includes('past') && !q.includes('day') && !q.includes('week') ? 6 : 0)
          ),
          response: "The LSTM model is predicting 184 commits for tomorrow. That's based on 5-day rolling engagement signals including PR opens, issue closures, and review activity — all trending upward right now."
        },
        {
          // model accuracy / r2 / confidence
          score: (
            (q.includes('confident') || q.includes('confidence') ? 7 : 0) +
            (q.includes('r2') || q.includes('r²') || q.includes('r-squared') ? 8 : 0) +
            (q.includes('accura') ? 6 : 0) +
            (q.includes('model') && q.includes('good') ? 4 : 0)
          ),
          response: "The model's R² score is 0.58 on the test set, meaning it explains 58% of the variance in next-day commits. For a metric as noisy as developer output, that's strong. Linear regression only hits 0.44 on the same data."
        },
        {
          // pull requests / PR
          score: (
            (q.includes('open pr') || q.includes('open pull') ? 8 : 0) +
            ((q.includes('pull request') || q.includes('pr')) && !q.includes('review') ? 6 : 0) +
            (q.includes('merge') ? 4 : 0) +
            (q.includes('how many') && q.includes('pr') ? 5 : 0)
          ),
          response: "There are currently 42 active PRs across the team — down 5.2% from last week. Usually a dip means the team is merging faster than they're opening, which is healthy. Watch if it drops below 30 — that could signal a slowdown in feature development."
        },
        {
          // reviews
          score: (
            (q.includes('review') ? 6 : 0) +
            (q.includes('code review') ? 5 : 0) +
            (q.includes('reviewing') ? 5 : 0)
          ),
          response: "The team has given 95 code reviews this week — up a strong 18.1%! Reviews are one of the best leading indicators for future commits. When engineers review more, they ship more. This surge suggests a productive sprint cycle is coming."
        },
        {
          // developer count / active devs
          score: (
            ((q.includes('how many') && q.includes('dev')) ? 8 : 0) +
            (q.includes('active developer') || q.includes('active dev') ? 8 : 0) +
            ((q.includes('team') && q.includes('size')) ? 6 : 0) +
            (q.includes('developer') && !q.includes('active') ? 4 : 0)
          ),
          response: "There are 87 active developers tracked this sprint — a 2.4% increase week over week. The top contributors account for nearly 60% of commit volume, which is a classic Pareto pattern. Want me to flag any outliers?"
        },
        {
          // burnout / stress
          score: (
            (q.includes('burnout') || q.includes('burn out') ? 8 : 0) +
            (q.includes('stress') ? 7 : 0) +
            (q.includes('overload') || q.includes('overwhelm') ? 7 : 0) +
            (q.includes('overtime') || q.includes('weekend') ? 5 : 0)
          ),
          response: "Burnout risk signals are mixed. Weekend commits are healthy — only 5–8/day vs 25+ on weekdays. But the spike to 32 commits on Fridays hints at end-of-sprint crunch culture. I'd recommend monitoring this pattern — if it persists 3+ weeks, it's worth a team conversation."
        },
        {
          // bottleneck / slowdown
          score: (
            (q.includes('bottleneck') ? 8 : 0) +
            (q.includes('slow') && !q.includes('slowly') ? 6 : 0) +
            (q.includes('delay') || q.includes('block') ? 6 : 0) +
            (q.includes('slowing down') ? 7 : 0)
          ),
          response: "The biggest bottleneck I see is PR review turnaround. With 42 open PRs and reviews slipping, there's likely a queue forming. If this continues, expect commit delays in 3–5 days. Pairing sessions or dedicated review blocks could fix it fast."
        },
        {
          // recommendations / improve
          score: (
            (q.includes('recommend') ? 7 : 0) +
            (q.includes('improve') ? 7 : 0) +
            (q.includes('should') && q.includes('focus') ? 7 : 0) +
            (q.includes('what') && q.includes('do') && q.includes('team') ? 5 : 0) +
            (q.includes('tip') || q.includes('advice') ? 6 : 0)
          ),
          response: "My top 3 data-driven recommendations: (1) Increase review volume — it's your strongest leading indicator. (2) Close open issues before creating new ones — issue closure has the highest positive weight. (3) Keep PR queue below 50 to avoid merge debt. Doing all three should push predicted commits north of 200."
        },
        {
          // manager / leadership
          score: (
            (q.includes('manager') ? 7 : 0) +
            (q.includes('leadership') || q.includes('leader') ? 7 : 0) +
            (q.includes('worried') || q.includes('concern') ? 6 : 0) +
            (q.includes('report') && q.includes('team') ? 5 : 0)
          ),
          response: "Tell your manager the team is in good shape, but flag two things: PR velocity dipped 5.2%, and Friday commit spikes hint at crunch culture. Neither is alarming yet, but both are worth bringing to 1:1s. The good news? Reviews are up 18% — which almost always precedes a productive week."
        },
        {
          // explain simply / non-technical
          score: (
            (q.includes('explain') && (q.includes('simple') || q.includes('plain') || q.includes('layman')) ? 8 : 0) +
            (q.includes('non-technical') || q.includes('non technical') ? 8 : 0) +
            (q.includes('like i') ? 6 : 0) +
            (q.includes('english') ? 5 : 0)
          ),
          response: "Here's the plain English version: We watch how often developers open PRs, close issues, and review each other's code. When those are high, our AI predicts more code will ship tomorrow. Right now the signals are strong — so expect a busy, productive day. Think of it like a sports warm-up; a good warm-up means a great game."
        },
        {
          // weekly summary / status
          score: (
            (q.includes('summary') ? 7 : 0) +
            (q.includes('this week') || q.includes('weekly') ? 7 : 0) +
            (q.includes('status') && !q.includes('team') ? 6 : 0) +
            (q.includes('overview') ? 6 : 0)
          ),
          response: "Weekly summary: 🟢 Reviews up 18.1% | 🔴 Open PRs down 5.2% | 🟢 Active devs up 2.4%. The LSTM model predicts 184 commits for tomorrow. Net assessment: the team is engaged and productive. Engagement is leading output — not lagging it."
        },
        {
          // build frequency / heatmap / builds
          score: (
            (q.includes('build frequen') ? 10 : 0) +
            (q.includes('heatmap') ? 10 : 0) +
            (q.includes('heat map') ? 9 : 0) +
            (q.includes('build') && q.includes('how often') ? 8 : 0) +
            (q.includes('build') && !q.includes('deploy') && !q.includes('fail') ? 4 : 0) +
            (q.includes('what is build') ? 8 : 0)
          ),
          response: "Build Frequency tracks how many CI/CD builds are triggered each day across the week, visualized as a heatmap. Darker cells = more builds. High build frequency generally means the team is actively shipping and deploying — it's a health signal for iteration speed. This week's peak was Thursday with 14 builds."
        },
        {
          // flow state / focus / deep work
          score: (
            (q.includes('flow state') || q.includes('flow') ? 8 : 0) +
            (q.includes('focus') && !q.includes('focus on') ? 7 : 0) +
            (q.includes('deep work') || q.includes('interruption') ? 8 : 0) +
            (q.includes('context switch') ? 8 : 0) +
            (q.includes('72%') ? 6 : 0)
          ),
          response: "Flow State Efficiency measures the % of coding time spent in uninterrupted deep work sessions (≥25 minutes). Currently at 72%, which is solid — industry average is around 60%. The team averages 4.2 hours of deep work daily, but has 9 context switches per day, which is slightly high. Reducing meeting fragmentation could push this above 80%."
        },
        {
          // deployment / build status / CI/CD
          score: (
            (q.includes('deploy') ? 7 : 0) +
            (q.includes('build fail') || q.includes('failed build') ? 9 : 0) +
            (q.includes('staging') || q.includes('production') ? 6 : 0) +
            (q.includes('ci') || q.includes('cd') || q.includes('pipeline') ? 6 : 0) +
            (q.includes('deployment log') || q.includes('build log') ? 9 : 0)
          ),
          response: "Current deployment status: ✅ Production on v2.4.1-rc (1m 42s) | ✅ Staging fix/auth-leak (58s) | ❌ Production v2.4.0-stable FAILED (null pointer exception, 2h ago) | 🔄 Preview feat/dark-mode in progress. The staging failure 2 hours ago should be investigated — it was only 12s before failing, suggesting an early crash."
        },
        {
          // velocity / coding speed
          score: (
            (q.includes('velocit') ? 8 : 0) +
            (q.includes('speed') && q.includes('code') ? 6 : 0) +
            (q.includes('output') && q.includes('team') ? 5 : 0) +
            (q.includes('daily') && (q.includes('commit') || q.includes('output')) ? 6 : 0)
          ),
          response: "Coding velocity is up +24% week-over-week. The team peaked Friday with 95 commits and 31 reviews — a strong finishing sprint. Monday–Wednesday tends to be slower (42–67 commits), which is typical ramp-up behavior. Thursday–Friday is when the team really ships. LSTM uses this daily pattern as a cyclical feature."
        },
        {
          // git activity / commits / recent activity
          score: (
            (q.includes('git') && q.includes('activit') ? 9 : 0) +
            (q.includes('recent commit') || q.includes('latest commit') ? 8 : 0) +
            (q.includes('what happened') || q.includes('what did') ? 5 : 0) +
            (q.includes('who committed') || q.includes('who pushed') ? 7 : 0)
          ),
          response: "Recent git activity: k.morris just pushed feat/lstm-fix (fixed Sigmoid activation, 12m ago), s.chen merged PR #842 (auth middleware, 48m ago), ci/cd flagged a build failure on staging-v4 (null pointer, 2h ago), and l.vadhanie pushed the new dashboard design 3h ago. All contributors are active — no inactive streaks over 24h."
        },
        {
          // what is LSTM / how does it work
          score: (
            (q.includes('what is lstm') || q.includes('how does lstm') ? 10 : 0) +
            (q.includes('lstm') && (q.includes('work') || q.includes('explain')) ? 8 : 0) +
            (q.includes('machine learning') || q.includes('model') && q.includes('how') ? 5 : 0) +
            (q.includes('forecast') && q.includes('how') ? 5 : 0)
          ),
          response: "LSTM stands for Long Short-Term Memory — it's a type of neural network that's great at learning patterns in sequential data over time. We feed it 5-day rolling window data (PRs opened, issues closed, reviews given, commit count) and it predicts next-day commit volume. Unlike simple regression, LSTM remembers whether Monday was slow because of a holiday or a burnout signal — context matters."
        },
        {
          // PR health / pr status
          score: (
            (q.includes('pr health') || q.includes('pr status') ? 10 : 0) +
            (q.includes('pull request') && (q.includes('health') || q.includes('status')) ? 9 : 0) +
            (q.includes('pr') && q.includes('cycle') ? 7 : 0) +
            (q.includes('merge time') || q.includes('review time') ? 6 : 0)
          ),
          response: "PR health is moderate. Average cycle time is 18.4 hours from first commit to merge — broken down as: 6.2h review phase, 4.1h testing/QA, 8.1h idle/waiting. The idle time is the biggest drag — PRs are sitting unattended before reviewers pick them up. If you can cut idle time to 4h, your total cycle drops to ~14h, which is excellent."
        },
      ];

      const best = intents.reduce((a, b) => (b.score > a.score ? b : a));
      let botResponse = best.score > 0
        ? best.response
        : "I'm trained on your team's GitHub activity and LSTM forecasting outputs. Try asking about commit predictions, PR health, review trends, developer counts, bottlenecks, or recommendations!";

      // --- NEW ML PIPELINE RESPONSES ---
      if (best.score > 0 && (q.includes('cluster') || q.includes('archetype') || q.includes('k-mean'))) {
        botResponse = "We use **K-Means Clustering** to automatically group developer metrics into behavioral 'archetypes' (like Team Lead, Reviewer, or Code Committer). It helps managers understand team topology.\n\n*Dashboard note: Try typing 'archetypes' in the search bar!*";
      } else if (best.score > 0 && (q.includes('burnout') || q.includes('isolation') || q.includes('anomaly'))) {
        botResponse = "Our **Isolation Forest** model tracks individual developer baselines. If a high performer suddenly drops output by >2 standard deviations, it throws a burnout alert. It acts as an early warning system.";
      } else if (best.score > 0 && (q.includes('bandit') || q.includes('rl') || q.includes('optimizer') || q.includes('sprint config'))) {
        botResponse = "We treat sprint configurations (like review caps) as arms in a slot machine. The **RL Thompson Sampling Optimizer** continuously learns which combination yields the highest team velocity, balancing exploration with exploitation.";
      } else if (best.score > 0 && (q.includes('vae') || q.includes('latent') || q.includes('team health'))) {
        botResponse = "The **Variational Autoencoder (VAE)** learns what a 'normal' week looks like for your team across all metrics. If a sprint's reconstruction error spikes, it flags a structural team-health anomaly — something Isolation Forest might miss on an individual level.";
      } else if (best.score > 0 && q.includes('shap')) {
        botResponse = "While absolute predictions come from our Transformer and LSTM, we map **XGBoost predictions with SHAP values** to provide explainability. They show exactly *why* a specific forecast was made (e.g. 'high PR volume pushed the prediction up 12%').";
      }

      const newBotMessage: Message = { id: (Date.now()+1).toString(), type: 'bot', content: botResponse };
      setMessages(prev => [...prev, newBotMessage]);
      setIsTyping(false);
    }, 1500 + Math.random() * 500);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') handleSend();
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
              <h3>DevInsight AI</h3>
              <span className="status-text">
                <span className="status-dot"></span> Online
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
                <p>{msg.content}</p>
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
                <div className="typing-dot delay-1"></div>
                <div className="typing-dot delay-2"></div>
                <div className="typing-dot delay-3"></div>
              </div>
            </div>
          )}
        </div>

        {/* Quick suggestion chips */}
        <div style={{padding:'0.5rem 1rem 0.25rem', display:'flex', flexWrap:'wrap', gap:'0.4rem', borderTop:'1px solid rgba(132,147,150,0.08)'}}>
          {["What's up?","Commits tomorrow?","PR health?","Burnout risk?","Top signals?","Recommendations"].map(chip => (
            <button key={chip}
              onClick={() => setInputValue(chip)}
              style={{
                background:'rgba(255,255,255,0.04)', border:'1px solid rgba(132,147,150,0.12)',
                color:'var(--text-muted)', fontSize:'0.68rem', padding:'0.25rem 0.7rem',
                borderRadius:'999px', cursor:'pointer', transition:'all 0.2s', fontFamily:'inherit',
              }}
              onMouseEnter={e => {
                const b = e.currentTarget;
                b.style.background='rgba(0,229,255,0.1)';
                b.style.color='#00e5ff';
                b.style.borderColor='#00e5ff';
              }}
              onMouseLeave={e => {
                const b = e.currentTarget;
                b.style.background='rgba(255,255,255,0.04)';
                b.style.color='var(--text-muted)';
                b.style.borderColor='rgba(132,147,150,0.12)';
              }}
            >{chip}</button>
          ))}
        </div>

        <div className="chat-input-row">
          <input 
            type="text" 
            placeholder="Ask anything about the team..." 
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyPress={handleKeyPress}
            className="chat-input-box"
          />
          <button onClick={handleSend} disabled={!inputValue.trim()} className="chat-send-btn">
            <Send size={20} />
          </button>
        </div>
      </div>
    </div>
  );
};

export default Chatbot;
