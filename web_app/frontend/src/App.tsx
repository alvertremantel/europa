import { useState } from 'react';
import axios from 'axios';
import { AttentionHeads, TextNeuronActivations } from 'circuitsvis';
import { Play, Loader2, BarChart3, Cpu, MessageSquare } from 'lucide-react';

interface AnalysisResult {
  tokens: string[];
  attention: number[][][][]; // [layer, head, query, key]
  activations: number[][][]; // [tokens, layer, neuron]
  logits: number[][]; // [pos, vocab]
  top_predictions: { token: string; confidence: number }[];
  config: {
    n_layers: number;
    n_heads: number;
    d_model: number;
  };
}

function App() {
  const [prompt, setPrompt] = useState('02000000 + 01000000 =');
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<'attention' | 'activations' | 'logits'>('attention');
  const [selectedLayer, setSelectedLayer] = useState(0);

  const analyze = async () => {
    setLoading(true);
    try {
      const response = await axios.post('/api/analyze', { prompt });
      setResult(response.data);
    } catch (error) {
      console.error('Error analyzing prompt:', error);
      alert('Failed to analyze prompt. Make sure the backend is running.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <header>
        <h1>Europa ALM-IS Explorer</h1>
        <p>Europa Arithmetic Language Model Interpretability Suite</p>
      </header>

      <main>
        <section className="input-section">
          <div className="input-group">
            <input
              type="text"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Enter arithmetic prompt (e.g. 02000000 + 01000000 =)"
              onKeyDown={(e) => e.key === 'Enter' && analyze()}
            />
            <button onClick={analyze} disabled={loading}>
              {loading ? <Loader2 className="spin" /> : <Play size={18} />}
              <span>Analyze</span>
            </button>
          </div>
        </section>

        {result && (
          <div className="results-container">
            <nav className="tabs">
              <button 
                className={activeTab === 'attention' ? 'active' : ''} 
                onClick={() => setActiveTab('attention')}
              >
                <MessageSquare size={16} /> Attention
              </button>
              <button 
                className={activeTab === 'activations' ? 'active' : ''} 
                onClick={() => setActiveTab('activations')}
              >
                <Cpu size={16} /> Activations
              </button>
              <button 
                className={activeTab === 'logits' ? 'active' : ''} 
                onClick={() => setActiveTab('logits')}
              >
                <BarChart3 size={16} /> Logit Lens
              </button>
            </nav>

            <div className="tab-content">
              {activeTab === 'attention' && (
                <div className="viz-box">
                  <div className="controls">
                    <label>Layer:</label>
                    <select value={selectedLayer} onChange={(e) => setSelectedLayer(parseInt(e.target.value))}>
                      {Array.from({ length: result.config.n_layers }).map((_, i) => (
                        <option key={i} value={i}>Layer {i}</option>
                      ))}
                    </select>
                  </div>
                  <AttentionHeads 
                    tokens={result.tokens} 
                    attention={result.attention[selectedLayer]} 
                  />
                </div>
              )}

              {activeTab === 'activations' && (
                <div className="viz-box">
                  <TextNeuronActivations 
                    tokens={result.tokens} 
                    activations={result.activations} 
                  />
                </div>
              )}

              {activeTab === 'logits' && (
                <div className="viz-box">
                  <h3>Predictions evolving across sequence</h3>
                  <div className="prediction-grid">
                    {result.tokens.map((token, i) => (
                      <div key={i} className="prediction-item">
                        <span className="token-label">{token}</span>
                        <div className="pred-info">
                          <span className="pred-token">→ {result.top_predictions[i].token}</span>
                          <span className="confidence">{(result.top_predictions[i].confidence * 100).toFixed(1)}%</span>
                        </div>
                        <div className="conf-bar-bg">
                          <div 
                            className="conf-bar-fill" 
                            style={{ width: `${result.top_predictions[i].confidence * 100}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </main>

      <style>{`
        :root {
          --primary: #3b82f6;
          --primary-hover: #2563eb;
          --bg: #f8fafc;
          --card: #ffffff;
          --text: #1e293b;
          --text-light: #64748b;
          --border: #e2e8f0;
        }

        body {
          margin: 0;
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
          background-color: var(--bg);
          color: var(--text);
        }

        .app-container {
          max-width: 1200px;
          margin: 0 auto;
          padding: 2rem;
        }

        header {
          margin-bottom: 2rem;
          text-align: center;
        }

        header h1 {
          margin: 0;
          font-size: 2.5rem;
          color: var(--primary);
        }

        header p {
          color: var(--text-light);
          margin-top: 0.5rem;
        }

        .input-section {
          background: var(--card);
          padding: 1.5rem;
          border-radius: 12px;
          box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
          margin-bottom: 2rem;
        }

        .input-group {
          display: flex;
          gap: 1rem;
        }

        input {
          flex: 1;
          padding: 0.75rem 1rem;
          border: 1px solid var(--border);
          border-radius: 8px;
          font-size: 1rem;
          outline: none;
          transition: border-color 0.2s;
        }

        input:focus {
          border-color: var(--primary);
        }

        button {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          padding: 0.75rem 1.5rem;
          background-color: var(--primary);
          color: white;
          border: none;
          border-radius: 8px;
          font-weight: 600;
          cursor: pointer;
          transition: background-color 0.2s;
        }

        button:hover:not(:disabled) {
          background-color: var(--primary-hover);
        }

        button:disabled {
          opacity: 0.7;
          cursor: not-allowed;
        }

        .spin {
          animation: spin 1s linear infinite;
        }

        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }

        .tabs {
          display: flex;
          gap: 0.5rem;
          margin-bottom: 1rem;
        }

        .tabs button {
          background: none;
          color: var(--text-light);
          padding: 0.5rem 1rem;
          border-bottom: 2px solid transparent;
          border-radius: 0;
        }

        .tabs button:hover {
          background: #f1f5f9;
        }

        .tabs button.active {
          color: var(--primary);
          border-bottom-color: var(--primary);
          background: none;
        }

        .viz-box {
          background: var(--card);
          padding: 1.5rem;
          border-radius: 12px;
          box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
          min-height: 500px;
        }

        .controls {
          margin-bottom: 1rem;
          display: flex;
          align-items: center;
          gap: 0.5rem;
        }

        select {
          padding: 0.25rem 0.5rem;
          border-radius: 4px;
          border: 1px solid var(--border);
        }

        .prediction-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
          gap: 1rem;
          margin-top: 1rem;
        }

        .prediction-item {
          padding: 1rem;
          background: #f8fafc;
          border-radius: 8px;
          border: 1px solid var(--border);
        }

        .token-label {
          font-weight: 700;
          color: var(--text-light);
          display: block;
          margin-bottom: 0.5rem;
        }

        .pred-info {
          display: flex;
          justify-content: space-between;
          margin-bottom: 0.25rem;
        }

        .pred-token {
          font-weight: 600;
          color: var(--primary);
        }

        .confidence {
          font-size: 0.875rem;
          color: var(--text-light);
        }

        .conf-bar-bg {
          height: 4px;
          background: #e2e8f0;
          border-radius: 2px;
          overflow: hidden;
        }

        .conf-bar-fill {
          height: 100%;
          background: var(--primary);
        }
      `}</style>
    </div>
  );
}

export default App;
