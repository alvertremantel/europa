import './App.css'
import { useAnalysisSession } from './hooks/useAnalysisSession'
import { EXAMPLE_PROMPTS } from './constants'

import { ActivationPanel } from './components/ActivationPanel'
import { AttentionPanel } from './components/AttentionPanel'
import { ErrorNotice } from './components/ErrorNotice'
import { LogitPanel } from './components/LogitPanel'
import { ModelStatusCard } from './components/ModelStatusCard'
import { OverviewMetrics } from './components/OverviewMetrics'
import { PromptBar } from './components/PromptBar'
import { SkeletonDashboard } from './components/SkeletonDashboard'
import { TokenPredictionTable } from './components/TokenPredictionTable'
import { NetworkPanel } from './components/network/NetworkPanel'

function App() {
  const {
    prompt,
    setPrompt,
    result,
    health,
    loading,
    loadingNetwork,
    error,
    networkError,
    selectedLayer,
    setSelectedLayer,
    activeDetailTab,
    networkControls,
    answerPrediction,
    submitPrompt,
    requestNetworkAnalysis,
    openDetailTab,
    setError,
  } = useAnalysisSession()

  const handleOpenDetailTab = (tab: typeof activeDetailTab) => {
    openDetailTab(tab)
    if (tab !== 'network' || !result) return
    if (!result.network) {
      void requestNetworkAnalysis()
    }
  }

  return (
    <div className="app-shell">
      <header className="hero">
        <div>
          <p className="hero__eyebrow">Europa ALM-IS</p>
          <h1>Mechanistic Interpretability Dashboard</h1>
          <p className="hero__lede">
            Compare prompt tokens, attention heads, residual activity, and next-token
            predictions in one responsive workspace.
          </p>
        </div>
        <ModelStatusCard health={health} result={result} loading={loading} />
      </header>

      <PromptBar
        prompt={prompt}
        loading={loading}
        examplePrompts={EXAMPLE_PROMPTS.map((example) => ({
          ...example,
          onSelect: () => {
            setPrompt(example.value)
            setError(null)
          },
        }))}
        onPromptChange={(value) => {
          setPrompt(value)
          if (error) {
            setError(null)
          }
        }}
        onSubmit={() => void submitPrompt()}
      />

      <ErrorNotice message={error} />

      {loading && !result ? <SkeletonDashboard /> : null}

      {result ? (
        <main className="dashboard">
          <OverviewMetrics result={result} answerPrediction={answerPrediction} />

          <div className="dashboard__primary">
            <TokenPredictionTable result={result} />

            <div className="detail-tab-strip" role="tablist" aria-label="Detail panels">
              <button
                type="button"
                className={activeDetailTab === 'attention' ? 'is-active' : ''}
                onClick={() => handleOpenDetailTab('attention')}
              >
                Attention
              </button>
              <button
                type="button"
                className={activeDetailTab === 'activations' ? 'is-active' : ''}
                onClick={() => handleOpenDetailTab('activations')}
              >
                Activations
              </button>
              <button
                type="button"
                className={activeDetailTab === 'logits' ? 'is-active' : ''}
                onClick={() => handleOpenDetailTab('logits')}
              >
                Logits
              </button>
              <button
                type="button"
                className={activeDetailTab === 'network' ? 'is-active' : ''}
                onClick={() => handleOpenDetailTab('network')}
              >
                Network
              </button>
            </div>

            <div className="dashboard__detail-grid">
              <div
                className={`detail-panel ${activeDetailTab === 'attention' ? 'detail-panel--active' : ''}`}
              >
                <AttentionPanel
                  result={result}
                  selectedLayer={selectedLayer}
                  onSelectedLayerChange={setSelectedLayer}
                />
              </div>

              <div
                className={`detail-panel ${activeDetailTab === 'activations' ? 'detail-panel--active' : ''}`}
              >
                <ActivationPanel result={result} />
              </div>

              <div
                className={`detail-panel detail-panel--full ${activeDetailTab === 'logits' ? 'detail-panel--active' : ''}`}
              >
                <LogitPanel result={result} />
              </div>

              <div
                className={`detail-panel detail-panel--full ${activeDetailTab === 'network' ? 'detail-panel--active' : ''}`}
              >
                <NetworkPanel
                  result={result}
                  controls={networkControls}
                  loading={loadingNetwork}
                  error={networkError}
                  onRequestNetwork={(controls) => void requestNetworkAnalysis(controls)}
                />
              </div>
            </div>
          </div>
        </main>
      ) : !loading ? (
        <section className="empty-state card">
          <h2>Analyze a prompt to populate the dashboard</h2>
          <p>
            Prompts should use reversed zero-padded arithmetic fields such as{' '}
            <code>02000000 + 01000000 =</code>. The backend appends <code>&lt;ans&gt;</code>{' '}
            automatically before running inference.
          </p>
        </section>
      ) : null}
    </div>
  )
}

export default App
