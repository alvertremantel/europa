import { useEffect, useMemo, useState } from 'react'
import axios from 'axios'

import './App.css'
import {
  analyzePrompt,
  getHealth,
  type AnalysisResult,
  type ApiErrorDetail,
  type HealthResponse,
} from './api'
import { ActivationPanel } from './components/ActivationPanel'
import { AttentionPanel } from './components/AttentionPanel'
import { ErrorNotice } from './components/ErrorNotice'
import { LogitPanel } from './components/LogitPanel'
import { ModelStatusCard } from './components/ModelStatusCard'
import { OverviewMetrics } from './components/OverviewMetrics'
import { PromptBar } from './components/PromptBar'
import { SkeletonDashboard } from './components/SkeletonDashboard'
import { TokenPredictionTable } from './components/TokenPredictionTable'

type DetailTab = 'attention' | 'activations' | 'logits'

const EXAMPLE_PROMPTS = [
  { label: 'Binary', value: '02000000 + 01000000 =' },
  { label: 'Three input', value: '03000000 + 02000000 + 01000000 =' },
  { label: 'Parentheses', value: '( 03000000 + 02000000 ) - 01000000 =' },
  { label: 'Negative input', value: '(-30000000) + 01000000 =' },
] as const

const DEFAULT_PROMPT = EXAMPLE_PROMPTS[0].value

function App() {
  const [prompt, setPrompt] = useState<string>(DEFAULT_PROMPT)
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [selectedLayer, setSelectedLayer] = useState(0)
  const [activeDetailTab, setActiveDetailTab] = useState<DetailTab>('attention')

  async function refreshHealth() {
    try {
      const nextHealth = await getHealth()
      setHealth(nextHealth)
    } catch {
      setHealth(null)
    }
  }

  useEffect(() => {
    let cancelled = false

    void getHealth()
      .then((nextHealth) => {
        if (!cancelled) {
          setHealth(nextHealth)
        }
      })
      .catch(() => {
        if (!cancelled) {
          setHealth(null)
        }
      })

    return () => {
      cancelled = true
    }
  }, [])

  const answerPrediction = useMemo(() => {
    if (!result) {
      return null
    }
    return result.top_predictions[result.answer_position] ?? null
  }, [result])

  const submitPrompt = async (nextPrompt = prompt) => {
    const cleanedPrompt = nextPrompt.trim()
    if (!cleanedPrompt) {
      setError('Enter an arithmetic prompt before analyzing.')
      return
    }

    setLoading(true)
    setError(null)

    try {
      const analysis = await analyzePrompt(cleanedPrompt)
      setPrompt(cleanedPrompt)
      setResult(analysis)
      setSelectedLayer((current) =>
        Math.min(current, Math.max(analysis.config.n_layers - 1, 0)),
      )
      void refreshHealth()
    } catch (caughtError) {
      setError(getErrorMessage(caughtError))
    } finally {
      setLoading(false)
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
                onClick={() => setActiveDetailTab('attention')}
              >
                Attention
              </button>
              <button
                type="button"
                className={activeDetailTab === 'activations' ? 'is-active' : ''}
                onClick={() => setActiveDetailTab('activations')}
              >
                Activations
              </button>
              <button
                type="button"
                className={activeDetailTab === 'logits' ? 'is-active' : ''}
                onClick={() => setActiveDetailTab('logits')}
              >
                Logits
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

function getErrorMessage(error: unknown): string {
  if (axios.isAxiosError<ApiErrorDetail>(error)) {
    return error.response?.data?.detail ?? 'Failed to analyze prompt. Check that the backend is running.'
  }
  if (error instanceof Error) {
    return error.message
  }
  return 'Failed to analyze prompt.'
}

export default App
