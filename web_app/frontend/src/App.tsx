import { useEffect, useMemo, useRef, useState } from 'react'
import axios from 'axios'

import './App.css'
import {
  analyzePrompt,
  getHealth,
  type AnalysisResult,
  type ApiErrorDetail,
  type HealthResponse,
  type NetworkControls,
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
import { NetworkPanel } from './components/network/NetworkPanel'

type DetailTab = 'attention' | 'activations' | 'logits' | 'network'

const EXAMPLE_PROMPTS = [
  { label: 'Binary', value: '02000000 + 01000000 =' },
  { label: 'Three input', value: '03000000 + 02000000 + 01000000 =' },
  { label: 'Parentheses', value: '( 03000000 + 02000000 ) - 01000000 =' },
  { label: 'Negative input', value: '(-30000000) + 01000000 =' },
] as const

const DEFAULT_PROMPT = EXAMPLE_PROMPTS[0].value

const DEFAULT_NETWORK_CONTROLS: NetworkControls = {
  mlp_threshold: 0,
  top_k: 5,
  top_neurons: 8,
  selected_token_index: null,
}

function App() {
  const [prompt, setPrompt] = useState<string>(DEFAULT_PROMPT)
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [loadingNetwork, setLoadingNetwork] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [networkError, setNetworkError] = useState<string | null>(null)
  const [selectedLayer, setSelectedLayer] = useState(0)
  const [activeDetailTab, setActiveDetailTab] = useState<DetailTab>('attention')
  const [networkControls, setNetworkControls] = useState<NetworkControls>(DEFAULT_NETWORK_CONTROLS)
  const abortRef = useRef<AbortController | null>(null)

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

    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller

    setLoading(true)
    setLoadingNetwork(false)
    setError(null)

    try {
      const includeNetwork = activeDetailTab === 'network'
      const analysis = await analyzePrompt(
        cleanedPrompt,
        {
          ...(includeNetwork ? networkControls : {}),
          include_network: includeNetwork,
        },
        controller.signal,
      )
      if (controller.signal.aborted) return
      setPrompt(cleanedPrompt)
      setResult(analysis)
      setNetworkError(null)
      setSelectedLayer((current) =>
        Math.min(current, Math.max(analysis.config.n_layers - 1, 0)),
      )
      void refreshHealth()
    } catch (caughtError) {
      if (!controller.signal.aborted) {
        setError(getErrorMessage(caughtError))
      }
    } finally {
      if (!controller.signal.aborted) {
        setLoading(false)
      }
    }
  }

  const requestNetworkAnalysis = async (nextControls = networkControls) => {
    const cleanedPrompt = prompt.trim()
    if (!cleanedPrompt || loading || loadingNetwork) {
      return
    }

    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller

    setLoading(false)
    setLoadingNetwork(true)
    setNetworkError(null)
    setNetworkControls(nextControls)

    try {
      const analysis = await analyzePrompt(
        cleanedPrompt,
        {
          ...nextControls,
          include_network: true,
        },
        controller.signal,
      )
      if (controller.signal.aborted) return
      setResult(analysis)
      setSelectedLayer((current) =>
        Math.min(current, Math.max(analysis.config.n_layers - 1, 0)),
      )
    } catch (caughtError) {
      if (!controller.signal.aborted) {
        setNetworkError(getErrorMessage(caughtError))
      }
    } finally {
      if (!controller.signal.aborted) {
        setLoadingNetwork(false)
      }
    }
  }

  const openDetailTab = (tab: DetailTab) => {
    setActiveDetailTab(tab)
    if (tab === 'network' && result && !result.network) {
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
                onClick={() => openDetailTab('attention')}
              >
                Attention
              </button>
              <button
                type="button"
                className={activeDetailTab === 'activations' ? 'is-active' : ''}
                onClick={() => openDetailTab('activations')}
              >
                Activations
              </button>
              <button
                type="button"
                className={activeDetailTab === 'logits' ? 'is-active' : ''}
                onClick={() => openDetailTab('logits')}
              >
                Logits
              </button>
              <button
                type="button"
                className={activeDetailTab === 'network' ? 'is-active' : ''}
                onClick={() => openDetailTab('network')}
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
