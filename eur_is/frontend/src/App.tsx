import './App.css'
import { useCallback, useEffect, useRef, useState } from 'react'
import { useAnalysisSession } from './hooks/useAnalysisSession'
import { EXAMPLE_PROMPTS } from './constants'
import { getPanelShortcutTarget, type DetailTab } from './keyboardShortcuts'
import { exportAnalysisDump } from './api'

import { ActivationPanel } from './components/ActivationPanel'
import { AttentionPanel } from './components/AttentionPanel'
import { ErrorNotice } from './components/ErrorNotice'
import { LogitPanel } from './components/LogitPanel'
import { ModelStatusCard } from './components/ModelStatusCard'
import { OverviewMetrics } from './components/OverviewMetrics'
import { PromptBriefingCard } from './components/PromptBriefingCard'
import { PromptBar } from './components/PromptBar'
import { SkeletonDashboard } from './components/SkeletonDashboard'
import { TokenPredictionTable } from './components/TokenPredictionTable'
import { NetworkPanel } from './components/network/NetworkPanel'

type DensityMode = 'comfortable' | 'compact'

const DENSITY_STORAGE_KEY = 'eur-is-density-mode'

function readDensityPreference(): DensityMode {
  if (typeof window === 'undefined') return 'compact'
  const stored = window.localStorage.getItem(DENSITY_STORAGE_KEY)
  return stored === 'comfortable' || stored === 'compact' ? stored : 'compact'
}

function isEditableTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false
  return (
    target.tagName === 'INPUT' ||
    target.tagName === 'TEXTAREA' ||
    target.tagName === 'SELECT' ||
    target.isContentEditable
  )
}

function App() {
  const promptInputRef = useRef<HTMLInputElement>(null)
  const [density, setDensity] = useState<DensityMode>(() => readDensityPreference())
  const [predictionCollapsed, setPredictionCollapsed] = useState(false)
  const [activationCollapsed, setActivationCollapsed] = useState(false)
  const [logitCollapsed, setLogitCollapsed] = useState(false)
  const [exporting, setExporting] = useState(false)
  const [exportError, setExportError] = useState<string | null>(null)
  const {
    prompt,
    setPrompt,
    result,
    health,
    loading,
    loadingNetwork,
    error,
    networkError,
    availableDetailTabs,
    selectedLayer,
    setSelectedLayer,
    activeDetailTab,
    networkControls,
    lastSubmittedPrompt,
    runtimeCapabilities,
    generatedAnswer,
    selectedAnswerTokenIndex,
    setSelectedAnswerTokenIndex,
    submitPrompt,
    requestNetworkAnalysis,
    openDetailTab,
    setError,
  } = useAnalysisSession()
  const matchTopRowHeights = predictionCollapsed && activationCollapsed && logitCollapsed

  const handleExportDump = useCallback(async () => {
    const exportPrompt = (lastSubmittedPrompt ?? '').trim()
    if (!exportPrompt) {
      setExportError('Run an analysis before exporting.')
      return
    }
    setExporting(true)
    setExportError(null)
    try {
      const { blob, filename } = await exportAnalysisDump(exportPrompt, {
        ...networkControls,
        output_format: 'zip',
      })
      const downloadUrl = URL.createObjectURL(blob)
      const link = document.createElement('a')
      const runtimeHint = result?.analysis_runtime ?? health?.analysis_runtime ?? 'runtime'
      const checkpointHint = (result?.checkpoint.path ?? health?.checkpoint.path ?? 'checkpoint')
        .split('/')
        .pop()
        ?.replace(/\.pt$/, '')
        ?? 'checkpoint'
      const fallbackName = `its-export-${new Date().toISOString().replace(/[:.]/g, '-')}-${checkpointHint}-${runtimeHint}.zip`
      link.href = downloadUrl
      link.download = filename ?? fallbackName
      document.body.append(link)
      link.click()
      link.remove()
      URL.revokeObjectURL(downloadUrl)
    } catch (caughtError: unknown) {
      if (caughtError instanceof Error) {
        setExportError(caughtError.message)
      } else {
        setExportError('Failed to export bundle.')
      }
    } finally {
      setExporting(false)
    }
  }, [health, lastSubmittedPrompt, networkControls, result])

  const handleOpenDetailTab = useCallback((tab: DetailTab) => {
    openDetailTab(tab)
    if (tab === 'network' && !runtimeCapabilities?.network_analysis) return
    if (tab !== 'network' || !result) return
    if (!result.network) {
      void requestNetworkAnalysis()
    }
  }, [openDetailTab, requestNetworkAnalysis, result, runtimeCapabilities?.network_analysis])

  useEffect(() => {
    window.localStorage.setItem(DENSITY_STORAGE_KEY, density)
  }, [density])

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.metaKey || event.ctrlKey || event.altKey || isEditableTarget(event.target)) {
        return
      }

      if (event.key === '/') {
        event.preventDefault()
        promptInputRef.current?.focus()
        promptInputRef.current?.select()
        return
      }

      if (event.key === '[' || event.key === ']') {
        event.preventDefault()
        const direction = event.key === '[' ? -1 : 1
        setSelectedLayer((current: number) => {
          const maxLayer = result ? Math.max(result.config.n_layers - 1, 0) : 0
          return Math.min(maxLayer, Math.max(0, current + direction))
        })
        return
      }

      const panelTarget = getPanelShortcutTarget(event.key)
      if (panelTarget) {
        event.preventDefault()
        if (panelTarget.tab) {
          handleOpenDetailTab(panelTarget.tab)
        }
        window.requestAnimationFrame(() => {
          document.getElementById(panelTarget.panelId)?.scrollIntoView({
            behavior: 'smooth',
            block: 'start',
          })
        })
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [handleOpenDetailTab, result, setSelectedLayer])

  return (
    <div className={`app-shell density-${density} ${loading ? 'is-loading' : ''}`}>
      <header className="hero">
        <div className="hero__intro">
          <p className="hero__eyebrow">Europa ATM-IS</p>
          <h1>Mechanistic Interpretability Dashboard</h1>
        </div>

        <div className="hero__main">

          <PromptBar
            inputRef={promptInputRef}
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

          <PromptBriefingCard result={result} />
        </div>
        <ModelStatusCard health={health} result={result} loading={loading}>
          <div className="status-card__extras">
            <label className="select-field">
              <span>Density</span>
              <select value={density} onChange={(event) => setDensity(event.target.value as DensityMode)}>
                <option value="compact">Compact</option>
                <option value="comfortable">Comfortable</option>
              </select>
            </label>
            {result ? (
              <button
                type="button"
                className="ghost-button export-button"
                onClick={() => void handleExportDump()}
                disabled={exporting}
              >
                {exporting ? 'Dumping…' : 'Dump data'}
              </button>
            ) : null}
            <div className="shortcut-hints" aria-label="Keyboard shortcuts">
              <span><kbd>/</kbd> prompt</span>
              <span><kbd>[</kbd><kbd>]</kbd> layer</span>
              <span><kbd>1</kbd>–<kbd>5</kbd> panels</span>
            </div>
            {exportError ? <p className="status-card__detail">{exportError}</p> : null}
          </div>
        </ModelStatusCard>
      </header>

      <ErrorNotice message={error} />

      {loading && !result ? <SkeletonDashboard /> : null}

      {result ? (
        <main className="dashboard">
          {loading ? <div className="loading-ribbon" aria-live="polite">Refreshing analysis…</div> : null}
          <OverviewMetrics result={result} generatedAnswer={generatedAnswer} />

          <div className="dashboard__primary">
            <div className="detail-tab-strip" role="tablist" aria-label="Detail panels">
              {availableDetailTabs.includes('attention') ? (
                <button
                  type="button"
                  className={activeDetailTab === 'attention' ? 'is-active' : ''}
                  onClick={() => handleOpenDetailTab('attention')}
                >
                  Attention
                </button>
              ) : null}
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
              {availableDetailTabs.includes('network') ? (
                <button
                  type="button"
                  className={activeDetailTab === 'network' ? 'is-active' : ''}
                  onClick={() => handleOpenDetailTab('network')}
                >
                  Network
                </button>
              ) : null}
            </div>

            <div className={`dashboard__detail-grid ${matchTopRowHeights ? 'dashboard__detail-grid--top-row-collapsed' : ''}`.trim()}>
              <div id="panel-predictions" className="dashboard__panel dashboard__panel--predictions">
                <TokenPredictionTable result={result} matchCollapsedHeight={matchTopRowHeights} onCollapsedStateChange={setPredictionCollapsed} />
              </div>

              {availableDetailTabs.includes('attention') ? (
                <div
                  id="panel-attention"
                  className={`detail-panel ${activeDetailTab === 'attention' ? 'detail-panel--active' : ''}`}
                >
                  <AttentionPanel
                    result={result}
                    selectedLayer={selectedLayer}
                    onSelectedLayerChange={setSelectedLayer}
                  />
                </div>
              ) : null}

              <div
                id="panel-activations"
                className={`detail-panel ${activeDetailTab === 'activations' ? 'detail-panel--active' : ''}`}
              >
                <ActivationPanel result={result} matchCollapsedHeight={matchTopRowHeights} onCollapsedStateChange={setActivationCollapsed} />
              </div>

              <div
                id="panel-logits"
                className={`detail-panel detail-panel--full ${activeDetailTab === 'logits' ? 'detail-panel--active' : ''}`}
              >
                <LogitPanel
                  result={result}
                  selectedAnswerTokenIndex={selectedAnswerTokenIndex}
                  onSelectedAnswerTokenIndexChange={setSelectedAnswerTokenIndex}
                  matchCollapsedHeight={matchTopRowHeights}
                  onCollapsedStateChange={setLogitCollapsed}
                />
              </div>

              {availableDetailTabs.includes('network') ? (
                <div
                  id="panel-network"
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
              ) : null}
            </div>
          </div>
        </main>
      ) : null}
    </div>
  )
}

export default App
