import { useMemo, useState } from 'react'

import type { AnalysisResult, NetworkControls } from '../../api'
import { AttentionActivityPanel } from './AttentionActivityPanel'
import { MlpActivityPanel } from './MlpActivityPanel'
import { NetworkInspector } from './NetworkInspector'
import { NetworkMap } from './NetworkMap'
import { ResidualStreamPanel } from './ResidualStreamPanel'
import type { MlpMetric, NetworkSelection } from './types'

interface NetworkPanelProps {
  result: AnalysisResult
  controls: NetworkControls
  loading: boolean
  error: string | null
  onRequestNetwork: (controls: NetworkControls) => void
}

export function NetworkPanel({
  result,
  controls,
  loading,
  error,
  onRequestNetwork,
}: NetworkPanelProps) {
  const network = result.network
  const [metric, setMetric] = useState<MlpMetric>('active_fraction_abs')
  const [selectedLayer, setSelectedLayer] = useState(0)
  const [selection, setSelection] = useState<NetworkSelection>({
    kind: 'attention',
    layer: 0,
    head: 0,
  })

  const clampedControls = useMemo(
    () => ({
      ...controls,
      selected_token_index: Math.min(
        controls.selected_token_index ?? result.answer_position,
        Math.max(result.tokens.length - 1, 0),
      ),
    }),
    [controls, result.answer_position, result.tokens.length],
  )

  if (!result.capabilities?.network_analysis) {
    return (
      <section className="card panel network-panel">
        <div className="panel__header">
          <div>
            <h2>Full-network CircuitVis</h2>
            <p>Full network analysis is unavailable for the loaded checkpoint mode.</p>
          </div>
        </div>
      </section>
    )
  }

  if (!network) {
    return (
      <section className="card panel network-panel">
        <div className="panel__header">
          <div>
            <h2>Full-network CircuitVis</h2>
            <p>
              Fetch compact cached summaries for MLP firing, attention activity, and
              residual streams.
            </p>
          </div>
          <button
            type="button"
            className="primary-button"
            disabled={loading}
            onClick={() => onRequestNetwork(clampedControls)}
          >
            {loading ? 'Loading network…' : 'Load network analysis'}
          </button>
        </div>
        {error ? <div className="network-empty network-empty--error">{error}</div> : null}
      </section>
    )
  }

  return (
    <section className="card panel network-panel">
      <div className="panel__header">
        <div>
          <h2>Full-network CircuitVis</h2>
          <p>
            CircuitsVis is used for attention patterns; custom heatmaps/SVG-like cards summarize
            MLP firing and residual stream projections without dumping raw tensors.
          </p>
        </div>
        <div className="network-controls network-controls--top">
          <label className="select-field">
            <span>Selected token</span>
            <select
              value={clampedControls.selected_token_index ?? result.answer_position}
              onChange={(event) =>
                onRequestNetwork({
                  ...clampedControls,
                  selected_token_index: Number.parseInt(event.target.value, 10),
                })
              }
            >
              {result.tokens.map((token, index) => (
                <option key={`${index}-${token}`} value={index}>
                  {index}: {token}
                </option>
              ))}
            </select>
          </label>
          <label className="select-field">
            <span>Top K</span>
            <select
              value={clampedControls.top_k}
              onChange={(event) =>
                onRequestNetwork({
                  ...clampedControls,
                  top_k: Number.parseInt(event.target.value, 10),
                })
              }
            >
              {[3, 5, 8, 10].map((value) => (
                <option key={value} value={value}>
                  {value}
                </option>
              ))}
            </select>
          </label>
        </div>
      </div>

      <div className="network-caveat">
        MLP “fires” = thresholded post-GELU activation; attention activity = attention
        probability summaries; residual stream contents = norms, deltas, top dimensions, and
        logit-lens projections.
      </div>
      {loading ? <div className="network-empty">Refreshing network analysis…</div> : null}
      {error ? <div className="network-empty network-empty--error">{error}</div> : null}

      <div className="network-layout">
        <div className="network-layout__main">
          <NetworkMap network={network} selection={selection} onSelectionChange={setSelection} />
          <MlpActivityPanel
            mlp={network.mlp}
            controls={network.controls}
            tokens={result.tokens}
            metric={metric}
            selection={selection}
            onMetricChange={setMetric}
            onSelectionChange={setSelection}
            onControlsChange={(nextControls) => onRequestNetwork(nextControls)}
          />
          <AttentionActivityPanel
            attention={network.attention}
            rawAttention={result.attention}
            tokens={result.tokens}
            selection={selection}
            selectedLayer={selectedLayer}
            onSelectedLayerChange={setSelectedLayer}
            onSelectionChange={setSelection}
          />
          <ResidualStreamPanel
            residual={network.residual}
            answerPosition={result.answer_position}
            tokens={result.tokens}
            selection={selection}
            onSelectionChange={setSelection}
          />
        </div>
        <NetworkInspector network={network} selection={selection} />
      </div>
    </section>
  )
}
