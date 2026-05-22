import type { MlpNetworkSummary, NetworkControls } from '../../api'
import type { MlpMetric, NetworkSelection } from './types'

interface MlpActivityPanelProps {
  mlp: MlpNetworkSummary
  controls: NetworkControls
  tokens: string[]
  metric: MlpMetric
  selection: NetworkSelection
  onMetricChange: (metric: MlpMetric) => void
  onSelectionChange: (selection: NetworkSelection) => void
  onControlsChange: (controls: NetworkControls) => void
}

const METRIC_LABELS: Record<MlpMetric, string> = {
  active_fraction_abs: 'Abs fraction',
  active_fraction_positive: 'Positive fraction',
  mean_abs_activation: 'Mean |activation|',
}

export function MlpActivityPanel({
  mlp,
  controls,
  tokens,
  metric,
  selection,
  onMetricChange,
  onSelectionChange,
  onControlsChange,
}: MlpActivityPanelProps) {
  const maxMetric = Math.max(
    ...mlp.layers.flatMap((layer) => layer.tokens.map((token) => metricValue(token, metric))),
    1,
  )
  const selectedLayer = selection.kind === 'mlp' ? selection.layer : 0
  const selectedToken = selection.kind === 'mlp' ? selection.tokenIndex : controls.selected_token_index ?? 0
  const selectedTokenSummary = mlp.layers[selectedLayer]?.tokens[selectedToken]

  return (
    <section className="network-section">
      <div className="network-section__header">
        <div>
          <h3>MLP firing</h3>
          <p>
            “Fires” means post-GELU hidden activation over the threshold; fallback rows show MLP
            output norms when hidden hooks are absent.
          </p>
        </div>
        <div className="network-controls">
          <label className="select-field">
            <span>Metric</span>
            <select value={metric} onChange={(event) => onMetricChange(event.target.value as MlpMetric)}>
              {Object.entries(METRIC_LABELS).map(([value, label]) => (
                <option key={value} value={value}>
                  {label}
                </option>
              ))}
            </select>
          </label>
          <label className="select-field">
            <span>Threshold</span>
            <input
              type="number"
              min="0"
              step="0.05"
              value={controls.mlp_threshold}
              onChange={(event) =>
                onControlsChange({
                  ...controls,
                  mlp_threshold: Number.parseFloat(event.target.value) || 0,
                })
              }
            />
          </label>
        </div>
      </div>

      <div className="network-heatmap">
        {mlp.layers.map((layer) => (
          <div key={layer.layer} className="network-heatmap__row">
            <span>Layer {layer.layer}</span>
            <div className="network-heatmap__cells">
              {tokens.map((token, tokenIndex) => {
                const summary = layer.tokens[tokenIndex]
                const value = summary ? metricValue(summary, metric) : 0
                const selected =
                  selection.kind === 'mlp' &&
                  selection.layer === layer.layer &&
                  selection.tokenIndex === tokenIndex
                return (
                  <button
                    key={`${layer.layer}-${tokenIndex}`}
                    type="button"
                    className={`network-heatmap__cell ${selected ? 'is-selected' : ''}`}
                    style={{ backgroundColor: `rgb(122 162 255 / ${0.12 + (value / maxMetric) * 0.78})` }}
                    title={`L${layer.layer} token ${tokenIndex} ${token}: ${value.toFixed(3)}`}
                    onClick={() => onSelectionChange({ kind: 'mlp', layer: layer.layer, tokenIndex })}
                  >
                    {value.toFixed(metric === 'mean_abs_activation' ? 2 : 2)}
                  </button>
                )
              })}
            </div>
          </div>
        ))}
      </div>

      <div className="network-bar-list">
        {mlp.layers.map((layer) => {
          const meanAbs = layer.layer_summary?.mean_abs_activation ?? 0
          const fraction = layer.layer_summary?.active_fraction_abs ?? 0
          return (
            <div key={layer.layer} className="network-bar-row">
              <span>Layer {layer.layer}</span>
              <div className="network-bar-row__track">
                <div style={{ width: `${Math.min(100, fraction * 100)}%` }} />
              </div>
              <small>abs frac {(fraction * 100).toFixed(1)}% · mean {meanAbs.toFixed(3)}</small>
            </div>
          )
        })}
      </div>

      <div className="network-list-card">
        <h4>Selected top neurons</h4>
        {selectedTokenSummary ? (
          <ol>
            {selectedTokenSummary.top_neurons.map((neuron) => (
              <li key={neuron.neuron_index}>
                <code>n{neuron.neuron_index}</code>
                <span>{neuron.value.toFixed(4)}</span>
              </li>
            ))}
          </ol>
        ) : (
          <p>No hidden-neuron data available for this selection.</p>
        )}
      </div>
    </section>
  )
}

function metricValue(
  token: { active_fraction_abs?: number; active_fraction_positive?: number; mean_abs_activation?: number },
  metric: MlpMetric,
): number {
  return token[metric] ?? 0
}
