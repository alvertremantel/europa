import { useMemo, useState } from 'react'

import { LazyTextNeuronActivations } from './circuitsvis/LazyTextNeuronActivations'

import type { AnalysisResult } from '../api'

interface ActivationPanelProps {
  result: AnalysisResult
}

export function ActivationPanel({ result }: ActivationPanelProps) {
  const [metric, setMetric] = useState<'l2' | 'max_abs'>('l2')
  const [selectedCell, setSelectedCell] = useState({ tokenIndex: result.answer_position, layerIndex: 0 })
  const clampedSelectedCell = {
    tokenIndex: Math.min(selectedCell.tokenIndex, Math.max(result.tokens.length - 1, 0)),
    layerIndex: Math.min(selectedCell.layerIndex, Math.max(result.config.n_layers - 1, 0)),
  }
  const heatmapValues =
    metric === 'l2'
      ? result.activation_summary.token_layer_l2
      : result.activation_summary.token_layer_max_abs
  const maxValue = useMemo(
    () => Math.max(...heatmapValues.flatMap((row) => row), 1),
    [heatmapValues],
  )
  const selectedValue = heatmapValues[clampedSelectedCell.tokenIndex]?.[clampedSelectedCell.layerIndex]
  const layerPeaks = useMemo(
    () =>
      Array.from({ length: result.config.n_layers }, (_, layerIndex) =>
        Math.max(...heatmapValues.map((row) => row[layerIndex] ?? 0), 0),
      ),
    [heatmapValues, result.config.n_layers],
  )

  return (
    <section className="card panel">
      <div className="panel__header">
        <div>
          <h2>Residual activations</h2>
          <p>Metric-switchable token/layer heatmap plus the raw CircuitsVis residual browser.</p>
        </div>
        <div className="panel__control-cluster">
          <label className="select-field">
            <span>Metric</span>
            <select value={metric} onChange={(event) => setMetric(event.target.value as typeof metric)}>
              <option value="l2">L2 norm</option>
              <option value="max_abs">Max |component|</option>
            </select>
          </label>
          <div className="activation-readout" aria-live="polite">
            <span className="meta-chip">Selected</span>
            <strong>
              T{clampedSelectedCell.tokenIndex} · L{clampedSelectedCell.layerIndex}
            </strong>
            <code>{selectedValue?.toFixed(3) ?? '—'}</code>
          </div>
        </div>
      </div>

      <div className="heatmap-legend" aria-label="Activation color scale">
        <span>0</span>
        <div className="heatmap-legend__bar" />
        <span>{maxValue.toFixed(2)}</span>
      </div>

      <div className="heatmap">
        <div className="heatmap__layer-peaks" aria-label="Layer peak values">
          <span />
          <div className="heatmap__cells">
            {layerPeaks.map((peak, layerIndex) => (
              <div key={`layer-peak-${layerIndex}`} className="heatmap__peak-cell">
                <span>L{layerIndex}</span>
                <strong>{peak.toFixed(1)}</strong>
              </div>
            ))}
          </div>
        </div>
        {heatmapValues.map((layerValues, tokenIndex) => (
          <div key={`token-${tokenIndex}`} className="heatmap__row">
            <div className="heatmap__label">
              <span>{tokenIndex}</span>
              <code>{result.tokens[tokenIndex]}</code>
            </div>
            <div className="heatmap__cells">
              {layerValues.map((value, layerIndex) => {
                const alpha = 0.15 + (value / maxValue) * 0.85
                const selected =
                  clampedSelectedCell.tokenIndex === tokenIndex && clampedSelectedCell.layerIndex === layerIndex
                return (
                  <button
                    key={`token-${tokenIndex}-layer-${layerIndex}`}
                    type="button"
                    className={`heatmap__cell ${selected ? 'is-selected' : ''}`}
                    style={{ backgroundColor: `rgb(83 109 254 / ${alpha})` }}
                    title={`Token ${tokenIndex}, layer ${layerIndex}: ${metric} ${value.toFixed(3)}`}
                    onClick={() => setSelectedCell({ tokenIndex, layerIndex })}
                  >
                    {value.toFixed(metric === 'l2' ? 2 : 3)}
                  </button>
                )
              })}
            </div>
          </div>
        ))}
      </div>

      <div className="circuitsvis-frame circuitsvis-frame--tall">
        <LazyTextNeuronActivations tokens={result.tokens} activations={result.activations} />
      </div>
    </section>
  )
}
