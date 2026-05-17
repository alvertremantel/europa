import { useMemo, useState, type CSSProperties } from 'react'

import { LazyAttentionHeads } from './circuitsvis/LazyAttentionHeads'

import type { AnalysisResult, AttentionHeadSummary } from '../api'

interface AttentionPanelProps {
  result: AnalysisResult
  selectedLayer: number
  onSelectedLayerChange: (layer: number) => void
}

export function AttentionPanel({
  result,
  selectedLayer,
  onSelectedLayerChange,
}: AttentionPanelProps) {
  const [selectedHead, setSelectedHead] = useState(0)
  const [matrixMetric, setMatrixMetric] = useState<'max_weight' | 'entropy' | 'mean_diagonal'>(
    'max_weight',
  )
  const clampedSelectedHead = Math.min(selectedHead, Math.max(result.config.n_heads - 1, 0))
  const metricRange = useMemo(() => {
    const values = result.attention_summary.heads.flatMap((layer) =>
      layer.map((head) => head[matrixMetric]),
    )
    return {
      min: Math.min(...values, 0),
      max: Math.max(...values, 1),
    }
  }, [matrixMetric, result.attention_summary.heads])
  const focusedHead = result.attention_summary.heads[selectedLayer]?.[clampedSelectedHead] ?? null

  return (
    <section className="card panel">
      <div className="panel__header">
        <div>
          <h2>Attention heads</h2>
          <p>Layer/head matrix plus a focused CircuitsVis view for the selected layer.</p>
        </div>

        <div className="panel__control-cluster">
          <label className="select-field">
            <span>Heat</span>
            <select
              value={matrixMetric}
              onChange={(event) => setMatrixMetric(event.target.value as typeof matrixMetric)}
            >
              <option value="max_weight">Max weight</option>
              <option value="entropy">Entropy</option>
              <option value="mean_diagonal">Self mass</option>
            </select>
          </label>
          <label className="select-field">
            <span>Layer</span>
            <select
              value={selectedLayer}
              onChange={(event) => onSelectedLayerChange(Number.parseInt(event.target.value, 10))}
            >
              {Array.from({ length: result.config.n_layers }, (_, index) => (
                <option key={index} value={index}>
                  Layer {index}
                </option>
              ))}
            </select>
          </label>
        </div>
      </div>

      <div className="attention-workbench">
        <div
          className="attention-matrix"
          aria-label="Attention head matrix"
          style={{ '--head-count': result.config.n_heads } as CSSProperties}
        >
          <div className="attention-matrix__corner">Layer</div>
          {Array.from({ length: result.config.n_heads }, (_, headIndex) => (
            <div key={`head-label-${headIndex}`} className="attention-matrix__head-label">
              H{headIndex}
            </div>
          ))}
          {result.attention_summary.heads.map((layer, layerIndex) => [
            <div key={`layer-${layerIndex}`} className="attention-matrix__layer-label">
              L{layerIndex}
            </div>,
            ...layer.map((head, headIndex) => (
              <HeadMatrixCell
                key={`${layerIndex}-${headIndex}`}
                head={head}
                layerIndex={layerIndex}
                headIndex={headIndex}
                metric={matrixMetric}
                range={metricRange}
                selected={selectedLayer === layerIndex && clampedSelectedHead === headIndex}
                onSelect={() => {
                  onSelectedLayerChange(layerIndex)
                  setSelectedHead(headIndex)
                }}
              />
            )),
          ])}
        </div>

        <aside className="attention-focus-card">
          <span className="meta-chip">Selected head</span>
          <h3>
            L{selectedLayer} · H{clampedSelectedHead}
          </h3>
          {focusedHead ? (
            <dl className="compact-kv-grid">
              <div>
                <dt>Entropy</dt>
                <dd>{focusedHead.entropy.toFixed(3)}</dd>
              </div>
              <div>
                <dt>Max weight</dt>
                <dd>{focusedHead.max_weight.toFixed(3)}</dd>
              </div>
              <div>
                <dt>Self mass</dt>
                <dd>{focusedHead.mean_diagonal.toFixed(3)}</dd>
              </div>
              <div>
                <dt>Strongest</dt>
                <dd>
                  <code>{focusedHead.strongest_pair.query_token}</code> →{' '}
                  <code>{focusedHead.strongest_pair.key_token}</code>
                </dd>
              </div>
            </dl>
          ) : (
            <p>No head summary for this selection.</p>
          )}
        </aside>
      </div>

      <div className="circuitsvis-frame">
        <LazyAttentionHeads tokens={result.tokens} attention={result.attention[selectedLayer]} />
      </div>
    </section>
  )
}

function HeadMatrixCell({
  head,
  layerIndex,
  headIndex,
  metric,
  range,
  selected,
  onSelect,
}: {
  head: AttentionHeadSummary
  layerIndex: number
  headIndex: number
  metric: 'max_weight' | 'entropy' | 'mean_diagonal'
  range: { min: number; max: number }
  selected: boolean
  onSelect: () => void
}) {
  const value = head[metric]
  const normalized = range.max === range.min ? 0 : (value - range.min) / (range.max - range.min)
  const alpha = 0.14 + Math.max(0, Math.min(1, normalized)) * 0.78

  return (
    <button
      type="button"
      className={`attention-matrix__cell ${selected ? 'is-selected' : ''}`}
      style={{ backgroundColor: `rgb(122 162 255 / ${alpha})` }}
      title={`L${layerIndex} H${headIndex}: ${metric.replace('_', ' ')} ${value.toFixed(3)}`}
      onClick={onSelect}
    >
      <strong>{value.toFixed(2)}</strong>
      <small>{head.strongest_pair.query_token}→{head.strongest_pair.key_token}</small>
    </button>
  )
}
