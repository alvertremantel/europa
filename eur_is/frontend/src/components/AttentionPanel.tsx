import { useMemo, useState, type CSSProperties } from 'react'

import { CollapsibleSection } from './CollapsibleSection'
import { LazyAttentionPattern } from './circuitsvis/LazyAttentionPattern'

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
  const [viewMode, setViewMode] = useState<'layer' | 'stack'>('layer')
  const allHeadIndices = useMemo(
    () => Array.from({ length: result.config.n_heads }, (_, index) => index),
    [result.config.n_heads],
  )
  const [selectedHeadsByLayer, setSelectedHeadsByLayer] = useState<Record<number, number>>({ 0: 0 })
  const [visibleHeadsByLayer, setVisibleHeadsByLayer] = useState<Record<number, number[]>>({})
  const [maxHeadsPerRow, setMaxHeadsPerRow] = useState(Math.min(Math.max(result.config.n_heads, 1), 4))
  const [matrixMetric, setMatrixMetric] = useState<'max_weight' | 'entropy' | 'mean_diagonal'>(
    'max_weight',
  )
  const selectedHead = selectedHeadsByLayer[selectedLayer] ?? 0
  const clampedSelectedHead = Math.min(selectedHead, Math.max(result.config.n_heads - 1, 0))
  const visibleHeads = useMemo(() => {
    const configured = visibleHeadsByLayer[selectedLayer]
    if (!configured || configured.length === 0) {
      return allHeadIndices
    }

    const normalized = configured
      .filter((headIndex) => headIndex >= 0 && headIndex < result.config.n_heads)
      .sort((left, right) => left - right)

    return normalized.length > 0 ? normalized : allHeadIndices
  }, [allHeadIndices, result.config.n_heads, selectedLayer, visibleHeadsByLayer])
  const graphColumns = Math.max(1, Math.min(maxHeadsPerRow, visibleHeads.length || 1))
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

  function setSelectedHeadForLayer(layerIndex: number, headIndex: number) {
    setSelectedHeadsByLayer((current) => ({
      ...current,
      [layerIndex]: Math.min(Math.max(headIndex, 0), Math.max(result.config.n_heads - 1, 0)),
    }))
  }

  function toggleVisibleHead(headIndex: number) {
    setVisibleHeadsByLayer((current) => {
      const existing = current[selectedLayer] ?? allHeadIndices
      const hasHead = existing.includes(headIndex)
      const next = hasHead ? existing.filter((value) => value !== headIndex) : [...existing, headIndex]
      const normalized = next
        .filter((value) => value >= 0 && value < result.config.n_heads)
        .sort((left, right) => left - right)

      return {
        ...current,
        [selectedLayer]: normalized.length > 0 ? normalized : [headIndex],
      }
    })
  }

  function showAllHeads() {
    setVisibleHeadsByLayer((current) => ({
      ...current,
      [selectedLayer]: allHeadIndices,
    }))
  }

  function ensureVisibleHead(layerIndex: number, headIndex: number) {
    setVisibleHeadsByLayer((current) => {
      const existing = current[layerIndex] ?? allHeadIndices
      if (existing.includes(headIndex)) {
        return current
      }

      return {
        ...current,
        [layerIndex]: [...existing, headIndex].sort((left, right) => left - right),
      }
    })
  }

  const stackLayers = useMemo(
    () => Array.from({ length: result.config.n_layers }, (_, layerIndex) => layerIndex),
    [result.config.n_layers],
  )

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
          <label className="select-field">
            <span>Head</span>
            <select
              value={clampedSelectedHead}
              onChange={(event) => {
                const headIndex = Number.parseInt(event.target.value, 10)
                setSelectedHeadForLayer(selectedLayer, headIndex)
                ensureVisibleHead(selectedLayer, headIndex)
              }}
            >
              {allHeadIndices.map((headIndex) => (
                <option key={headIndex} value={headIndex}>
                  Head {headIndex}
                </option>
              ))}
            </select>
          </label>
        </div>
      </div>

      <CollapsibleSection
        title="Attention matrix"
        summary="Layer/head overview matrix with selected-head summary."
      >
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
                    setSelectedHeadForLayer(layerIndex, headIndex)
                    ensureVisibleHead(layerIndex, headIndex)
                    onSelectedLayerChange(layerIndex)
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
      </CollapsibleSection>

      <CollapsibleSection
        title="Attention maps"
        summary="Layer and stack views for rendered attention heatmaps."
      >
        <div className="attention-visualization-controls">
        <div className="attention-visualization-controls__header">
          <div>
            <h3>{viewMode === 'layer' ? 'Layer attention maps' : 'Stack attention maps'}</h3>
            <p>
              {viewMode === 'layer'
                ? 'Choose any subset of heads in the current layer and control how many plots sit on each row.'
                : 'Show every head in every layer, with one layer per row across the full network.'}
            </p>
          </div>
          <div className="panel__control-cluster">
            <fieldset className="view-mode-toggle" aria-label="Attention view mode">
              <label className="view-mode-toggle__option">
                <input
                  type="radio"
                  name="attention-view-mode"
                  value="layer"
                  checked={viewMode === 'layer'}
                  onChange={() => setViewMode('layer')}
                />
                <span>Layer view</span>
              </label>
              <label className="view-mode-toggle__option">
                <input
                  type="radio"
                  name="attention-view-mode"
                  value="stack"
                  checked={viewMode === 'stack'}
                  onChange={() => setViewMode('stack')}
                />
                <span>Stack view</span>
              </label>
            </fieldset>
            {viewMode === 'layer' ? (
              <>
                <label className="select-field select-field--compact">
                  <span>Max per row</span>
                  <input
                    type="number"
                    min={1}
                    max={Math.max(result.config.n_heads, 1)}
                    value={maxHeadsPerRow}
                    onChange={(event) => {
                      const nextValue = Number.parseInt(event.target.value, 10)
                      if (Number.isNaN(nextValue)) {
                        return
                      }

                      setMaxHeadsPerRow(Math.min(Math.max(nextValue, 1), Math.max(result.config.n_heads, 1)))
                    }}
                  />
                </label>
                <button type="button" className="ghost-button" onClick={showAllHeads}>
                  Show all
                </button>
              </>
            ) : null}
          </div>
        </div>

        {viewMode === 'layer' ? (
          <div className="attention-head-toggle-list" role="group" aria-label="Visible attention heads">
            {allHeadIndices.map((headIndex) => (
              <label key={headIndex} className="attention-head-toggle">
                <input
                  type="checkbox"
                  checked={visibleHeads.includes(headIndex)}
                  onChange={() => toggleVisibleHead(headIndex)}
                />
                <span>H{headIndex}</span>
              </label>
            ))}
          </div>
        ) : null}
        </div>

        {viewMode === 'layer' ? (
          <div
            className="attention-graphs"
            style={{ '--attention-graphs-columns': graphColumns } as CSSProperties}
          >
            {visibleHeads.map((headIndex) => (
              <AttentionGraphCard
                key={`layer-${selectedLayer}-head-${headIndex}`}
                layerIndex={selectedLayer}
                headIndex={headIndex}
                tokens={result.tokens}
                attention={result.attention[selectedLayer][headIndex]}
                focused={clampedSelectedHead === headIndex}
                onFocus={() => {
                  setSelectedHeadForLayer(selectedLayer, headIndex)
                  ensureVisibleHead(selectedLayer, headIndex)
                }}
              />
            ))}
          </div>
        ) : (
          <div className="attention-stack">
            {stackLayers.map((layerIndex) => (
              <section key={`stack-layer-${layerIndex}`} className="attention-stack__layer">
                <div className="attention-stack__header">
                  <strong>Layer {layerIndex}</strong>
                  <span>{result.config.n_heads} heads</span>
                </div>
                <div
                  className="attention-stack__row"
                  style={{ '--head-count': result.config.n_heads } as CSSProperties}
                >
                  {allHeadIndices.map((headIndex) => (
                    <AttentionGraphCard
                      key={`stack-layer-${layerIndex}-head-${headIndex}`}
                      layerIndex={layerIndex}
                      headIndex={headIndex}
                      tokens={result.tokens}
                      attention={result.attention[layerIndex][headIndex]}
                      focused={selectedLayer === layerIndex && clampedSelectedHead === headIndex}
                      onFocus={() => {
                        onSelectedLayerChange(layerIndex)
                        setSelectedHeadForLayer(layerIndex, headIndex)
                        ensureVisibleHead(layerIndex, headIndex)
                      }}
                    />
                  ))}
                </div>
              </section>
            ))}
          </div>
        )}
      </CollapsibleSection>
    </section>
  )
}

function AttentionGraphCard({
  layerIndex,
  headIndex,
  tokens,
  attention,
  focused,
  onFocus,
}: {
  layerIndex: number
  headIndex: number
  tokens: string[]
  attention: number[][]
  focused: boolean
  onFocus: () => void
}) {
  return (
    <section className="attention-graph-card">
      <div className="attention-graph-card__header">
        <strong>L{layerIndex} · H{headIndex}</strong>
        <button
          type="button"
          className={`attention-head-chip ${focused ? 'is-selected' : ''}`}
          onClick={onFocus}
        >
          Focus
        </button>
      </div>
      <div className="circuitsvis-frame circuitsvis-frame--unrolled attention-graph-frame">
        <LazyAttentionPattern tokens={tokens} attention={attention} />
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
