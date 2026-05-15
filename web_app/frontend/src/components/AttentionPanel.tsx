import { AttentionHeads } from 'circuitsvis'

import type { AnalysisResult } from '../api'

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
  return (
    <section className="card panel">
      <div className="panel__header">
        <div>
          <h2>Attention heads</h2>
          <p>Selected layer view plus per-head compact summaries across the stack.</p>
        </div>

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

      <div className="head-summary-grid">
        {result.attention_summary.heads.map((layer, layerIndex) =>
          layer.map((head, headIndex) => (
            <button
              key={`${layerIndex}-${headIndex}`}
              type="button"
              className={`head-summary ${selectedLayer === layerIndex ? 'head-summary--selected' : ''}`}
              onClick={() => onSelectedLayerChange(layerIndex)}
            >
              <strong>
                L{layerIndex} · H{headIndex}
              </strong>
              <span>Entropy {head.entropy.toFixed(2)}</span>
              <span>Max {head.max_weight.toFixed(2)}</span>
              <span>
                {head.strongest_pair.query_token} → {head.strongest_pair.key_token}
              </span>
            </button>
          )),
        )}
      </div>

      <div className="circuitsvis-frame">
        <AttentionHeads tokens={result.tokens} attention={result.attention[selectedLayer]} />
      </div>
    </section>
  )
}
