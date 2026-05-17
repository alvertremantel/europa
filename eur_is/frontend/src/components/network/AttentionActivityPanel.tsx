import { AttentionHeads } from 'circuitsvis'

import type { AttentionNetworkSummary } from '../../api'
import type { NetworkSelection } from './types'

interface AttentionActivityPanelProps {
  attention: AttentionNetworkSummary
  rawAttention?: number[][][][] | null
  tokens: string[]
  selection: NetworkSelection
  selectedLayer: number
  onSelectedLayerChange: (layer: number) => void
  onSelectionChange: (selection: NetworkSelection) => void
}

export function AttentionActivityPanel({
  attention,
  rawAttention,
  tokens,
  selection,
  selectedLayer,
  onSelectedLayerChange,
  onSelectionChange,
}: AttentionActivityPanelProps) {
  const selectedLayerSummary = attention.layers[selectedLayer]

  return (
    <section className="network-section">
      <div className="network-section__header">
        <div>
          <h3>Attention activity</h3>
          <p>CircuitsVis attention patterns plus entropy, max-weight, and source-token summaries.</p>
        </div>
        <label className="select-field">
          <span>Layer</span>
          <select
            value={selectedLayer}
            onChange={(event) => onSelectedLayerChange(Number.parseInt(event.target.value, 10))}
          >
            {attention.layers.map((layer) => (
              <option key={layer.layer} value={layer.layer}>
                Layer {layer.layer}
              </option>
            ))}
          </select>
        </label>
      </div>

      <div className="network-head-grid">
        {attention.layers.flatMap((layer) =>
          layer.heads.map((head) => {
            const selected =
              selection.kind === 'attention' &&
              selection.layer === head.layer &&
              selection.head === head.head
            return (
              <button
                key={`${head.layer}-${head.head}`}
                type="button"
                className={`network-head-card ${selected ? 'is-selected' : ''}`}
                onClick={() => {
                  onSelectedLayerChange(head.layer)
                  onSelectionChange({ kind: 'attention', layer: head.layer, head: head.head })
                }}
              >
                <strong>
                  L{head.layer} · H{head.head}
                </strong>
                <span>Entropy {head.mean_entropy.toFixed(2)}</span>
                <span>Max {head.max_weight.toFixed(2)}</span>
                <span>
                  {head.strongest_pair.query_token} → {head.strongest_pair.key_token}
                </span>
              </button>
            )
          }),
        )}
      </div>

      {selectedLayerSummary?.availability === 'available' && rawAttention?.[selectedLayer] ? (
        <div className="circuitsvis-frame">
          <AttentionHeads tokens={tokens} attention={rawAttention[selectedLayer]} />
        </div>
      ) : (
        <div className="network-empty">Attention patterns are not available for this layer.</div>
      )}
    </section>
  )
}
