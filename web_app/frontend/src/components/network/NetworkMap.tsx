import type { NetworkAnalysis } from '../../api'
import type { CSSProperties } from 'react'
import type { NetworkSelection } from './types'

interface NetworkMapProps {
  network: NetworkAnalysis
  selection: NetworkSelection
  onSelectionChange: (selection: NetworkSelection) => void
}

export function NetworkMap({ network, selection, onSelectionChange }: NetworkMapProps) {
  const maxMlpMass = Math.max(
    ...network.mlp.layers.map((layer) => layer.layer_summary?.mean_abs_activation ?? 0),
    1,
  )
  const maxResidualDelta = Math.max(
    ...network.residual.layers.flatMap((layer) =>
      layer.tokens.map((token) => token.attention_delta_norm ?? 0),
    ),
    1,
  )

  return (
    <section className="network-section network-section--map">
      <div className="network-section__header">
        <h3>Full-network map</h3>
        <p>Attention heads, post-attention residual lanes, and MLP blocks by layer.</p>
      </div>

      <div className="network-map" role="list" aria-label="Network layer map">
        <div className="network-map__terminal">Input tokens</div>
        {network.attention.layers.map((attentionLayer) => {
          const mlpLayer = network.mlp.layers[attentionLayer.layer]
          const residualLayer = network.residual.layers[attentionLayer.layer]
          const mlpMass = mlpLayer?.layer_summary?.mean_abs_activation ?? 0
          const residualDelta = average(
            residualLayer?.tokens.map((token) => token.attention_delta_norm ?? 0) ?? [],
          )

          return (
            <article key={attentionLayer.layer} className="network-layer" role="listitem">
              <div className="network-layer__label">Layer {attentionLayer.layer}</div>
              <div className="network-layer__heads">
                {attentionLayer.heads.map((head) => {
                  const intensity = Math.min(1, head.max_weight)
                  const selected =
                    selection.kind === 'attention' &&
                    selection.layer === head.layer &&
                    selection.head === head.head
                  return (
                    <button
                      key={head.head}
                      type="button"
                      className={`network-head-node ${selected ? 'is-selected' : ''}`}
                      style={{ '--node-alpha': 0.18 + intensity * 0.72 } as CSSProperties}
                      title={`L${head.layer} H${head.head}: entropy ${head.mean_entropy.toFixed(2)}, max ${head.max_weight.toFixed(2)}`}
                      onClick={() => onSelectionChange({ kind: 'attention', layer: head.layer, head: head.head })}
                    >
                      H{head.head}
                    </button>
                  )
                })}
              </div>

              <button
                type="button"
                className={`network-residual-node ${
                  selection.kind === 'residual' && selection.layer === attentionLayer.layer
                    ? 'is-selected'
                    : ''
                }`}
                style={
                  { '--node-alpha': 0.16 + (residualDelta / maxResidualDelta) * 0.7 } as CSSProperties
                }
                onClick={() =>
                  onSelectionChange({
                    kind: 'residual',
                    layer: attentionLayer.layer,
                    tokenIndex: network.controls.selected_token_index ?? 0,
                  })
                }
              >
                resid Δ {residualDelta.toFixed(2)}
              </button>

              <button
                type="button"
                className={`network-mlp-node ${
                  selection.kind === 'mlp' && selection.layer === attentionLayer.layer
                    ? 'is-selected'
                    : ''
                }`}
                style={
                  { '--node-alpha': 0.16 + (mlpMass / maxMlpMass) * 0.7 } as CSSProperties
                }
                onClick={() =>
                  onSelectionChange({
                    kind: 'mlp',
                    layer: attentionLayer.layer,
                    tokenIndex: network.controls.selected_token_index ?? 0,
                  })
                }
              >
                MLP {formatMaybe(mlpMass)}
              </button>
            </article>
          )
        })}
        <div className="network-map__terminal">Final logits</div>
      </div>
    </section>
  )
}

function average(values: number[]): number {
  if (values.length === 0) {
    return 0
  }
  return values.reduce((sum, value) => sum + value, 0) / values.length
}

function formatMaybe(value: number): string {
  if (!Number.isFinite(value)) {
    return '—'
  }
  return value.toFixed(2)
}
