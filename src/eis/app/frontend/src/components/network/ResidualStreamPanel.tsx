import type { ResidualNetworkSummary } from '../../api'
import type { NetworkSelection } from './types'

interface ResidualStreamPanelProps {
  residual: ResidualNetworkSummary
  answerPosition: number
  tokens: string[]
  selection: NetworkSelection
  onSelectionChange: (selection: NetworkSelection) => void
}

export function ResidualStreamPanel({
  residual,
  answerPosition,
  tokens,
  selection,
  onSelectionChange,
}: ResidualStreamPanelProps) {
  const maxNorm = Math.max(
    ...residual.layers.flatMap((layer) => layer.tokens.map((token) => token.norm)),
    1,
  )
  const maxDelta = Math.max(
    ...residual.layers.flatMap((layer) =>
      layer.tokens.map((token) => token.attention_delta_norm ?? 0),
    ),
    1,
  )
  const selectedLayer = selection.kind === 'residual' ? selection.layer : 0
  const selectedToken = selection.kind === 'residual' ? selection.tokenIndex : answerPosition
  const selectedSummary = residual.layers[selectedLayer]?.tokens[selectedToken]

  return (
    <section className="network-section">
      <div className="network-section__header">
        <div>
          <h3>Residual stream after attention</h3>
          <p>
            Residual contents are summarized through norms, attention deltas, dimension loadings,
            and logit-lens projections.
          </p>
        </div>
      </div>

      <div className="network-two-column">
        <ResidualHeatmap
          title="Residual norm"
          residual={residual}
          tokens={tokens}
          maxValue={maxNorm}
          selection={selection}
          valueForToken={(token) => token.norm}
          onSelectionChange={onSelectionChange}
        />
        <ResidualHeatmap
          title="Attention delta norm"
          residual={residual}
          tokens={tokens}
          maxValue={maxDelta}
          selection={selection}
          valueForToken={(token) => token.attention_delta_norm ?? 0}
          onSelectionChange={onSelectionChange}
        />
      </div>

      <div className="network-list-card">
        <h4>Answer-position logit lens by layer</h4>
        <div className="network-logit-table">
          {residual.layers.map((layer) => {
            const answer = layer.tokens[answerPosition]
            return (
              <div key={layer.layer} className="network-logit-row">
                <strong>L{layer.layer}</strong>
                <div>
                  {(answer?.logit_lens_top_k ?? []).map((entry) => (
                    <span key={`${layer.layer}-${entry.token}`}>
                      <code>{entry.token}</code> {(entry.probability * 100).toFixed(1)}%
                    </span>
                  ))}
                </div>
              </div>
            )
          })}
        </div>
      </div>

      <div className="network-list-card">
        <h4>Selected residual dimensions</h4>
        {selectedSummary ? (
          <ol>
            {selectedSummary.top_dimensions.map((dimension) => (
              <li key={dimension.dimension}>
                <code>d{dimension.dimension}</code>
                <span>{dimension.value.toFixed(4)}</span>
              </li>
            ))}
          </ol>
        ) : (
          <p>Select a residual cell to inspect dimensions.</p>
        )}
      </div>
    </section>
  )
}

interface ResidualHeatmapProps {
  title: string
  residual: ResidualNetworkSummary
  tokens: string[]
  maxValue: number
  selection: NetworkSelection
  valueForToken: (token: ResidualNetworkSummary['layers'][number]['tokens'][number]) => number
  onSelectionChange: (selection: NetworkSelection) => void
}

function ResidualHeatmap({
  title,
  residual,
  tokens,
  maxValue,
  selection,
  valueForToken,
  onSelectionChange,
}: ResidualHeatmapProps) {
  return (
    <div className="network-list-card">
      <h4>{title}</h4>
      <div className="network-heatmap network-heatmap--compact">
        {residual.layers.map((layer) => (
          <div key={`${title}-${layer.layer}`} className="network-heatmap__row">
            <span>L{layer.layer}</span>
            <div className="network-heatmap__cells">
              {tokens.map((token, tokenIndex) => {
                const summary = layer.tokens[tokenIndex]
                const value = summary ? valueForToken(summary) : 0
                const selected =
                  selection.kind === 'residual' &&
                  selection.layer === layer.layer &&
                  selection.tokenIndex === tokenIndex
                return (
                  <button
                    key={`${title}-${layer.layer}-${tokenIndex}`}
                    type="button"
                    className={`network-heatmap__cell ${selected ? 'is-selected' : ''}`}
                    style={{ backgroundColor: `rgb(52 211 153 / ${0.12 + (value / maxValue) * 0.78})` }}
                    title={`L${layer.layer} token ${tokenIndex} ${token}: ${value.toFixed(3)}`}
                    onClick={() =>
                      onSelectionChange({ kind: 'residual', layer: layer.layer, tokenIndex })
                    }
                  >
                    {value.toFixed(2)}
                  </button>
                )
              })}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
