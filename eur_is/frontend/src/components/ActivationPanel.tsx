import { TextNeuronActivations } from 'circuitsvis'

import type { AnalysisResult } from '../api'

interface ActivationPanelProps {
  result: AnalysisResult
}

export function ActivationPanel({ result }: ActivationPanelProps) {
  const maxNorm = Math.max(...result.activation_summary.layer_peak_l2, 1)

  return (
    <section className="card panel">
      <div className="panel__header">
        <div>
          <h2>Residual activations</h2>
          <p>CircuitsVis residual browser with a layer/token norm heatmap.</p>
        </div>
      </div>

      <div className="heatmap">
        {result.activation_summary.token_layer_l2.map((layerNorms, tokenIndex) => (
          <div key={`token-${tokenIndex}`} className="heatmap__row">
            <div className="heatmap__label">
              <span>{tokenIndex}</span>
              <code>{result.tokens[tokenIndex]}</code>
            </div>
            <div className="heatmap__cells">
              {layerNorms.map((norm, layerIndex) => {
                const alpha = 0.15 + (norm / maxNorm) * 0.85
                return (
                  <div
                    key={`token-${tokenIndex}-layer-${layerIndex}`}
                    className="heatmap__cell"
                    style={{ backgroundColor: `rgb(83 109 254 / ${alpha})` }}
                    title={`Token ${tokenIndex}, layer ${layerIndex}: L2 ${norm.toFixed(3)}`}
                  >
                    {norm.toFixed(2)}
                  </div>
                )
              })}
            </div>
          </div>
        ))}
      </div>

      <div className="circuitsvis-frame circuitsvis-frame--tall">
        <TextNeuronActivations tokens={result.tokens} activations={result.activations} />
      </div>
    </section>
  )
}
