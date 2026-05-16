import type { AnalysisResult } from '../api'

interface LogitPanelProps {
  result: AnalysisResult
}

export function LogitPanel({ result }: LogitPanelProps) {
  const answerTopK = result.top_k_predictions[result.answer_position] ?? []

  return (
    <section className="card panel">
      <div className="panel__header">
        <div>
          <h2>Logit lens summary</h2>
          <p>Inspect the answer-position candidates, then compare the full token trajectory below.</p>
        </div>
      </div>

      <div className="answer-strip">
        {answerTopK.map((prediction) => (
          <article key={prediction.token} className="answer-strip__card">
            <strong>{prediction.token}</strong>
            <span>{(prediction.confidence * 100).toFixed(1)}%</span>
            <small>logit {prediction.logit?.toFixed(3) ?? '—'}</small>
          </article>
        ))}
      </div>

      <div className="position-grid">
        {result.tokens.map((token, tokenIndex) => (
          <article
            key={tokenIndex}
            className={`position-card ${tokenIndex === result.answer_position ? 'position-card--answer' : ''}`}
          >
            <div className="position-card__header">
              <span>Pos {tokenIndex}</span>
              <code>{token}</code>
            </div>
            <div className="position-card__rows">
              {(result.top_k_predictions[tokenIndex] ?? []).map((prediction) => (
                <div key={`${tokenIndex}-${prediction.token}`} className="position-card__row">
                  <code>{prediction.token}</code>
                  <span>{(prediction.confidence * 100).toFixed(1)}%</span>
                  <small>{prediction.logit?.toFixed(3) ?? '—'}</small>
                </div>
              ))}
            </div>
          </article>
        ))}
      </div>
    </section>
  )
}
