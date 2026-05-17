import type { AnalysisResult } from '../api'

interface TokenPredictionTableProps {
  result: AnalysisResult
}

export function TokenPredictionTable({ result }: TokenPredictionTableProps) {
  const answer = result.generated_answer
  return (
    <section className="card panel token-table-panel">
      <div className="panel__header">
        <div>
          <h2>Prediction matrix</h2>
          <p>Prompt-position next-token candidates with sticky token columns and confidence bars.</p>
        </div>
        <div className="prediction-summary" aria-label="Generated answer summary">
          <span className={`answer-badge ${answer.is_correct ? 'answer-badge--ok' : 'answer-badge--bad'}`}>
            {answer.is_correct ? 'Correct' : answer.is_valid_canonical ? 'Incorrect' : 'Invalid'}
          </span>
          <strong><code>{answer.text || '—'}</code></strong>
          <span>{answer.token_count} answer tokens</span>
        </div>
      </div>

      <div className="token-table-wrapper">
        <table className="token-table">
          <thead>
            <tr>
              <th>Pos</th>
              <th>Token</th>
              <th>Top prediction</th>
              <th>Confidence</th>
              <th>Top-k distribution</th>
            </tr>
          </thead>
          <tbody>
            {result.tokens.map((token, tokenIndex) => {
              const topPrediction = result.top_predictions[tokenIndex]
              const topK = result.top_k_predictions[tokenIndex] ?? []
              const isAnswerRow = tokenIndex === result.answer_position

              return (
                <tr key={tokenIndex} className={isAnswerRow ? 'is-answer-row' : ''}>
                  <td>{tokenIndex}</td>
                  <td>
                    <code>{token}</code>
                  </td>
                  <td>
                    <code>{topPrediction?.token ?? '—'}</code>
                  </td>
                  <td>
                    {topPrediction ? (
                      <div className="confidence-cell">
                        <span>{(topPrediction.confidence * 100).toFixed(1)}%</span>
                        <div className="confidence-bar confidence-bar--wide">
                          <div
                            className="confidence-bar__fill"
                            style={{ width: `${topPrediction.confidence * 100}%` }}
                          />
                        </div>
                      </div>
                    ) : (
                      '—'
                    )}
                  </td>
                  <td>
                    <div className="rank-list">
                      {topK.map((prediction) => (
                        <div key={`${tokenIndex}-${prediction.token}`} className="rank-list__item">
                          <div className="rank-list__meta">
                            <code>{prediction.token}</code>
                            <span>{(prediction.confidence * 100).toFixed(1)}%</span>
                          </div>
                          <div className="confidence-bar">
                            <div
                              className="confidence-bar__fill"
                              style={{ width: `${prediction.confidence * 100}%` }}
                            />
                          </div>
                        </div>
                      ))}
                    </div>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </section>
  )
}
