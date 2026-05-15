import type { AnalysisResult } from '../api'

interface TokenPredictionTableProps {
  result: AnalysisResult
}

export function TokenPredictionTable({ result }: TokenPredictionTableProps) {
  return (
    <section className="card panel token-table-panel">
      <div className="panel__header">
        <div>
          <h2>Token prediction table</h2>
          <p>Top-5 next-token candidates at each prompt position.</p>
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
              <th>Top-5 distribution</th>
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
                  <td>{topPrediction ? `${(topPrediction.confidence * 100).toFixed(1)}%` : '—'}</td>
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
