import type { AnalysisResult, GeneratedAnswerToken } from '../api'

interface LogitPanelProps {
  result: AnalysisResult
  selectedAnswerTokenIndex: number
  onSelectedAnswerTokenIndexChange: (value: number) => void
}

export function LogitPanel({
  result,
  selectedAnswerTokenIndex,
  onSelectedAnswerTokenIndexChange,
}: LogitPanelProps) {
  const selectedAnswerToken = result.generated_answer_top_k[selectedAnswerTokenIndex] ?? null
  const answerTopK = selectedAnswerToken?.top_predictions ?? []
  const hasGeneratedAnswer = result.generated_answer_top_k.length > 0

  return (
    <section className="card panel">
      <div className="panel__header">
        <div>
          <h2>Logit lens summary</h2>
          <p>
            Inspect one generated answer token at a time, then compare the full prompt-token
            trajectory below.
          </p>
        </div>
        {hasGeneratedAnswer ? (
          <label className="select-field">
            <span>Answer token</span>
            <select
              value={selectedAnswerTokenIndex}
              onChange={(event) => onSelectedAnswerTokenIndexChange(Number(event.target.value))}
            >
              {result.generated_answer_top_k.map((entry, index) => (
                <option key={`${index}-${entry.token}`} value={index}>
                  {formatAnswerTokenLabel(entry, index)}
                </option>
              ))}
            </select>
          </label>
        ) : null}
      </div>

      <p className="panel__subnote">
        {selectedAnswerToken
          ? `Viewing token ${selectedAnswerTokenIndex + 1} of ${result.generated_answer.token_count}: ${selectedAnswerToken.token}`
          : 'No generated answer tokens available.'}
      </p>

      {hasGeneratedAnswer ? (
        <div className="answer-token-timeline" aria-label="Generated answer token timeline">
          {result.generated_answer_top_k.map((entry, index) => (
            <button
              key={`${index}-${entry.token}`}
              type="button"
              className={`answer-token-timeline__item ${
                index === selectedAnswerTokenIndex ? 'is-selected' : ''
              }`}
              onClick={() => onSelectedAnswerTokenIndexChange(index)}
            >
              <span>#{index + 1}</span>
              <code>{entry.token}</code>
            </button>
          ))}
        </div>
      ) : null}

      <div className="answer-strip">
        {answerTopK.map((prediction) => (
          <article key={prediction.token} className="answer-strip__card">
            <div className="answer-strip__card-header">
              <strong><code>{prediction.token}</code></strong>
              <span>{(prediction.confidence * 100).toFixed(1)}%</span>
            </div>
            <div className="confidence-bar confidence-bar--wide">
              <div
                className="confidence-bar__fill"
                style={{ width: `${prediction.confidence * 100}%` }}
              />
            </div>
            <small>logit {prediction.logit?.toFixed(3) ?? '—'}</small>
          </article>
        ))}
      </div>

      <div className="trajectory-header">
        <h3>Prompt-token next-token trajectory</h3>
        <p>Separate from generated-answer token distributions above.</p>
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

function formatAnswerTokenLabel(entry: GeneratedAnswerToken, index: number): string {
  return `#${index + 1} · ${entry.token}`
}
