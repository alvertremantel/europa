import type { ReactNode } from 'react'

import type { AnalysisResult, HealthResponse } from '../api'

interface ModelStatusCardProps {
  health: HealthResponse | null
  result: AnalysisResult | null
  loading: boolean
  children?: ReactNode
}

export function ModelStatusCard({ health, result, loading, children }: ModelStatusCardProps) {
  const checkpoint = result?.checkpoint ?? health?.checkpoint ?? null

  return (
    <section className="card status-card">
      <div className="status-card__title-row">
        <h2>Server</h2>
        <div className="status-card__badges">
          <span className={`status-pill ${loading ? 'status-pill--busy' : 'status-pill--ok'}`}>
            {loading ? 'Analyzing' : health?.status ?? 'Offline'}
          </span>
          <span className="meta-chip">{checkpoint?.device ?? health?.device ?? 'unknown device'}</span>
        </div>
      </div>

      <div className="status-card__section">
        <h3>Loaded checkpoint</h3>
        <p className="status-card__path">{checkpoint?.path ?? 'Backend unavailable'}</p>
      </div>
      {health?.detail ? <p className="status-card__detail">{health.detail}</p> : null}

      <dl className="status-grid">
        <div>
          <dt>Epoch</dt>
          <dd>{checkpoint?.epoch ?? '—'}</dd>
        </div>
        <div>
          <dt>Exact match</dt>
          <dd>{formatPercent(checkpoint?.exact_match)}</dd>
        </div>
        <div>
          <dt>Val loss</dt>
          <dd>{formatDecimal(checkpoint?.val_loss)}</dd>
        </div>
        <div>
          <dt>Schema</dt>
          <dd>{checkpoint?.checkpoint_schema_version ?? '—'}</dd>
        </div>
      </dl>

      {children ? <div className="status-card__section status-card__section--extras">{children}</div> : null}
    </section>
  )
}

function formatPercent(value: number | null | undefined): string {
  return value == null ? '—' : `${(value * 100).toFixed(1)}%`
}

function formatDecimal(value: number | null | undefined): string {
  return value == null ? '—' : value.toFixed(4)
}
