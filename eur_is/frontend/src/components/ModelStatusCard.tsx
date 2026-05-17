import type { ReactNode } from 'react'

import type { AnalysisCapabilities, AnalysisResult, HealthResponse } from '../api'

interface ModelStatusCardProps {
  health: HealthResponse | null
  result: AnalysisResult | null
  loading: boolean
  children?: ReactNode
}

export function ModelStatusCard({ health, result, loading, children }: ModelStatusCardProps) {
  const checkpoint = result?.checkpoint ?? health?.checkpoint ?? null
  const positionEncoding = result?.position_encoding ?? health?.position_encoding ?? null
  const analysisRuntime = result?.analysis_runtime ?? health?.analysis_runtime ?? null
  const capabilities = result?.capabilities ?? health?.capabilities ?? null
  const limitations = buildCapabilityLimitations(capabilities)

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
        <div>
          <dt>Mode</dt>
          <dd>{formatPositionEncoding(positionEncoding)}</dd>
        </div>
        <div>
          <dt>Runtime</dt>
          <dd>{formatRuntime(analysisRuntime)}</dd>
        </div>
      </dl>

      {children ? <div className="status-card__section status-card__section--extras">{children}</div> : null}

      {limitations.length > 0 ? (
        <div className="status-card__detail">
          {limitations.map((limitation) => (
            <p key={limitation}>{limitation}</p>
          ))}
        </div>
      ) : null}
    </section>
  )
}

function buildCapabilityLimitations(capabilities: AnalysisCapabilities | null): string[] {
  if (!capabilities) {
    return []
  }

  const limitations: string[] = []
  if (!capabilities.network_analysis) {
    limitations.push('Full network analysis is unavailable for this checkpoint mode.')
  }
  if (!capabilities.attention_view) {
    limitations.push('Raw attention patterns are unavailable for this checkpoint mode.')
  }
  return limitations
}

function formatPositionEncoding(value: string | null | undefined): string {
  if (!value) return '—'
  if (value === 'digit_roles') return 'digit roles'
  return value
}

function formatRuntime(value: string | null | undefined): string {
  if (!value) return '—'
  if (value === 'native_pytorch') return 'native PyTorch'
  if (value === 'transformerlens') return 'TransformerLens'
  return value
}

function formatPercent(value: number | null | undefined): string {
  return value == null ? '—' : `${(value * 100).toFixed(1)}%`
}

function formatDecimal(value: number | null | undefined): string {
  return value == null ? '—' : value.toFixed(4)
}
