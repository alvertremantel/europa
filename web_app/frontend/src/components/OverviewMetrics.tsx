import { Activity, Brain, Hash, Layers3, Target } from 'lucide-react'

import type { AnalysisResult, TopPrediction } from '../api'

interface OverviewMetricsProps {
  result: AnalysisResult
  answerPrediction: TopPrediction | null
}

export function OverviewMetrics({ result, answerPrediction }: OverviewMetricsProps) {
  const cards = [
    {
      label: 'Tokens in prompt',
      value: result.tokens.length.toString(),
      icon: Hash,
      note: `Answer position: ${result.answer_position}`,
    },
    {
      label: 'Layers × heads',
      value: `${result.config.n_layers} × ${result.config.n_heads}`,
      icon: Layers3,
      note: `${result.config.d_model}-wide residual stream`,
    },
    {
      label: 'Answer prediction',
      value: answerPrediction?.token ?? '—',
      icon: Target,
      note: answerPrediction ? `${(answerPrediction.confidence * 100).toFixed(1)}% confidence` : 'No result yet',
    },
    {
      label: 'Peak residual norm',
      value: Math.max(...result.activation_summary.token_peak_l2).toFixed(2),
      icon: Activity,
      note: `Global max |resid| ${result.activation_summary.global_max_abs.toFixed(2)}`,
    },
    {
      label: 'Strongest attention',
      value: strongestHeadWeight(result).toFixed(3),
      icon: Brain,
      note: 'Max weight across all layers and heads',
    },
  ]

  return (
    <section className="metric-grid">
      {cards.map(({ icon: Icon, label, value, note }) => (
        <article key={label} className="card metric-card">
          <div className="metric-card__icon">
            <Icon size={18} />
          </div>
          <p>{label}</p>
          <strong>{value}</strong>
          <span>{note}</span>
        </article>
      ))}
    </section>
  )
}

function strongestHeadWeight(result: AnalysisResult): number {
  return result.attention_summary.heads.reduce((maxWeight, layer) => {
    return layer.reduce((layerMax, head) => Math.max(layerMax, head.max_weight), maxWeight)
  }, 0)
}
