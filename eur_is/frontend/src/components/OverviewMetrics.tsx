import { Activity, Brain, Hash, Layers3, Target } from 'lucide-react'

import type { AnalysisResult, GeneratedAnswer } from '../api'

interface OverviewMetricsProps {
  result: AnalysisResult
  generatedAnswer: GeneratedAnswer | null
}

export function OverviewMetrics({ result, generatedAnswer }: OverviewMetricsProps) {
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
      value: generatedAnswer?.text ?? '—',
      icon: Target,
      note: formatAnswerStatus(generatedAnswer),
    },
    {
      label: 'Peak residual norm',
      value: Math.max(...result.activation_summary.token_peak_l2).toFixed(2),
      icon: Activity,
      note: `Global max |resid| ${result.activation_summary.global_max_abs.toFixed(2)}`,
    },
    {
      label: 'Strongest attention',
      value: result.attention_summary ? strongestHeadWeight(result).toFixed(3) : '—',
      icon: Brain,
      note: result.attention_summary
        ? 'Max weight across all layers and heads'
        : 'Unavailable for this checkpoint mode',
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

function formatAnswerStatus(generatedAnswer: GeneratedAnswer | null): string {
  if (!generatedAnswer) {
    return 'No result yet'
  }
  if (generatedAnswer.is_correct) {
    return 'Mathematically correct'
  }
  if (generatedAnswer.is_valid_canonical) {
    return 'Mathematically incorrect'
  }
  return generatedAnswer.validation_error ?? 'Not a canonical answer'
}

function strongestHeadWeight(result: AnalysisResult): number {
  if (!result.attention_summary) {
    return 0
  }
  return result.attention_summary.heads.reduce((maxWeight, layer) => {
    return layer.reduce((layerMax, head) => Math.max(layerMax, head.max_weight), maxWeight)
  }, 0)
}
