import type { AnalysisResult, ProblemMetadata } from '../api'

interface PromptBriefingCardProps {
  result: AnalysisResult | null
}

const CURRICULUM_LABELS: Record<string, string> = {
  easy_binary_add_sub: 'Binary add/sub foundations',
  binary_mul_div: 'Binary mul/div focus',
  compositional_parentheses_three_input: 'Compositional reasoning',
  negative_input: 'Negative-input handling',
  other: 'Other prompt family',
}

const CATEGORY_LABELS: Record<string, string> = {
  binary: 'Binary',
  three_input: 'Three input',
  parentheses: 'Parentheses',
  negative_input: 'Negative input',
}

export function PromptBriefingCard({ result }: PromptBriefingCardProps) {
  if (!result) {
    return (
      <section className="card prompt-briefing">
        <div className="prompt-briefing__header">
          <div>
            <p className="hero__eyebrow">Briefing</p>
            <h2>Analyze a prompt to fill this panel</h2>
          </div>
        </div>

        <div className="prompt-briefing__empty-grid">
          <article className="prompt-briefing__callout">
            <strong>Curriculum fit</strong>
            <p>
              We’ll classify the expression by the training curriculum group and exact generated
              problem kind.
            </p>
          </article>
          <article className="prompt-briefing__callout">
            <strong>Model architecture</strong>
            <p>
              This panel also exposes the loaded model’s width, head layout, MLP size, context,
              and vocab details.
            </p>
          </article>
        </div>
      </section>
    )
  }

  const problem = result.problem
  const stats = [
    { label: 'Layers', value: result.config.n_layers.toString() },
    { label: 'Heads', value: result.config.n_heads.toString() },
    { label: 'Head width', value: result.config.d_head.toString() },
    { label: 'Residual', value: result.config.d_model.toString() },
    { label: 'MLP hidden', value: result.config.mlp_hidden?.toString() ?? '—' },
    { label: 'Context', value: result.config.sequence_length.toString() },
    { label: 'Vocab', value: result.config.vocab_size.toString() },
    {
      label: 'Dropout',
      value: result.config.dropout == null ? '—' : result.config.dropout.toFixed(2),
    },
  ]

  return (
    <section className="card prompt-briefing">
      <div className="prompt-briefing__header">
        <div>
          <p className="hero__eyebrow">Briefing</p>
          <h2>Prompt and model profile</h2>
        </div>
        {problem ? (
          <div className="prompt-briefing__badges">
            <span className="prompt-briefing__badge prompt-briefing__badge--primary">
              {labelForCurriculum(problem)}
            </span>
            <span className="prompt-briefing__badge">{labelForCategory(problem)}</span>
          </div>
        ) : null}
      </div>

      <div className="prompt-briefing__summary">
        <div className="prompt-briefing__callout">
          <strong>Training curriculum</strong>
          <p>{problem ? descriptionForProblem(problem) : 'Unable to classify this prompt.'}</p>
        </div>
        <div className="prompt-briefing__callout prompt-briefing__callout--code">
          <strong>Exact problem kind</strong>
          <code>{problem?.kind ?? 'unavailable'}</code>
        </div>
      </div>

      <div className="prompt-briefing__stat-grid">
        {stats.map((stat) => (
          <article key={stat.label} className="prompt-briefing__stat">
            <span>{stat.label}</span>
            <strong>{stat.value}</strong>
          </article>
        ))}
      </div>
    </section>
  )
}

function labelForCurriculum(problem: ProblemMetadata): string {
  return CURRICULUM_LABELS[problem.curriculum_group] ?? problem.curriculum_group
}

function labelForCategory(problem: ProblemMetadata): string {
  return CATEGORY_LABELS[problem.category] ?? problem.category
}

function descriptionForProblem(problem: ProblemMetadata): string {
  switch (problem.curriculum_group) {
    case 'easy_binary_add_sub':
      return 'This prompt lands in the easier binary addition/subtraction foundation bucket.'
    case 'binary_mul_div':
      return 'This prompt belongs to the binary multiplication/division curriculum focus.'
    case 'compositional_parentheses_three_input':
      return 'This prompt is treated as a compositional example alongside parentheses and three-input problems.'
    case 'negative_input':
      return 'This prompt exercises the negative-input slice of the training curriculum.'
    default:
      return 'This prompt does not map cleanly onto one of the named curriculum buckets.'
  }
}
