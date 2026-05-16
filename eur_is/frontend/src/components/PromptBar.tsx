import { Loader2, Play } from 'lucide-react'

interface ExamplePrompt {
  label: string
  value: string
  onSelect: () => void
}

interface PromptBarProps {
  prompt: string
  loading: boolean
  examplePrompts: ExamplePrompt[]
  onPromptChange: (value: string) => void
  onSubmit: () => void
}

export function PromptBar({
  prompt,
  loading,
  examplePrompts,
  onPromptChange,
  onSubmit,
}: PromptBarProps) {
  return (
    <section className="card prompt-bar">
      <div className="prompt-bar__header">
        <div>
          <h2>Prompt explorer</h2>
          <p>
            Use reversed zero-padded numbers. The backend appends <code>&lt;ans&gt;</code>{' '}
            automatically if it is missing.
          </p>
        </div>
        <div className="prompt-bar__examples" aria-label="Example prompts">
          {examplePrompts.map((example) => (
            <button key={example.label} type="button" className="ghost-button" onClick={example.onSelect}>
              {example.label}
            </button>
          ))}
        </div>
      </div>

      <div className="prompt-bar__controls">
        <input
          type="text"
          value={prompt}
          onChange={(event) => onPromptChange(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter') {
              onSubmit()
            }
          }}
          placeholder="02000000 + 01000000 ="
          aria-label="Arithmetic prompt"
        />

        <button type="button" className="primary-button" onClick={onSubmit} disabled={loading}>
          {loading ? <Loader2 className="spin" size={18} /> : <Play size={18} />}
          <span>{loading ? 'Analyzing…' : 'Analyze'}</span>
        </button>
      </div>
    </section>
  )
}
