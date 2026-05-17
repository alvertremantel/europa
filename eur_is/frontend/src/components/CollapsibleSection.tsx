import { useId, useState, type ReactNode } from 'react'

interface CollapsibleSectionProps {
  title: string
  defaultOpen?: boolean
  children: ReactNode
  summary?: ReactNode
  className?: string
  onOpenChange?: (open: boolean) => void
}

export function CollapsibleSection({
  title,
  defaultOpen = true,
  children,
  summary,
  className = '',
  onOpenChange,
}: CollapsibleSectionProps) {
  const [open, setOpen] = useState(defaultOpen)
  const contentId = useId()

  function handleToggle() {
    setOpen((current) => {
      const next = !current
      onOpenChange?.(next)
      return next
    })
  }

  return (
    <section className={`collapsible-section ${open ? 'is-open' : 'is-collapsed'} ${className}`.trim()}>
      <div className="collapsible-section__header">
        <div>
          <h3>{title}</h3>
          {summary ? <p>{summary}</p> : null}
        </div>
        <button
          type="button"
          className="collapsible-section__toggle"
          aria-expanded={open}
          aria-controls={contentId}
          onClick={handleToggle}
        >
          {open ? 'Collapse' : 'Expand'}
        </button>
      </div>
      {open ? <div id={contentId}>{children}</div> : null}
    </section>
  )
}
