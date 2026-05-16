interface ErrorNoticeProps {
  message: string | null
}

export function ErrorNotice({ message }: ErrorNoticeProps) {
  if (!message) {
    return null
  }

  return (
    <section className="error-banner" aria-live="polite">
      <strong>Analysis failed.</strong>
      <span>{message}</span>
    </section>
  )
}
