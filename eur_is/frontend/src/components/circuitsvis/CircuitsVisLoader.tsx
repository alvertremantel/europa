import { Component, type ReactNode } from 'react'

/**
 * Compact loading skeleton shown while a CircuitsVis embed loads.
 * Matches the .circuitsvis-frame border/background so the placeholder
 * fills the same region the finished embed will occupy.
 */
export function CircuitsVisLoader() {
  return (
    <div className="circuitsvis-placeholder" role="status" aria-live="polite" aria-busy="true">
      <div className="circuitsvis-placeholder__spinner" />
      <div className="circuitsvis-placeholder__body">
        <p className="circuitsvis-placeholder__label">Loading visualization…</p>
        <div className="circuitsvis-placeholder__skeleton" />
        <div className="circuitsvis-placeholder__skeleton circuitsvis-placeholder__skeleton--short" />
      </div>
    </div>
  )
}

/**
 * Local error boundary for a single CircuitsVis embed.
 * When a dynamic import or render fails the error is contained within
 * the embed region — the rest of the panel stays interactive.
 */
export class CircuitsVisErrorBoundary extends Component<
  { children: ReactNode; resetKey: string },
  { hasError: boolean; resetKey: string }
> {
  state = { hasError: false, resetKey: '' }

  static getDerivedStateFromError() {
    return { hasError: true }
  }

  static getDerivedStateFromProps(
    nextProps: { resetKey: string },
    prevState: { hasError: boolean; resetKey: string },
  ) {
    if (nextProps.resetKey !== prevState.resetKey) {
      return { hasError: false, resetKey: nextProps.resetKey }
    }

    return null
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="circuitsvis-placeholder circuitsvis-placeholder--error" role="alert">
          <div className="circuitsvis-placeholder__icon">&#9888;</div>
          <div className="circuitsvis-placeholder__body">
            <p className="circuitsvis-placeholder__label">Unable to load visualization.</p>
            <p className="circuitsvis-placeholder__note">
              The rest of this panel remains available.
            </p>
          </div>
        </div>
      )
    }

    return this.props.children
  }
}
