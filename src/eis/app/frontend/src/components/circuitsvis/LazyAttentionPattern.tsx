import { lazy, Suspense, type FC } from 'react'

import { CircuitsVisLoader, CircuitsVisErrorBoundary } from './CircuitsVisLoader'

interface AttentionPatternProps {
  tokens: string[]
  attention: number[][]
}

const AttentionPatternLazy = lazy(async (): Promise<{ default: FC<AttentionPatternProps> }> => {
  const mod = await import('circuitsvis')
  return { default: (mod as unknown as { AttentionPattern: FC<AttentionPatternProps> }).AttentionPattern }
})

export function LazyAttentionPattern(props: AttentionPatternProps) {
  const resetKey = `${props.tokens.join('\u0001')}|${props.attention.length}:${props.attention[0]?.length ?? 0}`

  return (
    <CircuitsVisErrorBoundary resetKey={resetKey}>
      <Suspense fallback={<CircuitsVisLoader />}>
        <AttentionPatternLazy {...props} />
      </Suspense>
    </CircuitsVisErrorBoundary>
  )
}
