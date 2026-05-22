import { lazy, Suspense, type FC } from 'react'

import { CircuitsVisLoader, CircuitsVisErrorBoundary } from './CircuitsVisLoader'

interface AttentionHeadsProps {
  tokens: string[]
  attention: number[][][]
}

const AttentionHeadsLazy = lazy(async (): Promise<{ default: FC<AttentionHeadsProps> }> => {
  const mod = await import('circuitsvis')
  return { default: mod.AttentionHeads }
})

export function LazyAttentionHeads(props: AttentionHeadsProps) {
  const resetKey = `${props.tokens.join('\u0001')}|${props.attention.length}:${props.attention[0]?.length ?? 0}:${props.attention[0]?.[0]?.length ?? 0}`

  return (
    <CircuitsVisErrorBoundary resetKey={resetKey}>
      <Suspense fallback={<CircuitsVisLoader />}>
        <AttentionHeadsLazy {...props} />
      </Suspense>
    </CircuitsVisErrorBoundary>
  )
}
