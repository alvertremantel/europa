import { lazy, Suspense, type FC } from 'react'

import { CircuitsVisLoader, CircuitsVisErrorBoundary } from './CircuitsVisLoader'

interface TextNeuronActivationsProps {
  tokens: string[]
  activations: number[][][]
}

const TextNeuronActivationsLazy = lazy(
  async (): Promise<{ default: FC<TextNeuronActivationsProps> }> => {
    const mod = await import('circuitsvis')
    return { default: mod.TextNeuronActivations }
  },
)

export function LazyTextNeuronActivations(props: TextNeuronActivationsProps) {
  const resetKey = `${props.tokens.join('\u0001')}|${props.activations.length}:${props.activations[0]?.length ?? 0}:${props.activations[0]?.[0]?.length ?? 0}`

  return (
    <CircuitsVisErrorBoundary resetKey={resetKey}>
      <Suspense fallback={<CircuitsVisLoader />}>
        <TextNeuronActivationsLazy {...props} />
      </Suspense>
    </CircuitsVisErrorBoundary>
  )
}
