import type { NetworkControls } from './types/api'

export const EXAMPLE_PROMPTS = [
  { label: 'Binary', value: '02000000 + 01000000 =' },
  { label: 'Three input', value: '03000000 + 02000000 + 01000000 =' },
  { label: 'Parentheses', value: '( 03000000 + 02000000 ) - 01000000 =' },
  { label: 'Negative input', value: '(-30000000) + 01000000 =' },
] as const

export const DEFAULT_PROMPT = EXAMPLE_PROMPTS[0].value

export const DEFAULT_NETWORK_CONTROLS: NetworkControls = {
  mlp_threshold: 0,
  top_k: 5,
  top_neurons: 8,
  selected_token_index: null,
}
