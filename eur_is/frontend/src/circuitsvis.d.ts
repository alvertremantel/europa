declare module 'circuitsvis' {
  import * as React from 'react';
  export interface AttentionHeadsProps {
    tokens: string[];
    attention: number[][][];
  }
  export const AttentionHeads: React.FC<AttentionHeadsProps>;
  export interface TextNeuronActivationsProps {
    tokens: string[];
    activations: number[][][];
  }
  export const TextNeuronActivations: React.FC<TextNeuronActivationsProps>;
}