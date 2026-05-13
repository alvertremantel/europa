declare module 'circuitsvis/dist/module/attention/AttentionHeads.js' {
  import * as React from 'react';
  export interface AttentionHeadsProps {
    tokens: string[];
    attention: number[][][];
  }
  export const AttentionHeads: React.FC<AttentionHeadsProps>;
}

declare module 'circuitsvis/dist/module/activations/TextNeuronActivations.js' {
  import * as React from 'react';
  export interface TextNeuronActivationsProps {
    tokens: string[];
    activations: number[][][];
  }
  export const TextNeuronActivations: React.FC<TextNeuronActivationsProps>;
}