export type NetworkSelection =
  | { kind: 'mlp'; layer: number; tokenIndex: number }
  | { kind: 'attention'; layer: number; head: number }
  | { kind: 'residual'; layer: number; tokenIndex: number }

export type MlpMetric = 'active_fraction_abs' | 'active_fraction_positive' | 'mean_abs_activation'
