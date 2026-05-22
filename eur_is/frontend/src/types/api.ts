export type PositionEncoding = 'fixed_meaning'

export type AnalysisRuntime = 'native_pytorch'

export interface AnalysisCapabilities {
  prompt_analysis: boolean
  generated_answer: boolean
  attention_view: boolean
  network_analysis: boolean
  circuitsvis_attention: boolean
}

export interface TopPrediction {
  token: string
  confidence: number
  logit?: number | null
}

export interface GeneratedAnswerToken {
  token: string
  top_predictions: TopPrediction[]
}

export interface GeneratedAnswer {
  text: string
  tokens: string[]
  token_count: number
  is_correct: boolean
  is_valid_canonical: boolean
  validation_error?: string | null
}

export interface StrongestAttentionPair {
  query_index: number
  key_index: number
  query_token: string
  key_token: string
  weight: number
}

export interface AttentionHeadSummary {
  entropy: number
  max_weight: number
  mean_diagonal: number
  strongest_pair: StrongestAttentionPair
}

export interface AttentionSummary {
  heads: AttentionHeadSummary[][]
}

export interface NetworkControls {
  mlp_threshold: number
  top_k: number
  top_neurons: number
  selected_token_index?: number | null
}

export interface NetworkWarningAvailability {
  warnings: string[]
}

export interface TopNeuronActivation {
  neuron_index: number
  value: number
  abs_value: number
}

export interface MlpTokenSummary {
  token_index: number
  token: string
  active_count_positive?: number
  active_fraction_positive?: number
  active_count_abs?: number
  active_fraction_abs?: number
  mean_abs_activation?: number
  max_activation?: number
  max_abs_activation?: number
  output_norm?: number
  top_neurons: TopNeuronActivation[]
}

export interface MlpLayerSummary {
  layer: number
  availability: string
  source_hook?: string | null
  layer_summary: Record<string, number> | null
  tokens: MlpTokenSummary[]
}

export interface MlpNetworkSummary {
  availability: string
  threshold: number
  layers: MlpLayerSummary[]
}

export interface AttentionArgmaxKey {
  query_index: number
  query_token: string
  key_index: number
  key_token: string
  weight: number
}

export interface AttentionHeadActivity {
  layer: number
  head: number
  mean_entropy: number
  entropy_by_query: number[]
  max_weight: number
  self_attention_mass: number
  previous_token_mass: number
  strongest_pair: StrongestAttentionPair
  argmax_keys: AttentionArgmaxKey[]
  result_norm_by_token?: number[] | null
}

export interface AttentionLayerActivity {
  layer: number
  availability: string
  source_hook?: string | null
  result_hook?: string | null
  heads: AttentionHeadActivity[]
}

export interface AttentionNetworkSummary {
  availability: string
  layers: AttentionLayerActivity[]
}

export interface ResidualDimensionSummary {
  dimension: number
  value: number
  abs_value: number
}

export interface LogitLensEntry {
  token: string
  probability: number
  logit: number
}

export interface ResidualTokenSummary {
  token_index: number
  token: string
  norm: number
  attention_delta_norm?: number | null
  cosine_to_previous_mid?: number | null
  cosine_to_final?: number | null
  top_dimensions: ResidualDimensionSummary[]
  logit_lens_top_k: LogitLensEntry[]
}

export interface ResidualLayerSummary {
  layer: number
  availability: string
  source_hook?: string | null
  pre_hook?: string | null
  tokens: ResidualTokenSummary[]
}

export interface ResidualNetworkSummary {
  availability: string
  layers: ResidualLayerSummary[]
}

export interface NetworkAnalysis {
  availability: NetworkWarningAvailability
  controls: NetworkControls
  mlp: MlpNetworkSummary
  attention: AttentionNetworkSummary
  residual: ResidualNetworkSummary
}

export interface ActivationSummary {
  token_layer_l2: number[][]
  token_layer_max_abs: number[][]
  layer_mean_l2: number[]
  layer_peak_l2: number[]
  token_peak_l2: number[]
  global_max_abs: number
}

export interface ModelConfig {
  n_layers: number
  n_heads: number
  d_model: number
  d_head: number
  mlp_hidden?: number | null
  sequence_length: number
  vocab_size: number
  dropout?: number | null
}

export interface ProblemMetadata {
  category: string
  kind: string
  curriculum_group: string
}

export interface CheckpointInfo {
  path: string
  device: string
  epoch: number | null
  exact_match: number | null
  val_loss: number | null
  train_loss: number | null
  checkpoint_schema_version: number | null
}

export interface RuntimeMetadata {
  position_encoding?: PositionEncoding | null
  analysis_runtime?: AnalysisRuntime | null
  capabilities?: AnalysisCapabilities | null
}

export interface AnalysisResult extends RuntimeMetadata {
  tokens: string[]
  attention?: number[][][][] | null
  activations: number[][][]
  logits: number[][]
  top_predictions: TopPrediction[]
  top_k_predictions: TopPrediction[][]
  attention_summary?: AttentionSummary | null
  activation_summary: ActivationSummary
  answer_position: number
  generated_answer: GeneratedAnswer
  generated_answer_top_k: GeneratedAnswerToken[]
  config: ModelConfig
  problem?: ProblemMetadata | null
  checkpoint: CheckpointInfo
  network?: NetworkAnalysis | null
}

export interface HealthResponse extends RuntimeMetadata {
  status: string
  device: string
  checkpoint: CheckpointInfo
  detail?: string | null
}

export interface ApiErrorDetail {
  detail?: string
}

export interface AnalyzePromptOptions extends Partial<NetworkControls> {
  include_network?: boolean
}

export interface ExportAnalysisOptions extends Partial<NetworkControls> {
  sections?: Array<'raw' | 'tables' | 'tensors' | 'markdown' | 'png'>
  output_format?: 'zip'
}
