import axios from 'axios'

export interface TopPrediction {
  token: string
  confidence: number
  logit?: number | null
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

export interface AnalysisResult {
  tokens: string[]
  attention: number[][][][]
  activations: number[][][]
  logits: number[][]
  top_predictions: TopPrediction[]
  top_k_predictions: TopPrediction[][]
  attention_summary: AttentionSummary
  activation_summary: ActivationSummary
  answer_position: number
  config: ModelConfig
  checkpoint: CheckpointInfo
}

export interface HealthResponse {
  status: string
  device: string
  checkpoint: CheckpointInfo
  detail?: string | null
}

export interface ApiErrorDetail {
  detail?: string
}

export async function analyzePrompt(prompt: string): Promise<AnalysisResult> {
  const response = await axios.post<AnalysisResult>('/api/analyze', { prompt })
  return response.data
}

export async function getHealth(): Promise<HealthResponse> {
  const response = await axios.get<HealthResponse>('/api/health')
  return response.data
}
