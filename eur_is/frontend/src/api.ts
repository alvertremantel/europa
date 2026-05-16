import axios from 'axios'
import type { AnalysisResult, AnalyzePromptOptions, HealthResponse } from './types/api'

export type {
  ActivationSummary,
  AnalysisResult,
  ApiErrorDetail,
  AttentionArgmaxKey,
  AttentionHeadActivity,
  AttentionHeadSummary,
  AttentionLayerActivity,
  AttentionNetworkSummary,
  AttentionSummary,
  AnalyzePromptOptions,
  CheckpointInfo,
  HealthResponse,
  LogitLensEntry,
  MlpLayerSummary,
  MlpNetworkSummary,
  MlpTokenSummary,
  ModelConfig,
  NetworkAnalysis,
  NetworkControls,
  NetworkWarningAvailability,
  ResidualDimensionSummary,
  ResidualLayerSummary,
  ResidualNetworkSummary,
  ResidualTokenSummary,
  StrongestAttentionPair,
  TopNeuronActivation,
  TopPrediction,
} from './types/api'

export async function analyzePrompt(
  prompt: string,
  options: AnalyzePromptOptions = {},
  signal?: AbortSignal,
): Promise<AnalysisResult> {
  const response = await axios.post<AnalysisResult>('/api/analyze', { prompt, ...options }, { signal })
  return response.data
}

export async function getHealth(): Promise<HealthResponse> {
  const response = await axios.get<HealthResponse>('/api/health')
  return response.data
}
