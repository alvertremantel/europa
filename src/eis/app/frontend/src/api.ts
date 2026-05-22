import axios from 'axios'
import type { AnalysisResult, AnalyzePromptOptions, ExportAnalysisOptions, HealthResponse } from './types/api'

export type {
  AnalysisCapabilities,
  AnalysisRuntime,
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
  GeneratedAnswer,
  GeneratedAnswerToken,
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
  ProblemMetadata,
  PositionEncoding,
  ResidualDimensionSummary,
  ResidualLayerSummary,
  ResidualNetworkSummary,
  RuntimeMetadata,
  ResidualTokenSummary,
  StrongestAttentionPair,
  TopNeuronActivation,
  TopPrediction,
  ExportAnalysisOptions,
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

export async function exportAnalysisDump(
  prompt: string,
  options: ExportAnalysisOptions = {},
  signal?: AbortSignal,
): Promise<{ blob: Blob; filename: string | null }> {
  try {
    const response = await axios.post('/api/export', { prompt, ...options }, { signal, responseType: 'blob' })
    const disposition = String(response.headers['content-disposition'] ?? '')
    const match = disposition.match(/filename="?([^";]+)"?/)
    return {
      blob: response.data as Blob,
      filename: match?.[1] ?? null,
    }
  } catch (error: unknown) {
    if (axios.isAxiosError(error) && error.response?.data instanceof Blob) {
      const text = await error.response.data.text()
      try {
        const parsed: unknown = JSON.parse(text)
        if (parsed && typeof parsed === 'object' && 'detail' in parsed) {
          return Promise.reject(new Error(String((parsed as Record<string, unknown>).detail), { cause: error }))
        }
      } catch {
        // If JSON parsing fails, fall through to the plain text message.
      }
      if (text.trim()) {
        throw new Error(text, { cause: error })
      }
    }
    throw error
  }
}
