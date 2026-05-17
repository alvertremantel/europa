import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import axios from 'axios'

import { analyzePrompt, getHealth } from '../api'
import { DEFAULT_NETWORK_CONTROLS, DEFAULT_PROMPT } from '../constants'
import type { AnalysisResult, ApiErrorDetail, HealthResponse, NetworkControls } from '../types/api'

type DetailTab = 'attention' | 'activations' | 'logits' | 'network'

export function useAnalysisSession() {
  const [prompt, setPrompt] = useState<string>(DEFAULT_PROMPT)
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [loadingNetwork, setLoadingNetwork] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [networkError, setNetworkError] = useState<string | null>(null)
  const [selectedLayer, setSelectedLayer] = useState(0)
  const [selectedAnswerTokenIndex, setSelectedAnswerTokenIndex] = useState(0)
  const [activeDetailTab, setActiveDetailTab] = useState<DetailTab>('attention')
  const [networkControls, setNetworkControls] = useState<NetworkControls>(DEFAULT_NETWORK_CONTROLS)
  const abortRef = useRef<AbortController | null>(null)

  const refreshHealth = useCallback(async () => {
    try {
      const nextHealth = await getHealth()
      setHealth(nextHealth)
    } catch {
      setHealth(null)
    }
  }, [])

  useEffect(() => {
    let cancelled = false

    void getHealth()
      .then((nextHealth) => {
        if (!cancelled) {
          setHealth(nextHealth)
        }
      })
      .catch(() => {
        if (!cancelled) {
          setHealth(null)
        }
      })

    return () => {
      cancelled = true
    }
  }, [])

  const generatedAnswer = useMemo(() => {
    if (!result) {
      return null
    }
    return result.generated_answer
  }, [result])

  const submitPrompt = useCallback(async (nextPrompt = prompt) => {
    const cleanedPrompt = nextPrompt.trim()
    if (!cleanedPrompt) {
      setError('Enter an arithmetic prompt before analyzing.')
      return
    }

    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller

    setLoading(true)
    setLoadingNetwork(false)
    setError(null)

    try {
      const includeNetwork = activeDetailTab === 'network'
      const analysis = await analyzePrompt(
        cleanedPrompt,
        {
          ...(includeNetwork ? networkControls : {}),
          include_network: includeNetwork,
        },
        controller.signal,
      )
      if (controller.signal.aborted) return
      setPrompt(cleanedPrompt)
      setResult(analysis)
      setNetworkError(null)
      setSelectedLayer((current) =>
        Math.min(current, Math.max(analysis.config.n_layers - 1, 0)),
      )
      setSelectedAnswerTokenIndex(0)
      void refreshHealth()
    } catch (caughtError) {
      if (!controller.signal.aborted) {
        setError(getErrorMessage(caughtError))
      }
    } finally {
      if (!controller.signal.aborted) {
        setLoading(false)
      }
    }
  }, [prompt, activeDetailTab, networkControls, refreshHealth])

  const requestNetworkAnalysis = useCallback(async (nextControls = networkControls) => {
    const cleanedPrompt = prompt.trim()
    if (!cleanedPrompt || loading || loadingNetwork) {
      return
    }

    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller

    setLoading(false)
    setLoadingNetwork(true)
    setNetworkError(null)
    setNetworkControls(nextControls)

    try {
      const analysis = await analyzePrompt(
        cleanedPrompt,
        {
          ...nextControls,
          include_network: true,
        },
        controller.signal,
      )
      if (controller.signal.aborted) return
      setResult(analysis)
      setSelectedLayer((current) =>
        Math.min(current, Math.max(analysis.config.n_layers - 1, 0)),
      )
      setSelectedAnswerTokenIndex((current) =>
        Math.min(current, Math.max(analysis.generated_answer_top_k.length - 1, 0)),
      )
    } catch (caughtError) {
      if (!controller.signal.aborted) {
        setNetworkError(getErrorMessage(caughtError))
      }
    } finally {
      if (!controller.signal.aborted) {
        setLoadingNetwork(false)
      }
    }
  }, [prompt, loading, loadingNetwork, networkControls])

  const openDetailTab = useCallback((tab: DetailTab) => {
    setActiveDetailTab(tab)
    // Network tab is loaded lazily; trigger fetch if not already present
    // (handled in the caller via setResult existence check)
  }, [])

  return {
    prompt,
    setPrompt,
    result,
    health,
    loading,
    loadingNetwork,
    error,
    networkError,
    selectedLayer,
    setSelectedLayer,
    activeDetailTab,
    setActiveDetailTab,
    networkControls,
    generatedAnswer,
    selectedAnswerTokenIndex,
    setSelectedAnswerTokenIndex,
    submitPrompt,
    requestNetworkAnalysis,
    openDetailTab,
    refreshHealth,
    setError,
    setNetworkError,
  }
}

function getErrorMessage(error: unknown): string {
  if (axios.isAxiosError<ApiErrorDetail>(error)) {
    return error.response?.data?.detail ?? 'Failed to analyze prompt. Check that the backend is running.'
  }
  if (error instanceof Error) {
    return error.message
  }
  return 'Failed to analyze prompt.'
}
