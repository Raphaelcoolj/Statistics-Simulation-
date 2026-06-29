'use client'

import { useState, useCallback, useEffect } from 'react'
import type {
  DatasetSchema,
  AnalysisRequest,
  AnalysisResult,
  ChartSuggestion,
} from '@/lib/types'

export type AnalysisMode = 'smart' | 'manual'

export interface InterpretResult {
  summary: string
  perAnalysis: { type: string; subject: string; interpretation: string }[]
  provider: string
  fallbackUsed: boolean
}

export interface AnalysisSession {
  id: string
  timestamp: number
  fileName: string
  schema: DatasetSchema
  result: AnalysisResult
  chartSuggestions: ChartSuggestion[]
  interpret: InterpretResult
}

export type PipelineStatus =
  | 'idle'
  | 'parsing'
  | 'profiling'
  | 'analysing'
  | 'interpreting'
  | 'done'
  | 'error'

const HISTORY_KEY = 'statlab_history'

function loadHistory(): AnalysisSession[] {
  if (typeof window === 'undefined') return []
  try {
    return JSON.parse(localStorage.getItem(HISTORY_KEY) ?? '[]')
  } catch {
    return []
  }
}

function saveHistory(sessions: AnalysisSession[]) {
  localStorage.setItem(HISTORY_KEY, JSON.stringify(sessions.slice(0, 20)))
}

export function useStatLab() {
  const [file, setFile] = useState<File | null>(null)
  const [schema, setSchema] = useState<DatasetSchema | null>(null)
  const [mode, setMode] = useState<AnalysisMode>('smart')
  const [manualRequest, setManualRequest] = useState<Partial<AnalysisRequest>>({})
  const [pipelineStatus, setPipelineStatus] = useState<PipelineStatus>('idle')
  const [pipelineError, setPipelineError] = useState<string | null>(null)
  const [currentSession, setCurrentSession] = useState<AnalysisSession | null>(null)
  const [history, setHistory] = useState<AnalysisSession[]>([])

  useEffect(() => {
    setHistory(loadHistory())
  }, [])

  const pushToHistory = useCallback((session: AnalysisSession) => {
    setHistory(prev => {
      const next = [session, ...prev.filter(s => s.id !== session.id)]
      saveHistory(next)
      return next
    })
  }, [])

  const runSmartPipeline = useCallback(async (csvFile: File, csvSchema: DatasetSchema) => {
    setPipelineStatus('profiling')
    setPipelineError(null)

    // Step 1: Profile
    const profileRes = await fetch('/api/profile', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ schema: csvSchema }),
    })
    const profileData = await profileRes.json()

    if (!profileData.success) {
      console.warn('[StatLab] Profiling failed, using rule-based fallback:', profileData.error)
    }

    const { analysisMap, chartSuggestions } = profileData.success
      ? profileData.output
      : {
          analysisMap: {
            modelType: 'linear' as const,
            dependentVariable: null,
            predictors: [],
            correlationPairs: [],
            hypothesisTests: [],
            descriptiveColumns: csvSchema.columns.map(c => c.name),
          },
          chartSuggestions: csvSchema.columns
            .filter(c => c.type === 'continuous')
            .map(c => ({
              chartType: 'histogram' as const,
              title: `Distribution of ${c.name}`,
              reason: 'Fallback histogram for continuous column',
              column: c.name,
            })),
        }

    const chartColumns = new Set(chartSuggestions.flatMap((s: ChartSuggestion) => [s.column, s.x, s.y].filter(Boolean) as string[]))

    const analyses: AnalysisRequest = {
      mode: 'smart',
      descriptive: {
        columns: [...new Set([...analysisMap.descriptiveColumns, ...chartColumns])],
        measures: ['central', 'spread', 'distribution'],
      },
      inferential: {
        correlationPairs: analysisMap.correlationPairs,
        hypothesisTests: analysisMap.hypothesisTests,
        regression: analysisMap.dependentVariable
          ? { dependent: analysisMap.dependentVariable, predictors: analysisMap.predictors }
          : undefined,
      },
      predictive: analysisMap.dependentVariable
        ? {
            dependent: analysisMap.dependentVariable,
            predictors: analysisMap.predictors,
            modelType: analysisMap.modelType,
          }
        : undefined,
    }

    // Step 2: Analyse
    setPipelineStatus('analysing')
    const form = new FormData()
    form.append('file', csvFile)
    form.append('analyses', JSON.stringify(analyses))
    const analyseRes = await fetch('/api/analyse', { method: 'POST', body: form })
    const analyseData = await analyseRes.json()
    if (!analyseData.success) throw new Error(analyseData.error ?? 'Analysis failed')

    const result: AnalysisResult = analyseData.result
    const finalSchema: DatasetSchema = analyseData.schema ?? csvSchema

    // Step 3: Interpret
    setPipelineStatus('interpreting')
    const interpretRes = await fetch('/api/interpret', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ schema: finalSchema, result }),
    })
    const interpretData = await interpretRes.json()
    // Always success per spec
    const interpret: InterpretResult = {
      summary: interpretData.summary ?? '',
      perAnalysis: interpretData.perAnalysis ?? [],
      provider: interpretData.provider ?? '',
      fallbackUsed: interpretData.fallbackUsed ?? false,
    }

    const session: AnalysisSession = {
      id: `${Date.now()}-${Math.random().toString(36).slice(2)}`,
      timestamp: Date.now(),
      fileName: csvFile.name,
      schema: finalSchema,
      result,
      chartSuggestions: chartSuggestions ?? result.chartSuggestions ?? [],
      interpret,
    }

    pushToHistory(session)
    setCurrentSession(session)
    setPipelineStatus('done')
    return session
  }, [pushToHistory])

  const runManualPipeline = useCallback(async (csvFile: File, csvSchema: DatasetSchema, analyses: AnalysisRequest) => {
    setPipelineStatus('analysing')
    setPipelineError(null)

    const form = new FormData()
    form.append('file', csvFile)
    form.append('analyses', JSON.stringify(analyses))
    const analyseRes = await fetch('/api/analyse', { method: 'POST', body: form })
    const analyseData = await analyseRes.json()
    if (!analyseData.success) throw new Error(analyseData.error ?? 'Analysis failed')

    const result: AnalysisResult = analyseData.result
    const finalSchema: DatasetSchema = analyseData.schema ?? csvSchema

    const session: AnalysisSession = {
      id: `${Date.now()}-${Math.random().toString(36).slice(2)}`,
      timestamp: Date.now(),
      fileName: csvFile.name,
      schema: finalSchema,
      result,
      chartSuggestions: result.chartSuggestions ?? [],
      interpret: { summary: '', perAnalysis: [], provider: '', fallbackUsed: false },
    }

    pushToHistory(session)
    setCurrentSession(session)
    setPipelineStatus('done')
    return session
  }, [pushToHistory])

  const submit = useCallback(async () => {
    if (!file || !schema) return null
    try {
      if (mode === 'smart') {
        return await runSmartPipeline(file, schema)
      } else {
        const analyses: AnalysisRequest = {
          mode: 'manual',
          ...manualRequest.descriptive && { descriptive: manualRequest.descriptive },
          ...manualRequest.inferential && { inferential: manualRequest.inferential },
          ...manualRequest.predictive && { predictive: manualRequest.predictive },
        }
        return await runManualPipeline(file, schema, analyses)
      }
    } catch (err) {
      setPipelineError(err instanceof Error ? err.message : 'An unexpected error occurred')
      setPipelineStatus('error')
      return null
    }
  }, [file, schema, mode, manualRequest, runSmartPipeline, runManualPipeline])

  const loadSession = useCallback((session: AnalysisSession) => {
    setCurrentSession(session)
    setPipelineStatus('done')
  }, [])

  const reset = useCallback(() => {
    setFile(null)
    setSchema(null)
    setMode('smart')
    setManualRequest({})
    setPipelineStatus('idle')
    setPipelineError(null)
    setCurrentSession(null)
  }, [])

  return {
    file, setFile,
    schema, setSchema,
    mode, setMode,
    manualRequest, setManualRequest,
    pipelineStatus, pipelineError,
    currentSession,
    history,
    submit,
    loadSession,
    reset,
  }
}
