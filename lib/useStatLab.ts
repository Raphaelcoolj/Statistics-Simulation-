'use client'

import { useState, useCallback, useEffect } from 'react'
import type {
  DatasetSchema,
  AnalysisRequest,
  AnalysisResult,
  ChartSuggestion,
  RelationshipSuggestion,
  PredictiveResult,
  ModelType,
} from '@/lib/types'
import { uncertainCodeColumns, applyCodeDetection } from '@/lib/utils/labels'

export type AnalysisMode = 'smart' | 'manual'

export interface InterpretResult {
  summary: string
  perAnalysis: { type: string; subject?: string; interpretation: string }[]
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
  relationshipSuggestions?: RelationshipSuggestion[]
  modelTrainingReport?: Record<string, unknown> | null
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
  const [codebook, setCodebook] = useState<Record<string, Record<string, string>> | null>(null)
  const [pipelineStatus, setPipelineStatus] = useState<PipelineStatus>('idle')
  const [pipelineError, setPipelineError] = useState<string | null>(null)
  const [currentSession, setCurrentSession] = useState<AnalysisSession | null>(null)
  const [history, setHistory] = useState<AnalysisSession[]>([])

  useEffect(() => {
    setHistory(loadHistory())
  }, [])

  /** Strip heavy fields before sending schema over the wire */
  const lightweightSchema = useCallback((s: DatasetSchema): DatasetSchema => ({
    ...s,
    fullData: undefined,
    sampleRows: s.sampleRows?.slice(0, 50),
    columns: s.columns.map(c => ({
      ...c,
      uniqueValues: c.uniqueValues && c.uniqueValues.length > 20
        ? c.uniqueValues.slice(0, 20)
        : c.uniqueValues,
    })),
  }), [])

  const pushToHistory = useCallback((session: AnalysisSession) => {
    setHistory(prev => {
      const next = [session, ...prev.filter(s => s.id !== session.id)]
      saveHistory(next)
      return next
    })
  }, [])

  // Optional AI inspection of columns the code-detection heuristics were unsure
  // about. Runs only when there are uncertain columns; never blocks on failure.
  const resolveUncertainCodes = useCallback(async (schema: DatasetSchema): Promise<DatasetSchema> => {
    const uncertain = uncertainCodeColumns(schema)
    if (uncertain.length === 0) return schema
    try {
      const res = await fetch('/api/detect-codes', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ columns: uncertain }),
      })
      const data = await res.json()
      if (data?.success && data.output?.columns) {
        const columns = applyCodeDetection(schema, data.output.columns)
        return { ...schema, columns }
      }
    } catch {
      // ignore — heuristic outcome stands
    }
    return schema
  }, [])

  const runSmartPipeline = useCallback(async (csvFile: File, csvSchema: DatasetSchema, codebook?: Record<string, Record<string, string>>) => {
    setPipelineStatus('profiling')
    setPipelineError(null)

    // Step 1: Profile
    const profileRes = await fetch('/api/profile', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ schema: lightweightSchema(csvSchema) }),
    })
    const profileData = await profileRes.json()

    if (!profileData.success) {
      console.warn('[StatLab] Profiling failed, using rule-based fallback:', profileData.error)
    }

    const { analysisMap, chartSuggestions, relationshipSuggestions } = profileData.success
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
          relationshipSuggestions: [],
        }

    // Derive relationships: use AI suggestions if available, otherwise build from analysisMap
    const relSuggestions: RelationshipSuggestion[] = relationshipSuggestions?.length > 0
      ? relationshipSuggestions
      : analysisMap.dependentVariable
        ? [{
            dependent: analysisMap.dependentVariable,
            predictors: analysisMap.predictors,
            modelType: analysisMap.modelType,
            reason: analysisMap.dependentVariable && analysisMap.predictors.length > 0
              ? `AI identified "${analysisMap.dependentVariable}" as target with ${analysisMap.predictors.length} predictor(s)`
              : 'Primary relationship from profiling',
          }]
        : []

    // Use first suggestion as primary model for the initial analysis
    const primaryRel = relSuggestions[0]

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
        regression: primaryRel
          ? { dependent: primaryRel.dependent, predictors: primaryRel.predictors }
          : undefined,
      },
      predictive: primaryRel
        ? {
            dependent: primaryRel.dependent,
            predictors: primaryRel.predictors,
            modelType: primaryRel.modelType,
          }
        : undefined,
    }

    // Step 2: Analyse
    setPipelineStatus('analysing')
    const form = new FormData()
    form.append('file', csvFile)
    form.append('analyses', JSON.stringify(analyses))
    if (codebook) form.append('codebook', JSON.stringify(codebook))
    if (primaryRel) form.append('model_training', JSON.stringify({ enabled: true }))
    const analyseRes = await fetch('/api/analyse', { method: 'POST', body: form })
    const analyseData = await analyseRes.json()
    if (!analyseData.success) throw new Error(analyseData.error ?? 'Analysis failed')

    const result: AnalysisResult = analyseData.result
    const modelTrainingReport = analyseData.modelTrainingReport ?? null
    let finalSchema: DatasetSchema = analyseData.schema ?? csvSchema
    finalSchema = await resolveUncertainCodes(finalSchema)

    // Step 3: Interpret
    setPipelineStatus('interpreting')
    const interpretRes = await fetch('/api/interpret', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ schema: lightweightSchema(finalSchema), result, modelTrainingReport }),
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
      relationshipSuggestions: relSuggestions,
      modelTrainingReport,
    }

    pushToHistory(session)
    setCurrentSession(session)
    setPipelineStatus('done')
    return session
  }, [pushToHistory, resolveUncertainCodes, lightweightSchema])

  const runManualPipeline = useCallback(async (csvFile: File, csvSchema: DatasetSchema, analyses: AnalysisRequest, codebook?: Record<string, Record<string, string>>, modelTraining?: Record<string, unknown>) => {
    setPipelineStatus('analysing')
    setPipelineError(null)

    const form = new FormData()
    form.append('file', csvFile)
    form.append('analyses', JSON.stringify(analyses))
    if (codebook) form.append('codebook', JSON.stringify(codebook))
    if (modelTraining?.enabled) form.append('model_training', JSON.stringify(modelTraining))
    const analyseRes = await fetch('/api/analyse', { method: 'POST', body: form })
    const analyseData = await analyseRes.json()
    if (!analyseData.success) throw new Error(analyseData.error ?? 'Analysis failed')

    const result: AnalysisResult = analyseData.result
    const modelTrainingReport = analyseData.modelTrainingReport ?? null
    let finalSchema: DatasetSchema = analyseData.schema ?? csvSchema
    finalSchema = await resolveUncertainCodes(finalSchema)

    const session: AnalysisSession = {
      id: `${Date.now()}-${Math.random().toString(36).slice(2)}`,
      timestamp: Date.now(),
      fileName: csvFile.name,
      schema: finalSchema,
      result,
      chartSuggestions: result.chartSuggestions ?? [],
      interpret: { summary: '', perAnalysis: [], provider: '', fallbackUsed: false },
      relationshipSuggestions: [],
      modelTrainingReport,
    }

    pushToHistory(session)
    setCurrentSession(session)
    setPipelineStatus('done')
    return session
  }, [pushToHistory, resolveUncertainCodes])

  const runPredictiveModel = useCallback(async (
    csvFile: File, csvSchema: DatasetSchema, dependent: string, predictors: string[], modelType?: string
  ): Promise<PredictiveResult | null> => {
    try {
      const analyses: AnalysisRequest = {
        mode: 'manual',
        predictive: { dependent, predictors, modelType: modelType as ModelType | undefined },
      }
      const form = new FormData()
      form.append('file', csvFile)
      form.append('analyses', JSON.stringify(analyses))
      if (codebook) form.append('codebook', JSON.stringify(codebook))
      const res = await fetch('/api/analyse', { method: 'POST', body: form })
      const data = await res.json()
      if (!data.success) throw new Error(data.error ?? 'Model failed')
      return data.result?.predictive ?? null
    } catch {
      return null
    }
  }, [codebook])

  const submit = useCallback(async () => {
    if (!file || !schema) return null
    try {
      if (mode === 'smart') {
        return await runSmartPipeline(file, schema, codebook ?? undefined)
      } else {
        const analyses: AnalysisRequest = {
          mode: 'manual',
          ...manualRequest.descriptive && { descriptive: manualRequest.descriptive },
          ...manualRequest.inferential && { inferential: manualRequest.inferential },
          ...manualRequest.predictive && { predictive: manualRequest.predictive },
        }
        return await runManualPipeline(file, schema, analyses, codebook ?? undefined, manualRequest.modelTraining)
      }
    } catch (err) {
      setPipelineError(err instanceof Error ? err.message : 'An unexpected error occurred')
      setPipelineStatus('error')
      return null
    }
  }, [file, schema, mode, manualRequest, codebook, runSmartPipeline, runManualPipeline])

  const loadSession = useCallback((session: AnalysisSession) => {
    setCurrentSession(session)
    setPipelineStatus('done')
  }, [])

  const reset = useCallback(() => {
    setFile(null)
    setSchema(null)
    setMode('smart')
    setManualRequest({})
    setCodebook(null)
    setPipelineStatus('idle')
    setPipelineError(null)
    setCurrentSession(null)
  }, [])

  return {
    file, setFile,
    schema, setSchema,
    mode, setMode,
    manualRequest, setManualRequest,
    codebook, setCodebook,
    pipelineStatus, pipelineError,
    currentSession,
    history,
    submit,
    loadSession,
    reset,
    runPredictiveModel,
  }
}
