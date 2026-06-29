'use client'

import { useState, useRef, useCallback, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import Papa from 'papaparse'
import { useStatLab } from '@/components/StatLabProvider'
import type { DatasetSchema, AnalysisRequest, Column } from '@/lib/types'

const PIPELINE_LABELS = {
  idle: '',
  parsing: 'Parsing CSV…',
  profiling: 'AI is profiling your dataset…',
  analysing: 'Running statistical computations…',
  interpreting: 'Generating AI interpretation…',
  done: 'Done',
  error: 'Error',
}

export default function HomePage() {
  const router = useRouter()
  
  const {
    file, setFile,
    schema, setSchema,
    mode, setMode,
    setManualRequest,
    pipelineStatus, pipelineError,
    submit,
  } = useStatLab()

  const [dragging, setDragging] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const parseFile = useCallback((f: File) => {
    setFile(f)
    Papa.parse(f, {
      header: true,
      skipEmptyLines: true,
      complete(results) {
        const fields = results.meta.fields ?? []
        const sampleRows = results.data as Record<string, string>[]

          const columns: Column[] = fields.map(name => {
          const values = sampleRows.map(r => r[name]).filter(Boolean)
          const numeric = values.every(v => !isNaN(Number(v)))
          return {
            name,
            type: numeric ? ('continuous' as const) : ('categorical' as const),
            uniqueValues: [...new Set(values)],
            sampleValues: values.slice(0, 5),
            nullCount: sampleRows.filter(r => !r[name]).length,
          }
        })

        const s = {
          fileName: f.name,
          rowCount: sampleRows.length,
          columnCount: fields.length,
          columns,
          sampleRows: sampleRows.slice(0, 200),
        }
        setSchema(s)
      },
    })
  }, [setFile, setSchema])

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setDragging(false)
    const f = e.dataTransfer.files[0]
    if (f && f.name.endsWith('.csv')) parseFile(f)
  }, [parseFile])

  const handleSubmit = useCallback(async () => {
    const session = await submit()
    if (session) {
      router.push('/analyse')
    }
  }, [submit, router])

  const isRunning = ['parsing', 'profiling', 'analysing', 'interpreting'].includes(pipelineStatus)
  const cols = schema?.columns ?? []

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-50 selection:bg-emerald-500/20 antialiased">
      {/* Home Navigation Header */}
      <header className="border-b border-zinc-800 px-6 py-4 flex items-center justify-between bg-zinc-900/80 backdrop-blur-md sticky top-0 z-50 shadow-sm">
        <div className="flex items-center gap-2.5">
          <div className="w-6 h-6 rounded bg-emerald-500 flex items-center justify-center shadow-sm">
            <svg className="w-3.5 h-3.5 text-zinc-950" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M3 13.5l5-5 4 4 5-6 4 3" />
            </svg>
          </div>
          <span className="font-semibold text-white tracking-tight text-base">StatLabs</span>
          <span className="text-xs text-zinc-500 font-mono">Stateless Statistical Analysis</span>
        </div>

        {/* Hover-Expanding Go-to Button */}
        <button 
          onClick={() => router.push('/analyse')}
          className="group flex items-center justify-center h-9 w-9 hover:w-32 bg-emerald-600 hover:bg-emerald-700 text-white rounded-lg transition-all duration-300 ease-in-out overflow-hidden shadow-sm font-medium text-xs whitespace-nowrap"
        >
          <span className="w-0 opacity-0 group-hover:w-20 group-hover:opacity-100 group-hover:mr-1 transition-all duration-300 ease-in-out text-center">
            To analyses
          </span>
          <svg className="w-3.5 h-3.5 transform group-hover:translate-x-0.5 transition-transform duration-200 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
          </svg>
        </button>
      </header>

      <main className="max-w-3xl mx-auto px-6 py-16 space-y-12">
        {/* Hero Section */}
        <div className="space-y-2.5">
          <h1 className="text-3xl font-bold tracking-tight text-white">
            Upload a dataset.{' '}
            <span className="text-emerald-400 font-semibold">Get instant insights.</span>
          </h1>
          <p className="text-zinc-400 text-sm max-w-lg leading-relaxed">
            Drop a CSV file, choose your analysis mode, and StatLabs will compute
            descriptive stats, correlations, regressions, and AI interpretations — all in your browser.
          </p>
        </div>

        {/* Drop Zone */}
        <div
          onDragOver={e => { e.preventDefault(); setDragging(true) }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          onClick={() => inputRef.current?.click()}
          className={`relative cursor-pointer rounded-xl border border-dashed transition-all p-12 flex flex-col items-center gap-3 text-center
            ${dragging ? 'border-emerald-500 bg-emerald-950/20' : 'border-zinc-800 bg-zinc-900/50 hover:border-emerald-500/50 hover:bg-zinc-900 shadow-sm'}`}
        >
          <input
            ref={inputRef}
            type="file"
            accept=".csv"
            className="hidden"
            onChange={e => {
              const f = e.target.files?.[0]
              if (f) parseFile(f)
            }}
          />
          <div className="w-10 h-10 rounded-lg bg-zinc-900 border border-zinc-800 flex items-center justify-center text-emerald-400 shadow-sm">
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 13h6m-3-3v6m5 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
          </div>
          {file ? (
            <div className="space-y-0.5">
              <p className="text-sm font-semibold text-emerald-400">{file.name}</p>
              <p className="text-xs text-zinc-500">{cols.length} columns detected · click to replace</p>
            </div>
          ) : (
            <div className="space-y-0.5">
              <p className="text-sm font-medium text-zinc-200">Drop your CSV here</p>
              <p className="text-xs text-zinc-500">or click to browse files</p>
            </div>
          )}
        </div>

        {/* Schema Preview */}
        {schema && (
          <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 overflow-hidden shadow-sm">
            <div className="px-4 py-3 border-b border-zinc-800 bg-zinc-900/80 flex items-center justify-between">
              <span className="text-xs font-medium text-zinc-400">Detected Schema</span>
              <span className="text-[11px] text-emerald-400 font-mono bg-emerald-950/50 border border-emerald-900/50 px-1.5 py-0.5 rounded">
                {cols.length} cols · {schema.rowCount.toLocaleString()} rows
              </span>
            </div>
            <div className="p-4 flex flex-wrap gap-1.5">
              {cols.map(col => (
                <span key={col.name} className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-zinc-900 border border-zinc-800 text-xs font-mono text-zinc-300">
                  <span className={`w-1.5 h-1.5 rounded-full ${col.type === 'continuous' ? 'bg-emerald-400' : 'bg-indigo-400'}`} />
                  <span className="text-zinc-100 font-medium">{col.name}</span>
                  <span className="text-zinc-500 text-[10px] lowercase">{col.type}</span>
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Mode Selection */}
        {schema && (
          <div className="space-y-3">
            <p className="text-[11px] font-bold text-zinc-500 uppercase tracking-wider">Analysis Mode</p>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {(['smart', 'manual'] as const).map(m => (
                <button
                  key={m}
                  type="button"
                  onClick={() => setMode(m)}
                  className={`rounded-xl border p-4 text-left transition-all ${
                    mode === m
                      ? 'border-emerald-500 bg-zinc-900 ring-1 ring-emerald-500 shadow-sm'
                      : 'border-zinc-800 bg-zinc-900/50 hover:border-zinc-700 hover:bg-zinc-900'
                  }`}
                >
                  <div className="flex items-start gap-2.5">
                    <div className={`mt-0.5 w-3.5 h-3.5 rounded-full border flex items-center justify-center flex-shrink-0 transition-colors ${
                      mode === m ? 'border-emerald-500' : 'border-zinc-700'
                    }`}>
                      {mode === m && <div className="w-1.5 h-1.5 rounded-full bg-emerald-500" />}
                    </div>
                    <div>
                      <p className={`text-sm font-medium ${mode === m ? 'text-emerald-400' : 'text-zinc-100'}`}>{m === 'smart' ? 'Smart Analyse' : 'Manual Mode'}</p>
                      <p className="text-xs text-zinc-400 mt-0.5 leading-normal">
                        {m === 'smart'
                          ? 'AI selects optimal analyses and charts automatically.'
                          : 'Configure columns, tests, and metrics manually.'}
                      </p>
                    </div>
                  </div>
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Configuration Setup Blocks */}
        {schema && mode === 'manual' && (
          <ManualConfig key={schema.fileName} schema={schema} onChange={setManualRequest} />
        )}

        {/* Pipeline Monitor States */}
        {isRunning && (
          <div className="rounded-xl border border-emerald-900/50 bg-emerald-950/20 px-4 py-3 flex items-center gap-3 animate-pulse">
            <div className="w-3.5 h-3.5 border-2 border-emerald-500 border-t-transparent rounded-full animate-spin flex-shrink-0" />
            <p className="text-xs font-medium text-emerald-400">{PIPELINE_LABELS[pipelineStatus]}</p>
          </div>
        )}

        {pipelineStatus === 'error' && pipelineError && (
          <div className="rounded-xl border border-red-900/50 bg-red-950/20 px-4 py-3">
            <p className="text-xs font-medium text-red-600">{pipelineError}</p>
          </div>
        )}

        {/* Action Trigger Button */}
        {schema && (
          <button
            onClick={handleSubmit}
            disabled={isRunning || !file}
            className="w-full py-3 rounded-xl bg-emerald-600 hover:bg-emerald-500 disabled:opacity-20 text-zinc-950 text-sm font-semibold transition-all shadow-md active:scale-[0.995]"
          >
            {isRunning ? PIPELINE_LABELS[pipelineStatus] : mode === 'smart' ? 'Run Smart Analysis' : 'Run Analysis'}
          </button>
        )}
      </main>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Manual Configuration Block Component
// ---------------------------------------------------------------------------
function ManualConfig({
  schema,
  onChange,
}: {
  schema: DatasetSchema
  onChange: (v: Partial<AnalysisRequest>) => void
}) {
  const cols = schema.columns
  const numericCols = cols.filter((c: Column) => c.type === 'continuous')

  const [descriptiveCols, setDescriptiveCols] = useState<string[]>([])
  const [corrA, setCorrA] = useState('')
  const [corrB, setCorrB] = useState('')
  const [corrPairs, setCorrPairs] = useState<[string, string][]>([])
  const [dependent, setDependent] = useState('')
  const [predictors, setPredictors] = useState<string[]>([])
  const [toast, setToast] = useState<string | null>(null)

  const showToast = (msg: string) => {
    setToast(msg)
    setTimeout(() => setToast(null), 4000)
  }

  useEffect(() => {
    if (dependent && predictors.length >= 1) {
      const predTypes = predictors.map((p: string) => cols.find((c: Column) => c.name === p)?.type)
      const hasContinuous = predTypes.some(t => t === 'continuous')
      const hasCategorical = predTypes.some(t => t === 'categorical' || t === 'binary')
      if (hasContinuous && hasCategorical) {
        showToast('Mixed column types for prediction — categorical predictors will be one-hot encoded.')
      }
    }
  }, [dependent, predictors, cols])

  useEffect(() => {
    onChange({
      mode: 'manual',
      descriptive: descriptiveCols.length > 0 ? {
        columns: descriptiveCols,
        measures: ['central', 'spread', 'distribution'],
      } : undefined,
      inferential: corrPairs.length > 0 || dependent ? {
        correlationPairs: corrPairs,
        regression: dependent ? { dependent, predictors } : undefined,
      } : undefined,
      predictive: dependent ? {
        dependent,
        predictors,
      } : undefined,
    })
  }, [descriptiveCols, corrPairs, dependent, predictors, onChange])

  const addPair = () => {
    if (corrA && corrB && corrA !== corrB) {
      const pairExists = corrPairs.some(p => (p[0] === corrA && p[1] === corrB) || (p[0] === corrB && p[1] === corrA))
      if (!pairExists) setCorrPairs(prev => [...prev, [corrA, corrB]])
      setCorrA('')
      setCorrB('')
    }
  }

  return (
    <div className="space-y-5 rounded-xl border border-zinc-800 bg-zinc-900/30 p-5 shadow-sm relative">
      {toast && (
        <div className="absolute top-3 right-3 left-3 z-10 rounded-lg bg-amber-950/90 border border-amber-800/50 px-4 py-2.5 text-xs text-amber-400 shadow-lg animate-in fade-in slide-in-from-top-1">
          {toast}
        </div>
      )}
      <p className="text-xs font-semibold text-zinc-300">Manual Configuration</p>

      {/* Descriptive Selector */}
      <div className="space-y-1.5">
        <label className="text-[10px] text-zinc-500 font-bold uppercase tracking-wider">Descriptive Columns</label>
        <div className="flex flex-wrap gap-1.5">
          {cols.map((col: Column) => {
            const isSelected = descriptiveCols.includes(col.name)
            return (
              <button
                key={col.name}
                type="button"
                onClick={() => setDescriptiveCols(p => isSelected ? p.filter(c => c !== col.name) : [...p, col.name])}
                className={`px-2.5 py-1 rounded-md text-xs font-mono transition-colors border ${
                  isSelected
                    ? 'bg-emerald-600 border-emerald-600 text-white shadow-sm'
                    : 'bg-zinc-900 border-zinc-800 text-zinc-400 hover:border-zinc-700'
                }`}
              >
                {col.name}
              </button>
            )
          })}
        </div>
      </div>

      {/* Metric Associations */}
      <div className="space-y-1.5">
        <label className="text-[10px] text-zinc-500 font-bold uppercase tracking-wider">Correlation Pairs</label>
        <div className="flex gap-2">
          <select value={corrA} onChange={e => setCorrA(e.target.value)} className="flex-1 bg-zinc-900 border border-zinc-800 rounded-lg px-2.5 py-1.5 text-xs text-zinc-300 focus:outline-none focus:border-zinc-700">
            <option value="">Column A</option>
            {numericCols.map((c: Column) => <option key={c.name} value={c.name} className="bg-zinc-900">{c.name}</option>)}
          </select>
          <select value={corrB} onChange={e => setCorrB(e.target.value)} className="flex-1 bg-zinc-900 border border-zinc-800 rounded-lg px-2.5 py-1.5 text-xs text-zinc-300 focus:outline-none focus:border-zinc-700">
            <option value="">Column B</option>
            {numericCols.map((c: Column) => <option key={c.name} value={c.name} className="bg-zinc-900">{c.name}</option>)}
          </select>
          <button type="button" onClick={addPair} className="px-3 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-xs text-zinc-950 font-semibold transition-colors shadow-sm">Add</button>
        </div>
        {corrPairs.length > 0 && (
          <div className="flex flex-wrap gap-1.5 mt-1.5">
            {corrPairs.map((pair, i) => (
              <span key={i} className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-zinc-900 border border-zinc-800 text-xs text-zinc-300 font-mono">
                {pair[0]} ↔ {pair[1]}
                <button type="button" onClick={() => setCorrPairs(p => p.filter((_, j) => j !== i))} className="ml-1 text-zinc-500 hover:text-zinc-200">×</button>
              </span>
            ))}
          </div>
        )}
      </div>

      {/* Regression Target */}
      <div className="space-y-1.5">
        <label className="text-[10px] text-zinc-500 font-bold uppercase tracking-wider">Regression Target</label>
        <select value={dependent} onChange={e => { setDependent(e.target.value); setPredictors([]) }} className="w-full bg-zinc-900 border border-zinc-800 rounded-lg px-2.5 py-1.5 text-xs text-zinc-300 focus:outline-none focus:border-zinc-700">
          <option value="">None</option>
          {cols.map((c: Column) => <option key={c.name} value={c.name} className="bg-zinc-900">{c.name}</option>)}
        </select>
        {dependent && (
          <div className="mt-2.5 space-y-1.5">
            <p className="text-[10px] text-zinc-500 font-bold uppercase tracking-wider">Predictors</p>
            <div className="flex flex-wrap gap-1.5">
              {cols.filter((c: Column) => c.name !== dependent).map((col: Column) => {
                const isPred = predictors.includes(col.name)
                return (
                  <button
                    key={col.name}
                    type="button"
                    onClick={() => setPredictors(p => isPred ? p.filter(c => c !== col.name) : [...p, col.name])}
                    className={`px-2.5 py-1 rounded-md text-xs font-mono transition-colors border ${
                      isPred
                        ? 'bg-emerald-600 border-emerald-600 text-white shadow-sm'
                        : 'bg-zinc-900 border-zinc-800 text-zinc-400 hover:border-zinc-700'
                    }`}
                  >
                    {col.name}
                  </button>
                )
              })}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}