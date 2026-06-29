'use client'

import { useEffect, useMemo, useState } from 'react'
import { useRouter } from 'next/navigation'
import {
  ScatterChart, Scatter, BarChart, Bar, LineChart, Line, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
} from 'recharts'
import { usePDFExport } from '@/lib/pdf'
import { useStatLab } from '@/components/StatLabProvider'
import type { AnalysisSession } from '@/lib/useStatLab'
import type { ChartSuggestion, DescriptiveResult, CorrelationResult, HypothesisResult, RegressionResult, ModelType } from '@/lib/types'

const ACCENT = ['#34d399', '#818cf8', '#f472b6', '#fb923c', '#38bdf8', '#a78bfa']
const BIN_COUNT = 10

export default function AnalysePage() {
  const router = useRouter()
  const { currentSession, history, loadSession } = useStatLab()
  const { exportPDF, isGenerating } = usePDFExport()
  const [activeSession, setActiveSession] = useState<AnalysisSession | null>(currentSession)
  const [tab, setTab] = useState<'analysis' | 'predictions'>('analysis')

  useEffect(() => {
    if (!currentSession && history.length === 0) {
      router.push('/')
    } else if (!activeSession && currentSession) {
      setActiveSession(currentSession)
    } else if (!activeSession && history.length > 0) {
      setActiveSession(history[0])
    }
  }, [currentSession, history, activeSession, router])

  const predictionScatterData = useMemo(() => {
    const session = currentSession ?? activeSession
    const predReg = session?.result?.predictive?.regressionResult
    if (!predReg?.predictions) return []
    const preds = predReg.predictions
    const residuals = predReg.residuals ?? []
    const step = Math.max(1, Math.floor(preds.length / 500))
    const data: { actual: number; predicted: number }[] = []
    for (let i = 0; i < preds.length; i += step) {
      data.push({
        actual: preds[i] + (residuals[i] ?? 0),
        predicted: preds[i],
      })
    }
    return data
  }, [currentSession, activeSession])

  if (!activeSession) {
    return (
      <div className="min-h-screen bg-zinc-950 flex items-center justify-center">
        <div className="w-5 h-5 border-2 border-emerald-400 border-t-transparent rounded-full animate-spin" />
      </div>
    )
  }

  const { result, chartSuggestions, interpret, schema, fileName, timestamp } = activeSession
  const descriptive: DescriptiveResult[] = result.descriptive ?? []
  const correlations: CorrelationResult[] = result.inferential?.correlations ?? []
  const hypothesisTests: HypothesisResult[] = result.inferential?.hypothesisTests ?? []
  const predictive = result.predictive
  const alignmentHasPredictive = !!(predictive?.modelType || predictive?.regressionResult)
  const predReg = predictive?.regressionResult
  const featureImportance = predReg?.featureImportance

  const handleExport = () => exportPDF({
    title: `${fileName} — Analysis Report`,
    fileName: `statlab-${fileName.replace('.csv', '')}.pdf`,
    includeTimestamp: true,
    sections: [
      ...(interpret.summary ? [{ elementId: 'ai-summary', title: 'AI Summary' }] : []),
      { elementId: 'descriptive-stats', title: 'Descriptive Statistics' },
      ...(chartSuggestions.length > 0 ? [{ elementId: 'charts-panel', title: 'Charts & Visualisations' }] : []),
      ...((correlations.length > 0 || hypothesisTests.length > 0) ? [{ elementId: 'inferential-panel', title: 'Inferential Analysis' }] : []),
      ...(alignmentHasPredictive ? [{ elementId: 'predictive-panel', title: 'Predictive Model' }] : []),
      ...(interpret.perAnalysis.length > 0 ? [{ elementId: 'ai-interpretation', title: 'AI Interpretation' }] : []),
    ],
  })

  return (
    <div className="min-h-screen bg-zinc-950 text-white">
      {/* Header */}
      <header className="border-b border-zinc-800 px-6 py-4 sticky top-0 z-30 bg-zinc-950/90 backdrop-blur-sm flex items-center justify-between">
        <div className="flex items-center gap-3">
          <button onClick={() => router.push('/')} className="text-zinc-500 hover:text-white transition-colors">
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
            </svg>
          </button>
          <div className="w-6 h-6 rounded bg-emerald-500 flex items-center justify-center">
            <svg className="w-3.5 h-3.5 text-zinc-950" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M3 13.5l5-5 4 4 5-6 4 3" />
            </svg>
          </div>
          <span className="font-semibold tracking-tight">StatLab</span>
          <span className="hidden sm:block text-zinc-600">/</span>
          <span className="hidden sm:block text-zinc-400 text-sm font-mono truncate max-w-xs">{fileName}</span>
        </div>
        <button
          onClick={handleExport}
          disabled={isGenerating}
          className="inline-flex items-center gap-2 px-4 py-2 text-xs font-semibold rounded-lg bg-emerald-600 text-white hover:bg-emerald-700 disabled:opacity-40 transition-all shadow-sm"
        >
          {isGenerating ? (
            <span className="w-3.5 h-3.5 border border-zinc-400 border-t-transparent rounded-full animate-spin" />
          ) : (
            <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
          )}
          Export PDF
        </button>
      </header>

      {/* Tab Navigation */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 pt-6">
        <div className="flex gap-1 border-b border-zinc-800">
          <button
            onClick={() => setTab('analysis')}
            className={`px-4 py-2.5 text-sm font-medium transition-colors border-b-2 -mb-[1px] ${
              tab === 'analysis'
                ? 'text-emerald-400 border-emerald-400'
                : 'text-zinc-500 border-transparent hover:text-zinc-300'
            }`}
          >
            Dataset Analysis
          </button>
          <button
            onClick={() => setTab('predictions')}
            disabled={!alignmentHasPredictive}
            className={`px-4 py-2.5 text-sm font-medium transition-colors border-b-2 -mb-[1px] ${
              !alignmentHasPredictive
                ? 'text-zinc-700 border-transparent cursor-not-allowed'
                : tab === 'predictions'
                  ? 'text-emerald-400 border-emerald-400'
                  : 'text-zinc-500 border-transparent hover:text-zinc-300'
            }`}
          >
            Predictions
          </button>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 py-6 space-y-8">
        {tab === 'analysis' ? (
          <>
            {/* AI Summary */}
            {interpret.summary && (
              <section id="ai-summary" className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-6 space-y-2">
                <div className="flex items-center gap-2 mb-3">
                  <div className="w-2 h-2 rounded-full bg-emerald-400" />
                  <span className="text-xs text-zinc-500 uppercase tracking-widest font-medium">AI Summary</span>
                  {interpret.provider && (
                    <span className="ml-auto text-xs text-zinc-600 font-mono">via {interpret.provider}</span>
                  )}
                </div>
                <p className="text-zinc-200 leading-relaxed">{interpret.summary}</p>
              </section>
            )}

            {/* Bento Grid Row 1 */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">

              {/* Descriptive Stats */}
              <div id="descriptive-stats" className="lg:col-span-2 rounded-xl border border-zinc-800 bg-zinc-900/50 overflow-hidden">
                <div className="px-5 py-4 border-b border-zinc-800 flex items-center justify-between">
                  <span className="text-sm font-semibold">Descriptive Statistics</span>
                  <span className="text-xs text-zinc-500 font-mono">{descriptive.length} columns</span>
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-zinc-800">
                        {['Column', 'Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Skewness', 'Outliers'].map(h => (
                          <th key={h} className="px-4 py-3 text-left text-xs text-zinc-500 font-medium">{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {descriptive.map((row, i) => (
                        <tr key={row.column} className={`border-b border-zinc-800/50 ${i % 2 === 0 ? '' : 'bg-zinc-900/30'}`}>
                          <td className="px-4 py-3 font-mono text-xs text-emerald-400">{row.column}</td>
                          <td className="px-4 py-3 text-zinc-300 font-mono text-xs">{fmt(row.mean)}</td>
                          <td className="px-4 py-3 text-zinc-300 font-mono text-xs">{fmt(row.median)}</td>
                          <td className="px-4 py-3 text-zinc-300 font-mono text-xs">{fmt(row.stdDev)}</td>
                          <td className="px-4 py-3 text-zinc-300 font-mono text-xs">{fmt(row.min)}</td>
                          <td className="px-4 py-3 text-zinc-300 font-mono text-xs">{fmt(row.max)}</td>
                          <td className="px-4 py-3 font-mono text-xs">
                            <SkewnessChip value={row.skewness} />
                          </td>
                          <td className="px-4 py-3 font-mono text-xs">
                            {row.outlierCount != null ? (
                              <span className={row.outlierCount > 0 ? 'text-amber-400' : 'text-zinc-600'}>
                                {row.outlierCount}
                              </span>
                            ) : (
                              <span className="text-zinc-600">—</span>
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  {descriptive.length === 0 && (
                    <p className="px-5 py-8 text-zinc-600 text-sm text-center">No descriptive results.</p>
                  )}
                </div>
              </div>

              {/* Dataset Info */}
              <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-5 space-y-4">
                <span className="text-sm font-semibold">Dataset</span>
                <div className="space-y-3">
                  <InfoRow label="File" value={schema.fileName} mono />
                  <InfoRow label="Rows" value={schema.rowCount?.toLocaleString() ?? '—'} mono />
                  <InfoRow label="Columns" value={String(schema.columnCount)} mono />
                  <InfoRow label="Analysed" value={new Date(timestamp).toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: 'numeric' })} />
                </div>
                <div className="pt-3 border-t border-zinc-800 space-y-1.5">
                  {schema.columns.map(col => (
                    <div key={col.name} className="flex items-center gap-2 text-xs">
                      <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${col.type === 'continuous' ? 'bg-emerald-400' : 'bg-violet-400'}`} />
                      <span className="font-mono text-zinc-300 truncate">{col.name}</span>
                      <span className="ml-auto text-zinc-600">{col.type}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Charts */}
            {chartSuggestions.length > 0 && (
              <section id="charts-panel" className="space-y-4">
                <SectionHeader title="Charts & Visualisations" count={chartSuggestions.length} />
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                  {chartSuggestions.map((s, i) => (
                    <ChartCard key={`${s.title}-${i}`} suggestion={s} activeSession={activeSession} index={i} />
                  ))}
                </div>
              </section>
            )}

            {/* Inferential */}
            {(correlations.length > 0 || hypothesisTests.length > 0) && (
              <section id="inferential-panel" className="space-y-4">
                <SectionHeader title="Inferential Analysis" />
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">

                  {correlations.length > 0 && (
                    <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 overflow-hidden">
                      <div className="px-5 py-4 border-b border-zinc-800">
                        <span className="text-sm font-semibold">Correlations</span>
                      </div>
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="border-b border-zinc-800">
                            {['Pair', 'r', 'Method', 'Interpretation'].map(h => (
                              <th key={h} className="px-4 py-3 text-left text-xs text-zinc-500">{h}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {correlations.map((c, i) => (
                            <tr key={`${c.columnA}-${c.columnB}-${i}`} className="border-b border-zinc-800/50">
                              <td className="px-4 py-3 font-mono text-xs text-zinc-300">{c.columnA} / {c.columnB}</td>
                              <td className="px-4 py-3 font-mono text-xs">
                                <span className={Math.abs(c.r) > 0.5 ? 'text-emerald-400' : 'text-zinc-400'}>
                                  {c.r?.toFixed(3)}
                                </span>
                              </td>
                              <td className="px-4 py-3 text-xs text-zinc-500 capitalize">{c.method}</td>
                              <td className="px-4 py-3 text-xs text-zinc-400">{c.interpretation}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}

                  {hypothesisTests.length > 0 && (
                    <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 overflow-hidden">
                      <div className="px-5 py-4 border-b border-zinc-800">
                        <span className="text-sm font-semibold">Hypothesis Tests</span>
                      </div>
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="border-b border-zinc-800">
                            {['Test', 'Statistic', 'p-value', 'Significant'].map(h => (
                              <th key={h} className="px-4 py-3 text-left text-xs text-zinc-500">{h}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {hypothesisTests.map((t, i) => (
                            <tr key={`${t.testType}-${i}`} className="border-b border-zinc-800/50">
                              <td className="px-4 py-3 text-xs text-zinc-300 capitalize">{t.testType}</td>
                              <td className="px-4 py-3 font-mono text-xs text-zinc-400">{t.statistic?.toFixed(4)}</td>
                              <td className="px-4 py-3 font-mono text-xs text-zinc-400">{t.pValue?.toFixed(4)}</td>
                              <td className="px-4 py-3">
                                <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${
                                  t.significant ? 'bg-emerald-900/40 text-emerald-400' : 'bg-zinc-800 text-zinc-500'
                                }`}>
                                  {t.significant ? 'Yes' : 'No'}
                                </span>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              </section>
            )}

            {/* AI Interpretation */}
            {interpret.perAnalysis.length > 0 && (
              <section id="ai-interpretation" className="space-y-4">
                <SectionHeader title="AI Interpretation" count={interpret.perAnalysis.length} />
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                  {interpret.perAnalysis.map((item, i) => (
                    <div key={`${item.subject}-${i}`} className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-5 space-y-2">
                      <div className="flex items-center gap-2">
                        <span className={`text-xs px-2 py-0.5 rounded-full font-mono ${typeColor(item.type)}`}>
                          {item.type}
                        </span>
                        <span className="text-xs font-mono text-zinc-400">{item.subject}</span>
                      </div>
                      <p className="text-sm text-zinc-300 leading-relaxed">{item.interpretation}</p>
                    </div>
                  ))}
                </div>
              </section>
            )}
          </>
        ) : (
          <>
            {/* Predictions Tab */}
            {alignmentHasPredictive && (
              <section id="predictive-panel" className="space-y-6">
                <div>
                  <SectionHeader title="Predictive Model" />
                  <div className="mt-4 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                    <StatCard label="Model" value={predictive.modelType ?? predReg?.modelType ?? '—'} />
                    <StatCard label="Train R²" value={fmt(predReg?.rSquared)} accent />
                    <StatCard label="Train RMSE" value={fmt(predReg?.rmse)} />
                    {predReg?.testMetrics && (
                      <>
                        <StatCard label="Test R²" value={fmt(predReg.testMetrics.rSquared)} accent />
                        <StatCard label="Test RMSE" value={fmt(predReg.testMetrics.rmse)} />
                        <StatCard label="Test Samples" value={String(predReg.testMetrics.sampleSize)} />
                      </>
                    )}
                    {predReg?.testMetrics?.accuracy != null && (
                      <StatCard label="Test Accuracy" value={`${(predReg.testMetrics.accuracy * 100).toFixed(1)}%`} accent />
                    )}
                    {predReg?.testMetrics?.precision != null && (
                      <StatCard label="Precision" value={`${(predReg.testMetrics.precision * 100).toFixed(1)}%`} />
                    )}
                    {predReg?.testMetrics?.recall != null && (
                      <StatCard label="Recall" value={`${(predReg.testMetrics.recall * 100).toFixed(1)}%`} />
                    )}
                    {predReg?.testMetrics?.f1 != null && (
                      <StatCard label="F1 Score" value={`${(predReg.testMetrics.f1 * 100).toFixed(1)}%`} accent />
                    )}
                    {predReg?.accuracy != null && !predReg?.testMetrics && (
                      <StatCard label="Accuracy" value={`${(predReg.accuracy * 100).toFixed(1)}%`} accent />
                    )}
                  </div>
                </div>

                {/* Model Explanation */}
                {predReg && (
                  <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-5">
                    <p className="text-xs text-zinc-500 uppercase tracking-widest mb-3">Interpretation</p>
                    <p className="text-sm text-zinc-300 leading-relaxed">
                      {buildModelExplanation(predictive.modelType ?? predReg.modelType, predReg)}
                    </p>
                  </div>
                )}

                {/* Prediction vs Actual Scatter */}
                {predictionScatterData.length > 0 && (
                  <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 overflow-hidden">
                    <div className="px-5 py-4 border-b border-zinc-800">
                      <p className="text-sm font-semibold">Predicted vs Actual</p>
                      <p className="text-xs text-zinc-500 mt-0.5">
                        {predictionScatterData.length < (predReg?.predictions?.length ?? 0)
                          ? `Sampled ${predictionScatterData.length} of ${predReg?.predictions?.length} points`
                          : `${predictionScatterData.length} data points`}
                      </p>
                    </div>
                    <div className="p-4 h-72">
                      <ResponsiveContainer width="100%" height="100%">
                        <ScatterChart>
                          <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                          <XAxis dataKey="actual" name="Actual" tick={{ fontSize: 10, fill: '#71717a' }} />
                          <YAxis dataKey="predicted" name="Predicted" tick={{ fontSize: 10, fill: '#71717a' }} />
                          <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ background: '#18181b', border: '1px solid #3f3f46', fontSize: 12 }} />
                          <Scatter data={predictionScatterData} fill="#34d399" fillOpacity={0.6} />
                        </ScatterChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                )}

                {/* Feature Importance */}
                {featureImportance && featureImportance.length > 0 && (
                  <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-5">
                    <p className="text-xs text-zinc-500 uppercase tracking-widest mb-4">Feature Importance</p>
                    <div className="space-y-2">
                      {featureImportance.map(item => (
                        <div key={item.feature} className="flex items-center gap-3">
                          <span className="text-xs font-mono text-zinc-400 w-28 truncate">{item.feature}</span>
                          <div className="flex-1 h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                            <div
                              className="h-full bg-emerald-500 rounded-full"
                              style={{ width: `${Math.min(100, Math.abs(item.importance) * 100)}%` }}
                            />
                          </div>
                          <span className="text-xs font-mono text-zinc-500 w-12 text-right">
                            {(item.importance * 100).toFixed(1)}%
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* VIF */}
                {predReg?.vif && predReg.vif.length > 0 && (
                  <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-5">
                    <p className="text-xs text-zinc-500 uppercase tracking-widest mb-4">Multicollinearity (VIF)</p>
                    <div className="space-y-2">
                      {predReg.vif.map(item => (
                        <div key={item.predictor} className="flex items-center gap-3">
                          <span className="text-xs font-mono text-zinc-400 w-40 truncate">{item.predictor}</span>
                          <div className="flex-1 h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                            <div
                              className={`h-full rounded-full ${item.value > 5 ? 'bg-red-500' : 'bg-emerald-500'}`}
                              style={{ width: `${Math.min(100, (item.value / 10) * 100)}%` }}
                            />
                          </div>
                          <span className={`text-xs font-mono w-12 text-right ${item.value > 5 ? 'text-red-400' : 'text-zinc-500'}`}>
                            {item.value === Infinity ? '∞' : item.value.toFixed(1)}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Model Note */}
                {predReg?.note && (
                  <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-4">
                    <p className="text-xs text-zinc-500 font-medium uppercase tracking-widest mb-1">Note</p>
                    <p className="text-sm text-zinc-400">{predReg.note}</p>
                  </div>
                )}
              </section>
            )}
          </>
        )}

        {/* History — always visible */}
        {history.length > 0 && (
          <section className="space-y-4 pt-4 border-t border-zinc-800">
            <SectionHeader title="Past Analyses" count={history.length} />
            <div className="space-y-2">
              {history.map(session => (
                <button
                  key={session.id}
                  onClick={() => {
                    loadSession(session)
                    setActiveSession(session)
                    window.scrollTo({ top: 0, behavior: 'smooth' })
                  }}
                  className={`w-full rounded-xl border text-left px-5 py-4 flex items-center gap-4 transition-colors ${
                    activeSession.id === session.id
                      ? 'border-emerald-700 bg-emerald-950/20'
                      : 'border-zinc-800 bg-zinc-900/30 hover:border-zinc-600'
                  }`}
                >
                  <div className="w-8 h-8 rounded-lg bg-zinc-800 flex items-center justify-center flex-shrink-0">
                    <svg className="w-4 h-4 text-zinc-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 17v-2m3 2v-4m3 4v-6M7 9l3-3 3 3 4-4" />
                    </svg>
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-zinc-200 truncate">{session.fileName}</p>
                    <p className="text-xs text-zinc-500">
                      {session.schema?.columnCount ?? 0} columns · {new Date(session.timestamp).toLocaleString()}
                    </p>
                  </div>
                  {activeSession.id === session.id && (
                    <span className="text-xs text-emerald-400 font-medium flex-shrink-0">Viewing</span>
                  )}
                </button>
              ))}
            </div>
          </section>
        )}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Chart card
// ---------------------------------------------------------------------------

function ChartCard({ suggestion, activeSession, index }: { suggestion: ChartSuggestion; activeSession: AnalysisSession; index: number }) {
  const color = ACCENT[index % ACCENT.length]
  const descriptive: DescriptiveResult[] = activeSession.result.descriptive ?? []
  const correlations = useMemo(
    () => activeSession.result.inferential?.correlations ?? [],
    [activeSession.result.inferential?.correlations]
  )
  const targetColumnName = suggestion.column || suggestion.x
  const targetMetrics = descriptive.find(d => d.column === targetColumnName)
  const { chartType, title, x, y } = suggestion

  // Determine skip reason before hooks
  const skipReason: string | null = (() => {
    const needsColumn = ['bar', 'pie', 'line', 'histogram', 'boxplot'].includes(chartType)
    if (needsColumn && !targetMetrics) return 'Column not found in descriptive results'
    if (['confusion_matrix', 'roc_curve'].includes(chartType)) return `"${chartType}" chart not yet implemented`
    return null
  })()

  const processedData: Record<string, unknown>[] = useMemo(() => {
    if ((chartType === 'bar' || chartType === 'pie' || chartType === 'line') && targetMetrics?.frequencyTable) {
      return Object.entries(targetMetrics.frequencyTable).map(([key, count]) => ({
        name: key,
        value: Number(count),
        mean: targetMetrics.mean ?? 0,
      }))
    }

    if (chartType === 'scatter') {
      const rows = activeSession.schema.sampleRows ?? []
      return rows.map((row, idx) => ({
        x: x != null && row[x] != null ? Number(row[x]) : idx,
        y: y != null && row[y] != null ? Number(row[y]) : 0,
      })).filter(d => !isNaN(d.x) && !isNaN(d.y))
    }

    if (chartType === 'histogram' && targetMetrics?.min != null && targetMetrics?.max != null && targetMetrics?.mean != null && targetMetrics?.stdDev != null && targetMetrics?.count != null) {
      return estimateHistogramBins(targetMetrics.min, targetMetrics.max, targetMetrics.mean, targetMetrics.stdDev, targetMetrics.count)
    }

    if (chartType === 'heatmap' && correlations.length > 0) {
      return []
    }

    if (chartType === 'boxplot' && targetMetrics?.min != null && targetMetrics?.max != null && targetMetrics?.median != null && targetMetrics?.iqr != null) {
      const q1 = targetMetrics.median - targetMetrics.iqr / 2
      const q3 = targetMetrics.median + targetMetrics.iqr / 2
      return [{
        name: targetColumnName ?? 'value',
        min: targetMetrics.min,
        q1: Math.max(q1, targetMetrics.min),
        median: targetMetrics.median,
        q3: Math.min(q3, targetMetrics.max),
        max: targetMetrics.max,
      }]
    }

    return (activeSession.schema.sampleRows ?? []).slice(0, 100).map((row, idx) => ({
      name: `Row ${idx + 1}`,
      x: x != null && row[x] != null ? Number(row[x]) : idx,
      y: y != null && row[y] != null ? Number(row[y]) : (targetMetrics?.mean ?? 0),
      value: targetColumnName != null && row[targetColumnName] != null ? Number(row[targetColumnName]) : 0,
      mean: targetMetrics?.mean ?? 0,
    }))
  }, [chartType, targetMetrics, x, y, targetColumnName, activeSession.schema.sampleRows, correlations])

  const heatmapData = useMemo(() => {
    if (chartType !== 'heatmap' || correlations.length === 0) return null
    return buildCorrelationGrid(correlations)
  }, [chartType, correlations])

  // Skip charts that can't find their data
  if (skipReason) {
    return <ChartEmpty title={title} reason={suggestion.reason} message={skipReason} />
  }

  // Skip charts with empty processed data (except heatmap which uses its own)
  if (processedData.length === 0 && chartType !== 'heatmap') {
    return <ChartEmpty title={title} reason={suggestion.reason} message="No data available for this chart" />
  }

  if (chartType === 'heatmap' && heatmapData) {
    return (
      <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 overflow-hidden">
        <div className="px-5 py-4 border-b border-zinc-800">
          <p className="text-sm font-semibold">{title}</p>
          {suggestion.reason && <p className="text-xs text-zinc-500 mt-0.5">{suggestion.reason}</p>}
        </div>
        <div className="p-4 overflow-x-auto">
          <table className="text-xs mx-auto">
            <thead>
              <tr>
                <th className="p-1" />
                {heatmapData.columns.map(col => (
                  <th key={col} className="p-1 font-mono text-zinc-500 font-medium max-w-[80px] truncate" title={col}>
                    {col.length > 8 ? col.slice(0, 8) + '\u2026' : col}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {heatmapData.matrix.map((row, i) => (
                <tr key={heatmapData.columns[i]}>
                  <td className="p-1 font-mono text-zinc-500 font-medium max-w-[80px] truncate text-right pr-2" title={heatmapData.columns[i]}>
                    {heatmapData.columns[i].length > 8 ? heatmapData.columns[i].slice(0, 8) + '\u2026' : heatmapData.columns[i]}
                  </td>
                  {row.map((r, j) => (
                    <td
                      key={j}
                      className="p-1 text-center font-mono rounded"
                      style={{
                        background: heatmapColor(r),
                        color: heatmapTextColor(r),
                        minWidth: 48,
                      }}
                      title={`${heatmapData.columns[i]} / ${heatmapData.columns[j]}: ${r.toFixed(3)}`}
                    >
                      {r.toFixed(2)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    )
  }

  if (chartType === 'boxplot' && processedData.length > 0 && 'q1' in processedData[0]) {
    const d = processedData[0] as { name: string; min: number; q1: number; median: number; q3: number; max: number }
    const range = d.max - d.min || 1
    const toPct = (v: number) => ((v - d.min) / range * 100)
    return (
      <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 overflow-hidden">
        <div className="px-5 py-4 border-b border-zinc-800">
          <p className="text-sm font-semibold">{title}</p>
          {suggestion.reason && <p className="text-xs text-zinc-500 mt-0.5">{suggestion.reason}</p>}
        </div>
        <div className="p-4 h-48 flex items-center justify-center">
          <div className="w-full max-w-xs">
            <p className="text-xs font-mono text-zinc-500 text-center mb-3">{d.name}</p>
            <div className="relative h-20">
              {/* IQR box */}
              <div
                className="absolute left-1/3 right-1/3 rounded border border-emerald-500 bg-emerald-500/20"
                style={{
                  top: `${100 - toPct(d.q1)}%`,
                  bottom: `${toPct(d.q3)}%`,
                }}
              >
                {/* Median line */}
                <div
                  className="absolute left-0 right-0 h-0.5 bg-emerald-400"
                  style={{ top: `${(toPct(d.median) - toPct(d.q3)) / (toPct(d.q1) - toPct(d.q3)) * 100}%` }}
                />
              </div>
              {/* Whisker top to q3 */}
              <div className="absolute left-1/2 w-px bg-zinc-500" style={{ top: `${100 - toPct(d.max)}%`, height: `${toPct(d.max) - toPct(d.q3)}%` }} />
              {/* Whisker q1 to bottom */}
              <div className="absolute left-1/2 w-px bg-zinc-500" style={{ top: `${100 - toPct(d.q1)}%`, height: `${toPct(d.q1) - toPct(d.min)}%` }} />
              {/* Whisker caps */}
              <div className="absolute left-[30%] right-[30%] h-px bg-zinc-500" style={{ top: `${100 - toPct(d.max)}%` }} />
              <div className="absolute left-[30%] right-[30%] h-px bg-zinc-500" style={{ top: `${100 - toPct(d.min)}%` }} />
              {/* Labels */}
              <span className="absolute -top-4 left-0 text-[10px] font-mono text-zinc-500">{d.max.toFixed(1)}</span>
              <span className="absolute text-[10px] font-mono text-emerald-400 right-0" style={{ top: `${100 - toPct(d.median)}%`, marginTop: -6 }}>{d.median.toFixed(1)}</span>
              <span className="absolute -bottom-4 left-0 text-[10px] font-mono text-zinc-500">{d.min.toFixed(1)}</span>
            </div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 overflow-hidden">
      <div className="px-5 py-4 border-b border-zinc-800">
        <p className="text-sm font-semibold">{title}</p>
        {suggestion.reason && <p className="text-xs text-zinc-500 mt-0.5">{suggestion.reason}</p>}
      </div>
      <div className="p-4 h-56">
        <ResponsiveContainer width="100%" height="100%" debounce={1}>
          {chartType === 'scatter' ? (
            <ScatterChart>
              <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
              <XAxis dataKey="x" name={x} tick={{ fontSize: 10, fill: '#71717a' }} />
              <YAxis dataKey="y" name={y ?? ''} tick={{ fontSize: 10, fill: '#71717a' }} />
              <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ background: '#18181b', border: '1px solid #3f3f46', fontSize: 12 }} />
              <Scatter data={processedData} fill={color} fillOpacity={0.8} />
            </ScatterChart>
          ) : chartType === 'line' ? (
            <LineChart data={processedData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
              <XAxis dataKey="name" tick={{ fontSize: 10, fill: '#71717a' }} />
              <YAxis tick={{ fontSize: 10, fill: '#71717a' }} />
              <Tooltip contentStyle={{ background: '#18181b', border: '1px solid #3f3f46', fontSize: 12 }} />
              <Line type="monotone" dataKey="value" stroke={color} strokeWidth={2} dot={false} />
            </LineChart>
          ) : chartType === 'pie' ? (
            <PieChart>
              <Pie data={processedData.slice(0, 6)} dataKey="value" nameKey="name" outerRadius={80} strokeWidth={0}>
                {processedData.slice(0, 6).map((_, i) => (
                  <Cell key={`cell-${i}`} fill={ACCENT[i % ACCENT.length]} />
                ))}
              </Pie>
              <Tooltip contentStyle={{ background: '#18181b', border: '1px solid #3f3f46', fontSize: 12 }} />
              <Legend iconSize={8} wrapperStyle={{ fontSize: 11, color: '#a1a1aa' }} />
            </PieChart>
          ) : chartType === 'histogram' ? (
            <BarChart data={processedData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
              <XAxis dataKey="name" tick={{ fontSize: 9, fill: '#71717a' }} angle={-20} textAnchor="end" height={40} />
              <YAxis tick={{ fontSize: 10, fill: '#71717a' }} />
              <Tooltip contentStyle={{ background: '#18181b', border: '1px solid #3f3f46', fontSize: 12 }} />
              <Bar dataKey="value" fill={color} radius={[2, 2, 0, 0]} />
            </BarChart>
          ) : (
            <BarChart data={processedData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
              <XAxis dataKey="name" tick={{ fontSize: 10, fill: '#71717a' }} />
              <YAxis tick={{ fontSize: 10, fill: '#71717a' }} />
              <Tooltip contentStyle={{ background: '#18181b', border: '1px solid #3f3f46', fontSize: 12 }} />
              <Bar dataKey="value" fill={color} radius={[3, 3, 0, 0]} />
            </BarChart>
          )}
        </ResponsiveContainer>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function fmt(v: number | null | undefined) {
  if (v == null || isNaN(v)) return '\u2014'
  return Math.abs(v) >= 1000
    ? v.toLocaleString(undefined, { maximumFractionDigits: 1 })
    : v.toFixed(3)
}

function SkewnessChip({ value }: { value?: number | null }) {
  if (value == null || isNaN(value)) return <span className="text-zinc-600">{'\u2014'}</span>
  const abs = Math.abs(value)
  const label = abs < 0.5 ? 'symmetric' : abs < 1 ? 'moderate' : 'skewed'
  const color = abs < 0.5 ? 'text-emerald-400' : abs < 1 ? 'text-amber-400' : 'text-red-400'
  return <span className={`${color}`}>{value.toFixed(2)} <span className="text-zinc-600">({label})</span></span>
}

function InfoRow({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="flex items-start justify-between gap-3">
      <span className="text-xs text-zinc-500">{label}</span>
      <span className={`text-xs text-zinc-300 text-right truncate max-w-[60%] ${mono ? 'font-mono' : ''}`}>{value}</span>
    </div>
  )
}

function SectionHeader({ title, count }: { title: string; count?: number }) {
  return (
    <div className="flex items-center gap-3">
      <h2 className="text-sm font-semibold text-zinc-300">{title}</h2>
      {count != null && (
        <span className="text-xs px-2 py-0.5 rounded-full bg-zinc-800 text-zinc-500 font-mono">{count}</span>
      )}
      <div className="flex-1 h-px bg-zinc-800" />
    </div>
  )
}

function StatCard({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-5 shadow-sm">
      <p className="text-xs text-zinc-400 uppercase tracking-widest mb-1.5">{label}</p>
      <p className={`text-2xl font-bold font-mono tracking-tight ${accent ? 'text-emerald-400' : 'text-white'}`}>{value}</p>
    </div>
  )
}

function typeColor(type: string) {
  const map: Record<string, string> = {
    descriptive: 'bg-emerald-900/40 text-emerald-400',
    inferential: 'bg-violet-900/40 text-violet-400',
    predictive: 'bg-indigo-900/40 text-indigo-400',
    correlation: 'bg-pink-900/40 text-pink-400',
    regression: 'bg-amber-900/40 text-amber-400',
  }
  return map[type.toLowerCase()] ?? 'bg-zinc-800 text-zinc-400'
}

// ---------------------------------------------------------------------------
// Distribution helpers
// ---------------------------------------------------------------------------

function normalPDF(x: number, mean: number, stdDev: number): number {
  const z = (x - mean) / stdDev
  return Math.exp(-0.5 * z * z) / (stdDev * Math.sqrt(2 * Math.PI))
}

function estimateHistogramBins(
  min: number, max: number, mean: number, stdDev: number, count: number, bins = BIN_COUNT
): { name: string; value: number }[] {
  if (min >= max || stdDev <= 0 || count <= 0) return []
  const binWidth = (max - min) / bins
  const result: { name: string; value: number }[] = []
  for (let i = 0; i < bins; i++) {
    const binMin = min + i * binWidth
    const binMax = binMin + binWidth
    const binMid = (binMin + binMax) / 2
    const pdf = normalPDF(binMid, mean, stdDev)
    const estimated = Math.round(pdf * binWidth * count)
    result.push({
      name: `${fmtShort(binMin)}\u2013${fmtShort(binMax)}`,
      value: Math.max(0, estimated),
    })
  }
  return result
}

function fmtShort(v: number): string {
  if (Math.abs(v) >= 1000) return v.toLocaleString(undefined, { maximumFractionDigits: 0 })
  if (Math.abs(v) >= 1) return v.toFixed(1)
  return v.toFixed(3)
}

function heatmapColor(r: number): string {
  const abs = Math.min(Math.abs(r), 1)
  if (r >= 0) {
    const g = Math.round(180 - abs * 120)
    const b = Math.round(200 - abs * 180)
    return `rgb(${Math.round(50 + abs * 50)}, ${g}, ${b})`
  } else {
    const rv = Math.round(200 - abs * 180)
    const g = Math.round(180 - abs * 120)
    return `rgb(${rv}, ${g}, ${Math.round(50 + abs * 50)})`
  }
}

function heatmapTextColor(r: number): string {
  const abs = Math.min(Math.abs(r), 1)
  let R: number, G: number, B: number
  if (r >= 0) {
    R = Math.round(50 + abs * 50)
    G = Math.round(180 - abs * 120)
    B = Math.round(200 - abs * 180)
  } else {
    R = Math.round(200 - abs * 180)
    G = Math.round(180 - abs * 120)
    B = Math.round(50 + abs * 50)
  }
  return (0.299 * R + 0.587 * G + 0.114 * B) > 128 ? '#18181b' : '#f4f4f5'
}

function buildCorrelationGrid(correlations: CorrelationResult[]): { columns: string[]; matrix: number[][] } {
  const colSet = new Set<string>()
  for (const c of correlations) {
    colSet.add(c.columnA)
    colSet.add(c.columnB)
  }
  const columns = Array.from(colSet)
  const n = columns.length
  const matrix: number[][] = Array.from({ length: n }, () => new Array(n).fill(0))
  const indexMap = new Map(columns.map((name, i) => [name, i]))
  for (const c of correlations) {
    const i = indexMap.get(c.columnA)!
    const j = indexMap.get(c.columnB)!
    matrix[i][j] = c.r
    matrix[j][i] = c.r
  }
  for (let i = 0; i < n; i++) matrix[i][i] = 1
  return { columns, matrix }
}

function ChartEmpty({ title, reason, message }: { title: string; reason?: string; message: string }) {
  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 overflow-hidden">
      <div className="px-5 py-4 border-b border-zinc-800">
        <p className="text-sm font-semibold">{title}</p>
        {reason && <p className="text-xs text-zinc-500 mt-0.5">{reason}</p>}
      </div>
      <div className="p-4 h-56 flex items-center justify-center">
        <p className="text-xs text-zinc-600">{message}</p>
      </div>
    </div>
  )
}

function buildModelExplanation(modelType: ModelType, reg: RegressionResult): string {
  const depName = reg.dependent
  const r2 = reg.rSquared ?? 0
  const r2Desc = r2 > 0.8 ? 'very strong' : r2 > 0.6 ? 'strong' : r2 > 0.4 ? 'moderate' : r2 > 0.2 ? 'weak' : 'very weak'
  const parts: string[] = []

  if (modelType === 'logistic') {
    parts.push(`The model predicts the probability of "${depName}" from ${reg.predictors.length} predictor(s).`)
    const metrics: string[] = []
    if (reg.accuracy != null) metrics.push(`accuracy ${(reg.accuracy * 100).toFixed(1)}%`)
    if (reg.testMetrics?.precision != null) metrics.push(`precision ${(reg.testMetrics.precision * 100).toFixed(1)}%`)
    if (reg.testMetrics?.recall != null) metrics.push(`recall ${(reg.testMetrics.recall * 100).toFixed(1)}%`)
    if (reg.testMetrics?.f1 != null) metrics.push(`F1 ${(reg.testMetrics.f1 * 100).toFixed(1)}%`)
    if (metrics.length > 0) parts.push(`Test set: ${metrics.join(', ')}.`)
    for (let i = 0; i < reg.predictors.length && i < reg.coefficients.length; i++) {
      const coef = reg.coefficients[i]
      const oddsRatio = Math.exp(coef)
      if (oddsRatio > 1.01) parts.push(`"${reg.predictors[i]}" increases odds of "${depName}" by ${((oddsRatio - 1) * 100).toFixed(1)}% per unit.`)
      else if (oddsRatio < 0.99) parts.push(`"${reg.predictors[i]}" decreases odds of "${depName}" by ${((1 - oddsRatio) * 100).toFixed(1)}% per unit.`)
    }
    parts.push(`Variance explained: ${(r2 * 100).toFixed(1)}% (${r2Desc}).`)
  } else if (modelType === 'timeseries') {
    if (reg.coefficients.length >= 1) {
      const slope = reg.coefficients[0]
      parts.push(`"${depName}" follows a ${slope > 0 ? 'upward' : slope < 0 ? 'downward' : 'flat'} trend (slope ${slope > 0 ? '+' : ''}${slope.toFixed(4)} per time unit).`)
    }
    parts.push(`Fit: R² ${(r2 * 100).toFixed(1)}% (${r2Desc}). RMSE ${reg.rmse?.toFixed(2) ?? 'N/A'}.`)
  } else if (modelType === 'randomforest') {
    parts.push(`Random forest with ${reg.predictors.length} predictor(s) predicting "${depName}".`)
    if (reg.featureImportance && reg.featureImportance.length > 0) {
      const top = reg.featureImportance.slice(0, 3)
      parts.push(`Top predictors: ${top.map(f => `"${f.feature}" (${(f.importance * 100).toFixed(1)}%)`).join(', ')}.`)
    }
    parts.push(`R² ${(r2 * 100).toFixed(1)}% (${r2Desc}). RMSE ${reg.rmse?.toFixed(2) ?? 'N/A'}.`)
  } else {
    for (let i = 0; i < reg.predictors.length && i < reg.coefficients.length; i++) {
      const coef = reg.coefficients[i]
      const absCoef = Math.abs(coef)
      const direction = coef > 0 ? 'increases' : 'decreases'
      if (absCoef >= 0.01) parts.push(`Each +1 in "${reg.predictors[i]}" ${direction} "${depName}" by ${absCoef.toFixed(2)} units.`)
      else parts.push(`"${reg.predictors[i]}" has negligible effect on "${depName}" (${direction} by ${absCoef.toFixed(4)}).`)
    }
    if (reg.coefficients.length > 0 && reg.intercept !== 0) {
      parts.push(`Baseline "${depName}" when all predictors are zero: ${reg.intercept.toFixed(2)}.`)
    }
    parts.push(`R² ${(r2 * 100).toFixed(1)}% (${r2Desc} fit). RMSE ${reg.rmse?.toFixed(2) ?? 'N/A'}.`)
  }

  return parts.join(' ')
}
