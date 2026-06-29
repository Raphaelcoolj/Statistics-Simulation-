'use client'

import { useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'
import {
  ScatterChart, Scatter, BarChart, Bar, LineChart, Line, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
} from 'recharts'
import { usePDFExport } from '@/lib/pdf'
import { useStatLab } from '@/components/StatLabProvider'
import type { AnalysisSession } from '@/lib/useStatLab'
import type { ChartSuggestion, DescriptiveResult, CorrelationResult, HypothesisResult } from '@/lib/types'

const ACCENT = ['#34d399', '#818cf8', '#f472b6', '#fb923c', '#38bdf8', '#a78bfa']

export default function AnalysePage() {
  const router = useRouter()
  const { currentSession, history, loadSession } = useStatLab()
  const { exportPDF, isGenerating } = usePDFExport()
  const [activeSession, setActiveSession] = useState<AnalysisSession | null>(currentSession)

  useEffect(() => {
    if (!currentSession && history.length === 0) {
      router.push('/')
    } else if (!activeSession && currentSession) {
      setActiveSession(currentSession)
    } else if (!activeSession && history.length > 0) {
      setActiveSession(history[0])
    }
  }, [currentSession, history, activeSession, router])

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

  const handleExport = () => {
    window.print()
  }

  return (
    <div className="min-h-screen bg-zinc-950 text-white selection:bg-emerald-500/30">
      
<style jsx global>{`
  @media print {
    @page {
      size: A4 portrait;
      margin: 20mm 16mm;
    }
    html, body {
      background-color: #ffffff !important;
      color: #0f172a !important;
      font-family: 'Segoe UI', system-ui, -apple-system, sans-serif !important;
      -webkit-print-color-adjust: exact !important;
      print-color-adjust: exact !important;
    }
    
    .print-page {
      display: flex !important;
      flex-direction: column !important;
      width: 100% !important;
      /* 🚀 FIX BORDERS: Use fluid min-height to guarantee page borders render fully on short pages */
      min-height: 250mm !important; 
      box-sizing: border-box !important;
      
      /* 🚀 FIX EXTRA BLANK PAGE: Force page breaks *after* sections, but prevent it on the final page */
      page-break-after: always !important;
      break-after: always !important;
    }
    
    /* Safely targets the final page section to ensure no blank trailing page appears */
    .print-page:last-child {
      page-break-after: avoid !important;
      break-after: avoid !important;
      min-height: auto !important;
    }

    .print-border-opaque {
      border: 2px solid #000000 !important;
    }

    .print-avoid-break {
      page-break-inside: avoid !important;
      break-inside: avoid !important;
    }

    /* Keep absolute sizing restrictions exclusively constrained to active chart drawings */
    .print-chart-container {
      width: 620px !important;
      height: 240px !important;
      display: block !important;
      page-break-inside: avoid !important;
      break-inside: avoid !important;
    }
    .print-chart-container .recharts-responsive-container {
      width: 100% !important;
      height: 100% !important;
    }
  }
`}</style>

      {/* ========================================================================= */}
      {/* SCREEN VIEW ONLY LAYER (Hidden during PDF Generation compilation)          */}
      {/* ========================================================================= */}
      <div className="print:hidden">
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
            <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            Export PDF
          </button>
        </header>

        <div className="max-w-7xl mx-auto px-4 sm:px-6 py-10 space-y-8">
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
                      {['Column', 'Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Skewness'].map(h => (
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

          {/* Inferential Analysis */}
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

          {/* Predictive Model */}
          {alignmentHasPredictive && (
            <section id="predictive-panel">
              <SectionHeader title="Predictive Model" />
              <div className="mt-4 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                <StatCard label="Model" value={predictive.modelType ?? predictive.regressionResult?.modelType ?? '—'} />
                <StatCard label="R²" value={fmt(predictive.regressionResult?.rSquared)} accent />
                <StatCard label="RMSE" value={fmt(predictive.regressionResult?.rmse)} />
                {predictive.regressionResult?.accuracy != null && (
                  <StatCard label="Accuracy" value={`${(predictive.regressionResult.accuracy * 100).toFixed(1)}%`} accent />
                )}
              </div>

              {predictive.regressionResult?.featureImportance && (
                <div className="mt-4 rounded-xl border border-zinc-800 bg-zinc-900/50 p-5">
                  <p className="text-xs text-zinc-500 uppercase tracking-widest mb-4">Feature Importance</p>
                  <div className="space-y-2">
                    {Object.entries(predictive.regressionResult.featureImportance).map(([feat, imp]) => (
                      <div key={feat} className="flex items-center gap-3">
                        <span className="text-xs font-mono text-zinc-400 w-28 truncate">{feat}</span>
                        <div className="flex-1 h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-emerald-500 rounded-full"
                            style={{ width: `${Math.min(100, (imp as number) * 100)}%` }}
                          />
                        </div>
                        <span className="text-xs font-mono text-zinc-500">{((imp as number) * 100).toFixed(1)}%</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
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

          {/* Past Analyses Session History */}
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


      {/* ========================================================================= */}
      {/* 🚀 HIGH-END PRINT EXPORT VIEW LAYER                                       */}
      {/* ========================================================================= */}
      <div className="hidden print:block text-slate-900 bg-white">
        
        {/* PAGE 1: APP STYLIZED EMERALD COVER PAGE */}
        <div className="print-page print-border-opaque p-16 justify-between relative overflow-hidden bg-white">
          <div className="absolute top-0 left-0 right-0 h-5 bg-[#059669]" />
          
          <div className="my-auto space-y-8">
            <div className="inline-flex items-center gap-2 px-3 py-1.5 bg-emerald-50 border border-emerald-200 text-[#059669] rounded-md font-bold text-xs uppercase tracking-wider">
              ⚡ StatLab Intelligent System Output
            </div>
            <h1 className="text-5xl font-black text-slate-900 tracking-tight leading-none">
              Statistical Analysis <br />
              <span className="text-[#059669]">Intelligence Report</span>
            </h1>
            <div className="h-1 w-20 bg-slate-900" />
            <p className="text-xl text-slate-500 font-medium">
              Target Dataset: <span className="font-bold text-slate-900 underline decoration-[#059669] decoration-2 underline-offset-8">{fileName}</span>
            </p>
          </div>

          <div className="pt-8 border-t-2 border-black flex justify-between items-center">
            <div>
              <p className="text-[10px] uppercase tracking-widest font-black text-slate-400">Compilation Date</p>
              <p className="text-sm font-bold text-slate-800">{new Date(timestamp).toLocaleDateString('en-GB', { day: 'numeric', month: 'long', year: 'numeric' })}</p>
            </div>
            <div className="text-right">
              <p className="text-[10px] uppercase tracking-widest font-black text-slate-400">Processing Node</p>
              <p className="text-sm font-bold text-[#059669]">Verified via Groq Pipeline</p>
            </div>
          </div>
        </div>

        {/* PAGE 2: EXECUTIVE BRIEFING & DESCRIPTIVE MATRIX */}
        <div className="print-page pt-4 space-y-6 bg-white">
          <div className="border-b-4 border-[#059669] pb-2 flex justify-between items-end">
            <h2 className="text-xl font-black tracking-tight text-slate-900 uppercase">01. Executive Briefing & Descriptive Analytics</h2>
            <span className="text-xs text-slate-400 font-mono">Page 2</span>
          </div>

          {interpret.summary && (
            <div className="bg-slate-50 border-2 border-black p-6 rounded-xl space-y-2 print-avoid-break">
              <h4 className="text-[10px] font-black uppercase text-[#059669] tracking-wider">Generative Summary Insights</h4>
              <p className="text-sm leading-relaxed text-slate-700 font-medium">{interpret.summary}</p>
            </div>
          )}

          <div className="space-y-3">
            <h3 className="text-sm font-bold text-slate-800 uppercase tracking-wide">Variable Distribution Properties</h3>
            <table className="w-full text-left border-collapse print-border-opaque">
              <thead>
                <tr className="bg-slate-900 text-white text-xs uppercase font-bold">
                  <th className="p-3 border-b border-black">Column Parameter</th>
                  <th className="p-3 border-b border-black text-right">Mean</th>
                  <th className="p-3 border-b border-black text-right">Median</th>
                  <th className="p-3 border-b border-black text-right">Std Dev</th>
                  <th className="p-3 border-b border-black text-right">Min</th>
                  <th className="p-3 border-b border-black text-right">Max</th>
                </tr>
              </thead>
              <tbody className="text-xs font-semibold divide-y divide-black">
                {descriptive.map((row, idx) => (
                  <tr key={idx} className={idx % 2 === 0 ? 'bg-white' : 'bg-slate-50'}>
                    <td className="p-3 font-bold text-slate-900 font-mono text-[#059669]">{row.column}</td>
                    <td className="p-3 text-right text-slate-700 font-mono">{fmt(row.mean)}</td>
                    <td className="p-3 text-right text-slate-700 font-mono">{fmt(row.median)}</td>
                    <td className="p-3 text-right text-slate-700 font-mono">{fmt(row.stdDev)}</td>
                    <td className="p-3 text-right text-slate-700 font-mono">{fmt(row.min)}</td>
                    <td className="p-3 text-right text-slate-700 font-mono">{fmt(row.max)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* PAGE 3: INFERENTIAL MATRIX */}
        {(correlations.length > 0 || hypothesisTests.length > 0) && (
          <div className="print-page pt-4 space-y-6 bg-white">
            <div className="border-b-4 border-[#059669] pb-2 flex justify-between items-end">
              <h2 className="text-xl font-black tracking-tight text-slate-900 uppercase">02. Inferential Data Architecture</h2>
              <span className="text-xs text-slate-400 font-mono">Page 3</span>
            </div>

            {correlations.length > 0 && (
              <div className="space-y-3 print-avoid-break">
                <h3 className="text-sm font-bold text-slate-800 uppercase tracking-wide">Spearman/Pearson Correlation Coefficients</h3>
                <table className="w-full text-left border-collapse print-border-opaque">
                  <thead>
                    <tr className="bg-slate-900 text-white text-xs uppercase font-bold">
                      <th className="p-3 border-b border-black">Target Pair Definition</th>
                      <th className="p-3 border-b border-black text-right">Coefficient (r)</th>
                      <th className="p-3 border-b border-black">Mathematical Model</th>
                    </tr>
                  </thead>
                  <tbody className="text-xs font-semibold divide-y divide-black">
                    {correlations.map((c, i) => (
                      <tr key={i} className="bg-white">
                        <td className="p-3 font-mono text-slate-900">{c.columnA} / {c.columnB}</td>
                        <td className="p-3 text-right font-mono text-[#059669] font-bold">{c.r?.toFixed(4)}</td>
                        <td className="p-3 text-slate-500 uppercase tracking-wider text-[10px]">{c.method}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {hypothesisTests.length > 0 && (
              <div className="space-y-3 print-avoid-break">
                <h3 className="text-sm font-bold text-slate-800 uppercase tracking-wide">Empirical Significance Metrics</h3>
                <table className="w-full text-left border-collapse print-border-opaque">
                  <thead>
                    <tr className="bg-slate-900 text-white text-xs uppercase font-bold">
                      <th className="p-3 border-b border-black">Test Matrix</th>
                      <th className="p-3 border-b border-black text-right">Statistic</th>
                      <th className="p-3 border-b border-black text-right">Asymp. Sig (p-value)</th>
                      <th className="p-3 border-b border-black text-center">Significant</th>
                    </tr>
                  </thead>
                  <tbody className="text-xs font-semibold divide-y divide-black">
                    {hypothesisTests.map((t, i) => (
                      <tr key={i} className="bg-white">
                        <td className="p-3 text-slate-900 capitalize font-bold">{t.testType}</td>
                        <td className="p-3 text-right text-slate-600 font-mono">{t.statistic?.toFixed(4)}</td>
                        <td className="p-3 text-right text-slate-600 font-mono">{t.pValue?.toFixed(4)}</td>
                        <td className="p-3 text-center">
                          <span className="text-[11px] font-black uppercase text-[#059669]">{t.significant ? 'YES' : 'NO'}</span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}

        {/* PAGE 4: PREDICTIVE MATRIX */}
        {alignmentHasPredictive && (
          <div className="print-page pt-4 space-y-6 bg-white">
            <div className="border-b-4 border-[#059669] pb-2 flex justify-between items-end">
              <h2 className="text-xl font-black tracking-tight text-slate-900 uppercase">03. Predictive Engine Mapping</h2>
              <span className="text-xs text-slate-400 font-mono">Page 4</span>
            </div>

            <div className="grid grid-cols-3 gap-4 print-avoid-break">
              <div className="print-border-opaque p-4 rounded-xl text-center">
                <p className="text-[10px] text-slate-400 font-bold uppercase tracking-wider mb-1">Regression Type</p>
                <p className="text-lg font-black text-slate-900 font-mono uppercase">{predictive.modelType ?? 'Linear'}</p>
              </div>
              <div className="print-border-opaque p-4 rounded-xl text-center">
                <p className="text-[10px] text-slate-400 font-bold uppercase tracking-wider mb-1">Variance Bound (R²)</p>
                <p className="text-lg font-black text-[#059669] font-mono">{fmt(predictive.regressionResult?.rSquared)}</p>
              </div>
              <div className="print-border-opaque p-4 rounded-xl text-center">
                <p className="text-[10px] text-slate-400 font-bold uppercase tracking-wider mb-1">Residual Deviation (RMSE)</p>
                <p className="text-lg font-black text-slate-900 font-mono">{fmt(predictive.regressionResult?.rmse)}</p>
              </div>
            </div>

            {predictive.regressionResult?.featureImportance && (
              <div className="print-border-opaque p-6 rounded-xl space-y-4 print-avoid-break">
                <h4 className="text-xs font-black uppercase tracking-wider text-slate-800">Evaluated Attribute Explanatory Influence Weights</h4>
                <div className="space-y-3">
                  {Object.entries(predictive.regressionResult.featureImportance).map(([feat, imp]) => (
                    <div key={feat} className="flex items-center gap-4 text-xs font-bold">
                      <span className="font-mono text-slate-700 w-32 truncate">{feat}</span>
                      <div className="flex-1 h-3 bg-slate-100 border border-black rounded-sm overflow-hidden">
                        <div
                          className="h-full bg-[#059669]"
                          style={{ width: `${Math.min(100, (imp as number) * 100)}%` }}
                        />
                      </div>
                      <span className="font-mono text-slate-900 w-12 text-right">{((imp as number) * 100).toFixed(1)}%</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* PAGE 5: AI CONTEXTUAL CARD LIST ANALYSIS */}
        {interpret.perAnalysis.length > 0 && (
          <div className="print-page pt-4 space-y-6 bg-white">
            <div className="border-b-4 border-[#059669] pb-2 flex justify-between items-end">
              <h2 className="text-xl font-black tracking-tight text-slate-900 uppercase">04. AI Structural Interpretations</h2>
              <span className="text-xs text-slate-400 font-mono">Page 5</span>
            </div>

            <div className="space-y-4">
              {interpret.perAnalysis.map((item, i) => (
                /* Added 'print-avoid-break' here so the card stays physically grouped together */
                <div key={i} className="print-border-opaque p-5 bg-white rounded-xl space-y-2 print-avoid-break">
                  <div className="flex justify-between items-center border-b border-slate-200 pb-2">
                    <span className="text-[10px] font-black text-[#059669] uppercase tracking-wider">Contextual Node Target</span>
                    <span className="text-[10px] font-mono font-bold px-2 py-0.5 bg-slate-100 border border-black uppercase">{item.type} : {item.subject}</span>
                  </div>
                  <p className="text-xs leading-relaxed text-slate-700 font-medium">{item.interpretation}</p>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* 🌟 NEW PAGE 6: DATA VISUALIZATIONS & CHARTS PRODUCTION ENGINE */}
        {chartSuggestions.length > 0 && (
          <div className="print-page pt-4 space-y-6 bg-white">
            <div className="border-b-4 border-[#059669] pb-2 flex justify-between items-end">
              <h2 className="text-xl font-black tracking-tight text-slate-900 uppercase">05. Visualizations & Graphical Artifacts</h2>
              <span className="text-xs text-slate-400 font-mono">Page 6</span>
            </div>
            
            <div className="space-y-6">
              {chartSuggestions.map((s, i) => (
                <div key={i} className="print-border-opaque p-4 rounded-xl bg-white space-y-2 print-avoid-break">
                  <div className="border-b border-slate-200 pb-1">
                    <h4 className="text-xs font-bold text-slate-900">{s.title}</h4>
                    {s.reason && <p className="text-[10px] text-slate-400 font-medium">{s.reason}</p>}
                  </div>
                  {/* Wraps chart in fixed sizing utility specifically configured for printer drivers */}
                  <div className="print-chart-container mx-auto">
                    <PrintChartCard suggestion={s} activeSession={activeSession} index={i} />
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Screen view components remaining untouched for full pipeline functional parity
// ---------------------------------------------------------------------------

function ChartCard({ suggestion, activeSession, index }: { suggestion: ChartSuggestion; activeSession: AnalysisSession; index: number }) {
  const color = ACCENT[index % ACCENT.length]
  const descriptive: DescriptiveResult[] = activeSession.result.descriptive ?? []
  const targetColumnName = suggestion.column || suggestion.x
  const targetMetrics = descriptive.find(d => d.column === targetColumnName)

  const { chartType, title, x, y } = suggestion

  const processedData = (() => {
    if ((chartType === 'bar' || chartType === 'pie' || chartType === 'line') && targetMetrics?.frequencyTable) {
      return Object.entries(targetMetrics.frequencyTable).map(([key, count]) => ({
        name: key,
        value: Number(count),
        mean: targetMetrics.mean ?? 0
      }))
    }

    if (chartType === 'scatter') {
      const preds = activeSession.result.predictive?.regressionResult?.predictions
      if (preds && Array.isArray(preds)) {
        return preds.slice(0, 80).map((pred: any, i: number) => {
          const row = activeSession.schema.sampleRows?.[i]
          return {
            x: row && x ? (Number(row[x]) || i) : i,
            y: Number(pred) || 0,
          }
        })
      }
    }

    return (activeSession.schema.sampleRows ?? []).slice(0, 15).map((row, idx) => ({
      name: `Row ${idx + 1}`,
      x: x && row[x] ? Number(row[x]) : idx,
      y: y && row[y] ? Number(row[y]) : (targetMetrics?.mean ?? 0),
      value: targetColumnName && row[targetColumnName] ? Number(row[targetColumnName]) : 0,
      mean: targetMetrics?.mean ?? 0
    }))
  })()

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
              <YAxis dataKey="y" name={y || 'Predicted Value'} tick={{ fontSize: 10, fill: '#71717a' }} />
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
// Dedicated static printer chart layout component (Enforces dimensions)
// ---------------------------------------------------------------------------
function PrintChartCard({ suggestion, activeSession, index }: { suggestion: ChartSuggestion; activeSession: AnalysisSession; index: number }) {
  const descriptive: DescriptiveResult[] = activeSession.result.descriptive ?? []
  const targetColumnName = suggestion.column || suggestion.x
  const targetMetrics = descriptive.find(d => d.column === targetColumnName)

  const { chartType, x, y } = suggestion

  const processedData = (() => {
    if ((chartType === 'bar' || chartType === 'pie' || chartType === 'line') && targetMetrics?.frequencyTable) {
      return Object.entries(targetMetrics.frequencyTable).map(([key, count]) => ({
        name: key,
        value: Number(count),
        mean: targetMetrics.mean ?? 0
      }))
    }

    if (chartType === 'scatter') {
      const preds = activeSession.result.predictive?.regressionResult?.predictions
      if (preds && Array.isArray(preds)) {
        return preds.slice(0, 80).map((pred: any, i: number) => {
          const row = activeSession.schema.sampleRows?.[i]
          return {
            x: row && x ? (Number(row[x]) || i) : i,
            y: Number(pred) || 0,
          }
        })
      }
    }

    return (activeSession.schema.sampleRows ?? []).slice(0, 15).map((row, idx) => ({
      name: `R ${idx + 1}`,
      x: x && row[x] ? Number(row[x]) : idx,
      y: y && row[y] ? Number(row[y]) : (targetMetrics?.mean ?? 0),
      value: targetColumnName && row[targetColumnName] ? Number(row[targetColumnName]) : 0,
      mean: targetMetrics?.mean ?? 0
    }))
  })()

  // High contrast palette tailored explicitly for physical document inks
  const PRINT_ACCENT = ['#059669', '#4f46e5', '#db2777', '#ca8a04', '#0284c7', '#7c3aed']
  const color = PRINT_ACCENT[index % PRINT_ACCENT.length]

  if (chartType === 'scatter') {
    return (
      <ScatterChart width={620} height={240} margin={{ top: 10, right: 20, left: 0, bottom: 10 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
        <XAxis dataKey="x" name={x} tick={{ fontSize: 9, fill: '#475569' }} />
        <YAxis dataKey="y" name={y || 'Predicted'} tick={{ fontSize: 9, fill: '#475569' }} />
        <Scatter data={processedData} fill={color} fillOpacity={0.8} />
      </ScatterChart>
    )
  }

  if (chartType === 'line') {
    return (
      <LineChart width={620} height={240} data={processedData} margin={{ top: 10, right: 20, left: 0, bottom: 10 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
        <XAxis dataKey="name" tick={{ fontSize: 9, fill: '#475569' }} />
        <YAxis tick={{ fontSize: 9, fill: '#475569' }} />
        <Line type="monotone" dataKey="value" stroke={color} strokeWidth={2} dot={true} />
      </LineChart>
    )
  }

  if (chartType === 'pie') {
    return (
      <PieChart width={620} height={240}>
        <Pie data={processedData.slice(0, 6)} dataKey="value" nameKey="name" outerRadius={70} stroke="#ffffff" strokeWidth={2}>
          {processedData.slice(0, 6).map((_, i) => (
            <Cell key={`cell-${i}`} fill={PRINT_ACCENT[i % PRINT_ACCENT.length]} />
          ))}
        </Pie>
        <Legend iconSize={8} wrapperStyle={{ fontSize: 10, color: '#1e293b' }} />
      </PieChart>
    )
  }

  return (
    <BarChart width={620} height={240} data={processedData} margin={{ top: 10, right: 20, left: 0, bottom: 10 }}>
      <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
      <XAxis dataKey="name" tick={{ fontSize: 9, fill: '#475569' }} />
      <YAxis tick={{ fontSize: 9, fill: '#475569' }} />
      <Bar dataKey="value" fill={color} radius={[2, 2, 0, 0]} />
    </BarChart>
  )
}

function fmt(v: number | null | undefined) {
  if (v == null || isNaN(v)) return '—'
  return Math.abs(v) >= 1000
    ? v.toLocaleString(undefined, { maximumFractionDigits: 1 })
    : v.toFixed(3)
}

function SkewnessChip({ value }: { value?: number | null }) {
  if (value == null || isNaN(value)) return <span className="text-zinc-600">—</span>
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

// Fixed target bg mapping token error
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