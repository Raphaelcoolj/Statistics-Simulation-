import * as ss from 'simple-statistics'
import type { Column, Row, DescriptiveResult } from '@/lib/types'
import { detectOutliersIQR } from '@/lib/stats/preprocessing'

function extractNumeric(data: Row[], columnName: string): number[] {
  const nums: number[] = []
  for (const row of data) {
    const val = row[columnName]
    if (val === null || val === undefined || val === '') continue
    const n = typeof val === 'number' ? val : Number(val)
    if (!isNaN(n)) nums.push(n)
  }
  return nums
}

function extractStringValues(data: Row[], columnName: string): string[] {
  const vals: string[] = []
  for (const row of data) {
    const val = row[columnName]
    if (val === null || val === undefined) continue
    vals.push(String(val))
  }
  return vals
}

function safeCompute<T>(fn: () => T): T | undefined {
  try { return fn() } catch { return undefined }
}

export function computeDescriptive(data: Row[], column: Column): DescriptiveResult {
  const base: DescriptiveResult = {
    column: column.name,
    count: 0,
    nullCount: column.nullCount ?? 0,
  }

  const type = column.type

  if (type === 'continuous' || type === 'ordinal') {
    const numeric = extractNumeric(data, column.name)
    if (numeric.length < 2) {
      base.count = numeric.length
      base.note = 'Insufficient data for statistical computation'
      return base
    }

    base.count = numeric.length
    base.mean = safeCompute(() => ss.mean(numeric))
    base.median = safeCompute(() => ss.median(numeric))

    const modeArr = safeCompute(() => ss.mode(numeric))
    base.mode = modeArr !== undefined ? (Array.isArray(modeArr) ? modeArr[0] : modeArr) : undefined

    base.stdDev = safeCompute(() => ss.standardDeviation(numeric))
    base.variance = safeCompute(() => ss.variance(numeric))
    base.min = safeCompute(() => ss.min(numeric))
    base.max = safeCompute(() => ss.max(numeric))
    if (base.min !== undefined && base.max !== undefined) {
      base.range = safeCompute(() => (base.max as number) - (base.min as number))
    }
    base.iqr = safeCompute(() => ss.interquartileRange(numeric))
    base.skewness = safeCompute(() => ss.sampleSkewness(numeric))
    base.kurtosis = safeCompute(() => ss.sampleKurtosis(numeric))
    base.outlierCount = safeCompute(() => detectOutliersIQR(data, column.name).count)

    return base
  }

  if (type === 'categorical' || type === 'binary') {
    const strVals = extractStringValues(data, column.name)
    base.count = strVals.length

    const freq: Record<string, number> = {}
    let maxFreq = 0
    let modeValue: string = ''
    for (const v of strVals) {
      freq[v] = (freq[v] || 0) + 1
      if (freq[v] > maxFreq) { maxFreq = freq[v]; modeValue = v }
    }
    base.mode = modeValue
    base.frequencyTable = freq

    return base
  }

  if (type === 'datetime') {
    const numeric = extractNumeric(data, column.name)
    base.count = numeric.length

    if (numeric.length > 0) {
      base.min = safeCompute(() => ss.min(numeric))
      base.max = safeCompute(() => ss.max(numeric))
    }

    return base
  }

  base.count = extractStringValues(data, column.name).length
  return base
}

export function computeAllDescriptive(
  data: Row[],
  columns: Column[],
  selectedColumns?: string[]
): DescriptiveResult[] {
  const results: DescriptiveResult[] = []
  for (const col of columns) {
    if (selectedColumns && !selectedColumns.includes(col.name)) continue
    results.push(computeDescriptive(data, col))
  }
  return results
}

export function getFrequencyTable(
  data: Row[],
  columnName: string
): Record<string, number> {
  const freq: Record<string, number> = {}
  for (const row of data) {
    const val = row[columnName]
    if (val === null || val === undefined) continue
    const key = String(val)
    freq[key] = (freq[key] || 0) + 1
  }

  const sorted: Record<string, number> = {}
  const entries = Object.entries(freq).sort((a, b) => b[1] - a[1])
  for (const [k, v] of entries) {
    sorted[k] = v
  }
  return sorted
}

export function getPercentiles(
  data: Row[],
  columnName: string,
  percentiles: number[]
): Record<number, number> {
  const numeric = extractNumeric(data, columnName)
  if (numeric.length === 0) return {}

  const result: Record<number, number> = {}
  for (const p of percentiles) {
    const pct = p / 100
    const val = safeCompute(() => ss.quantile(numeric, pct))
    if (val !== undefined) result[p] = val
  }
  return result
}