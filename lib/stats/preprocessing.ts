import * as ss from 'simple-statistics'
import type { Row, Column } from '@/lib/types'

function isMissing(val: unknown): boolean {
  if (val === null || val === undefined || val === '') return true
  if (typeof val === 'string' && ['NA', 'NaN', 'null', 'N/A', 'na', 'n/a'].includes(val)) return true
  return false
}

function extractNumeric(data: Row[], columnName: string): number[] {
  const nums: number[] = []
  for (const row of data) {
    const val = row[columnName]
    if (isMissing(val)) continue
    const n = typeof val === 'number' ? val : Number(val)
    if (!isNaN(n)) nums.push(n)
  }
  return nums
}

export function standardize(data: Row[], columns: Column[]): Row[] {
  const result = data.map(row => ({ ...row }))
  for (const col of columns) {
    if (col.type !== 'continuous') continue
    const vals = extractNumeric(result, col.name)
    if (vals.length < 2) continue
    const mean = ss.mean(vals)
    const std = ss.standardDeviation(vals)
    if (std === 0) continue
    for (const row of result) {
      const val = row[col.name]
      if (!isMissing(val)) {
        row[col.name] = (Number(val) - mean) / std
      }
    }
  }
  return result
}

export function computeStandardizationParams(
  data: Row[], columns: Column[]
): { column: string; mean: number; std: number }[] {
  const params: { column: string; mean: number; std: number }[] = []
  for (const col of columns) {
    if (col.type !== 'continuous') continue
    const vals = extractNumeric(data, col.name)
    if (vals.length < 2) continue
    const mean = ss.mean(vals)
    const std = ss.standardDeviation(vals)
    if (std === 0) continue
    params.push({ column: col.name, mean, std })
  }
  return params
}

export function applyStandardization(
  data: Row[], params: { column: string; mean: number; std: number }[]
): Row[] {
  const result = data.map(row => ({ ...row }))
  for (const { column, mean, std } of params) {
    for (const row of result) {
      const val = row[column]
      if (!isMissing(val)) {
        row[column] = (Number(val) - mean) / std
      }
    }
  }
  return result
}

export function oneHotEncode(
  data: Row[], columns: Column[]
): { data: Row[]; addedColumns: Record<string, string[]> } {
  const result = data.map(row => ({ ...row }))
  const addedColumns: Record<string, string[]> = {}

  for (const col of columns) {
    if (col.type !== 'categorical' && col.type !== 'binary') continue

    const categories = new Set<string>()
    for (const row of result) {
      const val = row[col.name]
      if (!isMissing(val)) categories.add(String(val))
    }

    if (categories.size <= 1) continue

    const sorted = Array.from(categories).sort()
    const encoded: string[] = []

    for (const cat of sorted.slice(1)) {
      const newCol = `${col.name}_${cat}`
      encoded.push(newCol)
      for (const row of result) {
        row[newCol] = String(row[col.name]) === cat ? 1 : 0
      }
    }

    addedColumns[col.name] = encoded
  }

  return { data: result, addedColumns }
}

export interface PreprocessResult {
  data: Row[]
  standardized: { column: string; mean: number; std: number }[]
  encoded: Record<string, string[]>
  droppedRows: number
}

export function preprocessForModel(
  data: Row[],
  dependent: string,
  predictors: string[],
  allColumns: Column[]
): PreprocessResult {
  const colMap = new Map(allColumns.map(c => [c.name, c]))
  const predCols = predictors.map(p => colMap.get(p)).filter(Boolean) as Column[]

  // Impute predictors using ALL rows (better stats), then drop dependent-missing rows
  let cleanData = applyMissingValueImputation(data, predCols)
  const preDropCount = cleanData.length
  cleanData = cleanData.filter(row => !isMissing(row[dependent]))
  const droppedRows = preDropCount - cleanData.length

  const { data: encodedData, addedColumns } = oneHotEncode(cleanData, predCols)

  const numericPredCols = predCols.filter(c => c.type === 'continuous')
  const stdParams = computeStandardizationParams(encodedData, numericPredCols)
  const standardizedData = applyStandardization(encodedData, stdParams)

  return {
    data: standardizedData,
    standardized: stdParams,
    encoded: addedColumns,
    droppedRows,
  }
}

export function trainTestSplit<T>(data: T[], testRatio = 0.2): { train: T[]; test: T[] } {
  const shuffled = [...data]
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]]
  }
  const splitIdx = Math.floor(shuffled.length * (1 - testRatio))
  return {
    train: shuffled.slice(0, splitIdx),
    test: shuffled.slice(splitIdx),
  }
}

export function computeVIF(
  data: Row[], predictors: string[], allColumns: Column[]
): { predictor: string; value: number }[] {
  const colMap = new Map(allColumns.map(c => [c.name, c]))
  const numericPreds = predictors.filter(p => {
    const col = colMap.get(p)
    return col && (col.type === 'continuous' || col.type === 'ordinal')
  })
  if (numericPreds.length < 2) return []

  const results: { predictor: string; value: number }[] = []
  for (const target of numericPreds) {
    const others = numericPreds.filter(p => p !== target)
    if (others.length === 0) continue

    const y = extractNumeric(data, target)
    const xMats = others.map(p => extractNumeric(data, p))
    const minLen = Math.min(y.length, ...xMats.map(col => col.length))
    if (minLen < 10) continue

    const yTrimmed = y.slice(0, minLen)
    const xTrimmed = xMats.map(col => col.slice(0, minLen))
    const X: number[][] = yTrimmed.map((_, i) => [1, ...xTrimmed.map(col => col[i])])
    const Xt = X[0].map((_, colIdx) => X.map(row => row[colIdx]))
    const XtX = Xt.map(row => X[0].map((_, j) => row.reduce((sum, _, k) => sum + row[k] * X[k][j], 0)))
    const XtY = Xt.map(row => row.reduce((sum, _, k) => sum + row[k] * yTrimmed[k], 0))

    const nEq = XtX.length
    const aug = XtX.map((row, i) => [...row, XtY[i]])
    let singular = false
    for (let col = 0; col < nEq; col++) {
      let maxRow = col
      for (let row = col + 1; row < nEq; row++) {
        if (Math.abs(aug[row][col]) > Math.abs(aug[maxRow][col])) maxRow = row
      }
      [aug[col], aug[maxRow]] = [aug[maxRow], aug[col]]
      const pivot = aug[col][col]
      if (Math.abs(pivot) < 1e-12) { singular = true; break }
      for (let row = col; row <= nEq; row++) aug[col][row] /= pivot
      for (let row = 0; row < nEq; row++) {
        if (row !== col) {
          const factor = aug[row][col]
          for (let j = col; j <= nEq; j++) aug[row][j] -= factor * aug[col][j]
        }
      }
    }

    if (singular) {
      results.push({ predictor: target, value: Infinity })
      continue
    }

    const beta = aug.map(row => row[nEq])
    const yPred = X.map(row => row.reduce((sum, xv, j) => sum + xv * beta[j], 0))
    const yMean = ss.mean(yTrimmed)
    const ssRes = yTrimmed.reduce((sum, yi, i) => sum + (yi - yPred[i]) ** 2, 0)
    const ssTot = yTrimmed.reduce((sum, yi) => sum + (yi - yMean) ** 2, 0)
    const rSquared = ssTot > 0 ? 1 - ssRes / ssTot : 0
    const vif = 1 - rSquared > 1e-10 ? 1 / (1 - rSquared) : Infinity
    results.push({ predictor: target, value: vif })
  }
  return results
}

export function detectOutliersIQR(
  data: Row[], column: string
): { count: number; lowerBound: number; upperBound: number } {
  const vals = extractNumeric(data, column)
  if (vals.length < 4) return { count: 0, lowerBound: 0, upperBound: 0 }
  const sorted = [...vals].sort((a, b) => a - b)
  const q1 = ss.quantile(sorted, 0.25)
  const q3 = ss.quantile(sorted, 0.75)
  const iqr = q3 - q1
  const lower = q1 - 1.5 * iqr
  const upper = q3 + 1.5 * iqr
  const count = vals.filter(v => v < lower || v > upper).length
  return { count, lowerBound: lower, upperBound: upper }
}

function applyMissingValueImputation(data: Row[], columns: Column[]): Row[] {
  const result = data.map(row => ({ ...row }))
  for (const col of columns) {
    if (col.type === 'continuous') {
      const vals = extractNumeric(result, col.name)
      const mean = vals.length > 0 ? ss.mean(vals) : 0
      for (const row of result) {
        if (isMissing(row[col.name])) row[col.name] = mean
      }
    } else if (col.type === 'ordinal') {
      const vals = extractNumeric(result, col.name)
      const median = vals.length > 0 ? ss.median(vals) : 0
      for (const row of result) {
        if (isMissing(row[col.name])) row[col.name] = median
      }
    } else {
      const strVals: string[] = []
      for (const row of result) {
        if (!isMissing(row[col.name])) strVals.push(String(row[col.name]))
      }
      const freq: Record<string, number> = {}
      let modeVal = ''
      let maxFreq = 0
      for (const v of strVals) {
        freq[v] = (freq[v] || 0) + 1
        if (freq[v] > maxFreq) { maxFreq = freq[v]; modeVal = v }
      }
      for (const row of result) {
        if (isMissing(row[col.name])) row[col.name] = modeVal
      }
    }
  }
  return result
}
