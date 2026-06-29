import * as ss from 'simple-statistics'
import { PolynomialRegression } from 'ml-regression'
import LogisticRegression from 'ml-logistic-regression'
import { RandomForestRegression, RandomForestClassifier } from 'ml-random-forest'
import type {
  Row, Column, ModelType, RegressionResult,
  PredictiveResult, AnalysisRequest
} from '@/lib/types'
import { preprocessForModel, trainTestSplit, computeVIF } from '@/lib/stats/preprocessing'

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

function safeCompute<T>(fn: () => T): T | null {
  try { return fn() } catch { return null }
}

function isMissing(val: unknown): boolean {
  if (val === null || val === undefined || val === '') return true
  if (typeof val === 'string' && ['NA', 'NaN', 'null', 'N/A', 'na', 'n/a'].includes(val)) return true
  return false
}

export function selectModel(
  dependent: Column,
  predictors: Column[],
  data: Row[]
): ModelType {
  if (dependent.type === 'binary') return 'logistic'

  if (predictors.length > 1) return 'multiple'

  const hasDatetime = predictors.some(p => p.type === 'datetime')
  if (hasDatetime) return 'timeseries'

  if (predictors.length === 1) {
    const y = extractNumeric(data, dependent.name)
    const x = extractNumeric(data, predictors[0].name)
    if (y.length >= 10 && x.length >= 10 && y.length === x.length) {
      const pairs = x.map((xv, i) => [xv, y[i]] as [number, number])
      const reg = safeCompute(() => ss.linearRegression(pairs))
      const regLine = reg ? ss.linearRegressionLine(reg) : null
      if (reg && regLine) {
        const rSquared = safeCompute(() => ss.rSquared(pairs, regLine)) ?? 0
        if (rSquared < 0.6) {
          const polyReg = safeCompute(() => new PolynomialRegression(x, y, 2))
          if (polyReg) {
            const polyPreds = x.map(xv => polyReg.predict(xv))
            const yMean = ss.mean(y)
            const ssRes = y.reduce((sum, yi, i) => sum + (yi - polyPreds[i]) ** 2, 0)
            const ssTot = y.reduce((sum, yi) => sum + (yi - yMean) ** 2, 0)
            const polyRSq = ssTot > 0 ? 1 - ssRes / ssTot : 0
            if (polyRSq > rSquared + 0.1) return 'polynomial'
          }
          return 'linear'
        }
      }
    }
  }

  if (dependent.type === 'continuous' && predictors.length === 1) return 'linear'

  if (data.length > 500 || predictors.length > 5) return 'randomforest'

  return 'linear'
}

export function runLinearRegression(
  data: Row[],
  dependent: string,
  predictor: string
): RegressionResult {
  const y = extractNumeric(data, dependent)
  const x = extractNumeric(data, predictor)
  const minLen = Math.min(y.length, x.length)
  const yTrimmed = y.slice(0, minLen)
  const xTrimmed = x.slice(0, minLen)

  const result: RegressionResult = {
    modelType: 'linear',
    dependent,
    predictors: [predictor],
    coefficients: [],
    intercept: 0,
    rSquared: 0,
    predictions: [],
  }

  if (yTrimmed.length < 10) {
    result.note = 'Insufficient data for predictive modelling (minimum 10 rows)'
    return result
  }

  const pairs = xTrimmed.map((xv, i) => [xv, yTrimmed[i]] as [number, number])
  const reg = safeCompute(() => ss.linearRegression(pairs))
  const regLine = reg ? ss.linearRegressionLine(reg) : null
  if (!reg || !regLine) return result

  result.intercept = reg.b
  result.coefficients = [reg.m]
  result.predictions = xTrimmed.map(regLine)
  result.rSquared = safeCompute(() => ss.rSquared(pairs, regLine)) ?? 0

  const residuals = yTrimmed.map((yi, i) => yi - result.predictions[i])
  result.residuals = residuals
  result.mse = residuals.reduce((s, r) => s + r * r, 0) / residuals.length
  result.rmse = Math.sqrt(result.mse)

  return result
}

export function runPolynomialRegression(
  data: Row[],
  dependent: string,
  predictor: string,
  degree: number = 2
): RegressionResult {
  const y = extractNumeric(data, dependent)
  const x = extractNumeric(data, predictor)
  const minLen = Math.min(y.length, x.length)
  const yTrimmed = y.slice(0, minLen)
  const xTrimmed = x.slice(0, minLen)

  const result: RegressionResult = {
    modelType: 'polynomial',
    dependent,
    predictors: [predictor],
    coefficients: [],
    intercept: 0,
    rSquared: 0,
    predictions: [],
  }

  if (yTrimmed.length < 10) {
    result.note = 'Insufficient data for predictive modelling (minimum 10 rows)'
    return result
  }

  const polyReg = safeCompute(() => new PolynomialRegression(xTrimmed, yTrimmed, degree))
  if (!polyReg) return result

  result.predictions = xTrimmed.map(xv => polyReg.predict(xv))
  const rawCoeffs = polyReg.coefficients
  result.coefficients = rawCoeffs ? Array.from(rawCoeffs as unknown as number[]) : []
  result.intercept = result.coefficients.length > 0 ? result.coefficients[0] : 0

  const yMean = ss.mean(yTrimmed)
  const ssRes = yTrimmed.reduce((sum, yi, i) => sum + (yi - result.predictions[i]) ** 2, 0)
  const ssTot = yTrimmed.reduce((sum, yi) => sum + (yi - yMean) ** 2, 0)
  result.rSquared = ssTot > 0 ? 1 - ssRes / ssTot : 0

  result.note = `Polynomial degree: ${degree}`

  const residuals = yTrimmed.map((yi, i) => yi - result.predictions[i])
  result.residuals = residuals
  result.mse = residuals.reduce((s, r) => s + r * r, 0) / residuals.length
  result.rmse = Math.sqrt(result.mse)

  return result
}

export function runMultipleRegression(
  data: Row[],
  dependent: string,
  predictors: string[]
): RegressionResult {
  const y = extractNumeric(data, dependent)

  const xMats = predictors.map(p => extractNumeric(data, p))
  const minLen = Math.min(y.length, ...xMats.map(col => col.length))
  const yTrimmed = y.slice(0, minLen)
  const xTrimmed = xMats.map(col => col.slice(0, minLen))

  const result: RegressionResult = {
    modelType: 'multiple',
    dependent,
    predictors,
    coefficients: [],
    intercept: 0,
    rSquared: 0,
    predictions: [],
  }

  if (yTrimmed.length < 10) {
    result.note = 'Insufficient data for predictive modelling (minimum 10 rows)'
    return result
  }

  const n = yTrimmed.length

  // Build X matrix with intercept column
  const X: number[][] = yTrimmed.map((_, i) => [1, ...xTrimmed.map(col => col[i])])

  // Normal equations: β = (XᵀX)⁻¹Xᵀy
  const Xt = X[0].map((_, colIdx) => X.map(row => row[colIdx]))
  const XtX = Xt.map(row => X[0].map((_, j) => row.reduce((sum, _, k) => sum + row[k] * X[k][j], 0)))
  const XtY = Xt.map(row => row.reduce((sum, _, k) => sum + row[k] * yTrimmed[k], 0))

  // Gaussian elimination
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
    result.modelType = 'linear'
    result.note = 'Fell back to linear regression due to multicollinearity in predictors'
    return runLinearRegression(data, dependent, predictors[0])
  }

  const beta = aug.map(row => row[nEq])
  result.intercept = beta[0]
  result.coefficients = beta.slice(1)
  result.predictions = X.map(row => row.reduce((sum, xv, j) => sum + xv * beta[j], 0))

  const yMean = ss.mean(yTrimmed)
  const ssRes = yTrimmed.reduce((sum, yi, i) => sum + (yi - result.predictions[i]) ** 2, 0)
  const ssTot = yTrimmed.reduce((sum, yi) => sum + (yi - yMean) ** 2, 0)
  result.rSquared = ssTot > 0 ? 1 - ssRes / ssTot : 0

  const nAdj = n
  const pAdj = predictors.length
  result.adjustedRSquared = 1 - (1 - result.rSquared) * (nAdj - 1) / (nAdj - pAdj - 1)

  const residuals = yTrimmed.map((yi, i) => yi - result.predictions[i])
  result.residuals = residuals
  result.mse = residuals.reduce((s, r) => s + r * r, 0) / residuals.length
  result.rmse = Math.sqrt(result.mse)

  return result
}

export function runLogisticRegression(
  data: Row[],
  dependent: string,
  predictors: string[]
): RegressionResult {
  const result: RegressionResult = {
    modelType: 'logistic',
    dependent,
    predictors,
    coefficients: [],
    intercept: 0,
    rSquared: 0,
    predictions: [],
  }

  // Encode dependent as 0/1
  const y: number[] = []
  const filteredData: number[][] = []

  for (const row of data) {
    const dv = row[dependent]
    if (isMissing(dv)) continue
    const dStr = String(dv).trim().toLowerCase()
    let encoded: number | null = null
    if (dStr === '1' || dStr === 'true' || dStr === 'yes') encoded = 1
    else if (dStr === '0' || dStr === 'false' || dStr === 'no') encoded = 0
    else { const dn = Number(dStr); if (!isNaN(dn)) encoded = dn }
    if (encoded === null) continue

    const xRow: number[] = []
    let skip = false
    for (let p = 0; p < predictors.length; p++) {
      const pv = row[predictors[p]]
      if (isMissing(pv)) { skip = true; break }
      const pn = typeof pv === 'number' ? pv : Number(pv)
      if (isNaN(pn)) { skip = true; break }
      xRow.push(pn)
    }
    if (skip) continue

    y.push(encoded)
    filteredData.push(xRow)
  }

  if (y.length < 10) {
    result.note = 'Insufficient data for predictive modelling (minimum 10 rows)'
    return result
  }

  try {
    const logreg = new LogisticRegression({ numSteps: 1000, learningRate: 0.1 })
    logreg.train(filteredData, y)
    const preds = logreg.predict(filteredData) as number[]
    result.predictions = preds

    const weights = (logreg as unknown as { weights: number[] }).weights
    if (weights && weights.length > 0) {
      result.intercept = weights[0] || 0
      result.coefficients = weights.slice(1)
    }

    const correct = preds.reduce((sum, p, i) => sum + (p === y[i] ? 1 : 0), 0)
    result.accuracy = correct / preds.length
    result.rSquared = undefined
    result.note = 'R² not applicable for logistic regression. Use accuracy instead.'
  } catch {
    result.note = 'Logistic regression failed to converge'
  }

  return result
}

export function runTimeSeriesRegression(
  data: Row[],
  dependent: string,
  timeColumn: string
): RegressionResult {
  const sorted = [...data].sort((a, b) => {
    const av = a[timeColumn]; const bv = b[timeColumn]
    if (av === null || av === undefined) return 1
    if (bv === null || bv === undefined) return -1
    const an = typeof av === 'number' ? av : Number(av)
    const bn = typeof bv === 'number' ? bv : Number(bv)
    return isNaN(an) ? 1 : isNaN(bn) ? -1 : an - bn
  })

  const y = extractNumeric(sorted, dependent)
  const result: RegressionResult = {
    modelType: 'timeseries',
    dependent,
    predictors: [timeColumn],
    coefficients: [],
    intercept: 0,
    rSquared: 0,
    predictions: [],
  }

  if (y.length < 10) {
    result.note = 'Insufficient data for predictive modelling (minimum 10 rows)'
    return result
  }

  const x = y.map((_, i) => i)
  const pairs = x.map((xv, i) => [xv, y[i]] as [number, number])
  const reg = safeCompute(() => ss.linearRegression(pairs))
  const regLine = reg ? ss.linearRegressionLine(reg) : null
  if (!reg || !regLine) return result

  result.intercept = reg.b
  result.coefficients = [reg.m]
  result.predictions = x.map(regLine)
  result.rSquared = safeCompute(() => ss.rSquared(pairs, regLine)) ?? 0

  const residuals = y.map((yi, i) => yi - result.predictions[i])
  result.residuals = residuals
  result.mse = residuals.reduce((s, r) => s + r * r, 0) / residuals.length
  result.rmse = Math.sqrt(result.mse)

  const lastIdx = x.length - 1
  const forecast = []
  for (let i = 1; i <= 5; i++) {
    const predIdx = lastIdx + i
    const predicted = regLine(predIdx)
    const interval = 1.96 * (result.rmse ?? 0)
    forecast.push({
      label: `Period ${predIdx + 1}`,
      predicted,
      lower: predicted - interval,
      upper: predicted + interval,
    })
  }
  result.predictions = [...result.predictions]
  return result
}

export function runRandomForest(
  data: Row[],
  dependent: string,
  predictors: string[],
  dependentType: Column['type']
): RegressionResult {
  const result: RegressionResult = {
    modelType: 'randomforest',
    dependent,
    predictors,
    coefficients: [],
    intercept: 0,
    rSquared: 0,
    predictions: [],
  }

  const y: number[] = []
  const xRows: number[][] = []

  for (const row of data) {
    const dv = row[dependent]
    if (isMissing(dv)) continue
    const dn = typeof dv === 'number' ? dv : Number(dv)
    if (isNaN(dn)) continue

    const xRow: number[] = []
    let skip = false
    for (const p of predictors) {
      const pv = row[p]
      if (isMissing(pv)) { skip = true; break }
      const pn = typeof pv === 'number' ? pv : Number(pv)
      if (isNaN(pn)) { skip = true; break }
      xRow.push(pn)
    }
    if (skip) continue
    y.push(dn)
    xRows.push(xRow)
  }

  if (y.length < 30) {
    result.note = 'Insufficient data for predictive modelling (minimum 30 rows for random forest)'
    return result
  }

  try {
    if (dependentType === 'binary') {
      const rf = new RandomForestClassifier({ nEstimators: 50, maxDepth: 10 } as Record<string, unknown>)
      rf.train(xRows, y)
      const preds = rf.predict(xRows) as number[]
      result.predictions = preds

      const correct = preds.reduce((sum, p, i) => sum + (p === y[i] ? 1 : 0), 0)
      result.accuracy = correct / preds.length
      result.rSquared = undefined
      result.note = 'R² not applicable for classification. Use accuracy instead.'
    } else {
      const rf = new RandomForestRegression({ nEstimators: 50, maxDepth: 10 } as Record<string, unknown>)
      rf.train(xRows, y)
      const preds = rf.predict(xRows) as number[]
      result.predictions = preds

      const yMean = ss.mean(y)
      const ssRes = y.reduce((sum, yi, i) => sum + (yi - preds[i]) ** 2, 0)
      const ssTot = y.reduce((sum, yi) => sum + (yi - yMean) ** 2, 0)
      result.rSquared = ssTot > 0 ? 1 - ssRes / ssTot : 0

      const residuals = y.map((yi, i) => yi - preds[i])
      result.residuals = residuals
      result.mse = residuals.reduce((s, r) => s + r * r, 0) / residuals.length
      result.rmse = Math.sqrt(result.mse)
    }

    // Permutation importance
    const importance: { feature: string; importance: number }[] = []
    const baselineMSE = result.mse ?? 0

    for (let p = 0; p < predictors.length; p++) {
      const shuffled = xRows.map(row => [...row])
      const colVals = shuffled.map(row => row[p])
      for (let i = shuffled.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [colVals[i], colVals[j]] = [colVals[j], colVals[i]]
      }
      for (let i = 0; i < shuffled.length; i++) shuffled[i][p] = colVals[i]

      try {
        let permPreds: number[]
        if (dependentType === 'binary') {
          const rfPerm = new RandomForestClassifier({ nEstimators: 50, maxDepth: 10 } as Record<string, unknown>)
          rfPerm.train(xRows, y)
          permPreds = rfPerm.predict(shuffled) as number[]
        } else {
          const rfPerm = new RandomForestRegression({ nEstimators: 50, maxDepth: 10 } as Record<string, unknown>)
          rfPerm.train(xRows, y)
          permPreds = rfPerm.predict(shuffled) as number[]
        }
        const permMSE = dependentType === 'binary'
          ? permPreds.reduce((sum, pv, i) => sum + (pv === y[i] ? 0 : 1), 0) / permPreds.length
          : permPreds.reduce((sum, pv, i) => sum + (pv - y[i]) ** 2, 0) / permPreds.length
        const imp = baselineMSE > 0 ? (permMSE - baselineMSE) / baselineMSE : 0
        importance.push({ feature: predictors[p], importance: imp })
      } catch { importance.push({ feature: predictors[p], importance: 0 }) }
    }

    importance.sort((a, b) => b.importance - a.importance)
    result.featureImportance = importance
  } catch {
    result.note = 'Random forest failed to train'
  }

  return result
}

function computeLinearPredictions(
  data: Row[], dependent: string, predictors: string[], intercept: number, coefficients: number[]
): number[] {
  const predictions: number[] = []
  for (const row of data) {
    const depVal = row[dependent]
    if (depVal === null || depVal === undefined || depVal === '') continue
    if (isNaN(Number(depVal))) continue
    let pred = intercept
    let valid = true
    for (let i = 0; i < predictors.length; i++) {
      const val = row[predictors[i]]
      if (val === null || val === undefined || val === '') { valid = false; break }
      const num = Number(val)
      if (isNaN(num)) { valid = false; break }
      pred += (coefficients[i] ?? 0) * num
    }
    if (valid) predictions.push(pred)
  }
  return predictions
}

export function runPredictive(
  data: Row[],
  columns: Column[],
  request: AnalysisRequest['predictive']
): PredictiveResult {
  const defaultResult: PredictiveResult = {
    modelType: 'linear',
    regressionResult: {
      modelType: 'linear',
      dependent: request?.dependent ?? '',
      predictors: request?.predictors ?? [],
      coefficients: [],
      intercept: 0,
      rSquared: 0,
      predictions: [],
    },
  }

  if (!request || !request.dependent || !request.predictors || request.predictors.length === 0) {
    defaultResult.regressionResult.note = 'No dependent or predictor variables specified'
    return defaultResult
  }

  const colMap = new Map(columns.map(c => [c.name, c]))
  const depCol = colMap.get(request.dependent)
  const predCols = request.predictors.map(p => colMap.get(p)).filter(Boolean) as Column[]
  if (!depCol || predCols.length === 0) {
    defaultResult.regressionResult.note = 'Dependent or predictor column not found in schema'
    return defaultResult
  }

  const modelType = request.modelType ?? selectModel(depCol, predCols, data)

  // Preprocess: impute, encode categoricals, standardize continuous
  const preprocessed = preprocessForModel(data, request.dependent, request.predictors, columns)

  // Build expanded predictor list: continuous stay as-is, categoricals replaced by encoded columns
  const expandedPredictors: string[] = []
  const preprocessNotes: string[] = []
  for (const p of request.predictors) {
    const col = colMap.get(p)
    if (!col) continue
    if (col.type === 'categorical' || col.type === 'binary') {
      const encoded = preprocessed.encoded[p]
      if (encoded && encoded.length > 0) {
        expandedPredictors.push(...encoded)
        preprocessNotes.push(`"${p}" one-hot encoded to ${encoded.length} columns`)
      }
    } else {
      expandedPredictors.push(p)
    }
  }

  if (preprocessed.standardized.length > 0) {
    preprocessNotes.push(`${preprocessed.standardized.length} continuous predictor(s) standardized (z-score)`)
  }

  // Train/test split (skip for timeseries which needs temporal order)
  const useTrainTest = modelType !== 'timeseries' && preprocessed.data.length >= 30
  const trainData = useTrainTest ? trainTestSplit(preprocessed.data).train : preprocessed.data
  const testData = useTrainTest ? trainTestSplit(preprocessed.data).test : []

  let regressionResult: RegressionResult
  switch (modelType) {
    case 'linear':
      regressionResult = runLinearRegression(trainData, request.dependent, request.predictors[0])
      break
    case 'polynomial':
      regressionResult = runPolynomialRegression(trainData, request.dependent, request.predictors[0])
      break
    case 'multiple':
      regressionResult = runMultipleRegression(trainData, request.dependent, expandedPredictors.length > 0 ? expandedPredictors : request.predictors)
      break
    case 'logistic':
      regressionResult = runLogisticRegression(trainData, request.dependent, expandedPredictors.length > 0 ? expandedPredictors : request.predictors)
      break
    case 'timeseries': {
      const timeCol = predCols.find(c => c.type === 'datetime')
      regressionResult = runTimeSeriesRegression(preprocessed.data, request.dependent, timeCol?.name ?? request.predictors[0])
      break
    }
    case 'randomforest':
      regressionResult = runRandomForest(trainData, request.dependent, expandedPredictors.length > 0 ? expandedPredictors : request.predictors, depCol.type)
      break
    default:
      regressionResult = runLinearRegression(trainData, request.dependent, request.predictors[0])
  }

  // Compute test metrics for linear models with coefficients
  if (useTrainTest && testData.length >= 10 && regressionResult.coefficients.length > 0) {
    const testPreds = computeLinearPredictions(testData, regressionResult.dependent, expandedPredictors.length > 0 ? expandedPredictors : request.predictors, regressionResult.intercept, regressionResult.coefficients)
    if (testPreds.length > 0) {
      regressionResult.testPredictions = testPreds

      const actualValues: number[] = []
      let ssRes = 0
      let ssTot = 0
      for (const row of testData) {
        const actual = typeof row[regressionResult.dependent] === 'number' ? row[regressionResult.dependent] : Number(row[regressionResult.dependent])
        if (!isNaN(actual as number)) {
          actualValues.push(actual as number)
          ssRes += ((actual as number) - testPreds[actualValues.length - 1]) ** 2
        }
      }
      if (actualValues.length >= 10) {
        const actualMean = ss.mean(actualValues)
        ssTot = actualValues.reduce((s, v) => s + (v - actualMean) ** 2, 0)
        const testRSq = ssTot > 0 ? 1 - ssRes / ssTot : 0
        const testRMse = Math.sqrt(ssRes / actualValues.length)
        regressionResult.testMetrics = {
          rSquared: Math.max(0, testRSq),
          rmse: testRMse,
          sampleSize: actualValues.length,
        }

        // Classification metrics for logistic
        if (regressionResult.modelType === 'logistic') {
          const isBinary = actualValues.every(v => v === 0 || v === 1)
          if (isBinary) {
            const roundedPreds = testPreds.map(p => Math.round(Math.max(0, Math.min(1, p))))
            let tp = 0, fp = 0, tn = 0, fn = 0
            for (let i = 0; i < actualValues.length; i++) {
              if (roundedPreds[i] === 1 && actualValues[i] === 1) tp++
              else if (roundedPreds[i] === 1 && actualValues[i] === 0) fp++
              else if (roundedPreds[i] === 0 && actualValues[i] === 0) tn++
              else if (roundedPreds[i] === 0 && actualValues[i] === 1) fn++
            }
            const accuracy = (tp + tn) / (tp + tn + fp + fn)
            const precision = tp + fp > 0 ? tp / (tp + fp) : 0
            const recall = tp + fn > 0 ? tp / (tp + fn) : 0
            const f1 = precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0
            regressionResult.testMetrics.accuracy = accuracy
            regressionResult.testMetrics.precision = precision
            regressionResult.testMetrics.recall = recall
            regressionResult.testMetrics.f1 = f1
          }
        }

        regressionResult.note = regressionResult.note
          ? `${regressionResult.note} | Test set: ${actualValues.length} samples`
          : `Test set: ${actualValues.length} samples`
      }
    }
  }

  // VIF check for multicollinearity (only for multiple regression)
  if ((modelType === 'multiple' || modelType === 'linear') && expandedPredictors.length >= 2) {
    const vifResults = computeVIF(preprocessed.data, expandedPredictors.length > 0 ? expandedPredictors : request.predictors, columns)
    if (vifResults.length > 0) {
      regressionResult.vif = vifResults
      const highVif = vifResults.filter(v => v.value > 5)
      if (highVif.length > 0) {
        const highNames = highVif.map(v => `${v.predictor} (${v.value.toFixed(1)})`).join(', ')
        regressionResult.note = regressionResult.note
          ? `${regressionResult.note} | High multicollinearity (VIF>5): ${highNames}`
          : `High multicollinearity (VIF>5): ${highNames}`
      }
    }
  }

  if (preprocessed.droppedRows > 0) {
    const notePrefix = `${preprocessed.droppedRows} row(s) dropped due to missing dependent variable values`
    regressionResult.note = regressionResult.note
      ? `${notePrefix}. ${regressionResult.note}`
      : notePrefix
  }

  if (preprocessNotes.length > 0) {
    regressionResult.note = regressionResult.note
      ? `${regressionResult.note} | Preprocessing: ${preprocessNotes.join('; ')}.`
      : `Preprocessing: ${preprocessNotes.join('; ')}.`
  }

  const predictiveResult: PredictiveResult = {
    modelType,
    regressionResult,
  }

  if (modelType === 'timeseries') {
    predictiveResult.forecast = regressionResult.predictions.slice(-5).map((p, i) => ({
      label: `Forecast ${i + 1}`,
      predicted: p,
      lower: p - 1.96 * (regressionResult.rmse ?? 0),
      upper: p + 1.96 * (regressionResult.rmse ?? 0),
    }))
  }

  return predictiveResult
}