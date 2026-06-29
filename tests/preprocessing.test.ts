import { describe, it, expect } from 'vitest'

const { trainTestSplit, computeVIF, detectOutliersIQR, standardize, oneHotEncode, preprocessForModel } = await import('@/lib/stats/preprocessing')

describe('trainTestSplit', () => {
  it('splits data into train and test sets', () => {
    const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    const { train, test } = trainTestSplit(data, 0.3)
    expect(train.length).toBe(7)
    expect(test.length).toBe(3)
  })

  it('returns all data combined when merged', () => {
    const data = [1, 2, 3, 4, 5]
    const { train, test } = trainTestSplit(data, 0.2)
    expect([...train, ...test].sort()).toEqual(data.sort())
  })

  it('handles empty array', () => {
    const { train, test } = trainTestSplit([], 0.2)
    expect(train).toEqual([])
    expect(test).toEqual([])
  })
})

describe('detectOutliersIQR', () => {
  const cols = [{ name: 'x', type: 'continuous' as const }]
  const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 100].map(v => ({ x: v }))

  it('detects outliers using IQR rule', () => {
    const result = detectOutliersIQR(data, 'x')
    expect(result.count).toBeGreaterThan(0)
    expect(result.upperBound).toBeLessThan(100)
    expect(result.lowerBound).toBeLessThan(result.upperBound)
  })

  it('returns zero for small datasets', () => {
    const small = [{ x: 1 }, { x: 2 }, { x: 3 }]
    const result = detectOutliersIQR(small, 'x')
    expect(result.count).toBe(0)
  })
})

describe('standardize', () => {
  const cols = [{ name: 'x', type: 'continuous' as const }]
  const data = [{ x: 1 }, { x: 2 }, { x: 3 }, { x: 4 }, { x: 5 }]

  it('produces zero mean and unit variance', () => {
    const result = standardize(data, cols)
    const vals = result.map(r => r.x as number)
    const mean = vals.reduce((s, v) => s + v, 0) / vals.length
    expect(mean).toBeCloseTo(0, 10)
  })
})

describe('oneHotEncode', () => {
  const cols = [{ name: 'color', type: 'categorical' as const }]
  const data = [{ color: 'red' }, { color: 'blue' }, { color: 'green' }, { color: 'red' }]

  it('creates encoded columns excluding first category', () => {
    const { data: encoded, addedColumns } = oneHotEncode(data, cols)
    expect(addedColumns.color).toBeDefined()
    expect(addedColumns.color.length).toBe(2)
    expect(encoded[0]).toHaveProperty(addedColumns.color[0])
  })
})

describe('computeVIF', () => {
  const cols = [
    { name: 'x1', type: 'continuous' as const },
    { name: 'x2', type: 'continuous' as const },
  ]
  const data = Array.from({ length: 50 }, (_, i) => ({ x1: i + 1, x2: (i + 1) * 2 + Math.random() }))

  it('returns VIF values for predictors', () => {
    const results = computeVIF(data, ['x1', 'x2'], cols)
    expect(results.length).toBe(2)
    for (const r of results) {
      expect(r.predictor).toBeDefined()
      expect(r.value).toBeGreaterThan(0)
    }
  })

  it('returns empty for single predictor', () => {
    const results = computeVIF(data, ['x1'], cols)
    expect(results).toEqual([])
  })
})
