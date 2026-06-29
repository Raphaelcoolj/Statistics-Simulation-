import { describe, it, expect } from 'vitest'

const { computeDescriptive, computeAllDescriptive } = await import('@/lib/stats/descriptive')

describe('computeDescriptive', () => {
  const col = { name: 'x', type: 'continuous' as const }
  const data = [{ x: 1 }, { x: 2 }, { x: 3 }, { x: 4 }, { x: 5 }, { x: 6 }, { x: 7 }, { x: 8 }, { x: 9 }, { x: 10 }]

  it('computes basic descriptive stats', () => {
    const result = computeDescriptive(data, col)
    expect(result.count).toBe(10)
    expect(result.mean).toBeCloseTo(5.5, 1)
    expect(result.median).toBe(5.5)
    expect(result.min).toBe(1)
    expect(result.max).toBe(10)
  })

  it('computes skewness for symmetric data', () => {
    const result = computeDescriptive(data, col)
    expect(result.skewness).toBeCloseTo(0, 1)
  })

  it('computes outlier count', () => {
    const withOutlier = [{ x: 1 }, { x: 2 }, { x: 3 }, { x: 4 }, { x: 5 }, { x: 6 }, { x: 7 }, { x: 8 }, { x: 9 }, { x: 100 }]
    const result = computeDescriptive(withOutlier, col)
    expect(result.outlierCount).toBeGreaterThan(0)
  })

  it('returns nullCount from schema', () => {
    const withNulls = [{ x: 1 }, { x: null }, { x: 3 }]
    const result = computeDescriptive(withNulls, { name: 'x', type: 'continuous' as const, nullCount: 1 })
    expect(result.nullCount).toBe(1)
  })
})

describe('computeAllDescriptive', () => {
  const cols = [
    { name: 'a', type: 'continuous' as const },
    { name: 'b', type: 'continuous' as const },
    { name: 'c', type: 'categorical' as const },
  ]
  const data = [
    { a: 1, b: 10, c: 'x' },
    { a: 2, b: 20, c: 'y' },
    { a: 3, b: 30, c: 'x' },
  ]

  it('computes stats for selected columns only', () => {
    const results = computeAllDescriptive(data, cols, ['a', 'b'])
    expect(results.length).toBe(2)
    expect(results[0].column).toBe('a')
    expect(results[1].column).toBe('b')
  })

  it('computes stats for all columns if no selection', () => {
    const results = computeAllDescriptive(data, cols)
    expect(results.length).toBe(3)
  })
})
