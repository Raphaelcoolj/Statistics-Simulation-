import { describe, it, expect } from 'vitest'

const { runLinearRegression, runMultipleRegression } = await import('@/lib/stats/predictive')

describe('runLinearRegression', () => {
  const data = Array.from({ length: 100 }, (_, i) => ({
    x: i + 1,
    y: (i + 1) * 2 + 5 + (Math.random() - 0.5),
  }))

  it('returns regression result with coefficients', () => {
    const result = runLinearRegression(data, 'y', 'x')
    expect(result.dependent).toBe('y')
    expect(result.predictors).toEqual(['x'])
    expect(result.coefficients.length).toBe(1)
    expect(result.intercept).toBeDefined()
    expect(result.rSquared).toBeGreaterThan(0)
    expect(result.rSquared).toBeLessThanOrEqual(1)
    expect(result.predictions.length).toBe(data.length)
    expect(result.residuals?.length).toBe(data.length)
  })

  it('produces near-perfect fit for linear data', () => {
    const perfect = Array.from({ length: 50 }, (_, i) => ({
      x: i,
      y: 3 * i + 7,
    }))
    const result = runLinearRegression(perfect, 'y', 'x')
    expect(result.coefficients[0]).toBeCloseTo(3, 1)
    expect(result.intercept).toBeCloseTo(7, 1)
    expect(result.rSquared).toBeGreaterThan(0.99)
  })
})

describe('runMultipleRegression', () => {
  const data = Array.from({ length: 100 }, (_, i) => ({
    x1: i + 1,
    x2: Math.random() * 50,
    y: (i + 1) * 2 + Math.random() * 50 * 0.3 + 1 + (Math.random() - 0.5),
  }))

  it('returns regression result with multiple coefficients', () => {
    const result = runMultipleRegression(data, 'y', ['x1', 'x2'])
    expect(result.coefficients.length).toBe(2)
    expect(result.rSquared).toBeGreaterThan(0)
    expect(result.predictions.length).toBe(data.length)
  })
})
