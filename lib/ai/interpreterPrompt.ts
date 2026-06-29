import type { DatasetSchema, AnalysisResult, InterpretResponseBody } from '@/lib/types'
import { callAI } from '@/lib/ai/providerChain'

export function buildInterpreterSystemPrompt(): string {
  return `You are a statistics tutor embedded in StatLab.
Explain computed statistical results in clear, accurate, 
plain English for non-statisticians.

Rules:
- Never recompute or change the numbers given to you
- Use exact numeric values provided
- Explain what results mean practically, not just statistically
- Flag statistical significance clearly
- Keep each interpretation to 2-4 sentences maximum
- Return ONLY valid JSON, no preamble, no markdown`
}

export function buildInterpreterUserPrompt(
  schema: DatasetSchema,
  result: AnalysisResult
): string {
  return `Interpret these results for: ${schema.fileName}
(${schema.rowCount} rows, ${schema.columnCount} columns)

${result.descriptive?.length ? `DESCRIPTIVE:\n${result.descriptive.map(d =>
  `${d.column}: mean=${d.mean ?? 'N/A'}, median=${d.median ?? 'N/A'}, ` +
  `stdDev=${d.stdDev ?? 'N/A'}, skewness=${d.skewness ?? 'N/A'}`
).join('\n')}` : ''}

${result.inferential?.correlations?.length ? `CORRELATIONS:\n${
  result.inferential.correlations.map(c =>
    `${c.columnA} vs ${c.columnB}: r=${c.r} (${c.method}, ${c.interpretation})`
  ).join('\n')}` : ''}

${result.inferential?.hypothesisTests?.length ? `HYPOTHESIS TESTS:\n${
  result.inferential.hypothesisTests.map(h =>
    `${h.testType}: stat=${h.statistic}, p=${h.pValue}, significant=${h.significant}`
  ).join('\n')}` : ''}

${result.predictive ? `PREDICTIVE MODEL:
type: ${result.predictive.modelType}
dependent: ${result.predictive.regressionResult.dependent}
predictors: ${result.predictive.regressionResult.predictors.join(', ')}
R²: ${result.predictive.regressionResult.rSquared ?? 'N/A'}
accuracy: ${result.predictive.regressionResult.accuracy ?? 'N/A'}` : ''}

Return ONLY this JSON:
{
  "summary": "2-3 sentence overall summary of key findings",
  "perAnalysis": [
    {
      "type": "descriptive|correlation|hypothesis|predictive|forecast",
      "subject": "column name or pair",
      "interpretation": "2-4 sentence explanation"
    }
  ]
}`
}

export function parseInterpreterResponse(
  raw: string
): Pick<InterpretResponseBody, 'summary' | 'perAnalysis'> {
  try {
    const cleaned = raw.replace(/```json|```/g, '').trim()
    const parsed = JSON.parse(cleaned)
    if (typeof parsed.summary !== 'string' || !parsed.summary) {
      throw new Error('Invalid summary')
    }
    if (!Array.isArray(parsed.perAnalysis)) {
      throw new Error('Invalid perAnalysis')
    }
    return { summary: parsed.summary, perAnalysis: parsed.perAnalysis }
  } catch {
    console.error('[StatLab Interpreter] Parse failed:', raw)
    return {
      summary: 'Analysis complete. Review the charts for full insights.',
      perAnalysis: [],
    }
  }
}

function validateInterpreterResponse(raw: string): boolean {
  try {
    const cleaned = raw.replace(/```json|```/g, '').trim()
    const parsed = JSON.parse(cleaned)
    return typeof parsed.summary === 'string' && parsed.summary.length > 0 && Array.isArray(parsed.perAnalysis)
  } catch {
    return false
  }
}

export async function runInterpreter(
  schema: DatasetSchema,
  result: AnalysisResult
): Promise<Pick<InterpretResponseBody, 'summary' | 'perAnalysis'> & { provider: string; fallbackUsed: boolean }> {
  const system = buildInterpreterSystemPrompt()
  const user = buildInterpreterUserPrompt(schema, result)
  const response = await callAI(system, user, validateInterpreterResponse, 'groq')
  const parsed = parseInterpreterResponse(response.content)
  return {
    ...parsed,
    provider: response.provider,
    fallbackUsed: response.fallbackUsed,
  }
}
