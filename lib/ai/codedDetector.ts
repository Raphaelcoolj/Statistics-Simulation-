import type { Column } from '@/lib/types'
import { callAI } from '@/lib/ai/providerChain'

export interface CodeDetectionResult {
  coded: boolean
  reason: string
  labels?: Record<string, string>
}

export interface CodeDetectorOutput {
  columns: Record<string, CodeDetectionResult>
}

export function buildCodeDetectorSystemPrompt(): string {
  return `You are a data-science assistant that detects whether numeric columns in a
dataset are *coded* variables — integer codes standing in for category or scale
labels (e.g. sex coded as 1=Male/2=Female, education coded 1=Primary/2=Secondary)
— rather than genuine numeric measurements.

Examine the column NAME and its SAMPLE VALUES.
A column is "coded" when:
- Its values are a small set of low/small integers (e.g. 0-10, 1-5) that map to
  named categories or ordinal levels rather than quantities you'd measure.
- Pairing the column name with the raw numbers implies a codebook (e.g. a
  column named "marital" holding 1,2,3, or "pain_level" holding 0-10, or an
  opaque column like "q1"/"v2" whose distinct integers spread coincidentally).

A column is NOT a code when it is a genuine continuous measurement (age, height,
score, count, revenue — where the integers are actual measured values) even if
the sample shown looks whole.

Return ONLY JSON, no markdown, matching:
{
  "columns": {
    "<column_name>": {
      "coded": true|false,
      "reason": "one sentence",
      "labels": { "<code>": "<human label>" }
    }
  }
}
- labels is REQUIRED when coded true (best-guess map of the observed codes),
  optional/omitted when false.
- Include EVERY column passed to you.`
}

export function buildCodeDetectorUserPrompt(columns: {
  name: string
  uniqueValues?: (string | number | boolean)[]
  sampleValues?: unknown[]
}[]): string {
  return `Decide, for each column, whether its numeric values are codes for
categories/scales or genuine measurements.

${columns.map(c =>
  `- ${c.name}` +
  (c.uniqueValues?.length ? ` | uniques: ${c.uniqueValues.slice(0, 12).join(', ')}` : '') +
  (c.sampleValues?.length ? ` | samples: ${c.sampleValues.slice(0, 6).join(', ')}` : '')
).join('\n')}

Return ONLY the JSON object described in the system prompt (a single
"columns" key mapping every column above to {coded, reason, labels?}).`
}

export function validateCodeDetectorResponse(raw: string): boolean {
  try {
    const cleaned = raw.replace(/```json|```/g, '').trim()
    const parsed = JSON.parse(cleaned)
    return !!parsed.columns && typeof parsed.columns === 'object'
  } catch {
    return false
  }
}

export function parseCodeDetectorResponse(raw: string): CodeDetectorOutput {
  try {
    const cleaned = raw.replace(/```json|```/g, '').trim()
    const parsed = JSON.parse(cleaned)
    return { columns: parsed.columns ?? {} }
  } catch {
    return { columns: {} }
  }
}

export async function runCodeDetector(
  columns: Column[]
): Promise<CodeDetectorOutput> {
  const payload = columns.map(c => ({
    name: c.name,
    uniqueValues: c.uniqueValues,
    sampleValues: c.sampleValues,
  }))
  const system = buildCodeDetectorSystemPrompt()
  const user = buildCodeDetectorUserPrompt(payload)
  const response = await callAI(system, user, validateCodeDetectorResponse, 'mistral')
  return parseCodeDetectorResponse(response.content)
}