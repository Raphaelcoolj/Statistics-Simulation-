import type { DatasetSchema, AnalysisResult, InterpretResponseBody } from '@/lib/types'
import { callAI } from '@/lib/ai/providerChain'

export function buildInterpreterSystemPrompt(): string {
  return `You are StatLab AI, a senior Data Scientist with 15+ years of experience in statistics, machine learning, experimentation, and business analytics.

Your role is not to simply describe charts or output numbers. Your responsibility is to think like an experienced data scientist consulting for a client.

Personality:
- Professional, precise, evidence-based, business-oriented
- Honest about uncertainty — never exaggerate findings
- Never fabricate statistics — every conclusion must come directly from the dataset and computed metrics

Writing Style:
- Write like a consultant delivering a report to executives
- Avoid robotic statements and repeating statistics unnecessarily
- Explain technical terms in plain English when appropriate
- Prioritize interpretation over numbers
- Highlight anomalies worth investigating

Rules:
- Never recompute or change the numbers given to you
- Use exact numeric values provided in the data
- Never infer causation from correlation — always flag this distinction
- Explain what results mean practically, not just statistically
- When feature importance or explainability data is provided, translate it into business actions
- When business translation is provided, reference it in your interpretation
- Flag statistical significance clearly
- Always include limitations when relevant
- Return ONLY valid JSON, no preamble, no markdown`
}

export function buildInterpreterUserPrompt(
  schema: DatasetSchema,
  result: AnalysisResult,
  modelTrainingReport?: Record<string, unknown> | null,
): string {
  let modelTrainingContext = '';
  if (modelTrainingReport) {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const mt = modelTrainingReport as Record<string, any>;
    const parts: string[] = [];
    if (mt.explainability?.summary) parts.push(`EXPLAINABILITY: ${mt.explainability.summary}`);
    if (mt.explainability?.consensusRanking?.length) {
      const topFeatures = mt.explainability.consensusRanking.slice(0, 5)
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        .map((f: any) => `${f.feature} (rank ${f.consensusRank})`).join(', ');
      parts.push(`TOP FEATURES (consensus): ${topFeatures}`);
    }
    if (mt.businessTranslation?.insights?.length) {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      parts.push(`BUSINESS TRANSLATION: ${mt.businessTranslation.insights.slice(0, 4).map((i: any) => i.text).join(' ')}`);
    }
    if (mt.businessTranslation?.confidence) parts.push(`MODEL CONFIDENCE: ${mt.businessTranslation.confidence}`);
    if (mt.recommendations?.length) {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      parts.push(`RECOMMENDATIONS: ${mt.recommendations.slice(0, 3).map((r: any) => `[${r.priority}] ${r.action}`).join(' | ')}`);
    }
    if (mt.bestModel) parts.push(`BEST MODEL: ${mt.bestModel.model} (score: ${mt.bestModel.score})`);
    if (parts.length > 0) modelTrainingContext = `\n\nMODEL TRAINING & EXPLAINABILITY:\n${parts.join('\n')}`;
  }

  return `You are StatLab AI, a senior Data Scientist. Analyse the following dataset and computed results, then produce a professional consulting report.

DATASET SCHEMA:
- File: ${schema.fileName}
- Rows: ${schema.rowCount?.toLocaleString()}
- Columns: ${schema.columnCount}
- Column details:
${schema.columns.map(col =>
  `  - ${col.name} (${col.type})` +
  (col.min !== undefined ? ` | range: ${col.min}–${col.max}` : '') +
  (col.uniqueValues?.length ? ` | samples: ${col.uniqueValues.slice(0, 6).join(', ')}` : '') +
  ` | nulls: ${col.nullCount ?? 0}`
).join('\n')}

COMPUTED ANALYSIS RESULTS:
${result.descriptive?.length ? `DESCRIPTIVE STATISTICS:\n${result.descriptive.map(d =>
  `${d.column}: mean=${d.mean ?? 'N/A'}, median=${d.median ?? 'N/A'}, stdDev=${d.stdDev ?? 'N/A'}, skewness=${d.skewness ?? 'N/A'}, outliers=${d.outlierCount ?? 'N/A'}`
).join('\n')}` : ''}

${result.inferential?.correlations?.length ? `CORRELATIONS:\n${result.inferential.correlations.map(c =>
  `${c.columnA} vs ${c.columnB}: r=${c.r} (${c.method}) — ${c.interpretation}`
).join('\n')}` : ''}

${result.inferential?.hypothesisTests?.length ? `HYPOTHESIS TESTS:\n${result.inferential.hypothesisTests.map(h =>
  `${h.testType}: statistic=${h.statistic?.toFixed(4)}, p=${h.pValue?.toFixed(4)}, significant=${h.significant}`
).join('\n')}` : ''}

${result.predictive ? `PREDICTIVE MODEL:
Type: ${result.predictive.modelType}
Dependent: ${result.predictive.regressionResult.dependent}
Predictors: ${result.predictive.regressionResult.predictors.join(', ')}
R²: ${result.predictive.regressionResult.rSquared ?? 'N/A'}
RMSE: ${result.predictive.regressionResult.rmse ?? 'N/A'}
MSE: ${result.predictive.regressionResult.mse ?? 'N/A'}
Accuracy: ${result.predictive.regressionResult.accuracy ?? 'N/A'}
${result.predictive.regressionResult.featureImportance ? `Feature Importance:\n${
  Object.entries(result.predictive.regressionResult.featureImportance)
    .sort(([,a], [,b]) => Number(b) - Number(a))
    .slice(0, 8)
    .map(([feat, imp]) => `  ${feat}: ${(Number(imp) * 100).toFixed(1)}%`)
    .join('\n')}` : ''}` : ''}
${modelTrainingContext}

RESPONSE FORMAT:
Return ONLY a raw valid JSON object (no markdown, no code fences):

{
  "summary": "Write a 4-6 sentence executive summary that covers: (1) what the dataset contains and its objective, (2) data quality notes (missing values, outliers, duplicates if notable), (3) the 2-3 most important findings with business impact, (4) model performance quality if applicable, (5) top recommendation, (6) confidence level (High/Medium/Low) with brief justification. Write like a consultant, not a robot.",

  "perAnalysis": [
    {
      "type": "descriptive|correlation|hypothesis|predictive|feature_importance|business_impact|data_quality|recommendation",
      "subject": "column name, pair, or topic (e.g., 'Age', 'Revenue vs Ad Spend', 'Model Performance', 'Data Quality')",
      "interpretation": "Write 2-4 sentences as a senior data scientist would: explain the distribution/skewness for descriptive, explain what r=0.81 means practically for correlations (never just state the number), translate p-values into plain-English significance for hypothesis tests, explain model metrics in business terms for predictive, explain why each feature matters for feature importance. Always connect findings to business actions when possible. Never fabricate statistics. Always note limitations or caveats when relevant."
    }
  ]
}

Generate perAnalysis entries for:
1. Each notable descriptive variable (focus on ones with interesting distributions, high skewness, or many outliers)
2. Each significant correlation pair (explain practical meaning, note correlation ≠ causation)
3. Each hypothesis test result (translate p-values to plain English)
4. The predictive model overall (explain R²/accuracy in business terms)
5. Top feature importances (group them, explain business impact)
6. Key business insights and recommendations (actionable, data-backed)

Prioritise quality over quantity — it is better to have 6-10 excellent interpretations than 20 shallow ones.`
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
      summary: 'Analysis complete. Review the charts and data tables for full insights.',
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
  result: AnalysisResult,
  modelTrainingReport?: Record<string, unknown> | null,
): Promise<Pick<InterpretResponseBody, 'summary' | 'perAnalysis'> & { provider: string; fallbackUsed: boolean }> {
  const system = buildInterpreterSystemPrompt()
  const user = buildInterpreterUserPrompt(schema, result, modelTrainingReport)
  const response = await callAI(system, user, validateInterpreterResponse, 'groq')
  const parsed = parseInterpreterResponse(response.content)
  return {
    ...parsed,
    provider: response.provider,
    fallbackUsed: response.fallbackUsed,
  }
}
