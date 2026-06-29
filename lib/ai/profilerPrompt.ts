import type { DatasetSchema, ProfilerOutput } from '@/lib/types'
import { callAI } from '@/lib/ai/providerChain'

export function buildProfilerSystemPrompt(): string {
  return `You are a senior data scientist assistant embedded in StatLab.
Your job is to analyse a dataset schema and determine the most 
meaningful statistical analyses and visualisations to run.

Rules:
- Select analyses based purely on data types and column semantics
- Never perform computation yourself
- Return ONLY valid JSON, no preamble, no markdown code fences

Model selection rules (apply in order):
1. Dependent column is binary → logistic
2. Multiple predictor columns → multiple
3. Datetime column exists → timeseries
4. One continuous predictor → linear
5. Non-linear patterns likely → polynomial
6. rowCount > 500 or many predictors → randomforest

Chart rules:
- Two continuous columns → scatter
- One continuous over time → line
- Categorical counts → bar
- Single continuous distribution → histogram
- Many column correlations → heatmap
- Binary classification output → confusion_matrix
- Regression predictions → scatter
- Group comparisons → boxplot
- Part of whole under 6 categories → pie

Identify the dependent variable by looking for column names
containing: price, salary, outcome, result, survived, target,
label, score, revenue, growth, mortality, diagnosis

For relationshipSuggestions, suggest 2-5 different dependent→predictor
relationships you can identify from the column names and types.
Each relationship should have a different dependent variable when possible.
Provide a short reason explaining why this relationship is meaningful.`
}

export function buildProfilerUserPrompt(schema: DatasetSchema): string {
  return `Analyse this dataset and return recommendations.

Dataset: ${schema.fileName}
Rows: ${schema.rowCount}
Columns: ${schema.columnCount}

Columns:
${schema.columns.map(col =>
  `- ${col.name} (${col.type})` +
  (col.min !== undefined ? ` range: ${col.min}–${col.max}` : '') +
  (col.uniqueValues?.length ? ` samples: ${col.uniqueValues.slice(0, 6).join(', ')}` : '') +
  ` nulls: ${col.nullCount ?? 0}`
).join('\n')}

Sample rows:
${JSON.stringify(schema.sampleRows.slice(0, 3), null, 2)}

Return ONLY this JSON:
{
  "analysisMap": {
    "modelType": "linear|logistic|polynomial|multiple|timeseries|randomforest",
    "dependentVariable": "column_name or null",
    "predictors": ["column_name"],
    "correlationPairs": [["col_a", "col_b"]],
    "hypothesisTests": [{ "type": "t-test|chi-square|anova", "columns": [] }],
    "descriptiveColumns": ["column_name"]
  },
  "chartSuggestions": [
    {
      "chartType": "scatter|line|bar|histogram|heatmap|boxplot|pie|confusion_matrix|roc_curve",
      "title": "descriptive title",
      "reason": "one sentence why",
      "x": "column or null",
      "y": "column or null",
      "column": "column or null",
      "series": []
    }
  ],
  "relationshipSuggestions": [
    {
      "dependent": "column_name",
      "predictors": ["col1", "col2"],
      "modelType": "linear|logistic|polynomial|multiple|timeseries|randomforest",
      "reason": "why this relationship is meaningful"
    }
  ]
}`
}

export function parseProfilerResponse(
  raw: string,
  schema: DatasetSchema
): ProfilerOutput {
  try {
    const cleaned = raw.replace(/```json|```/g, '').trim()
    const parsed = JSON.parse(cleaned)
    if (!parsed.analysisMap || !parsed.chartSuggestions) {
      throw new Error('Missing required fields')
    }
    return {
      analysisMap: parsed.analysisMap,
      chartSuggestions: parsed.chartSuggestions,
      relationshipSuggestions: parsed.relationshipSuggestions ?? [],
    }
  } catch {
    console.error('[StatLab Profiler] Parse failed:', raw)
    return {
      analysisMap: {
        modelType: 'linear',
        dependentVariable: null,
        predictors: [],
        correlationPairs: [],
        hypothesisTests: [],
        descriptiveColumns: schema.columns.map(c => c.name),
      },
      chartSuggestions: [],
      relationshipSuggestions: [],
    }
  }
}

function validateProfilerResponse(raw: string): boolean {
  try {
    const cleaned = raw.replace(/```json|```/g, '').trim()
    const parsed = JSON.parse(cleaned)
    return !!(
      parsed.analysisMap?.modelType &&
      Array.isArray(parsed.analysisMap.descriptiveColumns) &&
      Array.isArray(parsed.chartSuggestions)
    )
  } catch {
    return false
  }
}

export async function runProfiler(
  schema: DatasetSchema
): Promise<ProfilerOutput> {
  const system = buildProfilerSystemPrompt()
  const user = buildProfilerUserPrompt(schema)
  const response = await callAI(system, user, validateProfilerResponse, 'mistral')
  return parseProfilerResponse(response.content, schema)
}
