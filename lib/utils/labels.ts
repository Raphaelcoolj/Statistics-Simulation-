import type { Column, DatasetSchema } from '@/lib/types'

/** Columns the heuristic could not classify — candidates for AI inspection. */
export function uncertainCodeColumns(schema: DatasetSchema | null | undefined): Column[] {
  return (schema?.columns ?? []).filter(c => c.codeUncertain)
}

/**
 * Apply AI code-detection output onto the schema columns: sets `coded`, `labels`
 * and clears the uncertainty flag. Returns a new columns array (schema columns
 * are already merged by name).
 */
export function applyCodeDetection(
  schema: DatasetSchema | null | undefined,
  detection: Record<string, { coded: boolean; labels?: Record<string, string> }>,
): Column[] {
  if (!schema) return []
  const keys = Object.keys(detection ?? {})
  if (keys.length === 0) return schema.columns
  return schema.columns.map(col => {
    const d = detection[col.name]
    if (!d) return col
    return {
      ...col,
      coded: col.labels ? true : d.coded,
      codeUncertain: false,
      labels: d.labels && Object.keys(d.labels).length > 0 ? d.labels : col.labels,
      codeNote: d.coded && d.labels
        ? 'Values identified as codes by AI and mapped to human-readable labels.'
        : col.codeNote,
    }
  })
}

/**
 * Human-readable label for a coded value, e.g. sex: "1" -> "Male".
 * Falls back to the raw value when no codebook entry (or no codebook) exists.
 */
export function labelledValue(column: Column | undefined, value: unknown): string {
  if (!column?.labels || value === null || value === undefined) return String(value)
  const key = String(value)
  return column.labels[key] ?? String(value)
}

export function columnByName(
  schema: DatasetSchema | null | undefined,
  name: string | undefined | null,
): Column | undefined {
  if (!schema || name == null) return undefined
  return schema.columns.find(c => c.name === name)
}

/** Column whose values are mapped to human-readable labels via a codebook. */
export function labelledColumns(schema: DatasetSchema | null | undefined): Column[] {
  return (schema?.columns ?? []).filter(c => c.labels && Object.keys(c.labels).length > 0)
}