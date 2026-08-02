export function validateCSVFile(file: File | null): string | null {
  if (!file) return 'No file provided'
  const name = (file.name || '').toLowerCase()
  const isCsv = name.endsWith('.csv') || file.type === 'text/csv'
  const isExcel = name.endsWith('.xlsx') || name.endsWith('.xls')
  if (!isCsv && !isExcel) return 'File must be a CSV or Excel (.xlsx / .xls)'
  if (file.size === 0) return 'File is empty'
  if (file.size > 50 * 1024 * 1024) return 'File exceeds 50MB limit'
  return null
}

export function validateAnalysisRequest(raw: string): string | null {
  try {
    const parsed = JSON.parse(raw)
    if (!parsed.mode) return 'analyses.mode is required'
    if (!['smart', 'manual'].includes(parsed.mode)) {
      return 'analyses.mode must be smart or manual'
    }
    return null
  } catch {
    return 'Invalid JSON in analyses field'
  }
}

export function validateSchema(schema: unknown): string | null {
  if (!schema) return 'Schema is required'
  const s = schema as Record<string, unknown>
  if (!Array.isArray(s.columns)) return 'Schema must have columns array'
  if (!s.fileName) return 'Schema must have fileName'
  if (typeof s.rowCount !== 'number') return 'Schema must have rowCount'
  return null
}
