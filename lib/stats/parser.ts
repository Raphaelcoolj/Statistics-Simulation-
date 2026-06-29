import { parse as csvParse } from 'csv-parse/sync'
import type { Column, DatasetSchema, Row, MissingValueReport, MissingValueStrategy, MissingValueStrategyMap, ParsedDataset } from '@/lib/types'

export async function parseCSV(buffer: Buffer, fileName: string): Promise<ParsedDataset> {
  const content = buffer.toString('utf-8')
  const records = csvParse(content, {
    columns: true,
    skip_empty_lines: true,
    relax_quotes: true,
    relax_column_count: true,
  }) as Record<string, string>[]

  const headers = Object.keys(records[0] ?? {})
  const rowCount = records.length
  const columnCount = headers.length

  const columns: Column[] = headers.map(name => {
    const values: unknown[] = records.map(r => {
      const v = r[name]
      return v === '' || v === undefined || v === null ? null : v
    })
    return buildColumn(name, values)
  })

  const data: Row[] = records.map(r => {
    const row: Row = {}
    for (const key of headers) {
      row[key] = r[key] === '' || r[key] === undefined || r[key] === null ? null : r[key]
    }
    return row
  })

  const sampleRows = data.slice(0, 10) as Record<string, unknown>[]

  const schema: DatasetSchema = {
    fileName,
    rowCount,
    columnCount,
    columns,
    sampleRows,
  }

  const missingValueReport = detectMissingValues(data, columns)

  return { schema, data, missingValueReport }
}

const MISSING_VALUES = new Set([null, undefined, '', 'NA', 'NaN', 'null', 'N/A', 'na', 'n/a'])

function isMissing(value: unknown): boolean {
  if (value === null || value === undefined) return true
  if (typeof value === 'string' && MISSING_VALUES.has(value)) return true
  return false
}

function parseNumber(value: unknown): number | null {
  if (typeof value === 'number') return value
  if (typeof value === 'string') {
    const trimmed = value.trim()
    if (trimmed === '') return null
    const num = Number(trimmed)
    return isNaN(num) ? null : num
  }
  return null
}

function detectColumnType(values: unknown[]): Column['type'] {
  const nonNull = values.filter(v => !isMissing(v))
  if (nonNull.length === 0) return 'categorical'

  const datePattern = /^\d{4}[-\/]\d{1,2}[-\/]\d{1,2}|\d{1,2}[-\/]\d{1,2}[-\/]\d{4}|\d{1,2}\s[A-Za-z]{3}\s\d{4}/
  const allDates = nonNull.every(v => {
    if (typeof v === 'string') {
      const trimmed = v.trim()
      if (datePattern.test(trimmed)) return !isNaN(Date.parse(trimmed))
      return false
    }
    return v instanceof Date && !isNaN(v.getTime())
  })
  if (allDates) return 'datetime'

  const numericValues = nonNull.map(parseNumber).filter(v => v !== null)
  const allNumeric = numericValues.length === nonNull.length
  const uniqueSet = new Set(nonNull.map(v => String(v).trim()))
  const uniqueCount = uniqueSet.size

  if (uniqueCount === 2 && allNumeric) {
    const nums = numericValues as number[]
    const isBinary = nums.every(v => v === 0 || v === 1)
    if (isBinary) return 'binary'
  }
  if (uniqueCount === 2 && !allNumeric) {
    const lower = nonNull.map(v => String(v).trim().toLowerCase())
    const isBool = lower.every(v => v === 'true' || v === 'false' || v === '0' || v === '1' || v === 'yes' || v === 'no')
    if (isBool) return 'binary'
  }

  if (uniqueCount <= 15 && !allNumeric) return 'categorical'
  if (uniqueCount <= 15 && allNumeric) return 'ordinal'
  if (allNumeric) return 'continuous'

  return 'categorical'
}

function buildColumn(name: string, values: unknown[]): Column {
   const nonNull = values.filter(v => !isMissing(v))
   const type = detectColumnType(values)
   const nullCount = values.filter(v => isMissing(v)).length
   const sampleValues = nonNull.slice(0, 5)

   let uniqueValues: (string | number)[] | undefined
   let min: number | undefined
   let max: number | undefined
   let mean: number | undefined
   let median: number | undefined

   if (type === 'categorical' || type === 'ordinal' || type === 'binary') {
     const unique = Array.from(new Set(nonNull.map(v => String(v).trim())))
     uniqueValues = unique.slice(0, 20) as (string | number)[] | undefined
   }

   if (type === 'continuous' || type === 'ordinal') {
     const nums = nonNull.map(parseNumber).filter(v => v !== null) as number[]
     if (nums.length > 0) {
       min = Math.min(...nums)
       max = Math.max(...nums)
       
       // Calculate mean
       mean = nums.reduce((sum, val) => sum + val, 0) / nums.length
       
       // Calculate median
       const sorted = [...nums].sort((a, b) => a - b)
       const mid = Math.floor(sorted.length / 2)
       median = sorted.length % 2 !== 0 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2
     }
   }

   return { name, type, uniqueValues: uniqueValues as string[] | number[] | undefined, min, max, mean, median, sampleValues, nullCount }
 }

function detectMissingValues(data: Row[], columns: Column[]): MissingValueReport {
  const byColumn: MissingValueReport['byColumn'] = {}
  let totalMissing = 0

  for (const col of columns) {
    let count = 0
    for (const row of data) {
      if (isMissing(row[col.name])) count++
    }
    const percentage = data.length > 0 ? (count / data.length) * 100 : 0
    totalMissing += count

    let suggestedStrategy: MissingValueStrategy
    if (percentage > 50) {
      suggestedStrategy = 'drop_column'
    } else {
      switch (col.type) {
        case 'continuous':
          suggestedStrategy = 'mean'
          break
        case 'ordinal':
          suggestedStrategy = 'median'
          break
        case 'categorical':
        case 'binary':
          suggestedStrategy = 'mode'
          break
        case 'datetime':
          suggestedStrategy = 'drop_rows'
          break
        default:
          suggestedStrategy = 'mode'
      }
    }

    byColumn[col.name] = { count, percentage, suggestedStrategy }
  }

  const requiresAttention = Object.values(byColumn).some(c => c.percentage > 5)

  return { totalMissing, byColumn, requiresAttention }
}

function applyMissingValueStrategy(
  data: Row[],
  columns: Column[],
  strategies: MissingValueStrategyMap
): Row[] {
  let cleaned = data.map(row => ({ ...row }))

  const columnNamesToRemove = new Set<string>()

  for (const col of columns) {
    const strategy = strategies[col.name]
    if (!strategy) continue

    if (strategy === 'drop_column') {
      columnNamesToRemove.add(col.name)
      continue
    }

    if (strategy === 'drop_rows') {
      cleaned = cleaned.filter(row => !isMissing(row[col.name]))
      continue
    }

    if (strategy === 'mean' || strategy === 'median' || strategy === 'mode' || strategy === 'zero') {
      let replacement: unknown

      if (strategy === 'zero') {
        replacement = 0
      } else {
        const validValues = cleaned
          .map(row => row[col.name])
          .filter(v => !isMissing(v))
          .map(parseNumber)
          .filter(v => v !== null) as number[]

        if (validValues.length === 0) continue

        if (strategy === 'mean') {
          replacement = validValues.reduce((a, b) => a + b, 0) / validValues.length
        } else if (strategy === 'median') {
          const sorted = [...validValues].sort((a, b) => a - b)
          const mid = Math.floor(sorted.length / 2)
          replacement = sorted.length % 2 !== 0 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2
        } else {
          const freq: Record<number, number> = {}
          let maxFreq = 0
          let mode = validValues[0]
          for (const v of validValues) {
            freq[v] = (freq[v] || 0) + 1
            if (freq[v] > maxFreq) { maxFreq = freq[v]; mode = v }
          }
          replacement = mode
        }
      }

      for (const row of cleaned) {
        if (isMissing(row[col.name])) {
          row[col.name] = replacement as string | number | null
        }
      }
      continue
    }

    if (strategy === 'forward_fill') {
      let last: unknown = null
      for (const row of cleaned) {
        if (isMissing(row[col.name])) {
          if (last !== null) row[col.name] = last as string | number | null
        } else {
          last = row[col.name]
        }
      }
      continue
    }

    if (strategy === 'backward_fill') {
      let last: unknown = null
      for (let i = cleaned.length - 1; i >= 0; i--) {
        if (isMissing(cleaned[i][col.name])) {
          if (last !== null) cleaned[i][col.name] = last as string | number | null
        } else {
          last = cleaned[i][col.name]
        }
      }
      continue
    }
  }

  if (columnNamesToRemove.size > 0) {
    cleaned = cleaned.map(row => {
      const newRow: Row = {}
      for (const key of Object.keys(row)) {
        if (!columnNamesToRemove.has(key)) newRow[key] = row[key]
      }
      return newRow
    })
  }

  return cleaned
}

export { detectMissingValues, applyMissingValueStrategy }