import { NextRequest, NextResponse } from 'next/server'
import { runCodeDetector } from '@/lib/ai/codedDetector'
import type { Column } from '@/lib/types'

/**
 * POST /api/detect-codes
 * Inspects opaque numeric columns the heuristic rules were unsure about, using
 * AI to decide whether they hold category/scale codes and to suggest labels.
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const columns = (body?.columns ?? []) as Column[]
    if (!Array.isArray(columns) || columns.length === 0) {
      return NextResponse.json({ success: false, error: 'No columns provided' }, { status: 400 })
    }

    const output = await runCodeDetector(columns)

    return NextResponse.json({ success: true, output })
  } catch (err) {
    console.error('[StatLab CodeDetector]', err)
    // Never block the analysis pipeline — fall back to "no codes detected".
    return NextResponse.json({ success: true, output: { columns: {} } })
  }
}