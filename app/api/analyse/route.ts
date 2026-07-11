import { NextRequest, NextResponse } from 'next/server'
import { validateCSVFile } from '@/lib/utils/validation'
import { toErrorResponse } from '@/lib/utils/errors'

const PYTHON_BACKEND = process.env.PYTHON_BACKEND_URL ?? 'http://127.0.0.1:8000'

/**
 * POST /api/analyse
 *
 * Proxies to the FastAPI Python backend for all statistical computation.
 * The Python backend handles CSV parsing, missing value imputation,
 * descriptive statistics, inferential tests, and predictive modeling.
 *
 * @body multipart/form-data
 *   file: CSV file (required)
 *   analyses: JSON string of AnalysisRequest (required)
 *   strategies: JSON string of MissingValueStrategyMap (optional)
 */
export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData()

    const fileEntry = formData.get('file')
    const file = fileEntry instanceof File ? fileEntry : null
    const fileError = validateCSVFile(file)
    if (fileError) {
      return Response.json({ success: false, error: fileError }, { status: 400 })
    }

    const analysesRaw = formData.get('analyses')
    if (!analysesRaw || typeof analysesRaw !== 'string') {
      return Response.json({ success: false, error: 'Invalid analyses format' }, { status: 400 })
    }

    const proxyForm = new FormData()
    proxyForm.append('file', file as File)
    proxyForm.append('analyses', analysesRaw)

    const strategiesRaw = formData.get('strategies')
    if (strategiesRaw && typeof strategiesRaw === 'string') {
      proxyForm.append('strategies', strategiesRaw)
    }

    const controller = new AbortController()
    const timeout = setTimeout(() => controller.abort(), 60000)

    const res = await fetch(`${PYTHON_BACKEND}/analyse`, {
      method: 'POST',
      body: proxyForm,
      signal: controller.signal,
    })
    clearTimeout(timeout)

    if (!res.ok) {
      const errBody = await res.json().catch(() => ({}))
      return Response.json(
        { success: false, error: errBody.detail ?? `Python backend error (${res.status})` },
        { status: res.status },
      )
    }

    const data = await res.json()
    return NextResponse.json(data)
  } catch (err) {
    const msg = err instanceof Error ? err.message : 'Unknown error'
    if (msg.includes('aborted')) {
      return Response.json(
        { success: false, error: 'Analysis timed out. Try a smaller dataset.' },
        { status: 504 },
      )
    }
    return Response.json(toErrorResponse(err), { status: 502 })
  }
}
