// Live test of the AI coded-column detector. Requires an API key in .env.local.
// Run: npx tsx tests/codedDetector.live.test.ts  (or via next/compiler)
// This is NOT part of `npm test` (it makes a real network call).
import fs from 'fs'
import path from 'path'
import { runCodeDetector } from '@/lib/ai/codedDetector'
import type { Column } from '@/lib/types'

// Load .env.local into process.env (Next.js does this automatically; plain node doesn't).
const envPath = path.resolve(process.cwd(), '.env.local')
if (fs.existsSync(envPath)) {
  for (const line of fs.readFileSync(envPath, 'utf-8').split('\n')) {
    const m = line.match(/^\s*([A-Z0-9_]+)\s*=\s*(.+)\s*$/)
    if (m) process.env[m[1]] = m[2].replace(/^["']|["']$/g, '')
  }
}

function col(name: string, type: Column['type'], uniques: (string|number)[], samples: unknown[]): Column {
  return { name, type, uniqueValues: uniques, sampleValues: samples, nullCount: 0 }
}

async function main() {
  const columns: Column[] = [
    col('sex', 'ordinal', [1, 2], [1, 2, 1, 2]),
    col('marital', 'ordinal', [1, 2, 3], [1, 2, 3, 1]),
    col('pain_level', 'ordinal', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [0, 3, 7, 10, 2]),
    col('q1', 'ordinal', [1, 2, 3, 4, 5], [1, 3, 2, 5, 4]),
    col('age', 'continuous', [4, 15, 17, 21, 35, 48, 52, 64, 71, 83], [4, 15, 21, 48, 83]),
    col('score', 'continuous', [10, 32, 41, 55, 66, 72, 78, 80, 83, 90, 95, 96], [10, 55, 66, 90]),
  ]

  const out = await runCodeDetector(columns)
  console.log('Detector result:')
  for (const c of columns) {
    const r = out.columns[c.name]
    if (!r) {
      console.log(`  ${c.name.padEnd(12)} (no AI response)`)
      continue
    }
    const coded = r.coded ? 'CODED' : 'measurement'
    console.log(`  ${c.name.padEnd(12)} ${coded.padEnd(13)} ${r.reason}`)
    if (r.labels) console.log(`      labels: ${JSON.stringify(r.labels)}`)
  }
}

main().catch(e => {
  console.error('FAILED:', e.message)
  process.exit(1)
})