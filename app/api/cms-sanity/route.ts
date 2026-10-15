// Temporary local sanity check for the SethuCMS SDK. Delete once verified.
import { NextResponse } from 'next/server'
import { SethuClient } from '@sethucms/sdk'

export const dynamic = 'force-dynamic'

export async function GET() {
  if (process.env.NODE_ENV === 'production') {
    return NextResponse.json({ error: 'disabled in production' }, { status: 404 })
  }

  const baseUrl = process.env.SETHUCMS_BASE_URL
  const token = process.env.SETHUCMS_TOKEN
  const connectionId = process.env.SETHUCMS_CONNECTION_ID
  const collectionName = process.env.SETHUCMS_COLLECTION ?? 'projects'

  const missing = Object.entries({ SETHUCMS_BASE_URL: baseUrl, SETHUCMS_TOKEN: token, SETHUCMS_CONNECTION_ID: connectionId })
    .filter(([, v]) => !v)
    .map(([k]) => k)
  if (missing.length) return NextResponse.json({ ok: false, missing }, { status: 500 })

  const cms = new SethuClient({ baseUrl: baseUrl!, token: () => token! })
  const steps: Record<string, unknown> = {}

  const run = async (name: string, fn: () => Promise<unknown>) => {
    try {
      steps[name] = { ok: true, result: await fn() }
    } catch (e) {
      steps[name] = { ok: false, error: e instanceof Error ? `${e.name}: ${e.message}` : String(e) }
    }
  }

  await run('health', () => cms.isHealthy())
  await run('connections', async () => (await cms.connections()).map((c) => ({ id: c.id, name: c.name, adapter: c.adapterType, status: c.status })))
  await run('collections', async () => (await cms.collections(connectionId!)).map((c) => c.name))
  await run('list', async () => {
    const page = await cms.collection(connectionId!, collectionName).query().limit(5).list()
    return page
  })

  const ok = Object.values(steps).every((s) => (s as { ok: boolean }).ok)
  return NextResponse.json({ ok, steps }, { status: ok ? 200 : 502 })
}
