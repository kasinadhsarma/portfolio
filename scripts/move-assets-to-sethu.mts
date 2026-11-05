// One-time job: copy every image and PDF that still lives on Sanity's CDN into SethuCMS, then
// point the records at the copies. Safe to run again: records already moved are left alone.
//
//   node --env-file=.env.local scripts/move-assets-to-sethu.mts            (dry run: only lists)
//   SETHUCMS_EDITOR_TOKEN=... node --env-file=.env.local scripts/move-assets-to-sethu.mts --apply
//
// It needs an Editor token (SETHUCMS_EDITOR_TOKEN) to upload and update. Make one in the admin
// under Roles and access, and revoke it afterwards. After this runs, nothing refers to Sanity.
import { SethuClient } from '@sethucms/sdk'

const OLD_HOST = 'https://cdn.sanity.io/'
const MAX_BYTES = 4_000_000 // the gateway's upload limit
const TYPES: Record<string, string> = {
  png: 'image/png', jpg: 'image/jpeg', jpeg: 'image/jpeg', webp: 'image/webp', gif: 'image/gif',
  avif: 'image/avif', svg: 'image/svg+xml', pdf: 'application/pdf',
}

const apply = process.argv.includes('--apply')
const need = (name: string) => process.env[name] ?? (console.error(`Missing ${name}`), process.exit(1))
const baseUrl = need('SETHUCMS_BASE_URL').replace(/\/+$/, '')
const connectionId = need('SETHUCMS_CONNECTION_ID')
const token = apply ? need('SETHUCMS_EDITOR_TOKEN') : need('SETHUCMS_TOKEN')
const cms = new SethuClient({ baseUrl, token: () => token })

type Json = Record<string, unknown>
const copies = new Map<string, string>() // old link -> new link, so one file is uploaded once
let failed = 0

// Every object in a record that has a `url` on the old CDN.
function oldLinks(value: unknown, found: Json[] = []): Json[] {
  if (Array.isArray(value)) value.forEach((v) => oldLinks(v, found))
  else if (value && typeof value === 'object') {
    const o = value as Json
    if (typeof o.url === 'string' && o.url.startsWith(OLD_HOST)) found.push(o)
    else Object.values(o).forEach((v) => oldLinks(v, found))
  }
  return found
}

async function copy(url: string): Promise<string> {
  const known = copies.get(url)
  if (known) return known
  const res = await fetch(url)
  if (!res.ok) throw new Error(`the old link answered ${res.status}`)
  const bytes = Buffer.from(await res.arrayBuffer())
  if (bytes.length > MAX_BYTES) throw new Error(`larger than ${MAX_BYTES / 1_000_000} MB`)
  const name = new URL(url).pathname.split('/').pop() || 'file'
  const ext = name.split('.').pop()?.toLowerCase() ?? ''
  const contentType = TYPES[ext] ?? (res.headers.get('content-type') ?? '').split(';')[0]!.trim()

  const up = await fetch(`${baseUrl}/v1/connections/${encodeURIComponent(connectionId)}/files`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${token}`, 'Content-Type': contentType, 'X-File-Name': encodeURIComponent(name) },
    body: bytes,
  })
  if (!up.ok) throw new Error(`the gateway answered ${up.status}: ${(await up.text()).slice(0, 160)}`)
  const data = (await up.json()).data as { path: string }
  const fresh = `${baseUrl}${data.path}`
  copies.set(url, fresh)
  return fresh
}

let toMove = 0
for (const info of await cms.collections(connectionId)) {
  const col = cms.collection<Json>(connectionId, info.name)
  let records: Json[]
  try {
    records = (await col.query().limit(500).list()).data
  } catch (error) {
    console.log(`skip ${info.name}: cannot read it (${(error as Error).message})`)
    continue
  }

  for (const record of records) {
    const links = oldLinks(record)
    if (!links.length) continue
    toMove += links.length
    console.log(`${apply ? 'MOVE' : 'WOULD MOVE'} ${links.length} file(s) in ${info.name}/${record.id}`)
    if (!apply) continue

    try {
      for (const link of links) link.url = await copy(link.url as string)
      await col.update(String(record.id), record)
    } catch (error) {
      failed++
      console.log(`  FAILED: ${(error as Error).message} (the record keeps its old link)`)
    }
  }
}

console.log(
  apply
    ? `Done. ${copies.size} file(s) copied, ${failed} record(s) failed.`
    : `Dry run: ${toMove} file link(s) still on the old CDN. Re-run with --apply to move them.`,
)
