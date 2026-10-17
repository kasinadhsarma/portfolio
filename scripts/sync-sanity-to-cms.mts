// One-way sync: Sanity -> SethuCMS. Dry run by default; never deletes.
//   node --env-file=.env.local scripts/sync-sanity-to-cms.mts           (dry run)
//   node --env-file=.env.local scripts/sync-sanity-to-cms.mts --apply   (needs SETHUCMS_EDITOR_TOKEN)
//   add --force to rewrite records even when Sanity says they have not changed (for example after
//   this script starts storing more, like the image/file `url`).
// Records are matched on `sanity_id`; `sanity_updated_at` decides whether a record changed.
//
// Images and PDFs are copied out of Sanity into SethuCMS's own file storage (your database), and the
// records point at the new links. Files already copied are remembered in scripts/.sethu-files.json, so
// running again never uploads the same file twice. A file that cannot be copied (too big, a type SethuCMS
// does not accept) keeps its Sanity link and is reported.
//
// A collection that does not exist yet is created for you, with every field found in your Sanity
// documents. That one step needs an admin token (SETHUCMS_ADMIN_TOKEN); make it short-lived and
// revoke it afterwards. Records themselves are written with the editor token.
import { createClient } from '@sanity/client'
import { SethuClient } from '@sethucms/sdk'
import { readFileSync, writeFileSync } from 'node:fs'

const TYPES: Record<string, string> = {
  project: 'projects',
  resume: 'resume',
  skillCategory: 'skill_categories',
  achievement: 'achievements',
  certificate: 'certificates',
  resumeFile: 'resume_files',
}

const apply = process.argv.includes('--apply')
const force = process.argv.includes('--force')
const need = (k: string) => process.env[k] ?? (console.error(`Missing ${k}`), process.exit(1))

const projectId = need('NEXT_PUBLIC_SANITY_PROJECT_ID')
const dataset = need('NEXT_PUBLIC_SANITY_DATASET')
const sanity = createClient({
  projectId,
  dataset,
  apiVersion: '2024-01-01',
  useCdn: false,
})
const token = apply ? need('SETHUCMS_EDITOR_TOKEN') : need('SETHUCMS_TOKEN')
const connectionId = need('SETHUCMS_CONNECTION_ID')
const baseUrl = need('SETHUCMS_BASE_URL').replace(/\/+$/, '')
const cms = new SethuClient({ baseUrl, token: () => token })

// A Sanity asset reference as a public URL. The files stay on Sanity's CDN; only the link is stored.
//   image-<hash>-<W>x<H>-<ext>  ->  https://cdn.sanity.io/images/<project>/<dataset>/<hash>-<W>x<H>.<ext>
//   file-<hash>-<ext>           ->  https://cdn.sanity.io/files/<project>/<dataset>/<hash>.<ext>
function assetUrl(ref: string | undefined): string | null {
  const img = ref?.match(/^image-([a-f0-9]+)-(\d+x\d+)-(\w+)$/)
  if (img) return `https://cdn.sanity.io/images/${projectId}/${dataset}/${img[1]}-${img[2]}.${img[3]}`
  const file = ref?.match(/^file-([a-f0-9]+)-(\w+)$/)
  if (file) return `https://cdn.sanity.io/files/${projectId}/${dataset}/${file[1]}.${file[2]}`
  return null
}

// Flatten Sanity-only shapes: slug -> string, image/file assets -> asset ref + public url, drop system fields.
function flatten(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(flatten)
  if (value && typeof value === 'object') {
    const o = value as Record<string, unknown>
    if (o._type === 'slug') return o.current
    if (o._type === 'image' || o._type === 'file') {
      const ref = (o.asset as { _ref?: string } | undefined)?._ref
      return { asset: ref ?? null, url: assetUrl(ref), ...(o.alt ? { alt: o.alt } : {}) }
    }
    return Object.fromEntries(
      Object.entries(o).filter(([k]) => k !== '_key').map(([k, v]) => [k, flatten(v)]),
    )
  }
  return value
}

type Field = { name: string; type: string }

// ---- Moving images and PDFs into SethuCMS ---------------------------------

const MAX_FILE_BYTES = 4_000_000 // the gateway's upload limit
const TYPES_BY_EXT: Record<string, string> = {
  png: 'image/png', jpg: 'image/jpeg', jpeg: 'image/jpeg', webp: 'image/webp', gif: 'image/gif',
  avif: 'image/avif', svg: 'image/svg+xml', pdf: 'application/pdf',
}
type Moved = { url: string; fileId: string; name: string; contentType: string }
const CACHE_FILE = new URL('./.sethu-files.json', import.meta.url)
let moved: Record<string, Moved> = {}
try {
  moved = JSON.parse(readFileSync(CACHE_FILE, 'utf8'))
} catch {
  // first run
}
const cacheKey = (ref: string) => `${connectionId}:${ref}`
const wouldMove = new Set<string>()
let moveFailures = 0

// Objects like { asset: "image-...", url: "https://cdn.sanity.io/..." } anywhere in a record.
function sanityFiles(value: unknown, found: Record<string, unknown>[] = []): Record<string, unknown>[] {
  if (Array.isArray(value)) value.forEach((v) => sanityFiles(v, found))
  else if (value && typeof value === 'object') {
    const o = value as Record<string, unknown>
    if (typeof o.asset === 'string' && typeof o.url === 'string' && o.url.startsWith('https://cdn.sanity.io/')) found.push(o)
    else Object.values(o).forEach((v) => sanityFiles(v, found))
  }
  return found
}

async function copyToSethu(ref: string, url: string): Promise<Moved> {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`Sanity answered ${res.status}`)
  const bytes = Buffer.from(await res.arrayBuffer())
  if (bytes.length > MAX_FILE_BYTES) throw new Error(`larger than ${MAX_FILE_BYTES / 1_000_000} MB`)
  const ext = new URL(url).pathname.split('.').pop()?.toLowerCase() ?? ''
  const contentType = TYPES_BY_EXT[ext] ?? (res.headers.get('content-type') ?? '').split(';')[0]!.trim()
  // The original file name, when Sanity has it; otherwise the last part of the link.
  const original: string | null = await sanity.fetch('*[_id == $ref][0].originalFilename', { ref }).catch(() => null)
  const name = original || new URL(url).pathname.split('/').pop() || 'file'
  const up = await fetch(`${baseUrl}/v1/connections/${encodeURIComponent(connectionId)}/files`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${token}`, 'Content-Type': contentType, 'X-File-Name': encodeURIComponent(name) },
    body: bytes,
  })
  if (!up.ok) throw new Error(`the gateway answered ${up.status}: ${(await up.text()).slice(0, 160)}`)
  const data = (await up.json()).data as { id: string; path: string; name: string; contentType: string }
  return { url: `${baseUrl}${data.path}`, fileId: data.id, name: data.name, contentType: data.contentType }
}

// Point every Sanity file in a record at its copy in SethuCMS (copying it first when needed).
async function moveFiles(record: Record<string, unknown>): Promise<void> {
  for (const file of sanityFiles(record)) {
    const ref = file.asset as string
    let hit = moved[cacheKey(ref)]
    if (!hit) {
      if (!apply) {
        wouldMove.add(ref)
        continue
      }
      try {
        hit = await copyToSethu(ref, file.url as string)
      } catch (e) {
        moveFailures++
        console.log(`  could not copy ${ref} into SethuCMS: ${(e as Error).message} (the Sanity link is kept)`)
        continue
      }
      moved[cacheKey(ref)] = hit
      writeFileSync(CACHE_FILE, JSON.stringify(moved, null, 2))
    }
    Object.assign(file, { url: hit.url, fileId: hit.fileId, name: hit.name, contentType: hit.contentType })
  }
}

// Work out the field list of a new collection from the records that will go into it.
// Names the gateway would refuse (and "id", which is the record key) are left out with a warning;
// they are also removed from the records, or the gateway would refuse those records.
function inferFields(collection: string, records: Record<string, unknown>[]): { fields: Field[]; skipped: Set<string> } {
  const kinds = new Map<string, Set<string>>()
  const longest = new Map<string, number>()
  for (const record of records) {
    for (const [name, value] of Object.entries(record)) {
      const seen = kinds.get(name) ?? new Set<string>()
      kinds.set(name, seen)
      if (value === null || value === undefined) continue
      if (Array.isArray(value)) seen.add('array')
      else if (typeof value === 'object') seen.add('json')
      else if (typeof value === 'number') seen.add(Number.isInteger(value) ? 'integer' : 'decimal')
      else if (typeof value === 'boolean') seen.add('boolean')
      else {
        seen.add('string')
        longest.set(name, Math.max(longest.get(name) ?? 0, String(value).length))
      }
    }
  }
  const fields: Field[] = []
  const skipped = new Set<string>()
  for (const [name, seen] of kinds) {
    if (name === 'id' || !/^[A-Za-z_][A-Za-z0-9_]{0,62}$/.test(name)) {
      console.log(`  skipping field "${name}" in ${collection}: not allowed as a field name`)
      skipped.add(name)
      continue
    }
    let type = 'json'
    if (seen.size === 0) type = 'string'
    else if (seen.size === 1) type = [...seen][0]!
    else if (seen.size === 2 && seen.has('integer') && seen.has('decimal')) type = 'decimal'
    if (type === 'string' && (longest.get(name) ?? 0) > 500) type = 'text'
    fields.push({ name, type })
  }
  return { fields, skipped }
}

// Create the empty collection through the gateway (admin only). Firestore makes it with its first record.
async function declareCollection(collection: string, fields: Field[]): Promise<void> {
  const res = await fetch(`${baseUrl}/v1/admin/connections/${encodeURIComponent(connectionId)}/collections`, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${need('SETHUCMS_ADMIN_TOKEN')}`,
      'Content-Type': 'application/json',
      'Idempotency-Key': crypto.randomUUID(),
    },
    body: JSON.stringify({ name: collection, fields }),
  })
  if (!res.ok) throw new Error(`the gateway answered ${res.status}: ${(await res.text()).slice(0, 200)}`)
}

let changes = 0
let failures = 0
for (const [type, collection] of Object.entries(TYPES)) {
  const docs: Record<string, any>[] = await sanity.fetch(`*[_type == $type && !(_id in path("drafts.**"))]`, { type })
  const records = docs.map((d) => {
    const { _id, _type, _rev, _createdAt, _updatedAt, ...rest } = d
    return { ...(flatten(rest) as object), sanity_id: _id, sanity_updated_at: _updatedAt } as Record<string, unknown>
  })

  const col = cms.collection<Record<string, unknown>>(connectionId, collection)
  const existing = new Map<string, Record<string, any>>()
  let missing = false
  try {
    for (const r of (await col.query().limit(500).list()).data) existing.set(String(r.sanity_id), r)
  } catch (e) {
    missing = (e as { status?: number }).status === 404
    if (!missing) console.log(`[${type}] cannot read collection "${collection}": ${(e as Error).message}`)
  }

  if (missing && records.length === 0) {
    console.log(`[${type}] no documents in Sanity, so "${collection}" is not created yet`)
    continue
  }
  if (missing) {
    const { fields, skipped } = inferFields(collection, records)
    for (const record of records) for (const name of skipped) delete record[name]
    const summary = fields.map((f) => `${f.name}:${f.type}`).join(', ')
    if (!apply) {
      console.log(`WOULD CREATE collection ${collection} with fields: ${summary}`)
    } else {
      try {
        await declareCollection(collection, fields)
        console.log(`CREATED collection ${collection} (${fields.length} fields)`)
      } catch (e) {
        failures += docs.length
        changes += docs.length
        console.log(`[${type}] could not create collection "${collection}": ${(e as Error).message}`)
        continue
      }
    }
  }

  for (const [i, d] of docs.entries()) {
    const record = records[i]!
    const prev = existing.get(d._id)
    if (prev && !force && prev.sanity_updated_at === d._updatedAt) continue
    changes++
    console.log(`${prev ? 'UPDATE' : 'CREATE'} ${collection}/${d._id}`)
    await moveFiles(record)
    if (apply) {
      try {
        prev ? await col.update(String(prev.id), record) : await col.create(record)
      } catch (e) {
        failures++
        console.log(`  FAILED: ${(e as Error).message}`)
      }
    }
  }
  const sanityIds = new Set(docs.map((d) => d._id))
  for (const id of existing.keys()) if (!sanityIds.has(id)) console.log(`FLAG   ${collection}/${id} not in Sanity (not deleted)`)
}
console.log(apply ? `Applied ${changes - failures} of ${changes} change(s); ${failures} failed.` : `Dry run: ${changes} change(s). Re-run with --apply to write.`)
if (!apply && wouldMove.size > 0) console.log(`WOULD COPY ${wouldMove.size} image/PDF file(s) from Sanity into SethuCMS`)
if (moveFailures > 0) console.log(`${moveFailures} file(s) could not be copied and still use their Sanity link (see above).`)
if (failures > 0) {
  console.log(
    'If a collection was made by hand in Firebase it only knows the fields of its placeholder document, so records with other fields are refused. ' +
      'Delete that collection in Firebase, click "Read structure again" in the admin, then run this again and it will be created with every field.',
  )
}
