import 'server-only'
import { SethuClient, isSethuApiError, type RecordData } from '@sethucms/sdk'

// Read-only SethuCMS access. Server only: the token never reaches the client bundle.
// The client is not exported and only list/get are wrapped, so nothing can write.

export const CMS_REVALIDATE_SECONDS = 60

const baseUrl = process.env.SETHUCMS_BASE_URL
const token = process.env.SETHUCMS_TOKEN
const connectionId = process.env.SETHUCMS_CONNECTION_ID

export const isCmsConfigured = Boolean(baseUrl && token && connectionId)

// Next's fetch cache is opt-in per request, so wrap fetch to apply revalidate + tags.
const cachedFetch: typeof fetch = (input, init) =>
  fetch(input, { ...init, next: { revalidate: CMS_REVALIDATE_SECONDS, tags: ['sethucms'] } })

let client: SethuClient | null = null
function getClient(): SethuClient | null {
  if (!isCmsConfigured) return null
  client ??= new SethuClient({ baseUrl: baseUrl!, token: () => token!, fetch: cachedFetch })
  return client
}

function logCmsError(action: string, error: unknown) {
  // Only the stable code and request id are logged, never messages, urls or tokens.
  if (isSethuApiError(error)) {
    console.error(`[cms] ${action} failed: code=${error.code} requestId=${error.requestId ?? 'n/a'}`)
  } else {
    console.error(`[cms] ${action} failed: unexpected error`)
  }
}

/** One page of records. Returns [] if the gateway is down, misconfigured or errors. */
export async function cmsList<T extends RecordData = RecordData>(
  collection: string,
  options: { limit?: number; orderBy?: [field: string, direction?: 'asc' | 'desc'] } = {},
): Promise<T[]> {
  const cms = getClient()
  if (!cms) return []
  try {
    let query = cms.collection<T>(connectionId!, collection).query()
    if (options.orderBy) query = query.orderBy(options.orderBy[0], options.orderBy[1])
    if (options.limit) query = query.limit(options.limit)
    return (await query.list()).data
  } catch (error) {
    logCmsError(`list ${collection}`, error)
    return []
  }
}

/** One record by id. Returns null if missing, or if the gateway is down or errors. */
export async function cmsGet<T extends RecordData = RecordData>(
  collection: string,
  id: string | number,
): Promise<T | null> {
  const cms = getClient()
  if (!cms) return null
  try {
    return await cms.collection<T>(connectionId!, collection).get(id)
  } catch (error) {
    if (!(isSethuApiError(error) && error.code === 'NOT_FOUND')) logCmsError(`get ${collection}`, error)
    return null
  }
}
