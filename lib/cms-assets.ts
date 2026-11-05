// An image or file field in SethuCMS is stored as { url, alt? }, where `url` is the link to the
// file. New uploads in /sethu live in SethuCMS's own storage. Some older records still carry
// their original link on Sanity's CDN until scripts/move-assets-to-sethu.mts has copied them;
// those keep working in the meantime, with one warning so you know the job is still to do.
const LEGACY_HOST = 'cdn.sanity.io'

let warned = false

type AssetField = { url?: string | null } | null | undefined

/** The public link of an image or file field, or null if the field is empty or not usable. */
export function assetUrl(field: AssetField): string | null {
  const url = field?.url
  if (!url) return null

  try {
    const { protocol, hostname } = new URL(url)
    if (protocol !== 'https:' && protocol !== 'http:') return null
    if (hostname === LEGACY_HOST && !warned) {
      warned = true
      console.warn('[cms] some files are still on the old Sanity CDN; run scripts/move-assets-to-sethu.mts')
    }
    return url
  } catch {
    return null
  }
}
