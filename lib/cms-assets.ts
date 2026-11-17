// An image or file field in SethuCMS is stored as { url, alt? }, where `url` is the link to the
// file in SethuCMS's own storage (uploaded through the /sethu admin).

type AssetField = { url?: string | null } | null | undefined

/** The public link of an image or file field, or null if the field is empty or not usable. */
export function assetUrl(field: AssetField): string | null {
  const url = field?.url
  if (!url) return null

  try {
    const { protocol } = new URL(url)
    return protocol === 'https:' || protocol === 'http:' ? url : null
  } catch {
    return null
  }
}
