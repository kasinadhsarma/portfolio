// Images and files were uploaded to Sanity's asset CDN before the move to SethuCMS, and
// the records keep only the asset reference. This turns a reference into its public URL.
const ASSET_BASE = 'https://cdn.sanity.io'
const PROJECT = 'hnxv4yib'
const DATASET = 'production'

type ImageOptions = { width?: number; height?: number }

/** `image-<hash>-<w>x<h>-<ext>` or `file-<hash>-<ext>` -> CDN url, or null if unrecognised. */
export function assetUrl(ref: string | null | undefined, options: ImageOptions = {}): string | null {
  if (!ref) return null

  const image = /^image-([a-f0-9]+)-(\d+x\d+)-([a-z0-9]+)$/i.exec(ref)
  if (image) {
    const [, hash, size, ext] = image
    const params = new URLSearchParams()
    if (options.width) params.set('w', String(options.width))
    if (options.height) params.set('h', String(options.height))
    if (options.width && options.height) params.set('fit', 'crop')
    const query = params.size ? `?${params}` : ''
    return `${ASSET_BASE}/images/${PROJECT}/${DATASET}/${hash}-${size}.${ext}${query}`
  }

  const file = /^file-([a-f0-9]+)-([a-z0-9]+)$/i.exec(ref)
  if (file) return `${ASSET_BASE}/files/${PROJECT}/${DATASET}/${file[1]}.${file[2]}`

  return null
}
