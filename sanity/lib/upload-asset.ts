import { writeClient } from './writeClient'

export async function uploadAssetFromUrl(url: string, assetType: 'image' | 'file') {
  const res = await fetch(url)
  if (!res.ok) {
    throw new Error(`Failed to download asset from ${url}: ${res.status} ${res.statusText}`)
  }

  const buffer = Buffer.from(await res.arrayBuffer())
  const filename = decodeURIComponent(url.split('/').pop()?.split('?')[0] || `upload-${Date.now()}`)

  return writeClient.assets.upload(assetType, buffer, { filename })
}

export function slugify(input: string): string {
  return input
    .toLowerCase()
    .trim()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 96)
}
