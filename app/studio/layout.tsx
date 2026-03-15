/**
 * Studio-specific layout that provides metadata and viewport for the Sanity Studio.
 * Metadata/viewport must be exported from a server component, so they live here
 * rather than in the 'use client' page.tsx alongside NextStudio.
 */
export { metadata, viewport } from 'next-sanity/studio'

export default function StudioLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return children
}
