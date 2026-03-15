// Querying with "sanityFetch" will keep content automatically updated
// Before using it, import and render "<SanityLive />" in your layout, see
// https://github.com/sanity-io/next-sanity#live-content-api for more information.
import { defineLive } from "next-sanity/live";
import { client } from './client'

// Create a non-CDN client for live queries so content is always fresh
const liveClient = client.withConfig({ useCdn: false })

export const { sanityFetch, SanityLive } = defineLive({
  client: liveClient,
  // browserToken allows unauthenticated users to receive live content updates
  // Add NEXT_PUBLIC_SANITY_API_READ_TOKEN to .env.local if you have one
  ...(process.env.NEXT_PUBLIC_SANITY_API_READ_TOKEN && {
    browserToken: process.env.NEXT_PUBLIC_SANITY_API_READ_TOKEN,
  }),
});
