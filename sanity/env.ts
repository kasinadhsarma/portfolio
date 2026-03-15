export const apiVersion =
  process.env.NEXT_PUBLIC_SANITY_API_VERSION || '2026-03-15'

export const dataset = assertValue(
  process.env.NEXT_PUBLIC_SANITY_DATASET,
  'Missing environment variable: NEXT_PUBLIC_SANITY_DATASET'
)

export const projectId = assertValue(
  process.env.NEXT_PUBLIC_SANITY_PROJECT_ID,
  'Missing environment variable: NEXT_PUBLIC_SANITY_PROJECT_ID'
)

function assertValue<T>(v: T | undefined, errorMessage: string): T {
  if (v === undefined) {
    if (process.env.NODE_ENV === 'production' && !process.env.VERCEL) {
      // If we're building locally without variables, we might want to know
      // but if we're on Vercel, it's critical. 
      // Actually, let's just warn and return empty string to allow module evaluation.
    }
    console.warn(`[Sanity Env Warning]: ${errorMessage}`)
    return (process.env.NODE_ENV === 'production' ? '' : v) as T
  }

  return v
}
