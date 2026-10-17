import { NextResponse } from 'next/server'
import { getCertificates } from '@/lib/content'

export const dynamic = 'force-dynamic'

export async function GET() {
  return NextResponse.json(await getCertificates(), {
    headers: {
      'Cache-Control': 'public, s-maxage=3600, stale-while-revalidate=86400',
    },
  })
}
