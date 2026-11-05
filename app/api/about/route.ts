import { NextResponse } from 'next/server'
import { getAboutContent } from '@/lib/content'

export const dynamic = 'force-dynamic'

export async function GET() {
  return NextResponse.json(await getAboutContent(), {
    headers: {
      'Cache-Control': 'public, s-maxage=3600, stale-while-revalidate=86400',
    },
  })
}
