import { NextResponse } from 'next/server'
import { getActivities } from '@/lib/content'

export const dynamic = 'force-dynamic'

export async function GET() {
  return NextResponse.json(await getActivities(), {
    headers: {
      'Cache-Control': 'public, s-maxage=3600, stale-while-revalidate=86400',
    },
  })
}
