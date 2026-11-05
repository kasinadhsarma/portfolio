import { NextResponse } from 'next/server'
import { getSkills } from '@/lib/content'

export const dynamic = 'force-dynamic'

export async function GET() {
  return NextResponse.json(await getSkills(), {
    headers: {
      'Cache-Control': 'public, s-maxage=3600, stale-while-revalidate=86400',
    },
  })
}
