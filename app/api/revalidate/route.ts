import { NextRequest, NextResponse } from 'next/server'
import { revalidateTag } from 'next/cache'

// Webhook endpoint called by SethuCMS after any content change.
// Protected by a shared secret – set REVALIDATE_SECRET in your env.
export async function POST(request: NextRequest) {
  const secret = process.env.REVALIDATE_SECRET
  if (secret) {
    const auth = request.headers.get('x-revalidate-secret')
    if (auth !== secret) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
    }
  }

  revalidateTag('sethucms', 'max')

  return NextResponse.json({ revalidated: true, timestamp: Date.now() })
}
