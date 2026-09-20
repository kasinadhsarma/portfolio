import { NextResponse } from 'next/server'
import { client } from '@/sanity/lib/client'
import { writeClient } from '@/sanity/lib/writeClient'
import { uploadAssetFromUrl } from '@/sanity/lib/upload-asset'
import { isAuthorizedWrite } from '@/lib/api-auth'
import { SanityResumeFileCard } from '@/types/sanity'

export const dynamic = 'force-dynamic'

export async function GET() {
  try {
    const resumeFiles = await client.fetch<SanityResumeFileCard[]>(`
      *[_type == "resumeFile" && defined(file.asset)] | order(order asc) {
        label,
        "url": file.asset->url
      }
    `)

    return NextResponse.json(resumeFiles, {
      headers: {
        'Cache-Control': 'public, s-maxage=3600, stale-while-revalidate=86400',
      },
    })
  } catch (error) {
    console.error('Failed to fetch resume files from Sanity:', error)
    return NextResponse.json(
      { error: 'Failed to fetch resume files' },
      { status: 500 }
    )
  }
}

export async function POST(request: Request) {
  if (!isAuthorizedWrite(request)) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 403 })
  }

  try {
    const body = await request.json()
    const { label, fileUrl } = body ?? {}

    if (!label || !fileUrl) {
      return NextResponse.json(
        { error: 'label and fileUrl are required' },
        { status: 400 }
      )
    }

    const asset = await uploadAssetFromUrl(fileUrl, 'file')

    const created = await writeClient.create({
      _type: 'resumeFile',
      label,
      file: { _type: 'file', asset: { _type: 'reference', _ref: asset._id } },
    })

    return NextResponse.json(created, { status: 201 })
  } catch (error) {
    console.error('Failed to create resume file in Sanity:', error)
    return NextResponse.json(
      { error: 'Failed to create resume file' },
      { status: 500 }
    )
  }
}
