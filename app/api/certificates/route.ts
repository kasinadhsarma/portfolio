import { NextResponse } from 'next/server'
import { client } from '@/sanity/lib/client'
import { urlFor } from '@/sanity/lib/image'
import { writeClient } from '@/sanity/lib/writeClient'
import { uploadAssetFromUrl } from '@/sanity/lib/upload-asset'
import { isAuthorizedWrite } from '@/lib/api-auth'
import { SanityCertificate, SanityCertificateCard } from '@/types/sanity'

export const dynamic = 'force-dynamic'

const CATEGORIES = ['featured', 'cloud', 'work', 'practical']

export async function GET() {
  try {
    const certificates = await client.fetch<SanityCertificate[]>(`
      *[_type == "certificate"] | order(category asc, order asc) {
        _id,
        title,
        issuer,
        date,
        image,
        url,
        category
      }
    `)

    const transformed: SanityCertificateCard[] = certificates.map((cert) => ({
      _id: cert._id,
      title: cert.title,
      issuer: cert.issuer,
      date: cert.date ?? null,
      image: cert.image ? urlFor(cert.image).width(400).url() : null,
      url: cert.url ?? null,
      category: cert.category,
    }))

    return NextResponse.json(transformed, {
      headers: {
        'Cache-Control': 'public, s-maxage=3600, stale-while-revalidate=86400',
      },
    })
  } catch (error) {
    console.error('Failed to fetch certificates from Sanity:', error)
    return NextResponse.json(
      { error: 'Failed to fetch certificates' },
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
    const { title, issuer, category, date, url, imageUrl } = body ?? {}

    if (!title || !issuer || !category) {
      return NextResponse.json(
        { error: 'title, issuer, and category are required' },
        { status: 400 }
      )
    }
    if (!CATEGORIES.includes(category)) {
      return NextResponse.json(
        { error: `category must be one of: ${CATEGORIES.join(', ')}` },
        { status: 400 }
      )
    }

    let image
    if (imageUrl) {
      const asset = await uploadAssetFromUrl(imageUrl, 'image')
      image = { _type: 'image', asset: { _type: 'reference', _ref: asset._id }, alt: title }
    }

    const created = await writeClient.create({
      _type: 'certificate',
      title,
      issuer,
      category,
      date,
      url,
      ...(image && { image }),
    })

    return NextResponse.json(created, { status: 201 })
  } catch (error) {
    console.error('Failed to create certificate in Sanity:', error)
    return NextResponse.json(
      { error: 'Failed to create certificate' },
      { status: 500 }
    )
  }
}
