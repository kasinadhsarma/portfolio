import { NextResponse } from 'next/server'
import { client } from '@/sanity/lib/client'
import { urlFor } from '@/sanity/lib/image'
import { writeClient } from '@/sanity/lib/writeClient'
import { uploadAssetFromUrl, slugify } from '@/sanity/lib/upload-asset'
import { isAuthorizedWrite } from '@/lib/api-auth'
import { SanityProjectCard } from '@/types/sanity'

export const dynamic = 'force-dynamic'

const CATEGORIES = ['ai', 'web', 'cybersecurity', 'database', 'cloud', 'mobile', 'desktop', 'other']

export async function GET() {
  try {
    const projects = await client.fetch<SanityProjectCard[]>(`
      *[_type == "project"] | order(publishedAt desc, _createdAt desc) {
        _id,
        title,
        "slug": slug.current,
        description,
        "image": image.asset->url,
        technologies,
        category,
        github,
        liveUrl,
        featured,
        publishedAt
      }
    `)

    // Transform the data to include image URLs
    const transformedProjects = projects.map(project => ({
      ...project,
      image: project.image ? urlFor(project.image).width(600).height(400).url() : null,
    }))

    return NextResponse.json(transformedProjects, {
      headers: {
        'Cache-Control': 'public, s-maxage=3600, stale-while-revalidate=86400'
      }
    })
  } catch (error) {
    console.error('Failed to fetch projects from Sanity:', error)
    return NextResponse.json(
      { error: 'Failed to fetch projects' },
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
    const {
      title,
      category,
      slug,
      description,
      technologies,
      github,
      liveUrl,
      featured,
      imageUrl,
    } = body ?? {}

    if (!title || !Array.isArray(category) || category.length === 0) {
      return NextResponse.json(
        { error: 'title is required, and category must be a non-empty array' },
        { status: 400 }
      )
    }
    if (category.length > 3 || category.some((c: string) => !CATEGORIES.includes(c))) {
      return NextResponse.json(
        { error: `category must have 1-3 values from: ${CATEGORIES.join(', ')}` },
        { status: 400 }
      )
    }

    let image
    if (imageUrl) {
      const asset = await uploadAssetFromUrl(imageUrl, 'image')
      image = { _type: 'image', asset: { _type: 'reference', _ref: asset._id }, alt: title }
    }

    const created = await writeClient.create({
      _type: 'project',
      title,
      slug: { _type: 'slug', current: slug || slugify(title) },
      category,
      description,
      technologies,
      github,
      liveUrl,
      featured: Boolean(featured),
      status: 'development',
      publishedAt: new Date().toISOString(),
      ...(image && { image }),
    })

    return NextResponse.json(created, { status: 201 })
  } catch (error) {
    console.error('Failed to create project in Sanity:', error)
    return NextResponse.json(
      { error: 'Failed to create project' },
      { status: 500 }
    )
  }
}
