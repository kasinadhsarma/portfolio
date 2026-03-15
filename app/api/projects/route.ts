import { NextResponse } from 'next/server'
import { client } from '@/sanity/lib/client'
import { urlFor } from '@/sanity/lib/image'
import { SanityProjectCard } from '@/types/sanity'

export const dynamic = 'force-static'

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
