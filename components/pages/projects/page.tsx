import { SanityProjectCard } from "@/types/sanity"
import { client } from "@/sanity/lib/client"
import { urlFor } from "@/sanity/lib/image"
import { ProjectsHeader } from "@/components/pages/projects/projects-header"
import { ProjectsClient } from "./projects-client"
import { projectsPagePatterns as p } from "@/lib/responsive/pattrens/projects"

async function getProjects(): Promise<SanityProjectCard[]> {
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
    return projects.map(project => ({
      ...project,
      image: project.image ? urlFor(project.image).width(600).height(400).url() : null,
    }))
  } catch (error) {
    console.error('Failed to fetch projects from Sanity:', error)
    return []
  }
}

export default async function ProjectsPage() {
  const projects = await getProjects()

  return (
    <div className={p.container}>
      <ProjectsHeader />
      <ProjectsClient projects={projects} />
    </div>
  )
}
