
import { SanityProjectCard } from "@/types/sanity"
import { client } from "@/sanity/lib/client"
import { urlFor } from "@/sanity/lib/image"
import { ProjectsClient } from "./projects-client"

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
    <div className="container mx-auto space-y-6">
      <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
        <div>
          <div className="flex flex-col gap-2">
            <div className="flex items-center gap-4">
              <h1 className="text-4xl font-bold">Projects</h1>
            </div>
            <div className="flex items-center gap-4">
              <p className="text-muted-foreground">
                Explore my portfolio of personal and professional projects
              </p>
              <a
                href="https://wakatime.com/@9849b760-c9b2-46e7-b469-271f5faa6c63"
                target="_blank"
                rel="noopener noreferrer"
                className="hover:opacity-80 transition-opacity"
              >
                <img
                  src="https://wakatime.com/badge/user/9849b760-c9b2-46e7-b469-271f5faa6c63.svg"
                  alt="Total time coded since Aug 17 2024"
                  height="20"
                />
              </a>
            </div>
            <div className="h-1 w-16 bg-primary mt-2"></div>
          </div>
        </div>
      </div>

      <ProjectsClient projects={projects} />
    </div>
  )
}
