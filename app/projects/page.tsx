'use client'

import { useState } from "react"
import { ExternalLink, Github, Star, Code2, Brain, Shield, Database, Cloud } from "lucide-react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Badge } from "@/components/ui/badge"
import { SanityProjectCard } from "@/types/sanity"
import { client } from "@/sanity/lib/client"
import { urlFor } from "@/sanity/lib/image"

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

  const categories = [
    { id: 'all', label: 'All Projects', icon: Code2 },
    { id: 'ai', label: 'AI & ML', icon: Brain },
    { id: 'web', label: 'Web Development', icon: ExternalLink },
    { id: 'cybersecurity', label: 'Cybersecurity', icon: Shield },
    { id: 'database', label: 'Database', icon: Database },
    { id: 'cloud', label: 'Cloud Computing', icon: Cloud },
    { id: 'mobile', label: 'Mobile App', icon: Code2 },
    { id: 'desktop', label: 'Desktop App', icon: Code2 }
  ]

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

      <ProjectsClient projects={projects} categories={categories} />
    </div>
  )
}

interface Category {
  id: string
  label: string
  icon: typeof Code2
}

function ProjectsClient({
  projects,
  categories
}: {
  projects: SanityProjectCard[];
  categories: Category[]
}) {
  const [activeCategory, setActiveCategory] = useState('all')

  const filterProjects = (category: string) => {
    if (category === 'all') return projects
    return projects.filter(project => project.category === category)
  }

  return (
    <ScrollArea className="w-full">
      <Tabs defaultValue="all" onValueChange={setActiveCategory}>
        <TabsList className="inline-flex w-full md:w-auto">
          {categories.map(({ id, label, icon: Icon }) => (
            <TabsTrigger key={id} value={id} className="flex items-center gap-2">
              <Icon className="h-4 w-4" />
              {label}
            </TabsTrigger>
          ))}
        </TabsList>

        {categories.map(({ id }) => (
          <TabsContent key={id} value={id} className="mt-6">
            {filterProjects(id).length === 0 ? (
              <div className="text-center py-12 text-muted-foreground">
                No projects found in this category
              </div>
            ) : (
              <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
                {filterProjects(id).map(project => (
                  <ProjectCard key={project._id} project={project} />
                ))}
              </div>
            )}
          </TabsContent>
        ))}
      </Tabs>
    </ScrollArea>
  )
}

function ProjectCard({ project }: { project: SanityProjectCard }) {
  return (
    <Card className="overflow-hidden flex flex-col">
      {project.image && (
        <div className="aspect-video overflow-hidden">
          <img
            src={project.image}
            alt={project.title}
            className="w-full h-full object-cover transition-transform hover:scale-105"
          />
        </div>
      )}
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg">
            {project.title}
          </CardTitle>
          {project.featured && (
            <Badge variant="default" className="text-xs">
              Featured
            </Badge>
          )}
        </div>
      </CardHeader>
      <CardContent className="flex-grow space-y-4">
        <CardDescription className="line-clamp-3">
          {project.description || "A development project exploring various technologies and concepts."}
        </CardDescription>

        <div className="flex flex-wrap gap-2">
          {(project.technologies || []).map(tech => (
            <Badge key={tech} variant="secondary" className="capitalize">
              {tech}
            </Badge>
          ))}
          <Badge variant="outline" className="bg-primary/5 capitalize">
            {project.category.replace('-', ' ')}
          </Badge>
        </div>
      </CardContent>
      <CardFooter className="border-t bg-muted/50 pt-4">
        <div className="flex w-full justify-between">
          {project.github && (
            <a
              href={project.github}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors"
            >
              <Github className="h-4 w-4" />
              Source Code
            </a>
          )}
          {project.liveUrl && (
            <a
              href={project.liveUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors"
            >
              <ExternalLink className="h-4 w-4" />
              Live Demo
            </a>
          )}
        </div>
      </CardFooter>
    </Card>
  )
}
