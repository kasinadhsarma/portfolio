'use client'

import { useState } from "react"
import { ExternalLink, Github, Code2, Brain, Shield, Database, Cloud } from "lucide-react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Badge } from "@/components/ui/badge"
import { SanityProjectCard } from "@/types/sanity"

interface Category {
  id: string
  label: string
  icon: typeof Code2
}

export function ProjectsClient({ 
  projects
}: { 
  projects: SanityProjectCard[]
}) {
  const [activeCategory, setActiveCategory] = useState('all')

  const categories: Category[] = [
    { id: 'all', label: 'All Projects', icon: Code2 },
    { id: 'ai', label: 'AI & ML', icon: Brain },
    { id: 'web', label: 'Web Development', icon: ExternalLink },
    { id: 'cybersecurity', label: 'Cybersecurity', icon: Shield },
    { id: 'database', label: 'Database', icon: Database },
    { id: 'cloud', label: 'Cloud Computing', icon: Cloud },
    { id: 'mobile', label: 'Mobile App', icon: Code2 },
    { id: 'desktop', label: 'Desktop App', icon: Code2 }
  ]

  const filterProjects = (category: string) => {
    if (category === 'all') return projects
    return projects.filter(project => 
      Array.isArray(project.category) 
        ? project.category.includes(category)
        : project.category === category // backward compatibility
    )
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
          {(Array.isArray(project.category) ? project.category : [project.category]).map(cat => (
            <Badge key={cat} variant="outline" className="bg-primary/5 capitalize">
              {cat.replace('-', ' ')}
            </Badge>
          ))}
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