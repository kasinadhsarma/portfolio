'use client'

import { useState } from "react"
import { ExternalLink, Github, Code2, Shield, Database, Cloud } from "lucide-react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Badge } from "@/components/ui/badge"
import { SanityProjectCard } from "@/types/sanity"
import { projectsClientPatterns as p, projectCardPatterns as pc } from "@/lib/responsive/pattrens/projects"

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
    <Tabs defaultValue="all" onValueChange={setActiveCategory}>
      <div className={p.tabsScrollWrapper}>
        <TabsList className={p.tabsList}>
          {categories.map(({ id, label, icon: Icon }) => (
            <TabsTrigger key={id} value={id} className={p.tabsTrigger}>
              <Icon className="h-4 w-4" />
              {label}
            </TabsTrigger>
          ))}
        </TabsList>
      </div>

      {categories.map(({ id }) => (
        <TabsContent key={id} value={id} className={p.tabsContent}>
          {filterProjects(id).length === 0 ? (
            <div className={p.emptyState}>
              No projects found in this category
            </div>
          ) : (
            <div className={p.grid}>
              {filterProjects(id).map(project => (
                <ProjectCard key={project._id} project={project} />
              ))}
            </div>
          )}
        </TabsContent>
      ))}
    </Tabs>
  )
}

function ProjectCard({ project }: { project: SanityProjectCard }) {
  // Map category values to display labels
  const getCategoryLabel = (categoryValue: string) => {
    const categoryMap: Record<string, string> = {
      'web': 'Web Development',
      'cybersecurity': 'Cybersecurity',
      'database': 'Database',
      'cloud': 'Cloud Computing',
      'mobile': 'Mobile App',
      'desktop': 'Desktop App',
      'other': 'Other'
    }
    return categoryMap[categoryValue] || categoryValue
  }

  return (
    <Card className={pc.card}>
      {project.image && (
        <div className={pc.imageWrapper}>
          <img
            src={project.image}
            alt={project.title}
            className={pc.image}
          />
        </div>
      )}
      <CardHeader>
        <div className={pc.headerRow}>
          <CardTitle className={pc.title}>
            {project.title}
          </CardTitle>
          {project.featured && (
            <Badge variant="default" className={pc.featuredBadge}>
              Featured
            </Badge>
          )}
        </div>
      </CardHeader>
      <CardContent className={pc.content}>
        <CardDescription className={pc.description}>
          {project.description || "A development project exploring various technologies and concepts."}
        </CardDescription>

        <div className={pc.techRow}>
          {(project.technologies || []).map(tech => (
            <Badge key={tech} variant="secondary" className={pc.techBadge}>
              {tech}
            </Badge>
          ))}
          {(Array.isArray(project.category) ? project.category : [project.category]).map(cat => (
            <Badge key={cat} variant="outline" className={pc.categoryBadge}>
              {getCategoryLabel(cat)}
            </Badge>
          ))}
        </div>
      </CardContent>
      <CardFooter className={pc.footer}>
        <div className={pc.footerRow}>
          {project.github && (
            <a
              href={project.github}
              target="_blank"
              rel="noopener noreferrer"
              className={pc.footerLink}
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
              className={pc.footerLink}
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
