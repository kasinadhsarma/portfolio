'use client'

import { useEffect, useState } from "react"
import Image from "next/image"
import { ExternalLink, Github, Star, Code2, Brain, Shield, Database, Cloud, Moon, Sun } from "lucide-react"
import { useTheme } from "next-themes"
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Badge } from "@/components/ui/badge"
import { GitHubRepo } from "@/types/github"
import { Skeleton } from "@/components/ui/skeleton"

export default function ProjectsPage() {
  const [projects, setProjects] = useState<GitHubRepo[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [activeCategory, setActiveCategory] = useState('all')

  const categories = [
    { id: 'all', label: 'All Projects', icon: Code2 },
    { id: 'ai', label: 'AI & ML', icon: Brain },
    { id: 'web', label: 'Web Development', icon: ExternalLink },
    { id: 'cybersecurity', label: 'Cybersecurity', icon: Shield },
    { id: 'database', label: 'Database', icon: Database },
    { id: 'cloud', label: 'Cloud Computing', icon: Cloud }
  ]

  const fetchProjects = async () => {
      try {
        const response = await fetch('/api/projects')
        if (!response.ok) {
          const error = await response.json()
          throw new Error(error.error || 'Failed to fetch projects')
        }
        const data = await response.json()
        if (!Array.isArray(data)) {
          throw new Error('Invalid response format')
        }
        setProjects(data)
        setError(null)
      } catch (error) {
        console.error('Failed to fetch projects:', error)
        setError(error instanceof Error ? error.message : 'Failed to load projects')
      } finally {
        setLoading(false)
      }
    }

  useEffect(() => {
    fetchProjects()
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const filterProjects = (category: string) => {
    if (category === 'all') return projects

    const categoryMappings: Record<string, string[]> = {
      'ai': ['ai', 'machine-learning', 'deep-learning', 'neural-networks'],
      'web': ['web', 'javascript', 'typescript', 'react', 'nextjs'],
      'cybersecurity': ['security', 'cybersecurity', 'hacking', 'encryption'],
      'database': ['database', 'sql', 'mongodb', 'postgresql'],
      'cloud': ['cloud', 'aws', 'azure', 'gcp']
    }

    return projects.filter(project => {
      const relevantTopics = categoryMappings[category] || [category]
      const topics = project.topics || []
      return topics.some(topic => 
        relevantTopics.includes(topic.toLowerCase())
      ) || project.language?.toLowerCase() === category.toLowerCase()
    })
  }

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

      {error ? (
        <div className="flex flex-col items-center justify-center py-12 text-center">
          <div className="text-destructive mb-4">
            <Shield className="h-10 w-10" />
          </div>
          <p className="text-lg font-semibold text-destructive">{error}</p>
          <button 
            onClick={() => {
              setError(null)
              setLoading(true)
              fetchProjects()
            }}
            className="mt-4 inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground"
          >
            Try again
          </button>
        </div>
      ) : loading ? (
        <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
          {[...Array(6)].map((_, i) => (
            <Card key={i} className="overflow-hidden">
              <CardHeader>http://localhost:3000/research
                <Skeleton className="h-6 w-2/3" />
              </CardHeader>
              <CardContent className="space-y-4">
                <Skeleton className="h-20" />
                <div className="flex flex-wrap gap-2">
                  {[...Array(3)].map((_, j) => (
                    <Skeleton key={j} className="h-6 w-16" />
                  ))}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      ) : (
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
                      <ProjectCard key={project.id} project={project} />
                    ))}
                  </div>
                )}
              </TabsContent>
            ))}
          </Tabs>
        </ScrollArea>
      )}
    </div>
  )
}

function ProjectCard({ project }: { project: GitHubRepo }) {
  return (
    <Card className="overflow-hidden flex flex-col">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg">
            {project.name.replace(/-/g, ' ')}
          </CardTitle>
          <div className="flex items-center gap-2">
            <Star className="h-4 w-4" />
            <span className="text-sm">{project.stargazers_count}</span>
          </div>
        </div>
      </CardHeader>
      <CardContent className="flex-grow space-y-4">
        <CardDescription className="line-clamp-3">
          {project.description || "A development project exploring various technologies and concepts."}
        </CardDescription>
        
        <div className="flex flex-wrap gap-2">
          {(project.topics || []).map(topic => (
            <Badge key={topic} variant="secondary" className="capitalize">
              {topic}
            </Badge>
          ))}
          {project.language && (
            <Badge variant="outline" className="bg-primary/5">
              {project.language}
            </Badge>
          )}
        </div>
      </CardContent>
      <CardFooter className="border-t bg-muted/50 pt-4">
        <div className="flex w-full justify-between">
          <a
            href={project.html_url}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors"
          >
            <Github className="h-4 w-4" />
            Source Code
          </a>
          {project.homepage && (
            <a
              href={project.homepage}
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
